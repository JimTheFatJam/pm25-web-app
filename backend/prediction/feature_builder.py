"""
backend/prediction/feature_builder.py
============================================================================
Assembles the exact Baseline + Group D (157-feature) feature matrix for a
prediction request, from the static + dynamic predictor data already
fetched (see static_predictors.py, dynamic_predictors.py).

Per explicit instruction ("do not invent alternative feature
calculations; reuse the validated research-pipeline logic wherever
possible"), every transformation below CALLS the actual function from
the actual research pipeline module -- rolling averages, cyclical time
encoding, land-cover one-hot encoding, calendar/dry-days features,
physics features, wind-transport features, and Group D's tendency
features are none of them reimplemented here. This module's own code is
only the GLUE that gets raw fetched values into the shape those functions
expect, and reorders/validates the final result against the model's own
schema.

Requires config.settings.RESEARCH_PIPELINE_DIR to be set and importable
-- see model_loader.py, which performs that check at startup before this
module is ever called.
"""

import re

import numpy as np
import pandas as pd


def build_feature_matrix(
    cells_df, static_df, daily_df, hourly_df, requested_timestamps,
    landcover_scheme_raw, feature_order, log_fn=print,
):
    """
    cells_df: grid_id, grid_x_index, grid_y_index, centroid_latitude, centroid_longitude
    static_df: grid_id + elevation/landcover/ndvi/nighttime_lights (local + neighborhood) + population + distance_to_coast + road_density (local + neighborhood)
    daily_df: grid_id, date, aod/no2/so2/co/o3/hcho (local + neighborhood) -- includes the 7-day lookback window
    hourly_df: grid_id, datetime_utc, ERA5 variables + PBLH (local + neighborhood) -- includes the 24-hour lookback window
    requested_timestamps: the exact list of pandas Timestamps (UTC) the user actually requested (NOT including the lookback buffer)

    Returns (feature_df, diagnostic_info) where feature_df has exactly
    feature_order's columns in that exact order, one row per
    (grid_id, requested timestamp), plus grid_id/datetime_utc for joining
    back to the response.
    """
    # Imported here, not at module level, so a missing RESEARCH_PIPELINE_DIR
    # produces model_loader.py's clear startup error rather than an import
    # error at a random point in the request lifecycle.
    from prepare_features import (
        add_hourly_rolling_features, add_daily_rolling_features,
        one_hot_encode_landcover, add_cyclical_time_features,
        clip_negative_gas_values, apply_bidirectional_fallback,
        HOURLY_ERA5_VARS, DAILY_DYNAMIC_VARS,
    )
    from run_physics_transport_experiment import (
        add_physics_features, add_wind_transport_features, build_cell_adjacency,
        ADJACENCY_RADIUS_KM,
    )
    from build_additional_features import build_calendar_features, compute_consecutive_dry_days
    from advanced_feature_engineering_experiment import build_group_d
    from extract_era5_hourly_batched import derive_additional_features

    log_fn(f"Assembling feature matrix for {len(cells_df)} cell(s) x {len(requested_timestamps)} timestamp(s)...")

    # --------------------------------------------------------------
    # 1. Merge hourly (hourly_df already covers the 24h lookback) with
    #    daily (broadcast each day's value across its 24 hourly rows,
    #    matching prepare_features.py's own established merge-key
    #    pattern of "date" derived from the hourly datetime).
    # --------------------------------------------------------------
    hourly_df = hourly_df.copy()
    hourly_df["date"] = pd.to_datetime(hourly_df["datetime_utc"]).dt.date
    merged = hourly_df.merge(daily_df, on=["grid_id", "date"], how="left")

    # BUG FIX, confirmed by direct reproduction: a GEE-sourced column can
    # end up as object dtype even when every individual value is a valid
    # Python float or None (no lists, no strings) -- a pandas type-
    # inference quirk, plausibly triggered here by an entire hour's worth
    # of consecutive None values (a genuine data gap for that one hour,
    # confirmed in this same run's own log: "756 missing" = every cell x
    # every variable for that hour). np.exp() -- used inside the
    # Magnus-Tetens relative-humidity formula in derive_additional_features(),
    # called immediately below -- fails on an object-dtype array with
    # exactly "loop of ufunc does not support argument 0 of type float
    # which has no callable exp method", even though every value is a
    # real float: numpy's ufunc dispatch on object arrays tries calling
    # .exp() as a METHOD on each element, and a plain Python float has no
    # such method. The earlier coercion below (before clip/fallback) ran
    # too late to protect this step -- this one runs right after the
    # initial merge, before ANYTHING touches these columns numerically.
    for _var in HOURLY_ERA5_VARS + DAILY_DYNAMIC_VARS:
        for _scale in ("local", "neighborhood_mean"):
            _col = f"{_var}_{_scale}"
            if _col in merged.columns:
                merged[_col] = pd.to_numeric(merged[_col], errors="coerce")

    # --------------------------------------------------------------
    # 2. Derive relative_humidity / wind_speed_10m (Magnus-Tetens +
    #    magnitude) -- reused exactly via import, not reimplemented.
    # --------------------------------------------------------------
    merged = derive_additional_features(merged)
    log_fn("  Derived relative_humidity / wind_speed_10m (local + neighborhood).")

    # --------------------------------------------------------------
    # 2.5. BUG FIX -- this entire stage was missing before. Confirmed
    #    directly from prepare_features.py's own main() sequence: Stage
    #    1.5 (clip negative gas/AOD values) and Stage 2 (bidirectional
    #    local<->neighborhood fallback) MUST run before Stage 3 (rolling
    #    averages), in that exact order -- otherwise a rolling mean would
    #    still be pulled down by retrieval noise clipping is meant to
    #    remove, and a fallback substitution could copy a negative value
    #    across scales. This is also the direct source of the 15
    #    "<var>_local_filled_from_neighborhood" indicator columns the
    #    model expects (confirmed via FALLBACK_VARS = HOURLY_ERA5_VARS +
    #    DAILY_DYNAMIC_VARS + ["ndvi", "population"]) -- only the hourly/
    #    daily portion runs here, since ndvi/population aren't merged in
    #    until static_df is (step 5 below); the fallback function
    #    processes each variable independently, so splitting this call in
    #    two produces an identical result to one combined call.
    # --------------------------------------------------------------
    # BUG FIX: GEE-sourced columns (built row-by-row in dynamic_predictors.py
    # from values that can be a real float or Python None depending on
    # per-pixel GEE results) can end up as object dtype rather than a
    # clean float64, depending on the exact mix/order of values received.
    # clip_negative_gas_values/apply_bidirectional_fallback then hit
    # "Invalid value '[]' for dtype 'float64'" on such a column in some
    # pandas versions -- confirmed reproducible with a deliberately
    # object-dtype input, and confirmed this coercion neutralizes it
    # regardless of the exact mechanism. pd.to_numeric(errors="coerce")
    # forces a clean float64 (any truly non-numeric value becomes NaN,
    # not a crash) before these steps ever see the data.
    for _var in HOURLY_ERA5_VARS + DAILY_DYNAMIC_VARS:
        for _scale in ("local", "neighborhood_mean"):
            _col = f"{_var}_{_scale}"
            if _col in merged.columns:
                merged[_col] = pd.to_numeric(merged[_col], errors="coerce")

    merged = clip_negative_gas_values(merged)
    merged = apply_bidirectional_fallback(merged, HOURLY_ERA5_VARS + DAILY_DYNAMIC_VARS)
    log_fn("  Clipped negative gas/AOD values and applied local<->neighborhood fallback (hourly/daily variables).")

    # --------------------------------------------------------------
    # 3. Rolling averages -- 28 hourly (7 vars x 2 scales x 2 windows) +
    #    24 daily (6 vars x 2 scales x 2 windows) = 52 columns, computed
    #    by the research pipeline's own functions, not re-derived.
    # --------------------------------------------------------------
    merged = merged.sort_values(["grid_id", "datetime_utc"])
    merged = add_hourly_rolling_features(merged)
    merged = add_daily_rolling_features(merged)
    log_fn("  Computed hourly (3h/24h) and daily (3d/7d) rolling averages.")

    # --------------------------------------------------------------
    # 4. Cyclical time encoding.
    # --------------------------------------------------------------
    merged = add_cyclical_time_features(merged)

    # --------------------------------------------------------------
    # 5. Merge in static predictors (elevation, land cover, NDVI,
    #    nighttime lights, population, distance to coast, road density).
    # --------------------------------------------------------------
    merged = merged.merge(static_df, on="grid_id", how="left")

    # Second half of the same fallback fix (see step 2.5 above) -- ndvi
    # and population are also in FALLBACK_VARS, but weren't available
    # until just now. Same function, same logic, just called once these
    # two variables' local/neighborhood_mean columns actually exist.
    for _var in ("ndvi", "population"):
        for _scale in ("local", "neighborhood_mean"):
            _col = f"{_var}_{_scale}"
            if _col in merged.columns:
                merged[_col] = pd.to_numeric(merged[_col], errors="coerce")
    merged = apply_bidirectional_fallback(merged, ["ndvi", "population"])
    log_fn("  Applied local<->neighborhood fallback (NDVI, population).")

    # --------------------------------------------------------------
    # 6. Land cover one-hot encoding, using the EXACT scheme the model
    #    was trained with (landcover_encoding_scheme.txt, loaded at
    #    startup by model_loader.py) -- never re-derived from whatever
    #    classes happen to appear in this request's own small cell set,
    #    which would silently produce a different (wrong) set of one-hot
    #    columns than the model expects.
    # --------------------------------------------------------------
    if landcover_scheme_raw is None:
        raise FeatureBuildError(
            "landcover_encoding_scheme.txt was not found at model load time -- cannot apply "
            "land-cover one-hot encoding consistently with how the model was trained. "
            "See the README.txt beside the deployed model artifacts (config.settings.MODEL_DIR)."
        )
    scheme = _parse_landcover_scheme(landcover_scheme_raw)
    merged = one_hot_encode_landcover(merged, scheme)
    log_fn("  Applied land-cover one-hot encoding using the trained model's own scheme.")

    # --------------------------------------------------------------
    # 7. Calendar / dry-days features (9 columns) -- computed on each
    #    row's LOCAL date, matching the research pipeline's own
    #    established correction (not UTC).
    # --------------------------------------------------------------
    merged = merged.merge(cells_df[["grid_id", "centroid_latitude", "centroid_longitude"]], on="grid_id", how="left")
    # BUG FIX, caught by re-inspecting build_calendar_features' own
    # implementation directly: it calls add_local_datetime(df,
    # timezone_lookup), which does df.merge(timezone_lookup, ...) --
    # passing None (as this code did) crashes immediately with a
    # TypeError the moment a real request reaches this stage, not caught
    # by any test so far since no request had gotten this far yet. This
    # deployment is Philippines-only, so every cell maps to the same
    # single timezone -- matching config.settings.LOCAL_TIMEZONE_NAME
    # ("Asia/Manila"), not a hardcoded string independent of that setting.
    import config.settings as _settings
    timezone_lookup = pd.DataFrame({
        "grid_id": cells_df["grid_id"].unique(),
        "timezone": _settings.LOCAL_TIMEZONE_NAME,
    })
    # build_calendar_features ALSO returns only a fresh DataFrame
    # (grid_id, datetime_utc, + 8 calendar columns) -- confirmed directly
    # from its own return statement -- not the full input DataFrame back.
    # Same fix as compute_consecutive_dry_days below: merge the result
    # in, don't assign over `merged`, or every other already-computed
    # feature (rolling averages, static predictors, land cover encoding)
    # would be silently destroyed.
    calendar_result = build_calendar_features(merged, country_code="PH", timezone_lookup=timezone_lookup)
    # BUG FIX, confirmed by direct reproduction: build_calendar_features'
    # own return statement builds its result via
    # pd.DataFrame({"datetime_utc": df["datetime_utc"].values, ...}) --
    # .values on a tz-aware Series, wrapped in a dict-based DataFrame
    # constructor, silently strips the UTC tz label (confirmed: dtype
    # goes from "datetime64[us, UTC]" to "datetime64[us]"). The
    # underlying wall-clock values are still correct UTC times, just
    # missing the label -- tz_localize (not tz_convert) re-attaches it
    # without changing the actual instant, making this side mergeable
    # again against `merged`'s still-tz-aware datetime_utc column.
    calendar_result["datetime_utc"] = pd.to_datetime(calendar_result["datetime_utc"]).dt.tz_localize("UTC")
    calendar_cols = [c for c in calendar_result.columns if c not in ("grid_id", "datetime_utc")]
    merged = merged.merge(calendar_result[["grid_id", "datetime_utc"] + calendar_cols],
                           on=["grid_id", "datetime_utc"], how="left")
    # BUG FIX, caught by re-inspecting compute_consecutive_dry_days'
    # actual signature directly: it requires THREE arguments
    # (existing_features_df, existing_metadata_df, timezone_lookup), not
    # two -- the timezone_lookup argument was missing entirely, which
    # would have raised a TypeError the moment a real request reached
    # this stage. existing_features_df must carry grid_id +
    # precipitation_local (used internally to bucket by local calendar
    # day per the WMO dry-day definition); existing_metadata_df must
    # carry datetime_utc, same row count/order (concatenated
    # positionally inside the function, not merged by key).
    # compute_consecutive_dry_days returns ONLY grid_id/datetime_utc/
    # consecutive_dry_days (confirmed directly from its own return
    # statement) -- NOT the full input DataFrame back. Must merge this
    # result back into `merged`, not assign over it, or every other
    # feature already computed above (rolling averages, cyclical
    # encoding, static predictors) would be silently destroyed.
    dry_days_result = compute_consecutive_dry_days(
        merged[["grid_id", "precipitation_local"]], merged[["datetime_utc"]], timezone_lookup,
    )
    merged = merged.merge(dry_days_result[["grid_id", "datetime_utc", "consecutive_dry_days"]],
                           on=["grid_id", "datetime_utc"], how="left")
    log_fn("  Built calendar/dry-days features (9 columns).")

    # --------------------------------------------------------------
    # 8. Physics + wind-transport features -- adjacency computed fresh
    #    for just this request's cells (cheap at this scale; the
    #    production version would reuse a precomputed/cached adjacency
    #    table for the full grid, per the deployment plan's Section 4,
    #    explicitly deferred for this diagnostic stage).
    # --------------------------------------------------------------
    centroids = cells_df[["grid_id", "centroid_latitude", "centroid_longitude"]].drop_duplicates(subset=["grid_id"])
    adjacency = build_cell_adjacency(centroids, ADJACENCY_RADIUS_KM)
    merged = add_physics_features(merged)
    merged = add_wind_transport_features(merged, adjacency)
    log_fn("  Built physics (5) and wind-transport (6) features.")

    # --------------------------------------------------------------
    # 9. Group D tendency features (5 columns) -- the actual deployment
    #    target's own defining addition on top of Baseline.
    # --------------------------------------------------------------
    merged = build_group_d(merged)
    log_fn("  Built Group D tendency features (5 columns).")

    # --------------------------------------------------------------
    # 10. Restrict to only the ACTUALLY requested timestamps (drop the
    #     lookback-only rows now that rolling/tendency features have
    #     been computed from them) and reorder columns to match the
    #     model's exact schema.
    # --------------------------------------------------------------
    merged["datetime_utc"] = pd.to_datetime(merged["datetime_utc"])
    requested_set = set(pd.to_datetime(requested_timestamps))
    result = merged[merged["datetime_utc"].isin(requested_set)].copy()

    missing_cols = [c for c in feature_order if c not in result.columns]
    if missing_cols:
        raise FeatureBuildError(
            f"Feature construction did not produce {len(missing_cols)} column(s) the model "
            f"expects: {missing_cols}. This means the request-scoped feature pipeline has "
            f"drifted from the training pipeline's actual output -- do not proceed with "
            f"inference using a misaligned feature set."
        )

    feature_matrix = result[["grid_id", "datetime_utc"] + feature_order].reset_index(drop=True)
    n_rows_with_nan = feature_matrix[feature_order].isna().any(axis=1).sum()
    log_fn(f"  Feature matrix assembled: {len(feature_matrix)} rows x {len(feature_order)} features "
           f"({n_rows_with_nan} row(s) contain at least one missing value).")

    diagnostic_info = {
        "rows": len(feature_matrix),
        "columns": len(feature_order),
        "rows_with_missing_values": int(n_rows_with_nan),
    }
    return feature_matrix, diagnostic_info


def _parse_landcover_scheme(raw_text):
    """Correctly parses landcover_encoding_scheme.txt, per
    prepare_features.py's own report-writing code (confirmed by reading
    it directly, then verified by round-tripping a simulated report
    through this parser and checking the result matches exactly).

    The real scheme dict structure -- confirmed directly from
    one_hot_encode_landcover()'s own implementation, not the earlier
    integration-testing guess -- is PER-COLUMN, not a single flat list:
    {column_name: [sorted list of class codes kept separate for that
    column]}, e.g. {"landcover_local": [10, 12, 17],
    "landcover_neighborhood_mean": [13, 17]}. Different columns can (and
    in practice do) retain different classes, since the decision is based
    on each class's own prevalence in that specific column.

    The .txt file itself is a human-readable report (padded columns,
    percentage strings, "KEEP"/"-> other" decision text) written by
    prepare_features.py for a person to read -- not a clean data format.
    This parser extracts the same decisions the report displays, rather
    than assuming a simpler format that was never actually confirmed
    against the file this project produces.

    Format read directly from the writing code:
      "{column_name}:"                                  -- starts a new column section
      "  {code:>2} ({label:<32}) {pct_str:<30} {decision}"  -- one class per line, decision is "KEEP" or "-> other"
    """
    scheme = {}
    current_col = None
    col_header_re = re.compile(r"^(\S.*):$")
    entry_re = re.compile(r"^\s+(\d+)\s+\(.*\).*\s(KEEP|-> other)\s*$")

    for line in raw_text.splitlines():
        header_match = col_header_re.match(line)
        if header_match:
            current_col = header_match.group(1)
            scheme[current_col] = []
            continue
        entry_match = entry_re.match(line)
        if entry_match and current_col is not None:
            code = int(entry_match.group(1))
            decision = entry_match.group(2)
            if decision == "KEEP":
                scheme[current_col].append(code)

    if not scheme:
        raise FeatureBuildError(
            "Could not parse any column sections from landcover_encoding_scheme.txt. "
            "Expected lines like 'landcover_local:' followed by indented class entries "
            "ending in 'KEEP' or '-> other' -- check the actual file's format against "
            "_parse_landcover_scheme() in feature_builder.py if prepare_features.py's own "
            "report-writing code has since changed."
        )
    for col in scheme:
        scheme[col] = sorted(scheme[col])
    return scheme


class FeatureBuildError(Exception):
    """Raised when the feature matrix cannot be assembled correctly or
    does not match the model's expected schema."""
