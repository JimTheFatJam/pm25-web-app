"""
backend/prediction/pipeline.py
============================================================================
Orchestrates the 7-stage prediction lifecycle, per the deployment plan's
Section 7 (narrowed to only the "group_D" config's real-inference path --
no SHAP, no job-queue infrastructure, per this task's explicit "do not
implement yet" list).

    [1/7] Validating request
    [2/7] Preparing selected cells
    [3/7] Loading static predictors
    [4/7] Retrieving dynamic GEE predictors
    [5/7] Building feature matrix
    [6/7] Running Baseline + Group D inference
    [7/7] Preparing prediction response
"""

from datetime import date, timezone

import pandas as pd

import config.settings as settings
from config.settings import DIAGNOSTIC_TEST_MAX_ROWS, MIN_PREDICTION_DATE
from backend.prediction import job_state, static_predictors, dynamic_predictors, feature_builder, inference
from backend.prediction.job_state import check_cancelled, set_stage, log


class PredictionRequestError(Exception):
    """A clean, user-facing validation failure -- distinct from an
    unexpected internal error. Caught separately in app.py so the
    frontend gets a specific, actionable message."""


def run_prediction_job(job_id, cell_ids, start_datetime, end_datetime, grid_gdf, loaded_model, temporal_resolution="hourly", availability=None):
    """The full pipeline, run in a background thread by job_state.run_job_in_background.

    Thin wrapper (behaviour-preserving) around run_prediction_core: the
    web job path binds the core's callbacks to job_state (stage/log/
    cancel), retains feature rows under the job id, and uses the existing
    per-timestamp GEE retrieval functions -- exactly what this function
    did before the core was extracted. The mobile path (backend/mobile.py)
    calls the core directly with its own callbacks/retrieval functions.
    """
    return run_prediction_core(
        cell_ids, start_datetime, end_datetime, grid_gdf, loaded_model,
        temporal_resolution=temporal_resolution, availability=availability,
        log_fn=lambda m: log(job_id, m),
        cancel_check=lambda: check_cancelled(job_id),
        set_stage_fn=lambda i: set_stage(job_id, i),
        retain_key=job_id,
    )


def run_prediction_core(cell_ids, start_datetime, end_datetime, grid_gdf, loaded_model, temporal_resolution="hourly",
                        availability=None, log_fn=print, cancel_check=lambda: None, set_stage_fn=lambda i: None,
                        retain_key=None, fetch_daily_fn=None, fetch_hourly_fn=None):
    """The full 7-stage pipeline, independent of the job registry.

    Callbacks: log_fn(message), cancel_check() (raise to abort),
    set_stage_fn(stage_index). retain_key: key under which the built
    feature rows are handed to explanation_store (None = don't retain).
    fetch_daily_fn / fetch_hourly_fn: retrieval functions with the exact
    signatures/return schemas of dynamic_predictors.fetch_daily_predictors
    and fetch_hourly_predictors; default to those functions (the web
    path), the mobile path injects the batched equivalents.

    Everything below this docstring is the previous run_prediction_job
    body with its job_state calls replaced by the callbacks -- no
    numerical step changed.
    grid_gdf: the full official grid GeoDataFrame (already loaded once at
    Flask startup by grid/grid_provider.py) -- used to look up
    grid_x_index/grid_y_index/centroid lat-lon for the requested cell_ids.

    availability: the dict returned by
    data_availability.check_dynamic_data_availability(), computed ONCE at
    app startup and passed in here (see backend/app.py) -- never
    recomputed per request. None if that startup check itself failed
    (e.g. GEE auth/network failure), in which case every request is
    refused with a clear error rather than silently allowing an
    unverified date range.

    start_datetime/end_datetime arrive already timezone-aware (Philippine
    local time, +08:00 -- the frontend's own established convention).
    They are deliberately NOT converted to UTC wholesale here: the
    min-date/year checks below intentionally stay in local time, since
    that's the calendar date the user actually selected on the date
    picker (converting to UTC first could shift the date across midnight
    and silently check the wrong day). Only what actually needs UTC --
    the hourly timestamp index used for GEE queries and returned to the
    frontend -- is explicitly converted.
    """
    if fetch_daily_fn is None:
        fetch_daily_fn = dynamic_predictors.fetch_daily_predictors
    if fetch_hourly_fn is None:
        fetch_hourly_fn = dynamic_predictors.fetch_hourly_predictors

    # ---------------- Stage 1: Validating request ----------------
    set_stage_fn(0)
    cancel_check()

    # Explicit UTC conversion, not passed as an extra tz= argument to
    # pd.date_range -- start_datetime/end_datetime are already
    # timezone-aware (+08:00), and pd.date_range raises
    # "Inferred time zone not equal to passed time zone" if given
    # tz-aware inputs alongside a DIFFERENT explicit tz kwarg. Converting
    # first, then calling date_range with no further tz argument, avoids
    # that conflict entirely -- confirmed directly by reproducing both
    # the failure and this fix against the exact +08:00 values the
    # frontend sends.
    start_datetime_utc = start_datetime.astimezone(timezone.utc)
    end_datetime_utc = end_datetime.astimezone(timezone.utc)

    n_cells = len(cell_ids)
    timestamps = pd.date_range(start_datetime_utc, end_datetime_utc, freq="h")
    n_timestamps = len(timestamps)
    n_rows = n_cells * n_timestamps
    log_fn(f"Cells = {n_cells}")
    log_fn(f"Timestamps = {n_timestamps}")
    log_fn(f"Prediction rows = {n_cells} x {n_timestamps} = {n_rows}")

    if n_rows > DIAGNOSTIC_TEST_MAX_ROWS:
        raise PredictionRequestError(
            f"Requested {n_rows} prediction rows ({n_cells} cells x {n_timestamps} timestamps), "
            f"which exceeds this diagnostic-testing stage's deliberately small ceiling of "
            f"{DIAGNOSTIC_TEST_MAX_ROWS} rows (see config.settings.DIAGNOSTIC_TEST_MAX_ROWS). "
            f"This is NOT the eventual production limit -- it exists only so a mistaken large "
            f"request can't trigger a huge number of live GEE calls before real stress testing "
            f"has set MAX_PREDICTION_ROWS. Reduce the number of cells or the time range."
        )

    min_date = date.fromisoformat(MIN_PREDICTION_DATE)
    if start_datetime.date() < min_date:
        raise PredictionRequestError(
            f"Requested start date {start_datetime.date()} is before MIN_PREDICTION_DATE "
            f"({min_date}) -- see the deployment plan, Section 3.1.1. The model was only "
            f"trained on data from this date onward."
        )

    # Replaces the earlier hardcoded PREDICTION_UNAVAILABLE_ZONE_DAYS = 5
    # ("provisional, ERA5T-derived") -- confirmed wrong: this pipeline's
    # ERA5/ERA5-Land collections have no faster-latency ERA5T variant on
    # GEE, and are only available up to roughly 3 months behind real
    # time. availability["latest_valid_date"] is the live-queried bound
    # across every dynamic dataset the model actually requires (see
    # data_availability.py), computed once at startup -- not recomputed
    # here.
    if availability is None:
        raise PredictionRequestError(
            "The dynamic-data availability check has not yet completed successfully -- either it is "
            "still running in the background (try again in a moment), or it failed at startup (see "
            "the backend console log for the specific GEE error, and restart the backend once fixed). "
            "Refusing rather than guessing a date range."
        )
    latest_valid_date = availability["latest_valid_date"]
    if end_datetime.date() > latest_valid_date:
        raise PredictionRequestError(
            f"Requested end date {end_datetime.date()} is after the latest date for which all "
            f"required dynamic predictors are currently available ({latest_valid_date}), "
            f"determined from live GEE data as of the last backend startup "
            f"(binding dataset: '{availability['binding_dataset']}'). Choose an earlier date range."
        )

    log_fn("Request validated.")

    # ---------------- Stage 2: Preparing selected cells ----------------
    set_stage_fn(1)
    cancel_check()

    cells_df = grid_gdf[grid_gdf["grid_id"].isin(cell_ids)][
        ["grid_id", "grid_x_index", "grid_y_index", "geometry"]
    ].copy()
    if len(cells_df) != len(cell_ids):
        found = set(cells_df["grid_id"])
        missing = [c for c in cell_ids if c not in found]
        raise PredictionRequestError(f"{len(missing)} requested cell ID(s) not found in the grid: {missing}")

    centroids_wgs84 = cells_df.geometry.centroid.to_crs("EPSG:4326")
    cells_df["centroid_longitude"] = centroids_wgs84.x.values
    cells_df["centroid_latitude"] = centroids_wgs84.y.values
    cells_df = cells_df.drop(columns=["geometry"])
    log_fn(f"Resolved {len(cells_df)} cell(s): {list(cells_df['grid_id'])}")

    prediction_year = start_datetime.year

    # ---------------- Stage 3: Loading static predictors ----------------
    set_stage_fn(2)
    cancel_check()

    population_df = static_predictors.load_population(cells_df, settings.ANALYSIS_CRS, prediction_year, log_fn=log_fn)
    coastline_df = static_predictors.load_coastline_distance(cells_df, settings.ANALYSIS_CRS, log_fn=log_fn)
    road_density_df = static_predictors.load_road_density(cells_df, log_fn=log_fn)
    gee_static_df = static_predictors.fetch_gee_static_layers(cells_df, settings.ANALYSIS_CRS, prediction_year, log_fn=log_fn)

    static_df = population_df.merge(coastline_df, on="grid_id").merge(road_density_df, on="grid_id").merge(gee_static_df, on="grid_id")
    log_fn(f"Static predictors ready for {len(static_df)} cell(s).")

    # ---------------- Stage 4: Retrieving dynamic GEE predictors ----------------
    set_stage_fn(3)
    cancel_check()

    daily_df = fetch_daily_fn(
        cells_df, settings.ANALYSIS_CRS, start_datetime_utc.date(), end_datetime_utc.date(),
        log_fn=log_fn, cancel_check=cancel_check,
    )
    hourly_df = fetch_hourly_fn(
        cells_df, settings.ANALYSIS_CRS, start_datetime_utc, end_datetime_utc,
        log_fn=log_fn, cancel_check=cancel_check,
    )

    # ---------------- Stage 5: Building feature matrix ----------------
    set_stage_fn(4)
    cancel_check()

    feature_matrix, diag = feature_builder.build_feature_matrix(
        cells_df, static_df, daily_df, hourly_df, timestamps,
        loaded_model["landcover_scheme_raw"], loaded_model["feature_order"],
        log_fn=log_fn,
    )

    # Application-layer plumbing only (SHAP explanations): hand the
    # already-built feature rows to the bounded retention store so a
    # single (cell, hour) row can be explained later on demand. This
    # neither reads back into nor alters the feature matrix, inference,
    # or the result; any failure is logged and ignored so the prediction
    # itself is never affected.
    try:
        from backend.prediction import explanation_store
        if retain_key is None:
            raise RuntimeError("no retain_key")
        outcome = explanation_store.attach(retain_key, feature_matrix, loaded_model["feature_order"])
        log_fn(f"Feature rows for SHAP explanations: {outcome['reason']} "
                    f"({outcome['rows']} row(s), {outcome['bytes'] / 1e6:.1f} MB).")
    except Exception as _e:  # noqa: BLE001 -- explanations are optional
        log_fn(f"Feature rows for SHAP explanations: not retained ({type(_e).__name__}: {_e}).")

    # ---------------- Stage 6: Running Baseline + Group D inference ----------------
    set_stage_fn(5)
    cancel_check()

    predictions_df = inference.predict(feature_matrix, loaded_model, log_fn=log_fn)

    # ---------------- Stage 7: Preparing prediction response ----------------
    set_stage_fn(6)
    cancel_check()

    # BUG FIX: temporal_resolution was silently ignored before -- every
    # request produced hourly output regardless of what the user
    # selected. The model itself always runs hourly internally (its
    # feature engineering -- rolling averages, Group D tendency features
    # -- genuinely requires hourly granularity, so this is NOT changed).
    # When "daily" is requested, the hourly predictions are averaged per
    # (cell, LOCAL calendar day) after inference completes, not before --
    # exactly matching how daily-average PM2.5 is conventionally reported
    # (mean of hourly concentrations). Grouped by Philippine local date,
    # not raw UTC date, to avoid a single Manila day splitting across two
    # UTC-date buckets near the day boundary.
    if temporal_resolution == "daily":
        log_fn(f"Averaging {len(predictions_df)} hourly prediction(s) into daily values "
                     f"(temporal_resolution=daily) -- grouped by Philippine local calendar date.")
        local_dt = predictions_df["datetime_utc"].dt.tz_convert(settings.LOCAL_TIMEZONE)
        predictions_df = predictions_df.copy()
        predictions_df["_local_date"] = local_dt.dt.date
        daily = (
            predictions_df.groupby(["grid_id", "_local_date"], as_index=False)["pm25_prediction"].mean()
        )
        # Representative timestamp per day: local midnight, converted back
        # to UTC -- consistent, unambiguous start-of-day reference rather
        # than an arbitrary hour, since this value no longer represents
        # any single hour's measurement.
        daily["datetime_utc"] = pd.to_datetime(daily["_local_date"]).dt.tz_localize(settings.LOCAL_TIMEZONE).dt.tz_convert("UTC")
        predictions_df = daily.drop(columns=["_local_date"])
        log_fn(f"Daily averaging complete: {len(predictions_df)} daily value(s).")

    predictions_df["datetime_utc"] = predictions_df["datetime_utc"].astype(str)
    predictions_df["pm25_prediction"] = predictions_df["pm25_prediction"].astype(float)
    # Explicit native-type conversion -- pandas/numpy types (int64,
    # float64) are NOT JSON-serializable by Flask's jsonify(), confirmed
    # directly (a bare numpy.int64 in a dict raises TypeError at
    # jsonify() time, not at construction time -- would have surfaced
    # only when the frontend polled a completed job, not caught by any
    # syntax check). record-orient dicts from a DataFrame can carry
    # numpy scalars in any column with an int/float dtype, so every
    # value is coerced explicitly here rather than trusting to_dict()'s
    # own output types.
    predictions_records = [
        {"grid_id": str(r["grid_id"]), "datetime_utc": str(r["datetime_utc"]), "pm25_prediction": float(r["pm25_prediction"])}
        for r in predictions_df.to_dict(orient="records")
    ]
    result = {
        "cell_ids": [str(c) for c in cells_df["grid_id"]],
        "timestamps": sorted(set(predictions_df["datetime_utc"].tolist())),
        "predictions": predictions_records,
        "row_count": int(len(predictions_df)),
        "rows_with_missing_values": int(diag["rows_with_missing_values"]),
        "temporal_resolution": temporal_resolution,
    }
    log_fn(f"Response prepared: {len(predictions_df)} prediction(s).")
    return result
