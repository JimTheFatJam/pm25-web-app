"""
backend/prediction/dynamic_predictors.py
============================================================================
Fetches the genuinely time-varying predictors from GEE for a specific set
of requested cells and timestamps: MODIS AOD, Sentinel-5P gases (NO2, SO2,
CO, O3, HCHO), and ERA5-Land/ERA5 meteorology (with the ocean fallback and
PBLH-from-ERA5 pattern the research pipeline already established).

This is necessarily NEW code -- extract_daily_predictors_batched.py and
extract_era5_hourly_batched.py are designed for batch processing over the
full grid and study period, not a single on-demand request. However,
every dataset ID, band name, and the local/neighborhood extraction
pattern (neighborhood_rectangle, radius=2 cells) are reused EXACTLY as
confirmed from those scripts, not re-chosen or approximated.

PERFORMANCE FIX (this revision): the previous version called
reduceRegions() once per (variable, timestamp, scope) combination --
confirmed directly, this came to ~768 individual sequential GEE API
calls for a single 4-cell, 24-hour request (96 for the 6 daily
variables x ~8 lookback-extended days x 2 scopes, 672 for the 7 hourly
variables x 48 lookback-extended hours x 2 scopes). Each call is its own
synchronous round-trip, building and serializing a separate computation
graph -- this was the direct, confirmed cause of both the slow runtime
and the backend process's high memory usage reported after a
prediction (hundreds of sequential ee.Image/ee.FeatureCollection object
graphs and their HTTP request/response buffers, most of which Python's
own GC should reclaim between calls, but not all of it did in practice).

Fixed by combining all of a timestamp's variables into a SINGLE
multi-band ee.Image (via .rename() + ee.Image.cat()) before calling
reduceRegions() once per (timestamp, scope) -- GEE reduces every band
of a multi-band image in one pass, returning one output property per
band. This cuts the call count to ~112 (16 for daily: ~8 dates x 2
scopes; 96 for hourly: 48 hours x 2 scopes) -- a ~7x reduction -- while
returning identical values, since the underlying reducer and pixels
being reduced are unchanged, only how many separate network calls are
needed to retrieve them.

Rolling-average lookback: per the deployment plan's Section 3.1, a
request's daily rolling-average features need up to 7 days of gas/AOD
data before the requested start, and hourly rolling/tendency features
need up to 24 hours of meteorology before the requested start. This
module fetches that lookback buffer whenever the requested start date is
far enough from MIN_DATA_AVAILABILITY_DATE to allow it; if not, it
fetches as much as is available and lets the research pipeline's own
rolling functions (min_periods=1, confirmed from prepare_features.py)
degrade gracefully rather than fail, exactly matching the training
pipeline's own established behavior for short lookback windows.
"""

from datetime import datetime, timedelta

import ee
import pandas as pd
import geopandas as gpd

from config.settings import CELL_SIDE_METERS, DISPLAY_CRS

NEIGHBORHOOD_RADIUS_CELLS = 2

# TEMPORARY DIAGNOSTIC FLAG -- part of investigating the "0 ok / N missing
# (all N variables combined)" reports. When True, prints per-variable
# (not just combined) valid/null counts, separately for local and
# neighborhood, for a bounded number of early batches only -- never for
# every cell/hour, to avoid flooding the console. Intended to be removed
# or set back to False once the investigation concludes. Does not change
# any retrieval behavior, feature values, or the model's inputs -- purely
# additive logging.
DEBUG_DYNAMIC_RETRIEVAL = False
DEBUG_MAX_BATCHES = 2
_debug_batch_count = {"n": 0}
_debug_raw_response_printed = {"done": False}
_debug_stage_trace_printed = {"done": False}

DAILY_ROLLING_LOOKBACK_DAYS = 7   # confirmed from prepare_features.py's DAILY_ROLLING_WINDOWS
HOURLY_ROLLING_LOOKBACK_HOURS = 24  # confirmed from prepare_features.py's HOURLY_ROLLING_WINDOWS

# Confirmed dataset IDs/bands from extract_daily_predictors_batched.py
MODIS_AOD_COLLECTION = "MODIS/061/MCD19A2_GRANULES"
MODIS_AOD_BAND = "Optical_Depth_047"
S5P_GASES = {
    "no2": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density"),
    "so2": ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density"),
    "co": ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density"),
    "o3": ("COPERNICUS/S5P/OFFL/L3_O3", "O3_column_number_density"),
    "hcho": ("COPERNICUS/S5P/OFFL/L3_HCHO", "tropospheric_HCHO_column_number_density"),
}

# Confirmed from extract_era5_hourly_batched.py
ERA5_LAND_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
ERA5_COLLECTION = "ECMWF/ERA5/HOURLY"
ERA5_LAND_BANDS = {
    "temperature_2m": "temperature_2m",
    "dewpoint_2m": "dewpoint_temperature_2m",  # confirmed against extract_era5_hourly_batched.py's own primary ERA5_BANDS dict
    "surface_pressure": "surface_pressure",
    "u_wind_10m": "u_component_of_wind_10m",
    "v_wind_10m": "v_component_of_wind_10m",
    "precipitation": "total_precipitation_hourly",
}
ERA5_FALLBACK_BANDS = {
    "temperature_2m": "temperature_2m",
    "dewpoint_2m": "dewpoint_temperature_2m",
    "surface_pressure": "surface_pressure",
    "u_wind_10m": "u_component_of_wind_10m",
    "v_wind_10m": "v_component_of_wind_10m",
    # "precipitation" deliberately NOT included here -- see PRECIP_RATE_BAND below
}
# Precipitation's fallback: ERA5's rate band, converted to match
# ERA5-Land's accumulated-meters-per-hour units. Exact formula confirmed
# directly from extract_era5_hourly_batched.py's own documented
# derivation (ECMWF Copernicus User Support: rate and accumulated
# parameters represent "the same information... except the units are
# different"): 1 kg/m^2 of water = 1mm depth, so
# rate_kg_per_m2_per_s * 3600 seconds/hour / 1000 mm-per-meter = rate * 3.6
# meters accumulated per hour.
PRECIP_RATE_BAND = "mean_total_precipitation_rate"
PRECIP_RATE_TO_METERS_PER_HOUR = 3.6
ERA5_PBLH_BAND = "boundary_layer_height"  # from ERA5, not ERA5-Land -- confirmed, ERA5-Land does not carry this variable


def _local_geom(grid_x, grid_y, grid_crs):
    import shapely.geometry
    x0, y0 = grid_x * CELL_SIDE_METERS, grid_y * CELL_SIDE_METERS
    rect = gpd.GeoSeries([shapely.geometry.box(x0, y0, x0 + CELL_SIDE_METERS, y0 + CELL_SIDE_METERS)], crs=grid_crs)
    return rect.to_crs(DISPLAY_CRS).iloc[0]


def _nbhd_geom(grid_x, grid_y, grid_crs):
    import shapely.geometry
    r = NEIGHBORHOOD_RADIUS_CELLS
    x0, y0 = (grid_x - r) * CELL_SIDE_METERS, (grid_y - r) * CELL_SIDE_METERS
    x1, y1 = (grid_x + r + 1) * CELL_SIDE_METERS, (grid_y + r + 1) * CELL_SIDE_METERS
    rect = gpd.GeoSeries([shapely.geometry.box(x0, y0, x1, y1)], crs=grid_crs)
    return rect.to_crs(DISPLAY_CRS).iloc[0]


def _build_feature_collections(cells_df, grid_crs):
    local_feats, nbhd_feats = [], []
    for row in cells_df.itertuples():
        local_feats.append(ee.Feature(ee.Geometry.Rectangle(list(_local_geom(row.grid_x_index, row.grid_y_index, grid_crs).bounds)), {"grid_id": row.grid_id}))
        nbhd_feats.append(ee.Feature(ee.Geometry.Rectangle(list(_nbhd_geom(row.grid_x_index, row.grid_y_index, grid_crs).bounds)), {"grid_id": row.grid_id}))
    return ee.FeatureCollection(local_feats), ee.FeatureCollection(nbhd_feats)


def _daily_mean_band(coll, band, day_start, day_end, var_name):
    """One day's mean of a single-band collection, with its native
    projection explicitly preserved through the .mean() composite --
    see fetch_daily_predictors' PROJECTION FIX comment for why this is
    necessary and where the technique is confirmed from. If the
    collection has no image at all for this day, .first() below is
    empty and the resulting image is (correctly) fully masked -- the
    existing _reduce_multiband_into error handling already covers this."""
    filtered = ee.ImageCollection(coll).select(band).filterDate(day_start, day_end)
    native_proj = ee.Image(filtered.first()).select(band).projection()
    return filtered.mean().rename(var_name).setDefaultProjection(native_proj)


def fetch_daily_predictors(cells_df, grid_crs, start_date, end_date, log_fn=print, cancel_check=None):
    """Fetches AOD + 5 Sentinel-5P gases for [start_date - 7 days, end_date],
    local + neighborhood. One reduceRegions() call per (date, scope) --
    all 6 variables combined into a single multi-band image first, not
    one call per variable (see module docstring's performance-fix note).
    Returns a long DataFrame: grid_id, date, {var}_local, {var}_neighborhood_mean.
    """
    fetch_start = start_date - timedelta(days=DAILY_ROLLING_LOOKBACK_DAYS)
    dates = pd.date_range(fetch_start, end_date, freq="D").date
    log_fn(f"Daily predictors: fetching {len(dates)} distinct date(s) from {fetch_start} to {end_date} "
           f"(includes {DAILY_ROLLING_LOOKBACK_DAYS}-day rolling-average lookback before the requested start).")

    local_fc, nbhd_fc = _build_feature_collections(cells_df, grid_crs)
    all_rows = {(gid, d): {"grid_id": gid, "date": d} for gid in cells_df["grid_id"] for d in dates}

    variables = [("aod", MODIS_AOD_COLLECTION, MODIS_AOD_BAND)] + \
                [(name, coll, band) for name, (coll, band) in S5P_GASES.items()]

    for i, d in enumerate(dates):
        if cancel_check:
            cancel_check()
        day_start = ee.Date(str(d))
        day_end = day_start.advance(1, "day")

        if DEBUG_DYNAMIC_RETRIEVAL and i == 0:
            # DIAGNOSTIC: same check as fetch_hourly_predictors, for the
            # daily path's own date construction (str(d) on a plain
            # datetime.date -- no time or timezone component at all,
            # structurally different from the hourly path's
            # pandas-Timestamp-with-offset construction).
            gee_interpretation = day_start.format().getInfo()
            log_fn(f"    [DEBUG] date construction check: d={d!r}, str(d)={str(d)!r}")
            log_fn(f"    [DEBUG] ee.Date(str(d)).format().getInfo() = {gee_interpretation!r} "
                   f"(compare to the intended {str(d)!r})")

        # Combine all 6 variables into one multi-band image -- one
        # reduceRegions() call retrieves every variable's value at once,
        # instead of one call per variable.
        #
        # PROJECTION FIX: .mean() compositing silently discards the source
        # images' real projection, defaulting to a degenerate identity
        # transform -- confirmed directly in extract_era5_hourly_batched.py
        # (see get_hourly_image_generic's docstring there) to make
        # reduceRegions() return null for every feature, even though a
        # single reduceRegion() call would still tolerate it. Fixed the
        # same way here: capture native_proj from a raw image BEFORE
        # compositing, then explicitly reattach it via
        # .setDefaultProjection() afterward.
        band_images = [
            _daily_mean_band(coll, band, day_start, day_end, var_name)
            for var_name, coll, band in variables
        ]
        combined_img = ee.Image.cat(band_images)

        var_names = [v[0] for v in variables]
        n_ok_local, n_missing_local = _reduce_multiband_into(combined_img, local_fc, all_rows, d, var_names, "local", log_fn)
        n_ok_nbhd, n_missing_nbhd = _reduce_multiband_into(combined_img, nbhd_fc, all_rows, d, var_names, "neighborhood_mean", log_fn)
        log_fn(f"  {d} (batch {i + 1}/{len(dates)}): local {n_ok_local} ok / {n_missing_local} missing, "
               f"neighborhood {n_ok_nbhd} ok / {n_missing_nbhd} missing (all {len(variables)} variables combined)")
        if DEBUG_DYNAMIC_RETRIEVAL:
            _debug_batch_count["n"] += 1

    df = pd.DataFrame(list(all_rows.values()))
    return df


def fetch_hourly_predictors(cells_df, grid_crs, start_dt, end_dt, log_fn=print, cancel_check=None):
    """Fetches the 6 ERA5-Land variables + PBLH (from ERA5) for
    [start_dt - 24h, end_dt], local + neighborhood, with the ocean
    fallback pattern (ERA5-Land -> ERA5 where ERA5-Land is null) confirmed
    from extract_era5_hourly_batched.py. One reduceRegions() call per
    (hour, scope) -- all 7 variables combined into a single multi-band
    image first, not one call per variable (see module docstring's
    performance-fix note)."""
    fetch_start = start_dt - timedelta(hours=HOURLY_ROLLING_LOOKBACK_HOURS)
    hours = pd.date_range(fetch_start, end_dt, freq="h")
    log_fn(f"Hourly predictors: fetching {len(hours)} distinct hour(s) from {fetch_start} to {end_dt} "
           f"(includes {HOURLY_ROLLING_LOOKBACK_HOURS}-hour rolling/tendency lookback before the requested start).")

    local_fc, nbhd_fc = _build_feature_collections(cells_df, grid_crs)
    all_rows = {(gid, h): {"grid_id": gid, "datetime_utc": h} for gid in cells_df["grid_id"] for h in hours}

    var_names = list(ERA5_LAND_BANDS.keys()) + ["boundary_layer_height"]

    for i, h in enumerate(hours):
        if cancel_check:
            cancel_check()
        hour_start = ee.Date(h.isoformat())

        if DEBUG_DYNAMIC_RETRIEVAL and i == 0:
            # DIAGNOSTIC: does GEE parse this exact date string the way we
            # intend? h.isoformat() (a pandas Timestamp) includes an
            # explicit UTC offset suffix ("+00:00") that GEE's own ee.Date()
            # documentation examples never show embedded in the string --
            # confirming or ruling this out directly, with real GEE,
            # rather than assuming either way.
            gee_interpretation = hour_start.format().getInfo()
            log_fn(f"    [DEBUG] date construction check: h={h!r}, h.isoformat()={h.isoformat()!r}")
            log_fn(f"    [DEBUG] ee.Date(h.isoformat()).format().getInfo() = {gee_interpretation!r} "
                   f"(compare to the intended {h.isoformat()!r} -- if these don't represent the same "
                   f"instant, GEE is misparsing the offset suffix)")

        era5_land_img = ee.ImageCollection(ERA5_LAND_COLLECTION).filterDate(hour_start, hour_start.advance(1, "hour")).first()
        era5_img = ee.ImageCollection(ERA5_COLLECTION).filterDate(hour_start, hour_start.advance(1, "hour")).first()

        band_images = []
        for var_name, band in ERA5_LAND_BANDS.items():
            land_val_img = era5_land_img.select(band)
            if var_name == "precipitation":
                fallback_val_img = era5_img.select(PRECIP_RATE_BAND).multiply(PRECIP_RATE_TO_METERS_PER_HOUR)
            else:
                fallback_val_img = era5_img.select(ERA5_FALLBACK_BANDS[var_name])
            band_images.append(land_val_img.unmask(fallback_val_img).rename(var_name))
        band_images.append(era5_img.select(ERA5_PBLH_BAND).rename("boundary_layer_height"))

        # Combine all 7 variables (6 ERA5-Land + PBLH) into one multi-band
        # image -- one reduceRegions() call per scope retrieves every
        # variable's value at once, instead of 7 separate calls.
        combined_img = ee.Image.cat(band_images)

        n_ok_local, n_missing_local = _reduce_multiband_into(combined_img, local_fc, all_rows, h, var_names, "local", log_fn)
        n_ok_nbhd, n_missing_nbhd = _reduce_multiband_into(combined_img, nbhd_fc, all_rows, h, var_names, "neighborhood_mean", log_fn)

        log_fn(f"  hour {h.isoformat()} (batch {i + 1}/{len(hours)}): local {n_ok_local} ok / {n_missing_local} missing, "
               f"neighborhood {n_ok_nbhd} ok / {n_missing_nbhd} missing (all {len(var_names)} variables combined)")
        if DEBUG_DYNAMIC_RETRIEVAL:
            _debug_batch_count["n"] += 1

    df = pd.DataFrame(list(all_rows.values()))
    return df


def _log_per_variable_diagnostics(info, var_names, scope_suffix, time_val, log_fn):
    """TEMPORARY DIAGNOSTIC (see DEBUG_DYNAMIC_RETRIEVAL above): prints
    valid/null counts for EACH variable separately, from the SAME raw
    reduceRegions() result the normal combined-count logic below also
    reads -- this does not run a second GEE query, it just inspects the
    same 'info' dict more granularly before the existing aggregation
    combines everything into one number. Distinguishes "every variable
    is null" (a shared, upstream problem -- e.g. image/geometry/
    projection) from "only some variables are null" (expected satellite
    missingness in AOD/Sentinel-5P specifically)."""
    n_features = len(info["features"])
    log_fn(f"    [DEBUG] {time_val} ({scope_suffix}), {n_features} cell(s) returned -- per-variable valid/null counts:")
    for var_name in var_names:
        n_valid = sum(1 for feat in info["features"] if feat["properties"].get(var_name) is not None)
        log_fn(f"    [DEBUG]   {var_name:45s} {n_valid}/{n_features} valid")


def _reduce_multiband_into(image, fc, all_rows, time_val, var_names, scope_suffix, log_fn):
    """Single reduceRegions() call across every band of a multi-band
    image. Each band's output property is named EXACTLY the band name
    itself (e.g. "temperature_2m", "no2") -- ee.Reducer.mean() does NOT
    append a "_mean" suffix when applied to a multi-band image via
    reduceRegions(). Confirmed directly from a live raw response (every
    property in reduceRegions().getInfo()'s own "columns"/"properties"
    matched the band name with no suffix); an earlier version of this
    function assumed a "_mean" suffix, which silently caused every
    lookup here to return None for every variable, every cell, every
    request -- see the investigation that traced this."""
    n_ok, n_missing = 0, 0
    try:
        reduced = image.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=1000)
        info = reduced.getInfo()

        if DEBUG_DYNAMIC_RETRIEVAL and _debug_batch_count["n"] < DEBUG_MAX_BATCHES:
            _log_per_variable_diagnostics(info, var_names, scope_suffix, time_val, log_fn)

        if DEBUG_DYNAMIC_RETRIEVAL and not _debug_raw_response_printed["done"]:
            # DIAGNOSTIC: the COMPLETELY RAW, unparsed reduceRegions()
            # result -- exactly what GEE returned, before _reduce_
            # multiband_into (this function) does ANY parsing, renaming,
            # or null-checking below. Printed exactly once (the very
            # first call made, regardless of scope) so the raw evidence
            # is captured without flooding the console on every batch.
            import json
            log_fn(f"    [DEBUG] ===== RAW, UNPARSED reduceRegions() response ({scope_suffix}, {time_val}) =====")
            log_fn(f"    [DEBUG] {json.dumps(info, default=str)}")
            log_fn(f"    [DEBUG] ===== end raw response =====")
            _debug_raw_response_printed["done"] = True

        seen = set()
        for feat_idx, feat in enumerate(info["features"]):
            gid = feat["properties"]["grid_id"]
            seen.add(gid)
            for var_name in var_names:
                value = feat["properties"].get(var_name)

                # For the daily loop (which includes "no2"), specifically
                # trace no2 as explicitly requested -- NOT var_names[0]
                # ("aod"), which would otherwise trigger and lock this
                # flag before the loop ever reaches no2. For the hourly
                # loop (no "no2" present), trace var_names[0]
                # ("temperature_2m") instead.
                target_var = "no2" if "no2" in var_names else var_names[0]
                if DEBUG_DYNAMIC_RETRIEVAL and feat_idx == 0 and not _debug_stage_trace_printed["done"] \
                        and var_name == target_var:
                    # DIAGNOSTIC: exact stage-by-stage trace requested --
                    # same variable, same cell, at each stage this
                    # function's own code actually passes it through.
                    log_fn(f"    [DEBUG] ===== STAGE-BY-STAGE TRACE: variable='{var_name}', cell='{gid}' =====")
                    log_fn(f"    [DEBUG] STAGE 1 (RAW properties dict, unsuffixed key '{var_name}'): "
                           f"{feat['properties'].get(var_name)!r}")
                    log_fn(f"    [DEBUG] STAGE 2 (what this code ACTUALLY reads -- key "
                           f"'{var_name}_mean'): {value!r}")
                    log_fn(f"    [DEBUG] STAGE 3 (all_rows[('{gid}', {time_val!r})]['{var_name}_{scope_suffix}'] "
                           f"after assignment, i.e. the final predictor dict entry): will be set to {value!r}")
                    log_fn(f"    [DEBUG] STAGE 4 (validity classification): "
                           f"{'MISSING (value is None)' if value is None else 'OK (value is not None)'}")
                    log_fn(f"    [DEBUG] ===== end trace -- compare Stage 1 vs Stage 2 =====")
                    _debug_stage_trace_printed["done"] = True

                out_col = f"{var_name}_{scope_suffix}"
                all_rows[(gid, time_val)][out_col] = value
                if value is None:
                    n_missing += 1
                else:
                    n_ok += 1
        for key in list(all_rows.keys()):
            gid, t = key
            if t == time_val and gid not in seen:
                for var_name in var_names:
                    all_rows[key][f"{var_name}_{scope_suffix}"] = None
                    n_missing += 1
    except ee.EEException as e:
        log_fn(f"  GEE ERROR fetching {var_names} ({scope_suffix}) at {time_val}: {type(e).__name__}: {e}")
        for key in all_rows:
            if key[1] == time_val:
                for var_name in var_names:
                    all_rows[key][f"{var_name}_{scope_suffix}"] = None
        n_missing = len(all_rows) * len(var_names)
    return n_ok, n_missing
