"""
backend/prediction/dynamic_predictors_batched.py
============================================================================
BATCHED variants of dynamic_predictors.fetch_hourly_predictors /
fetch_daily_predictors -- a PERFORMANCE change only, approved for the
mobile endpoint's cold requests. Measured before this module existed: a
single cell / single hour needed 50 hourly + 16 daily reduceRegions()
calls (~2.4 s and ~0.8 s each, ~136 s in total).

What is identical to dynamic_predictors.py (imported from it, not
re-declared): collection IDs, band names, the per-hour image construction
(ERA5-Land band .unmask(ERA5 fallback), precipitation rate x 3.6, PBLH
from ERA5), the per-date construction (_daily_mean_band: same-day .mean()
with the native projection captured BEFORE compositing and re-attached),
the cell/neighbourhood geometries (_build_feature_collections), the
reducer (ee.Reducer.mean()), the scale (1000 m), the band order within
each timestamp (so the first band -- and therefore reduceRegions'
default projection -- is the same ERA5-Land temperature band / AOD band
as before), and the strict same-timestamp temporal semantics (no ±3 h /
±5 d fallback).

What differs: instead of one reduceRegions() per (timestamp, scope), the
per-timestamp images of a whole window are concatenated into one
multi-band image whose bands are named "<var>__<k>" (k = position in the
chunk), reduced ONCE per scope, and the result parsed back into the same
long-format rows. Missing-image semantics are reproduced explicitly: the
existing code raises (and records None for every variable of that
timestamp) when a collection has no image for an hour/date, because
.first() is null; here that condition is tested server-side
(ee.Algorithms.If) and the timestamp's bands are fully masked instead --
same outcome (None for all variables of that timestamp), without failing
the rest of the chunk.

Output schema is exactly that of the originals: long DataFrames with
grid_id, datetime_utc / date, and {var}_local, {var}_neighborhood_mean.
tests/test_batched_retrieval.py compares both implementations on real
cells/dates (live GEE) with numerical tolerances.
"""

from datetime import timedelta

import ee
import pandas as pd

from backend.prediction.dynamic_predictors import (
    DAILY_ROLLING_LOOKBACK_DAYS, HOURLY_ROLLING_LOOKBACK_HOURS,
    MODIS_AOD_COLLECTION, MODIS_AOD_BAND, S5P_GASES,
    ERA5_LAND_COLLECTION, ERA5_COLLECTION, ERA5_LAND_BANDS, ERA5_FALLBACK_BANDS,
    PRECIP_RATE_BAND, PRECIP_RATE_TO_METERS_PER_HOUR, ERA5_PBLH_BAND,
    _build_feature_collections,
)

# Chunk sizes: one mobile request (1 hour + 24 h lookback = 25 hours; 1 day
# + 7 d lookback = 8 dates) fits in a single call per scope. Longer
# windows are split into consecutive chunks of these sizes.
HOURLY_CHUNK_HOURS = 25
DAILY_CHUNK_DAYS = 8

HOURLY_VAR_NAMES = list(ERA5_LAND_BANDS.keys()) + ["boundary_layer_height"]
DAILY_VARIABLES = [("aod", MODIS_AOD_COLLECTION, MODIS_AOD_BAND)] + \
                  [(name, coll, band) for name, (coll, band) in S5P_GASES.items()]
DAILY_VAR_NAMES = [v[0] for v in DAILY_VARIABLES]


def _band_names(var_names, k):
    return [f"{v}__{k}" for v in var_names]


def _masked_placeholder(band_names):
    """A fully-masked image with the given band names -- reduces to null
    for every feature, reproducing the original code's 'None for every
    variable of this timestamp' outcome when a collection has no image."""
    return ee.Image.constant([0] * len(band_names)).rename(band_names).updateMask(ee.Image.constant(0))


def _hourly_image(h, k):
    """The SAME per-hour construction as fetch_hourly_predictors, with the
    output bands suffixed by the chunk position k."""
    hour_start = ee.Date(h.isoformat())
    hour_end = hour_start.advance(1, "hour")
    land_col = ee.ImageCollection(ERA5_LAND_COLLECTION).filterDate(hour_start, hour_end)
    era5_col = ee.ImageCollection(ERA5_COLLECTION).filterDate(hour_start, hour_end)
    era5_land_img = land_col.first()
    era5_img = era5_col.first()

    band_images = []
    for var_name, band in ERA5_LAND_BANDS.items():
        land_val_img = era5_land_img.select(band)
        if var_name == "precipitation":
            fallback_val_img = era5_img.select(PRECIP_RATE_BAND).multiply(PRECIP_RATE_TO_METERS_PER_HOUR)
        else:
            fallback_val_img = era5_img.select(ERA5_FALLBACK_BANDS[var_name])
        band_images.append(land_val_img.unmask(fallback_val_img).rename(f"{var_name}__{k}"))
    band_images.append(era5_img.select(ERA5_PBLH_BAND).rename(f"boundary_layer_height__{k}"))
    combined = ee.Image.cat(band_images)

    names = _band_names(HOURLY_VAR_NAMES, k)
    has_both = land_col.size().gt(0).And(era5_col.size().gt(0))
    return ee.Image(ee.Algorithms.If(has_both, combined, _masked_placeholder(names)))


def _daily_image(d, k):
    """The SAME per-date construction as fetch_daily_predictors /
    _daily_mean_band, bands suffixed by k. As in the original, a date on
    which ANY variable's collection has no image yields None for ALL six
    variables of that date."""
    day_start = ee.Date(str(d))
    day_end = day_start.advance(1, "day")
    band_images, presence = [], None
    for var_name, coll, band in DAILY_VARIABLES:
        filtered = ee.ImageCollection(coll).select(band).filterDate(day_start, day_end)
        native_proj = ee.Image(filtered.first()).select(band).projection()
        band_images.append(filtered.mean().rename(f"{var_name}__{k}").setDefaultProjection(native_proj))
        present = filtered.size().gt(0)
        presence = present if presence is None else presence.And(present)
    combined = ee.Image.cat(band_images)
    names = _band_names(DAILY_VAR_NAMES, k)
    return ee.Image(ee.Algorithms.If(presence, combined, _masked_placeholder(names)))


def _reduce_chunk_into(image, fc, all_rows, times, var_names, scope_suffix, log_fn):
    """One reduceRegions() for a whole chunk. Parses "<var>__<k>" back to
    (grid_id, times[k]) rows -- the same None-for-missing convention as
    dynamic_predictors._reduce_multiband_into (cells absent from the
    response are None for every variable/timestamp of the chunk)."""
    n_ok = n_missing = 0
    try:
        reduced = image.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=1000)
        info = reduced.getInfo()
        seen = set()
        for feat in info["features"]:
            props = feat["properties"]
            gid = props["grid_id"]
            seen.add(gid)
            for k, t in enumerate(times):
                for var_name in var_names:
                    value = props.get(f"{var_name}__{k}")
                    all_rows[(gid, t)][f"{var_name}_{scope_suffix}"] = value
                    if value is None:
                        n_missing += 1
                    else:
                        n_ok += 1
        for (gid, t) in list(all_rows.keys()):
            if t in set(times) and gid not in seen:
                for var_name in var_names:
                    all_rows[(gid, t)][f"{var_name}_{scope_suffix}"] = None
                    n_missing += 1
    except ee.EEException as e:
        log_fn(f"  GEE ERROR (batched) fetching {var_names} ({scope_suffix}) for {len(times)} timestamp(s) "
               f"starting {times[0]}: {type(e).__name__}: {e}")
        for (gid, t) in all_rows:
            if t in set(times):
                for var_name in var_names:
                    all_rows[(gid, t)][f"{var_name}_{scope_suffix}"] = None
        n_missing = sum(1 for (gid, t) in all_rows if t in set(times)) * len(var_names)
    return n_ok, n_missing


def fetch_hourly_predictors_batched(cells_df, grid_crs, start_dt, end_dt, log_fn=print, cancel_check=None):
    """Drop-in equivalent of dynamic_predictors.fetch_hourly_predictors:
    same window ([start_dt - 24h, end_dt]), same rows/columns, chunked
    multi-band retrieval (2 reduceRegions calls per HOURLY_CHUNK_HOURS)."""
    fetch_start = start_dt - timedelta(hours=HOURLY_ROLLING_LOOKBACK_HOURS)
    hours = pd.date_range(fetch_start, end_dt, freq="h")
    log_fn(f"Hourly predictors (batched): fetching {len(hours)} distinct hour(s) from {fetch_start} to {end_dt} "
           f"(includes {HOURLY_ROLLING_LOOKBACK_HOURS}-hour rolling/tendency lookback before the requested start).")

    local_fc, nbhd_fc = _build_feature_collections(cells_df, grid_crs)
    all_rows = {(gid, h): {"grid_id": gid, "datetime_utc": h} for gid in cells_df["grid_id"] for h in hours}

    n_calls = 0
    for c0 in range(0, len(hours), HOURLY_CHUNK_HOURS):
        if cancel_check:
            cancel_check()
        chunk = list(hours[c0:c0 + HOURLY_CHUNK_HOURS])
        combined = ee.Image.cat([_hourly_image(h, k) for k, h in enumerate(chunk)])
        ok_l, miss_l = _reduce_chunk_into(combined, local_fc, all_rows, chunk, HOURLY_VAR_NAMES, "local", log_fn)
        ok_n, miss_n = _reduce_chunk_into(combined, nbhd_fc, all_rows, chunk, HOURLY_VAR_NAMES, "neighborhood_mean", log_fn)
        n_calls += 2
        log_fn(f"  hours {chunk[0].isoformat()} .. {chunk[-1].isoformat()} ({len(chunk)} hours, 2 calls): "
               f"local {ok_l} ok / {miss_l} missing, neighborhood {ok_n} ok / {miss_n} missing "
               f"(all {len(HOURLY_VAR_NAMES)} variables combined)")
    log_fn(f"Hourly predictors (batched): {n_calls} reduceRegions call(s) for {len(hours)} hour(s).")
    return pd.DataFrame(list(all_rows.values()))


def fetch_daily_predictors_batched(cells_df, grid_crs, start_date, end_date, log_fn=print, cancel_check=None):
    """Drop-in equivalent of dynamic_predictors.fetch_daily_predictors:
    same window ([start_date - 7 d, end_date]), same rows/columns, chunked
    multi-band retrieval (2 reduceRegions calls per DAILY_CHUNK_DAYS)."""
    fetch_start = start_date - timedelta(days=DAILY_ROLLING_LOOKBACK_DAYS)
    dates = list(pd.date_range(fetch_start, end_date, freq="D").date)
    log_fn(f"Daily predictors (batched): fetching {len(dates)} distinct date(s) from {fetch_start} to {end_date} "
           f"(includes {DAILY_ROLLING_LOOKBACK_DAYS}-day rolling-average lookback before the requested start).")

    local_fc, nbhd_fc = _build_feature_collections(cells_df, grid_crs)
    all_rows = {(gid, d): {"grid_id": gid, "date": d} for gid in cells_df["grid_id"] for d in dates}

    n_calls = 0
    for c0 in range(0, len(dates), DAILY_CHUNK_DAYS):
        if cancel_check:
            cancel_check()
        chunk = dates[c0:c0 + DAILY_CHUNK_DAYS]
        combined = ee.Image.cat([_daily_image(d, k) for k, d in enumerate(chunk)])
        ok_l, miss_l = _reduce_chunk_into(combined, local_fc, all_rows, chunk, DAILY_VAR_NAMES, "local", log_fn)
        ok_n, miss_n = _reduce_chunk_into(combined, nbhd_fc, all_rows, chunk, DAILY_VAR_NAMES, "neighborhood_mean", log_fn)
        n_calls += 2
        log_fn(f"  dates {chunk[0]} .. {chunk[-1]} ({len(chunk)} dates, 2 calls): "
               f"local {ok_l} ok / {miss_l} missing, neighborhood {ok_n} ok / {miss_n} missing "
               f"(all {len(DAILY_VAR_NAMES)} variables combined)")
    log_fn(f"Daily predictors (batched): {n_calls} reduceRegions call(s) for {len(dates)} date(s).")
    return pd.DataFrame(list(all_rows.values()))
