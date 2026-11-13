"""
backend/export.py -- CSV / GeoJSON export of a COMPLETED prediction job.

Formatting only. Reads the job's already-retained result (see
backend/prediction/pipeline.py's `result` dict and job_state.py) plus the
selected cells' geometry (backend/cell_geometry.py) -- it never re-runs a
prediction, never touches GEE or the model, and adds no variables beyond
what the result already carries. datetime_local is the same instant as
datetime_utc expressed in Asia/Manila (config.settings.LOCAL_TIMEZONE) --
a representation of an existing value, not a new one.

Both writers are generators yielding text chunks so Flask can stream the
response instead of building the whole file in memory (a 200,000-row
GeoJSON is ~60 MB).

Row semantics are exactly the result's: hourly results carry one row per
(cell, UTC hour); daily results carry one row per (cell, Philippine local
day), whose datetime_utc is the local-midnight representative timestamp
the pipeline already assigns. temporal_resolution is included in every
row so the meaning is unambiguous without the filename.
"""

import csv
import io
import json
import math
from datetime import datetime, timezone

import pandas as pd

from config.settings import LOCAL_TIMEZONE

CSV_COLUMNS = [
    "grid_id", "grid_x_index", "grid_y_index",
    "centroid_latitude", "centroid_longitude",
    "datetime_utc", "datetime_local", "temporal_resolution", "pm25_prediction_ugm3",
]

CSV_CHUNK_ROWS = 1000


class ExportError(ValueError):
    """A user-facing export request problem (bad format / unknown timestamp)."""


def parse_result_timestamp(ts_str):
    """The result stores timestamps as str(pd.Timestamp) (e.g.
    '2026-08-24 22:00:00+00:00'); normalise to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def format_utc_iso(ts):
    return ts.isoformat()  # e.g. 2026-08-24T22:00:00+00:00


def format_local_iso(ts):
    return ts.tz_convert(LOCAL_TIMEZONE).isoformat()  # e.g. 2026-08-25T06:00:00+08:00


def _clean_value(v):
    """NaN/None -> None so CSV writes an empty field and JSON writes null
    (never the non-standard NaN token)."""
    if v is None:
        return None
    try:
        if math.isnan(v):
            return None
    except TypeError:
        pass
    return float(v)


def build_cells_meta(selected, grid_crs):
    """One dict per grid_id: indices, centroid lat/lon, and the closed
    polygon ring [[lon, lat], ...] (5 points) in EPSG:4326. `selected`
    is the subset of the loaded grid for the job's cell_ids."""
    from backend.cell_geometry import cell_corners_wgs84, cell_centroids_wgs84
    lons, lats = cell_corners_wgs84(selected, grid_crs)
    clats, clons = cell_centroids_wgs84(selected, grid_crs)
    meta = {}
    for i, row in enumerate(selected.itertuples()):
        ring = [[lons[i][j], lats[i][j]] for j in range(4)]
        ring.append(ring[0])
        meta[row.grid_id] = {
            "grid_x_index": int(row.grid_x_index),
            "grid_y_index": int(row.grid_y_index),
            "centroid_latitude": clats[i],
            "centroid_longitude": clons[i],
            "ring": ring,
        }
    return meta


def _rows(result, cells_meta, timestamp=None):
    """Yields flat dicts in CSV_COLUMNS order for every prediction record
    (optionally only those at `timestamp`, the result's own string)."""
    resolution = result.get("temporal_resolution", "hourly")
    ts_cache = {}
    for rec in result["predictions"]:
        ts_str = rec["datetime_utc"]
        if timestamp is not None and ts_str != timestamp:
            continue
        if ts_str not in ts_cache:
            ts = parse_result_timestamp(ts_str)
            ts_cache[ts_str] = (format_utc_iso(ts), format_local_iso(ts))
        utc_iso, local_iso = ts_cache[ts_str]
        meta = cells_meta.get(rec["grid_id"], {})
        yield {
            "grid_id": rec["grid_id"],
            "grid_x_index": meta.get("grid_x_index"),
            "grid_y_index": meta.get("grid_y_index"),
            "centroid_latitude": meta.get("centroid_latitude"),
            "centroid_longitude": meta.get("centroid_longitude"),
            "datetime_utc": utc_iso,
            "datetime_local": local_iso,
            "temporal_resolution": resolution,
            "pm25_prediction_ugm3": _clean_value(rec["pm25_prediction"]),
        }


def iter_csv(result, cells_meta):
    """Generator of CSV text chunks (header first)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    n = 0
    for row in _rows(result, cells_meta):
        writer.writerow({k: ("" if row[k] is None else row[k]) for k in CSV_COLUMNS})
        n += 1
        if n % CSV_CHUNK_ROWS == 0:
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)
    yield buf.getvalue()


def iter_geojson(result, cells_meta, metadata, timestamp=None):
    """Generator of GeoJSON text chunks: a FeatureCollection with one
    Feature per (cell, timestamp) -- or per cell for the single frame
    `timestamp` -- carrying the actual 1km cell polygon and the same
    properties as the CSV columns. `metadata` is emitted as a top-level
    foreign member (permitted by RFC 7946 section 6.1)."""
    if timestamp is not None and timestamp not in set(result["timestamps"]):
        raise ExportError(f"timestamp {timestamp!r} is not one of this job's result timestamps.")
    head = {"type": "FeatureCollection", "metadata": metadata}
    head_text = json.dumps(head, allow_nan=False)
    # Open the collection with metadata, then stream features.
    yield head_text[:-1] + ', "features": ['
    first = True
    for row in _rows(result, cells_meta, timestamp=timestamp):
        ring = cells_meta.get(row["grid_id"], {}).get("ring")
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]} if ring else None,
            "properties": {k: row[k] for k in CSV_COLUMNS},
        }
        text = json.dumps(feature, allow_nan=False)
        yield text if first else ",\n" + text
        first = False
    yield "]}"


def build_filename(result, job_id, fmt):
    """pm25_{resolution}_{startLocal}_to_{endLocal}_{n}cells_{jobid8}.{ext}
    -- start/end are Philippine local calendar dates of the first/last
    result timestamp."""
    resolution = result.get("temporal_resolution", "hourly")
    timestamps = sorted(result["timestamps"])
    if timestamps:
        start_local = parse_result_timestamp(timestamps[0]).tz_convert(LOCAL_TIMEZONE).date().isoformat()
        end_local = parse_result_timestamp(timestamps[-1]).tz_convert(LOCAL_TIMEZONE).date().isoformat()
    else:
        start_local = end_local = "nodata"
    n_cells = len(result.get("cell_ids", []))
    ext = "csv" if fmt == "csv" else "geojson"
    return f"pm25_{resolution}_{start_local}_to_{end_local}_{n_cells}cells_{str(job_id)[:8]}.{ext}"


def build_metadata(result, job_id, model_info=None):
    """Job/result metadata for the GeoJSON foreign member. model_info is
    the (already-loaded) ensemble_config.json's descriptive fields --
    nothing here is computed from the model itself."""
    meta = {
        "job_id": str(job_id),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "temporal_resolution": result.get("temporal_resolution", "hourly"),
        "row_count": int(result.get("row_count", len(result.get("predictions", [])))),
        "rows_with_missing_values": int(result.get("rows_with_missing_values", 0)),
        "cell_count": len(result.get("cell_ids", [])),
        "timestamp_count": len(result.get("timestamps", [])),
        "crs": "EPSG:4326",
        "units": "ug/m3",
        "note": ("pm25_prediction_ugm3 is the deployed ensemble model's prediction. Daily rows are the mean of "
                 "modelled hourly predictions for the Philippine local day (not a separately validated daily "
                 "target); datetime_utc for daily rows is that day's local-midnight representative timestamp."),
    }
    if model_info:
        meta["model"] = {k: model_info.get(k) for k in ("trained_utc", "weighting_method", "target_transform", "inverse_transform")
                         if k in model_info}
    return meta
