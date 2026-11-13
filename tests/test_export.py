"""
tests/test_export.py -- CSV / GeoJSON prediction export (backend/export.py
+ backend/cell_geometry.py), against a small synthetic grid and a
synthetic completed-job result shaped exactly like pipeline.py's `result`.
No GEE, model, or Flask app import needed.
"""

import csv
import io
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import export
from backend.cell_geometry import cell_corners_wgs84, cell_centroids_wgs84

TEST_CRS = "EPSG:3857"


def make_grid(n=3):
    cells = []
    for i in range(n):
        for j in range(n):
            cells.append({
                "grid_id": f"T{i}{j}", "grid_x_index": 13500 + i, "grid_y_index": 1500 + j,
                "geometry": box((13500 + i) * 1000, (1500 + j) * 1000, (13501 + i) * 1000, (1501 + j) * 1000),
            })
    return gpd.GeoDataFrame(cells, crs=TEST_CRS)


def make_result(resolution="hourly", with_nan=False):
    # Same string form pipeline.py produces: str(pd.Timestamp) with +00:00.
    if resolution == "hourly":
        timestamps = ["2026-08-24 22:00:00+00:00", "2026-08-24 23:00:00+00:00"]
    else:
        timestamps = ["2026-08-15 16:00:00+00:00", "2026-08-16 16:00:00+00:00"]  # local-midnight reps
    cell_ids = ["T00", "T01"]
    preds = []
    for c in cell_ids:
        for k, t in enumerate(timestamps):
            v = float("nan") if (with_nan and c == "T01" and k == 0) else 10.0 + k
            preds.append({"grid_id": c, "datetime_utc": t, "pm25_prediction": v})
    return {
        "cell_ids": cell_ids, "timestamps": timestamps, "predictions": preds,
        "row_count": len(preds), "rows_with_missing_values": 1 if with_nan else 0,
        "temporal_resolution": resolution,
    }


def _meta(grid, result):
    selected = grid[grid["grid_id"].isin(result["cell_ids"])]
    return export.build_cells_meta(selected, grid.crs)


def test_cell_corners_and_centroids_shapes():
    grid = make_grid()
    lons, lats = cell_corners_wgs84(grid, grid.crs)
    assert len(lons) == len(grid) and all(len(c) == 4 for c in lons)
    clats, clons = cell_centroids_wgs84(grid, grid.crs)
    assert len(clats) == len(grid)
    # centroid lies within the corner bounding box
    assert min(lons[0]) <= clons[0] <= max(lons[0]) and min(lats[0]) <= clats[0] <= max(lats[0])
    # empty input is handled
    assert cell_corners_wgs84(grid.iloc[0:0], grid.crs) == ([], [])


def test_csv_columns_order_and_timestamps():
    grid, result = make_grid(), make_result()
    text = "".join(export.iter_csv(result, _meta(grid, result)))
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0].keys()) == export.CSV_COLUMNS
    assert len(rows) == 4
    r = rows[0]
    assert r["grid_id"] == "T00" and r["grid_x_index"] == "13500" and r["grid_y_index"] == "1500"
    assert r["datetime_utc"] == "2026-08-24T22:00:00+00:00"
    assert r["datetime_local"] == "2026-08-25T06:00:00+08:00"   # Asia/Manila, same instant
    assert r["temporal_resolution"] == "hourly"
    assert float(r["pm25_prediction_ugm3"]) == 10.0
    assert float(r["centroid_latitude"]) and float(r["centroid_longitude"])


def test_csv_nan_becomes_empty_field():
    grid, result = make_grid(), make_result(with_nan=True)
    rows = list(csv.DictReader(io.StringIO("".join(export.iter_csv(result, _meta(grid, result))))))
    nan_row = [r for r in rows if r["grid_id"] == "T01"][0]
    assert nan_row["pm25_prediction_ugm3"] == ""


def test_csv_chunking_preserves_every_row():
    grid = make_grid()
    result = make_result()
    # Inflate well past one chunk boundary.
    many = []
    for k in range(export.CSV_CHUNK_ROWS * 2 + 7):
        many.append({"grid_id": "T00", "datetime_utc": result["timestamps"][0], "pm25_prediction": float(k)})
    result["predictions"] = many
    rows = list(csv.DictReader(io.StringIO("".join(export.iter_csv(result, _meta(grid, result))))))
    assert len(rows) == len(many)
    assert [float(r["pm25_prediction_ugm3"]) for r in rows] == [float(k) for k in range(len(many))]


def test_geojson_valid_with_closed_rings_and_metadata():
    grid, result = make_grid(), make_result(with_nan=True)
    meta = export.build_metadata(result, "job-1234-abcd", model_info={"trained_utc": "2026-08-12T03:00:25+00:00", "weighting_method": "inverse_rmse"})
    text = "".join(export.iter_geojson(result, _meta(grid, result), meta))
    doc = json.loads(text)  # would raise on NaN tokens or malformed streaming joins
    assert doc["type"] == "FeatureCollection"
    assert doc["metadata"]["job_id"] == "job-1234-abcd"
    assert doc["metadata"]["model"]["trained_utc"].startswith("2026-08-12")
    assert len(doc["features"]) == 4
    f = doc["features"][0]
    ring = f["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]
    assert list(f["properties"].keys()) == export.CSV_COLUMNS
    nan_feat = [x for x in doc["features"] if x["properties"]["grid_id"] == "T01"][0]
    assert nan_feat["properties"]["pm25_prediction_ugm3"] is None


def test_geojson_timestamp_filter_and_rejection():
    grid, result = make_grid(), make_result()
    meta = export.build_metadata(result, "j")
    doc = json.loads("".join(export.iter_geojson(result, _meta(grid, result), meta, timestamp=result["timestamps"][1])))
    assert len(doc["features"]) == 2
    assert all(x["properties"]["datetime_utc"] == "2026-08-24T23:00:00+00:00" for x in doc["features"])
    try:
        list(export.iter_geojson(result, _meta(grid, result), meta, timestamp="1999-01-01 00:00:00+00:00"))
        assert False, "expected ExportError for an unknown timestamp"
    except export.ExportError:
        pass


def test_daily_rows_keep_representative_timestamp_and_label():
    grid, result = make_grid(), make_result(resolution="daily")
    rows = list(csv.DictReader(io.StringIO("".join(export.iter_csv(result, _meta(grid, result))))))
    assert rows[0]["temporal_resolution"] == "daily"
    assert rows[0]["datetime_utc"] == "2026-08-15T16:00:00+00:00"
    assert rows[0]["datetime_local"] == "2026-08-16T00:00:00+08:00"  # local midnight


def test_filename_convention():
    result = make_result()
    assert export.build_filename(result, "0123456789abcdef", "csv") == "pm25_hourly_2026-08-25_to_2026-08-25_2cells_01234567.csv"
    daily = make_result(resolution="daily")
    assert export.build_filename(daily, "0123456789abcdef", "geojson") == "pm25_daily_2026-08-16_to_2026-08-17_2cells_01234567.geojson"


def test_clean_value():
    assert export._clean_value(None) is None
    assert export._clean_value(float("nan")) is None
    assert export._clean_value(3) == 3.0 and isinstance(export._clean_value(3), float)
    assert not math.isnan(export._clean_value(1.5))
