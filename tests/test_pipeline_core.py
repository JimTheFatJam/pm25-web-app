"""
tests/test_pipeline_core.py -- the behaviour-preserving pipeline refactor:
run_prediction_job (web job wrapper, the pre-existing entry point) and
run_prediction_core (mobile entry point) must produce IDENTICAL results
for identical inputs, and the wrapper must still drive job_state (stages,
logs, retention key = job id) exactly as before.

Static/dynamic retrieval is replaced by deterministic fakes (no GEE), but
the REAL research feature engineering and the REAL deployed model run, so
this is skipped when the research pipeline / artifacts are unavailable.
"""

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import LOCAL_TIMEZONE, MODEL_XGBOOST_PATH, RESEARCH_PIPELINE_DIR
import config.settings as settings


def _available():
    return MODEL_XGBOOST_PATH.exists() and RESEARCH_PIPELINE_DIR is not None and Path(RESEARCH_PIPELINE_DIR).is_dir()


pytestmark = pytest.mark.skipif(not _available(), reason="research pipeline / model artifacts not available")

GRID_CRS = "ESRI:102028"
CELLS = ["PHFULL_-122_2895", "PHFULL_-121_2895"]


@pytest.fixture(scope="module")
def loaded():
    from backend.prediction.model_loader import load_models
    return load_models()


@pytest.fixture(scope="module")
def grid():
    rows = []
    for gid in CELLS:
        gx, gy = int(gid.split("_")[1]), int(gid.split("_")[2])
        rows.append({"grid_id": gid, "grid_x_index": gx, "grid_y_index": gy,
                     "geometry": box(gx * 1000, gy * 1000, (gx + 1) * 1000, (gy + 1) * 1000)})
    g = gpd.GeoDataFrame(rows, crs=GRID_CRS)
    settings.ANALYSIS_CRS = g.crs
    return g


HOURLY_VARS = ["temperature_2m", "dewpoint_2m", "surface_pressure", "u_wind_10m", "v_wind_10m", "precipitation", "boundary_layer_height"]
DAILY_VARS = ["aod", "no2", "so2", "co", "o3", "hcho"]
BASE = {"temperature_2m": 300.0, "dewpoint_2m": 295.0, "surface_pressure": 101000.0, "u_wind_10m": 1.0, "v_wind_10m": -0.5,
        "precipitation": 0.0002, "boundary_layer_height": 600.0, "aod": 250.0, "no2": 3e-5, "so2": 2e-4, "co": 0.03, "o3": 0.12, "hcho": 1e-4}


def fake_hourly(cells_df, grid_crs, start_dt, end_dt, log_fn=print, cancel_check=None):
    from backend.prediction.dynamic_predictors import HOURLY_ROLLING_LOOKBACK_HOURS
    hours = pd.date_range(start_dt - pd.Timedelta(hours=HOURLY_ROLLING_LOOKBACK_HOURS), end_dt, freq="h")
    rows = []
    for ci, gid in enumerate(cells_df["grid_id"]):
        for h in hours:
            r = {"grid_id": gid, "datetime_utc": h}
            hi = int(h.timestamp() // 3600)  # absolute hour -> identical values regardless of window
            for v in HOURLY_VARS:
                val = BASE[v] * (1 + 0.01 * np.sin(hi / 3 + ci))
                r[f"{v}_local"] = val
                r[f"{v}_neighborhood_mean"] = val * 1.01
            rows.append(r)
    return pd.DataFrame(rows)


def fake_daily(cells_df, grid_crs, start_date, end_date, log_fn=print, cancel_check=None):
    from backend.prediction.dynamic_predictors import DAILY_ROLLING_LOOKBACK_DAYS
    dates = pd.date_range(start_date - pd.Timedelta(days=DAILY_ROLLING_LOOKBACK_DAYS), end_date, freq="D").date
    rows = []
    for ci, gid in enumerate(cells_df["grid_id"]):
        for d in dates:
            r = {"grid_id": gid, "date": d}
            di = d.toordinal()  # absolute day -> identical values regardless of window
            for v in DAILY_VARS:
                val = None if (d == date(2026, 9, 5) and v == "aod") else BASE[v] * (1 + 0.05 * np.cos(di + ci))  # one genuine gap
                r[f"{v}_local"] = val
                r[f"{v}_neighborhood_mean"] = None if val is None else val * 0.98
            rows.append(r)
    return pd.DataFrame(rows)


def _patch_static(monkeypatch):
    from backend.prediction import static_predictors as sp
    def population(cells_df, crs, year, log_fn=print):
        return pd.DataFrame({"grid_id": cells_df["grid_id"], "population_local": 5000.0, "population_neighborhood_mean": 4000.0})
    def coast(cells_df, crs, log_fn=print):
        return pd.DataFrame({"grid_id": cells_df["grid_id"], "distance_to_coast_km": 3.9})
    def roads(cells_df, log_fn=print):
        return pd.DataFrame({"grid_id": cells_df["grid_id"], "road_length_local_m": 12000.0,
                             "road_length_neighborhood_m": 150000.0, "road_density_neighborhood_m_per_km2": 6000.0})
    def gee_static(cells_df, crs, year, log_fn=print):
        return pd.DataFrame({"grid_id": cells_df["grid_id"], "elevation_local": 40.0, "elevation_neighborhood_mean": 55.0,
                             "landcover_local": 13, "landcover_neighborhood_mode": 13, "landcover_neighborhood_mode_fraction": 0.7,
                             "ndvi_local": 3000.0, "ndvi_neighborhood_mean": 3500.0,
                             "nighttime_lights_local": 20.0, "nighttime_lights_neighborhood_mean": 15.0})
    monkeypatch.setattr(sp, "load_population", population)
    monkeypatch.setattr(sp, "load_coastline_distance", coast)
    monkeypatch.setattr(sp, "load_road_density", roads)
    monkeypatch.setattr(sp, "fetch_gee_static_layers", gee_static)


def _patch_dynamic(monkeypatch):
    from backend.prediction import dynamic_predictors as dp
    monkeypatch.setattr(dp, "fetch_hourly_predictors", fake_hourly)
    monkeypatch.setattr(dp, "fetch_daily_predictors", fake_daily)


def test_job_wrapper_and_core_produce_identical_results(monkeypatch, loaded, grid):
    from backend.prediction import pipeline, job_state, explanation_store
    _patch_static(monkeypatch); _patch_dynamic(monkeypatch)
    availability = {"latest_valid_date": date(2026, 9, 10), "binding_dataset": "no2"}
    start = datetime(2026, 9, 10, 20, 0, tzinfo=LOCAL_TIMEZONE)
    end = datetime(2026, 9, 10, 23, 0, tzinfo=LOCAL_TIMEZONE)

    explanation_store._reset_for_tests()
    job_id = job_state.create_job()
    via_job = pipeline.run_prediction_job(job_id, CELLS, start, end, grid, loaded, "hourly", availability)
    job = job_state.get_job(job_id)
    assert job["stage_index"] == 6 and any("Feature matrix assembled" in l for l in job["log_lines"])
    assert explanation_store.has_job(job_id)  # retained under the job id, as before

    core_log = []
    via_core = pipeline.run_prediction_core(CELLS, start, end, grid, loaded, "hourly", availability,
                                            log_fn=core_log.append, retain_key="mobile:test",
                                            fetch_daily_fn=fake_daily, fetch_hourly_fn=fake_hourly)
    assert explanation_store.has_job("mobile:test")

    assert via_job["row_count"] == via_core["row_count"] == len(CELLS) * 4
    assert via_job["timestamps"] == via_core["timestamps"]
    assert via_job["rows_with_missing_values"] == via_core["rows_with_missing_values"]
    a = {(r["grid_id"], r["datetime_utc"]): r["pm25_prediction"] for r in via_job["predictions"]}
    b = {(r["grid_id"], r["datetime_utc"]): r["pm25_prediction"] for r in via_core["predictions"]}
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], f"prediction differs for {k}"
        assert np.isfinite(a[k]) and a[k] >= 0
    # the same messages were logged by both paths (job log has stage lines in addition)
    assert [l for l in job["log_lines"] if not l.startswith("[")] == core_log
    explanation_store._reset_for_tests()
    with job_state._lock:
        job_state._jobs.pop(job_id, None)


def test_core_single_hour_matches_last_row_of_multi_hour_run(monkeypatch, loaded, grid):
    """The mobile path requests [T*, T*]; its prediction must equal the T*
    row of a longer request ending at T* (same lookback context)."""
    from backend.prediction import pipeline, explanation_store
    _patch_static(monkeypatch); _patch_dynamic(monkeypatch)
    availability = {"latest_valid_date": date(2026, 9, 10), "binding_dataset": "no2"}
    t_star = datetime(2026, 9, 10, 23, 0, tzinfo=LOCAL_TIMEZONE)
    long_run = pipeline.run_prediction_core(CELLS[:1], datetime(2026, 9, 10, 20, 0, tzinfo=LOCAL_TIMEZONE), t_star, grid, loaded,
                                            "hourly", availability, log_fn=lambda m: None,
                                            fetch_daily_fn=fake_daily, fetch_hourly_fn=fake_hourly)
    single = pipeline.run_prediction_core(CELLS[:1], t_star, t_star, grid, loaded, "hourly", availability, log_fn=lambda m: None,
                                          fetch_daily_fn=fake_daily, fetch_hourly_fn=fake_hourly)
    assert len(single["predictions"]) == 1
    last = [r for r in long_run["predictions"] if r["datetime_utc"] == single["predictions"][0]["datetime_utc"]][0]
    assert abs(last["pm25_prediction"] - single["predictions"][0]["pm25_prediction"]) < 1e-9
    explanation_store._reset_for_tests()
