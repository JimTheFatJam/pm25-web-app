"""
tests/test_mobile.py -- backend/mobile.py request flow with the expensive
parts (availability, pipeline, explainer) injected as fakes: coordinate
validation, point -> cell, strict T*, availability states, response
assembly (classification, guidance, SHAP top-3), cache hit/miss, in-flight
de-duplication under concurrent identical requests, and error mapping.
"""

import sys
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import mobile
from config.settings import LOCAL_TIMEZONE, MOBILE_CACHE_MAX_ENTRIES
from config.pm25_classification import MOBILE_CLASSIFICATION_SCHEME_ID

# A small synthetic grid in the REAL grid CRS (ESRI:102028, metres) around
# Cebu City so lat/lon -> cell uses the genuine projection path.
GRID_CRS = "ESRI:102028"


def make_grid():
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
    x, y = tr.transform(123.9131, 10.3039)  # Cebu City centroid used in the live runs
    gx0, gy0 = int(np.floor(x / 1000)), int(np.floor(y / 1000))
    cells = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            gx, gy = gx0 + i, gy0 + j
            gid = "PH000000001" if (i, j) == (1, 1) else f"PHFULL_{gx}_{gy}"  # one training-style id
            cells.append({"grid_id": gid, "grid_x_index": gx, "grid_y_index": gy,
                          "geometry": box(gx * 1000, gy * 1000, (gx + 1) * 1000, (gy + 1) * 1000)})
    return gpd.GeoDataFrame(cells, crs=GRID_CRS), (gx0, gy0)


def ready_availability(latest=date(2026, 9, 10)):
    return {"status": "ready", "error": None, "refresh_error": None,
            "result": {"latest_valid_date": latest, "binding_dataset": "no2"}}


def fake_explanation(n=6):
    contribs = [{"feature": f"f{i}", "display_name": f"Feature {i}", "value": 1.0, "shap_log": s}
                for i, s in enumerate([0.5, -0.4, 0.3, -0.2, 0.1, 0.05])]
    return {"contributions": contribs, "n_missing_features": 4, "n_features": 157}


def make_service(pm25=12.3, availability=None, compute_delay=0.0, counter=None, explanation=None):
    grid, _ = make_grid()
    index = mobile.CellIndex(grid)

    def compute(grid_id, tstar_local):
        if counter is not None:
            counter["n"] += 1
        time.sleep(compute_delay)
        return {"pm25": pm25, "retain_key": f"mobile:{grid_id}", "n_rows": 1, "compute_seconds": compute_delay}

    def explain(retain_key, grid_id, tstar_iso):
        return fake_explanation() if explanation is None else explanation

    return mobile.MobileService(index, lambda: availability or ready_availability(), compute, explain, cache_max_entries=3)


# ---------------- coordinates / cell lookup ----------------

def test_invalid_coordinates():
    svc = make_service()
    for body in [None, {}, {"lat": "x", "lon": 1}, {"lat": 91, "lon": 0}, {"lat": 0, "lon": 181}, {"lat": float("nan"), "lon": 0}]:
        payload, status = svc.handle(body)
        assert status == 400 and payload["status"] == "error" and payload["error_code"] in ("invalid_request", "invalid_coordinates")


def test_point_maps_to_correct_cell_and_training_id_style():
    grid, (gx0, gy0) = make_grid()
    index = mobile.CellIndex(grid)
    assert index.cell_for_point(10.3039, 123.9131) == f"PHFULL_{gx0}_{gy0}"
    # a point one cell north-east resolves to the training-style id in the synthetic grid
    from pyproj import Transformer
    tr = Transformer.from_crs(GRID_CRS, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform((gx0 + 1) * 1000 + 500, (gy0 + 1) * 1000 + 500)
    assert index.cell_for_point(lat, lon) == "PH000000001"


def test_outside_grid_is_404():
    svc = make_service()
    payload, status = svc.handle({"lat": 14.6, "lon": 121.0})  # Manila: not in the 3x3 synthetic grid
    assert status == 404 and payload["error_code"] == "outside_coverage"


# ---------------- T* ----------------

def test_strict_tstar_is_2300_local_on_latest_valid_date():
    t = mobile.StrictTStarStrategy().select({"latest_valid_date": date(2026, 9, 10)})
    assert t == datetime(2026, 9, 10, 23, 0, tzinfo=LOCAL_TIMEZONE)
    assert t.astimezone(timezone.utc) == datetime(2026, 9, 10, 15, 0, tzinfo=timezone.utc)
    assert mobile.StrictTStarStrategy().select(None) is None


def test_availability_pending_and_failed():
    pending = make_service(availability={"status": "pending", "result": None, "error": None})
    p, s = pending.handle({"lat": 10.3039, "lon": 123.9131})
    assert s == 503 and p["error_code"] == "availability_pending"
    failed = make_service(availability={"status": "failed", "result": None, "error": "boom"})
    p, s = failed.handle({"lat": 10.3039, "lon": 123.9131})
    assert s == 503 and p["error_code"] == "availability_failed"


# ---------------- successful reading ----------------

def test_successful_reading_has_classification_guidance_and_top3():
    svc = make_service(pm25=12.34)
    p, s = svc.handle({"lat": 10.3039, "lon": 123.9131})
    assert s == 200 and p["status"] == "ok"
    assert p["pm25_ugm3"] == 12.3 and p["units"] == "µg/m³"
    assert p["as_of_local"] == "2026-09-10T23:00:00+08:00" and p["as_of_utc"] == "2026-09-10T15:00:00+00:00"
    assert p["as_of_local_label"].endswith("PHT") and "2026" in p["as_of_local_label"]
    c = p["classification"]
    assert c["scheme_id"] == MOBILE_CLASSIFICATION_SCHEME_ID == "us_epa_aqi_pm25_24h"
    assert c["category"] == "Moderate" and c["guidance"] and "not an official" in c["basis_note"]
    assert "not a live measurement" in p["data_note"]
    e = p["explanation"]
    assert e["available"] and len(e["top_factors"]) == 3
    assert [f["direction"] for f in e["top_factors"]] == ["higher", "lower", "higher"]
    assert e["top_factors"][0]["label"] == "Feature 0" and "attribution" in e["note"]
    assert p["n_missing_features"] == 4 and p["from_cache"] is False and p["staleness_hours"] > 0
    assert p["timestamp_strategy"] == "strict_same_timestamp"


def test_categories_across_epa_bands():
    for value, label in [(5.0, "Good"), (20.0, "Moderate"), (40.0, "Unhealthy for Sensitive Groups"),
                         (100.0, "Unhealthy"), (200.0, "Very Unhealthy"), (300.0, "Hazardous")]:
        p, s = make_service(pm25=value).handle({"lat": 10.3039, "lon": 123.9131})
        assert s == 200 and p["classification"]["category"] == label and p["classification"]["guidance"]


def test_nan_prediction_is_unavailable_not_500():
    p, s = make_service(pm25=float("nan")).handle({"lat": 10.3039, "lon": 123.9131})
    assert s == 200 and p["status"] == "unavailable" and p["pm25_ugm3"] is None and p["classification"] is None


def test_explanation_failure_does_not_break_reading():
    p, s = make_service(explanation={"error": "no rows retained"}).handle({"lat": 10.3039, "lon": 123.9131})
    assert s == 200 and p["status"] == "ok" and p["explanation"]["available"] is False and p["explanation"]["error"]


def test_compute_failure_is_clean_500():
    svc = make_service()
    def boom(grid_id, t): raise RuntimeError("GEE down")
    svc.compute_prediction = boom
    p, s = svc.handle({"lat": 10.3039, "lon": 123.9131})
    assert s == 500 and p["error_code"] == "prediction_failed" and "GEE down" in p["message"]


# ---------------- cache ----------------

def test_cache_miss_then_hit_and_tstar_advance():
    counter = {"n": 0}
    avail = ready_availability()
    svc = make_service(counter=counter, availability=avail)
    p1, _ = svc.handle({"lat": 10.3039, "lon": 123.9131})
    p2, _ = svc.handle({"lat": 10.3039, "lon": 123.9131})
    assert counter["n"] == 1 and p1["from_cache"] is False and p2["from_cache"] is True
    assert svc.cache.stats["hits"] >= 1 and svc.cache.stats["misses"] == 1
    # T* advances -> new key -> recomputed
    avail["result"]["latest_valid_date"] = date(2026, 9, 11)
    p3, _ = svc.handle({"lat": 10.3039, "lon": 123.9131})
    assert counter["n"] == 2 and p3["from_cache"] is False and p3["as_of_local"].startswith("2026-09-11")


def test_cache_is_bounded():
    grid, (gx0, gy0) = make_grid()
    svc = make_service()
    from pyproj import Transformer
    tr = Transformer.from_crs(GRID_CRS, "EPSG:4326", always_xy=True)
    for i in range(-1, 2):
        for j in range(-1, 2):
            lon, lat = tr.transform((gx0 + i) * 1000 + 500, (gy0 + j) * 1000 + 500)
            assert svc.handle({"lat": lat, "lon": lon})[1] == 200
    assert len(svc.cache) == 3  # cache_max_entries=3 in make_service
    assert MOBILE_CACHE_MAX_ENTRIES >= 100


def test_concurrent_identical_requests_compute_once():
    counter = {"n": 0}
    svc = make_service(counter=counter, compute_delay=0.4)
    results = []
    def worker():
        results.append(svc.handle({"lat": 10.3039, "lon": 123.9131}))
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert counter["n"] == 1
    assert all(s == 200 and p["pm25_ugm3"] == 12.3 for p, s in results)
    assert svc.cache.stats["waits"] == 7 and svc.cache.stats["computations"] == 1


def test_concurrent_failure_propagates_to_waiters_and_allows_retry():
    calls = {"n": 0}
    svc = make_service(compute_delay=0.2)
    def flaky(grid_id, t):
        calls["n"] += 1
        time.sleep(0.2)
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return {"pm25": 7.0, "retain_key": None, "n_rows": 1}
    svc.compute_prediction = flaky
    results = []
    threads = [threading.Thread(target=lambda: results.append(svc.handle({"lat": 10.3039, "lon": 123.9131}))) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert all(s == 500 for _, s in results) and calls["n"] == 1
    p, s = svc.handle({"lat": 10.3039, "lon": 123.9131})  # retry succeeds and gets cached
    assert s == 200 and p["pm25_ugm3"] == 7.0 and calls["n"] == 2


def test_periodic_availability_refresh_advances_tstar_through_the_production_wiring(monkeypatch):
    """Regression guard for the refresh path as deployed (app.py wires the
    service to data_availability.get_availability_state, and the hourly
    loop calls refresh_availability_once): a request after the refresh
    discovers a newer latest_valid_date must select the new T*, miss the
    (cell, T*) cache and recompute; an identical request before it is a
    cache hit with the old T*. A failed refresh keeps the last good bound."""
    from backend.prediction import data_availability as da

    counter = {"n": 0}
    grid, _ = make_grid()

    def compute(grid_id, tstar_local):
        counter["n"] += 1
        return {"pm25": 9.0, "retain_key": None, "n_rows": 1, "compute_seconds": 0.0}

    svc = mobile.MobileService(mobile.CellIndex(grid), da.get_availability_state, compute, lambda *a: None, cache_max_entries=3)
    saved = dict(da._state)
    try:
        latest = {"d": date(2026, 9, 11)}
        monkeypatch.setattr(da, "check_dynamic_data_availability",
                            lambda log_fn=print: {"latest_valid_date": latest["d"], "binding_dataset": "no2"})
        assert da.refresh_availability_once(log_fn=lambda m: None)

        p1, s1 = svc.handle({"lat": 10.3039, "lon": 123.9131})
        p2, _ = svc.handle({"lat": 10.3039, "lon": 123.9131})
        assert s1 == 200 and p1["as_of_local_label"] == "Sep 11, 2026, 11:00 PM PHT"
        assert p1["from_cache"] is False and p2["from_cache"] is True and counter["n"] == 1

        # GEE now covers Sep 12 -> the hourly refresh advances the bound.
        latest["d"] = date(2026, 9, 12)
        assert da.refresh_availability_once(log_fn=lambda m: None)
        p3, _ = svc.handle({"lat": 10.3039, "lon": 123.9131})
        assert p3["as_of_local_label"] == "Sep 12, 2026, 11:00 PM PHT"
        assert p3["as_of_utc"] == "2026-09-12T15:00:00+00:00"
        assert p3["from_cache"] is False and counter["n"] == 2

        # A failed refresh keeps the last good bound (Sep 12), served from cache.
        def boom(log_fn=print):
            raise da.DataAvailabilityError("GEE down")
        monkeypatch.setattr(da, "check_dynamic_data_availability", boom)
        assert not da.refresh_availability_once(log_fn=lambda m: None)
        p4, _ = svc.handle({"lat": 10.3039, "lon": 123.9131})
        assert p4["as_of_local_label"] == "Sep 12, 2026, 11:00 PM PHT" and p4["from_cache"] is True
        assert da.get_availability_state()["refresh_error"] is not None
    finally:
        with da._lock:
            da._state.clear()
            da._state.update(saved)
