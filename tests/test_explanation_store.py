"""
tests/test_explanation_store.py -- bounded feature-row retention:
exact row round-trip, budget accounting, oldest-first eviction, oversize
job rejection, disabled budget, discard on job eviction, and lookup error
paths. No model or GEE involved.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config.settings as settings
from backend.prediction import explanation_store as es

FEATURES = [f"f{i}" for i in range(157)]


def make_matrix(n_cells=3, n_hours=4, start="2026-08-24 00:00:00+00:00", seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    ts = pd.date_range(start, periods=n_hours, freq="h")
    for c in range(n_cells):
        for t in ts:
            rows.append({"grid_id": f"C{c}", "datetime_utc": t, **{f: rng.rand() for f in FEATURES}})
    df = pd.DataFrame(rows)
    df.loc[0, "f3"] = np.nan  # a missing feature value survives retention as NaN
    return df


def setup_function(_):
    es._reset_for_tests()
    settings.SHAP_FEATURE_RETENTION_MAX_MB = 64


def test_round_trip_exact_row():
    df = make_matrix()
    out = es.attach("job-a", df, FEATURES)
    assert out["retained"] and out["rows"] == 12
    # per-row cost ~ 157*4 + 8 bytes
    assert 12 * 157 * 4 <= out["bytes"] <= 12 * 157 * 4 + 12 * 8 + 12 * 8 + 64
    target = df[(df["grid_id"] == "C1") & (df["datetime_utc"] == pd.Timestamp("2026-08-24 02:00:00+00:00"))].iloc[0]
    row, order = es.get_row("job-a", "C1", "2026-08-24 02:00:00+00:00")
    assert order == FEATURES
    np.testing.assert_allclose(row, target[FEATURES].to_numpy(dtype=np.float32), rtol=0, atol=0)
    # ISO 8601 and naive (UTC-assumed) forms resolve to the same row
    row2, _ = es.get_row("job-a", "C1", "2026-08-24T02:00:00+00:00")
    row3, _ = es.get_row("job-a", "C1", "2026-08-24 02:00:00")
    np.testing.assert_array_equal(row, row2)
    np.testing.assert_array_equal(row, row3)
    nan_row, _ = es.get_row("job-a", "C0", "2026-08-24 00:00:00+00:00")
    assert np.isnan(nan_row[3])


def test_lookup_error_paths():
    es.attach("job-a", make_matrix(), FEATURES)
    assert es.get_row("nope", "C0", "2026-08-24 00:00:00+00:00")[0] is None
    assert es.get_row("job-a", "ZZ", "2026-08-24 00:00:00+00:00")[0] is None
    assert es.get_row("job-a", "C0", "2026-08-24 00:30:00+00:00")[0] is None  # not an hourly row
    assert es.get_row("job-a", "C0", "not-a-date")[0] is None
    assert es.hourly_timestamps("job-a") == [f"2026-08-24T0{h}:00:00+00:00" for h in range(4)]
    assert es.hourly_timestamps("nope") is None


def test_budget_evicts_oldest_first():
    one_job_bytes = es.attach("j1", make_matrix(), FEATURES)["bytes"]
    es._reset_for_tests()
    # Budget fits exactly two such jobs (plus a little slack), not three.
    settings.SHAP_FEATURE_RETENTION_MAX_MB = (2 * one_job_bytes + 100) / (1024 * 1024)
    assert es.attach("j1", make_matrix(seed=1), FEATURES)["retained"]
    assert es.attach("j2", make_matrix(seed=2), FEATURES)["retained"]
    out = es.attach("j3", make_matrix(seed=3), FEATURES)
    assert out["retained"] and "evicted 1" in out["reason"]
    assert not es.has_job("j1") and es.has_job("j2") and es.has_job("j3")
    s = es.stats()
    assert s["jobs"] == 2 and s["total_bytes"] == 2 * one_job_bytes <= s["budget_bytes"]


def test_oversize_job_is_not_retained_and_others_survive():
    one_job_bytes = es.attach("j1", make_matrix(), FEATURES)["bytes"]
    settings.SHAP_FEATURE_RETENTION_MAX_MB = (one_job_bytes + 100) / (1024 * 1024)
    big = make_matrix(n_cells=6)  # twice the size -> exceeds the whole budget
    out = es.attach("big", big, FEATURES)
    assert not out["retained"] and "exceed" in out["reason"]
    assert es.has_job("j1") and not es.has_job("big")


def test_disabled_budget_retains_nothing():
    settings.SHAP_FEATURE_RETENTION_MAX_MB = 0
    out = es.attach("j1", make_matrix(), FEATURES)
    assert not out["retained"] and "disabled" in out["reason"]
    assert es.stats()["jobs"] == 0


def test_discard_and_job_state_eviction_hook():
    es.attach("j1", make_matrix(), FEATURES)
    es.discard("j1")
    assert not es.has_job("j1") and es.stats()["total_bytes"] == 0
    es.discard("j1")  # idempotent

    # job_state's own eviction drops the store too. Isolated from any jobs
    # other tests may have created: start from an empty registry.
    from backend.prediction import job_state
    with job_state._lock:
        job_state._jobs.clear()
    ids = [job_state.create_job() for _ in range(job_state.MAX_RETAINED_JOBS)]
    for jid in ids:
        job_state.set_complete(jid, {"ok": True})
    es.attach(ids[0], make_matrix(), FEATURES)
    assert es.has_job(ids[0])
    # Eviction only happens once the registry EXCEEDS MAX_RETAINED_JOBS at
    # the next create_job(): the (N+1)th job is admitted first, then the
    # oldest terminal job is evicted the time after that.
    job_state.create_job()
    job_state.create_job()
    assert ids[0] not in job_state._jobs
    assert not es.has_job(ids[0])
    with job_state._lock:
        job_state._jobs.clear()


def test_attach_never_raises_on_bad_input():
    out = es.attach("bad", pd.DataFrame({"grid_id": ["a"], "datetime_utc": ["x"]}), FEATURES)
    assert not out["retained"] and "failed" in out["reason"]
