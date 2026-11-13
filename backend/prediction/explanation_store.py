"""
backend/prediction/explanation_store.py
============================================================================
Bounded, memory-aware retention of completed jobs' feature rows, so that a
single (cell, hour) row can later be explained with SHAP on demand.

This exists because feature_builder.build_feature_matrix() already
produces exactly the model-input row for every (cell, hour) -- SHAP needs
nothing more -- but pipeline.py previously discarded that matrix after
inference. Retaining the FULL matrix of every job unbounded could consume
hundreds of MB (200,000 rows x 157 float64 = ~250 MB per job), so:

  - rows are stored as float32 (XGBoost consumes float32 internally; the
    test suite checks float32 vs float64 SHAP agree within a tolerance),
  - the index is two small int32 arrays (cell position, timestamp
    position), not a per-row Python dict,
  - a single process-wide byte budget (config.settings
    .SHAP_FEATURE_RETENTION_MAX_MB) bounds total retention; the oldest
    jobs are evicted first when a new job would exceed it, and a job
    larger than the whole budget is not retained at all.

Estimated cost: 157 x 4 B + 8 B index = ~636 B/row -> 1,200 rows (50 cells
x 24 h) = 0.8 MB; 36,000 rows = 23 MB; the 200,000-row ceiling = 127 MB
(exceeds the 64 MB default, so not retained).

STRICTLY NON-CRITICAL TO PREDICTION: every public function here swallows
its own errors and reports an outcome; pipeline.py calls attach() inside
a try/except as well. If this module fails, is disabled, or evicts a job,
the prediction result is untouched -- only /api/prediction_explanation
answers "unavailable" for that job.

Thread-safety: one lock, same pattern as job_state.py. Never calls into
job_state while holding it (job_state calls discard() while holding ITS
lock, so the reverse must never happen).
"""

import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

import config.settings as settings

_lock = threading.Lock()
_stores = OrderedDict()   # job_id -> store dict (insertion order = age)
_total_bytes = 0


def budget_bytes():
    mb = getattr(settings, "SHAP_FEATURE_RETENTION_MAX_MB", 0) or 0
    return int(mb * 1024 * 1024)


def attach(job_id, feature_matrix, feature_order):
    """Retain a completed job's feature rows. feature_matrix is exactly
    feature_builder.build_feature_matrix()'s first return value
    (grid_id, datetime_utc, + feature_order columns). Returns a dict
    {"retained": bool, "reason": str, "bytes": int, "rows": int}."""
    global _total_bytes
    try:
        budget = budget_bytes()
        n_rows = int(len(feature_matrix))
        if budget <= 0:
            return {"retained": False, "reason": "retention disabled (SHAP_FEATURE_RETENTION_MAX_MB <= 0)", "bytes": 0, "rows": n_rows}

        X = np.ascontiguousarray(feature_matrix[feature_order].to_numpy(dtype=np.float32))
        grid_ids = feature_matrix["grid_id"].astype(str).to_numpy()
        # tz-aware UTC -> int64 nanoseconds since epoch (drop tz first so
        # numpy never has to represent it).
        ts_ns = pd.to_datetime(feature_matrix["datetime_utc"], utc=True).dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype("int64")

        cell_ids = list(dict.fromkeys(grid_ids.tolist()))
        cell_pos = {c: i for i, c in enumerate(cell_ids)}
        row_cell = np.fromiter((cell_pos[g] for g in grid_ids), dtype=np.int32, count=n_rows)
        uniq_ts = np.unique(ts_ns)
        row_ts = np.searchsorted(uniq_ts, ts_ns).astype(np.int32)

        nbytes = int(X.nbytes + row_cell.nbytes + row_ts.nbytes + uniq_ts.nbytes)
        if nbytes > budget:
            return {"retained": False, "rows": n_rows, "bytes": nbytes,
                    "reason": f"job's feature rows ({nbytes / 1e6:.1f} MB) exceed the whole retention budget "
                              f"({budget / 1e6:.0f} MB); explanation unavailable for this job"}

        store = {"X": X, "cell_ids": cell_ids, "cell_pos": cell_pos, "uniq_ts": uniq_ts,
                 "row_cell": row_cell, "row_ts": row_ts, "nbytes": nbytes, "rows": n_rows,
                 "feature_order": list(feature_order)}
        with _lock:
            if job_id in _stores:
                _total_bytes -= _stores.pop(job_id)["nbytes"]
            evicted = []
            while _stores and _total_bytes + nbytes > budget:
                old_id, old = _stores.popitem(last=False)  # oldest first
                _total_bytes -= old["nbytes"]
                evicted.append(old_id)
            _stores[job_id] = store
            _total_bytes += nbytes
        reason = "retained" + (f" (evicted {len(evicted)} older job(s) to fit)" if evicted else "")
        return {"retained": True, "reason": reason, "bytes": nbytes, "rows": n_rows}
    except Exception as e:  # noqa: BLE001 -- must never propagate into the prediction job
        return {"retained": False, "reason": f"retention failed: {type(e).__name__}: {e}", "bytes": 0, "rows": 0}


def discard(job_id):
    """Drop a job's rows (called by job_state when it evicts the job)."""
    global _total_bytes
    with _lock:
        store = _stores.pop(job_id, None)
        if store is not None:
            _total_bytes -= store["nbytes"]


def has_job(job_id):
    with _lock:
        return job_id in _stores


def get_row(job_id, grid_id, datetime_utc):
    """Returns (row_float32_ndarray, feature_order) or (None, reason).
    datetime_utc: any string pandas can parse (the result's own
    'YYYY-MM-DD HH:MM:SS+00:00' form or ISO 8601); naive = UTC."""
    with _lock:
        store = _stores.get(job_id)
        if store is None:
            return None, "no feature rows are retained for this job (not retained, evicted, or disabled)"
        ci = store["cell_pos"].get(str(grid_id))
        if ci is None:
            return None, f"grid_id {grid_id!r} is not part of this job"
        try:
            ts = pd.Timestamp(datetime_utc)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            ts_ns = int(ts.value)
        except Exception as e:  # noqa: BLE001
            return None, f"could not parse datetime_utc {datetime_utc!r}: {e}"
        pos = np.searchsorted(store["uniq_ts"], ts_ns)
        if pos >= len(store["uniq_ts"]) or store["uniq_ts"][pos] != ts_ns:
            return None, f"timestamp {datetime_utc!r} is not one of this job's hourly feature rows"
        idx = np.flatnonzero((store["row_cell"] == ci) & (store["row_ts"] == pos))
        if len(idx) == 0:
            return None, "no feature row for that cell/timestamp"
        return store["X"][idx[0]].copy(), store["feature_order"]


def hourly_timestamps(job_id):
    """The UTC hourly timestamps retained for a job (ISO strings), or None."""
    with _lock:
        store = _stores.get(job_id)
        if store is None:
            return None
        return [pd.Timestamp(int(t), unit="ns", tz="UTC").isoformat() for t in store["uniq_ts"]]


def stats():
    with _lock:
        return {"jobs": len(_stores), "total_bytes": _total_bytes, "budget_bytes": budget_bytes()}


def _reset_for_tests():
    global _total_bytes
    with _lock:
        _stores.clear()
        _total_bytes = 0
