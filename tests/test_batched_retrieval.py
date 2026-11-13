"""
tests/test_batched_retrieval.py -- numerical equivalence of the batched
GEE retrieval (backend/prediction/dynamic_predictors_batched.py) against
the existing per-timestamp retrieval (dynamic_predictors.py), on REAL
grid cells and a real window, via LIVE Google Earth Engine.

Skipped automatically when GEE cannot be initialised or the real grid is
absent. When it runs it also prints call counts and wall-clock timings for
both implementations (the numbers quoted in the mobile latency report).

Equivalence criteria: identical row keys and columns; identical
None/NaN positions; values equal within rtol=1e-6 / atol=1e-9.
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import _ee_windows_compat  # noqa: F401  (must precede `import ee`)
import ee  # noqa: E402

from config.settings import GEE_PROJECT_ID, OFFICIAL_GRID_PATH  # noqa: E402


def _gee_ready():
    if not OFFICIAL_GRID_PATH.exists():
        return False
    try:
        ee.Initialize(project=GEE_PROJECT_ID) if GEE_PROJECT_ID else ee.Initialize()
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gee_ready(), reason="live GEE / real grid not available")

# Two real cells from the live web runs (Cebu City and Batangas City), and a
# window fully inside the availability bound observed on 2026-09-20.
TEST_CELLS = ["PHFULL_-122_2895", "PHFULL_-445_3264"]
T_END = datetime(2026, 9, 10, 9, 0, tzinfo=timezone.utc)  # 17:00 PHT


@pytest.fixture(scope="module")
def cells():
    from grid.grid_provider import load_grid_with_fallback
    import config.settings as settings
    grid = load_grid_with_fallback()
    settings.ANALYSIS_CRS = grid.crs
    sub = grid[grid["grid_id"].isin(TEST_CELLS)][["grid_id", "grid_x_index", "grid_y_index"]].reset_index(drop=True)
    assert len(sub) == len(TEST_CELLS)
    return sub, grid.crs


def _compare(df_a, df_b, key_cols, label):
    a = df_a.sort_values(key_cols).reset_index(drop=True)
    b = df_b.sort_values(key_cols).reset_index(drop=True)
    assert list(a.columns) == list(b.columns), f"{label}: column sets differ"
    assert len(a) == len(b), f"{label}: row counts differ"
    for c in key_cols:
        assert (a[c].astype(str) == b[c].astype(str)).all(), f"{label}: key column {c} differs"
    value_cols = [c for c in a.columns if c not in key_cols]
    n_values = n_none = 0
    for c in value_cols:
        va = pd.to_numeric(a[c], errors="coerce").to_numpy(dtype=float)
        vb = pd.to_numeric(b[c], errors="coerce").to_numpy(dtype=float)
        na, nb = np.isnan(va), np.isnan(vb)
        assert (na == nb).all(), f"{label}: missing-value positions differ in {c}"
        np.testing.assert_allclose(va[~na], vb[~nb], rtol=1e-6, atol=1e-9, err_msg=f"{label}: values differ in {c}")
        n_values += len(va); n_none += int(na.sum())
    return n_values, n_none


def test_hourly_batched_equals_existing(cells):
    from backend.prediction import dynamic_predictors as dp
    from backend.prediction import dynamic_predictors_batched as dpb
    cells_df, crs = cells
    quiet = lambda m: None
    t = time.time(); existing = dp.fetch_hourly_predictors(cells_df, crs, T_END, T_END, log_fn=quiet); t_existing = time.time() - t
    t = time.time(); batched = dpb.fetch_hourly_predictors_batched(cells_df, crs, T_END, T_END, log_fn=quiet); t_batched = time.time() - t
    n_values, n_none = _compare(existing, batched, ["grid_id", "datetime_utc"], "hourly")
    n_hours = len(existing) // len(cells_df)
    print(f"\nHOURLY equivalence OK: {len(cells_df)} cells x {n_hours} hours, {n_values} values ({n_none} missing in both). "
          f"existing: {2 * n_hours} calls in {t_existing:.1f}s | batched: 2 calls in {t_batched:.1f}s")


def test_daily_batched_equals_existing(cells):
    from backend.prediction import dynamic_predictors as dp
    from backend.prediction import dynamic_predictors_batched as dpb
    cells_df, crs = cells
    quiet = lambda m: None
    d = T_END.date()
    t = time.time(); existing = dp.fetch_daily_predictors(cells_df, crs, d, d, log_fn=quiet); t_existing = time.time() - t
    t = time.time(); batched = dpb.fetch_daily_predictors_batched(cells_df, crs, d, d, log_fn=quiet); t_batched = time.time() - t
    n_values, n_none = _compare(existing, batched, ["grid_id", "date"], "daily")
    n_dates = len(existing) // len(cells_df)
    print(f"\nDAILY equivalence OK: {len(cells_df)} cells x {n_dates} dates, {n_values} values ({n_none} missing in both). "
          f"existing: {2 * n_dates} calls in {t_existing:.1f}s | batched: 2 calls in {t_batched:.1f}s")


def test_missing_hour_semantics_match(cells):
    """An hour with NO ERA5 image (far beyond availability): existing code
    records None for every variable; batched must do the same instead of
    failing the whole chunk."""
    from backend.prediction import dynamic_predictors as dp
    from backend.prediction import dynamic_predictors_batched as dpb
    cells_df, crs = cells
    quiet = lambda m: None
    future = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(days=30)
    # Only the requested hour (lookback also missing): both should be all-None.
    existing = dp.fetch_hourly_predictors(cells_df.iloc[:1], crs, future, future, log_fn=quiet)
    batched = dpb.fetch_hourly_predictors_batched(cells_df.iloc[:1], crs, future, future, log_fn=quiet)
    _compare(existing, batched, ["grid_id", "datetime_utc"], "hourly-missing")
    vals = [c for c in batched.columns if c not in ("grid_id", "datetime_utc")]
    assert batched[vals].isna().all().all()
