"""
reproduce_real_request.py
============================================================================
Direct reproduction of a real prediction request's dynamic-data fetch --
unlike diagnose_dynamic_retrieval.py (which reimplemented similar logic
inline for a single isolated timestamp), this script calls the ACTUAL
production functions, dynamic_predictors.fetch_hourly_predictors() and
fetch_daily_predictors(), directly, across a realistic multi-hour /
multi-day span matching the scale of the requests that actually failed
(~21 cells; a 48-hour hourly window; a 9-day daily window).

Why this script exists: the previous diagnostic tested ONE isolated
timestamp in a short-lived script and found ERA5/ERA5-Land/Sentinel-5P
all valid, live, at the exact date/hour that failed in the real app.
That rules out data availability, geometry, and the ee.Image.cat()/crs=
question as the cause. What it does NOT rule out is something that only
emerges during the real function's own longer loop across many
hours/days -- e.g. GEE-side rate limiting after many rapid sequential
calls, a state bug across loop iterations, or a transient network issue
that a single isolated call wouldn't hit. The original reports failed at
DIFFERENT batch positions in different requests (batch 1/9 daily,
batch 31/48 hourly) -- inconsistent with a fixed, reproducible per-
timestamp cause, and consistent with something loop-dependent instead.

This calls fetch_hourly_predictors()/fetch_daily_predictors() with
DEBUG_DYNAMIC_RETRIEVAL enabled and no batch cap, so EVERY batch gets a
full per-variable valid/null breakdown -- not just the first two, since
finding exactly where (if anywhere) it breaks down is the whole point.
Each log line is timestamped so a rate-limiting/slowdown pattern would
be directly visible.

Makes NO changes to the application, model, feature schema, or
prediction logic. Read-only diagnostics; does not write any prediction
results anywhere.
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import _ee_windows_compat  # noqa -- must precede `import ee`, matches app.py's order

import ee
from config.settings import GEE_PROJECT_ID, LOCAL_TIMEZONE
import config.settings as _settings
from grid.grid_provider import load_grid_with_fallback
from backend.prediction import dynamic_predictors as dp

print("Initializing GEE...")
if GEE_PROJECT_ID:
    ee.Initialize(project=GEE_PROJECT_ID)
else:
    ee.Initialize()
print("  OK.\n")

print("Loading the real grid file...")
GRID = load_grid_with_fallback()
_settings.ANALYSIS_CRS = GRID.crs
grid_crs = GRID.crs
print(f"  Loaded {len(GRID):,} cells.\n")

N_CELLS = int(sys.argv[1]) if len(sys.argv) > 1 else 21  # matches 126/6 and 147/7 from the original reports
sample_cells = GRID.iloc[:N_CELLS][["grid_id", "grid_x_index", "grid_y_index"]].reset_index(drop=True)
print(f"Using {len(sample_cells)} real grid cells (matching the original reports' implied cell count).\n")

# Timestamped log wrapper -- makes any slowdown-before-failure pattern
# directly visible, and preserves every line (nothing is dropped or
# summarized) so the exact failing batch, if any, is fully visible.
_t_start = time.time()


def timestamped_log(msg):
    elapsed = time.time() - _t_start
    print(f"[t+{elapsed:6.1f}s] {msg}")


# Full detail for every batch -- overriding the default 2-batch cap,
# since finding exactly where (if anywhere) this breaks down is the
# entire purpose of this script.
dp.DEBUG_DYNAMIC_RETRIEVAL = True
dp.DEBUG_MAX_BATCHES = 10_000
dp._debug_batch_count["n"] = 0


# ============================================================
# Reproduction 1: hourly, 48-hour window ending 2026-08-24T22:00 UTC --
# matches the original "batch 31/48" report exactly (24h lookback + 24h
# requested = 48 total hourly batches).
# ============================================================
print("=" * 70)
print("REPRODUCTION 1: fetch_hourly_predictors(), 48-hour window ending 2026-08-24T22:00 UTC")
print("(calls the REAL, actual production function directly -- not a reimplementation)")
print("=" * 70)

end_dt = datetime(2026, 8, 24, 22, 0, tzinfo=timezone.utc)
start_dt = end_dt - timedelta(hours=23)  # + the function's own 24h lookback = 48 total hours

try:
    hourly_df = dp.fetch_hourly_predictors(sample_cells, grid_crs, start_dt, end_dt, log_fn=timestamped_log)
    print(f"\nCompleted. Result shape: {hourly_df.shape}")
    era5_cols = [c for c in hourly_df.columns if c not in ("grid_id", "datetime_utc")]
    n_total = len(hourly_df) * len(era5_cols)
    n_null = hourly_df[era5_cols].isna().sum().sum()
    print(f"Overall: {n_total - n_null}/{n_total} (cell, hour, variable) values valid across the full 48-hour request.")
except Exception as e:
    print(f"\nEXCEPTION during fetch_hourly_predictors(): {type(e).__name__}: {e}")

# ============================================================
# Reproduction 2: daily, 9-day window ending 2026-08-17 -- matches the
# original "batch 1/9" report exactly (7-day lookback + up to 2 requested
# days = 9 total daily batches).
# ============================================================
print()
print("=" * 70)
print("REPRODUCTION 2: fetch_daily_predictors(), 9-day window ending 2026-08-17")
print("(calls the REAL, actual production function directly -- not a reimplementation)")
print("=" * 70)

dp._debug_batch_count["n"] = 0
dp._debug_raw_response_printed["done"] = False
dp._debug_stage_trace_printed["done"] = False
end_date = datetime(2026, 8, 17).date()
start_date = end_date - timedelta(days=1)  # + the function's own 7-day lookback = 9 total daily batches

try:
    daily_df = dp.fetch_daily_predictors(sample_cells, grid_crs, start_date, end_date, log_fn=timestamped_log)
    print(f"\nCompleted. Result shape: {daily_df.shape}")
    gas_cols = [c for c in daily_df.columns if c not in ("grid_id", "date")]
    n_total = len(daily_df) * len(gas_cols)
    n_null = daily_df[gas_cols].isna().sum().sum()
    print(f"Overall: {n_total - n_null}/{n_total} (cell, date, variable) values valid across the full 9-day request.")
except Exception as e:
    print(f"\nEXCEPTION during fetch_daily_predictors(): {type(e).__name__}: {e}")

print()
print("=" * 70)
print("Reproduction complete. Look for:")
print("  1. Whether EITHER real function reproduces a '0 ok / N missing' batch at all.")
print("  2. If so, at which batch number/timestamp, and whether elapsed time (t+Xs) shows a")
print("     slowdown beforehand (would support a rate-limiting/throttling explanation).")
print("  3. Whether any EXCEPTION was printed (would appear as 'GEE ERROR fetching...' in the")
print("     normal log, or as an uncaught EXCEPTION line above) -- this changes the diagnosis")
print("     from 'silently returns null' to 'GEE actively rejected the request'.")
print("=" * 70)
