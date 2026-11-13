"""
backend/prediction/data_availability.py
============================================================================
Determines the latest Philippine calendar date for which a prediction can
actually be constructed, based on the REAL, LIVE availability of every
dynamic GEE dataset the trained model requires -- not an assumed,
hardcoded cutoff.

============================================================================
REVISION: the previous version hung indefinitely at startup. Root cause,
traced (not assumed):

The previous version called
    ee.ImageCollection(collection_id).sort("system:time_start", False).first()
on the COMPLETELY UNFILTERED collection. This pattern is already used
safely elsewhere in this codebase (static_predictors.py, for NDVI and
VIIRS nighttime lights) -- but those are small, MONTHLY-cadence
collections with at most a few hundred images total. Applied instead to
MODIS/061/MCD19A2_GRANULES (daily granules, multiple per day globally,
since 2000) or ECMWF/ERA5_LAND/HOURLY and ECMWF/ERA5/HOURLY (hourly,
since 1950/1940), the same pattern asks GEE to sort a collection that can
hold many millions of images, with no index to short-circuit that sort.
Every official GEE ImageCollection example that combines .sort() with
"most recent image" first narrows the collection with .filterDate() (or
another filter) -- see e.g. developers.google.com/earth-engine/guides/
ic_info and ic_iterating -- specifically because .sort() over a large,
unfiltered collection is expensive; this was the confirmed cause of the
startup hang, not merely a slow-but-eventually-returning query.

FIX: for each dataset, filter to a small, RECENT date window first (a
few hundred images at most), sort only within that bounded window, and
only widen the window (a bounded number of times) if it comes back
empty -- never sort the full, unfiltered collection. A hard per-query
timeout (via a worker thread, see _run_with_timeout) is kept as a
defensive backstop in case some other collection behaves unexpectedly,
but the windowing fix is what actually makes each query fast; the
timeout is not a substitute for it.

ARCHITECTURE: this check is no longer run at blocking, module-level
Flask startup. It is launched in a background daemon thread (the same
pattern already used for prediction jobs in job_state.py) immediately
after the app starts, so the app itself is never blocked from serving
requests. See backend/app.py's AvailabilityState for how the result is
shared with request handlers once it completes.
============================================================================

Run ONCE per process lifetime (from a background thread at startup, see
backend/app.py); the result must be cached by the caller and never
re-queried per prediction request.
"""

import threading
import time
from datetime import datetime, timedelta, timezone

import ee

from config.settings import LOCAL_TIMEZONE
from backend.prediction.dynamic_predictors import (
    MODIS_AOD_COLLECTION, S5P_GASES, ERA5_LAND_COLLECTION, ERA5_COLLECTION,
)

# Every dynamic collection the trained Baseline + Group D feature schema
# actually requires -- confirmed exact IDs from dynamic_predictors.py,
# not re-chosen here. All must have data at a given timestamp for a
# prediction there to be constructible, so the binding constraint is
# whichever of these has the EARLIEST "latest available" timestamp.
DAILY_COLLECTIONS = {"aod": MODIS_AOD_COLLECTION}
DAILY_COLLECTIONS.update({name: coll for name, (coll, band) in S5P_GASES.items()})
HOURLY_COLLECTIONS = {"era5_land": ERA5_LAND_COLLECTION, "era5": ERA5_COLLECTION}

# Expanding search windows (days back from now), tried in order until one
# yields an image. 14 days covers fast-turnaround products (Sentinel-5P)
# in one shot; 120 days comfortably covers ERA5/ERA5-Land's confirmed
# ~3-month lag; 400 days is a generous final fallback for an unexpectedly
# delayed product. Each window is still tiny relative to these
# collections' full multi-decade history, so sorting within it stays fast
# regardless of which window ends up being needed.
SEARCH_WINDOWS_DAYS = (14, 120, 400)

# Hard wall-clock bound per individual GEE query, as a defensive backstop
# only -- the windowing above is what actually keeps queries fast. If a
# query is still this slow, something is genuinely wrong with that
# specific collection/window, and this raises a clear, diagnosable error
# instead of hanging the caller.
PER_QUERY_TIMEOUT_SECONDS = 30


class DataAvailabilityError(Exception):
    """Raised when the availability check itself cannot be completed --
    e.g. GEE not initialized, an auth/network failure, a query timing
    out, or a dataset returning no image within the widest search
    window. This is distinct from the normal, expected case of a
    dataset's latest data simply being some months old; callers must
    NOT fall back to a silently-assumed date range when this is raised
    (see this module's docstring)."""


def _run_with_timeout(fn, timeout_seconds, description):
    """Runs fn() in a background thread with a hard wall-clock deadline
    on how long THIS function waits for it.

    IMPORTANT, confirmed by testing: concurrent.futures.ThreadPoolExecutor
    used as a context manager does NOT work for this -- its __exit__ calls
    shutdown(wait=True), which blocks until the worker thread finishes
    regardless of whether future.result(timeout=...) already raised,
    completely defeating the timeout when the underlying call is
    genuinely hung. A raw daemon thread with .join(timeout=...) does not
    have this problem: join() returns after at most timeout_seconds
    either way, and an abandoned daemon thread cannot block this
    function from returning, nor block eventual process exit."""
    result_box = {}

    def _worker():
        try:
            result_box["value"] = fn()
        except Exception as e:
            result_box["error"] = e

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        raise DataAvailabilityError(
            f"GEE query for {description} did not complete within {timeout_seconds}s -- "
            f"treating this as a failure rather than continuing to wait."
        )
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("value")


def _latest_timestamp(collection_id, log_fn=print):
    """The most recent system:time_start in a collection, as a UTC
    datetime -- found by filtering to a small, bounded, recent window
    FIRST (expanding it, a bounded number of times, only if empty), then
    sorting within that window. Never sorts the full, unfiltered
    collection -- see module docstring for exactly why.

    The entire per-window GEE interaction (filterDate, sort, first, and
    the final getInfo()) is wrapped by the SAME timeout-protected call --
    confirmed by testing that wrapping only the final .getInfo() is not
    enough: an exception raised by an earlier step (e.g. .first() itself)
    would otherwise escape uncaught and, in the background-thread caller,
    leave the shared state stuck at "pending" indefinitely rather than
    transitioning to "failed"."""
    now = datetime.now(timezone.utc)
    for window_days in SEARCH_WINDOWS_DAYS:
        window_start_iso = (now - timedelta(days=window_days)).isoformat()
        now_iso = now.isoformat()

        def _query(coll_id=collection_id, start_iso=window_start_iso, end_iso=now_iso):
            filtered = ee.ImageCollection(coll_id).filterDate(ee.Date(start_iso), ee.Date(end_iso))
            img = filtered.sort("system:time_start", False).first()
            return img.get("system:time_start").getInfo()

        try:
            millis = _run_with_timeout(_query, PER_QUERY_TIMEOUT_SECONDS, f"{collection_id} ({window_days}d window)")
        except DataAvailabilityError:
            raise
        except Exception as e:
            raise DataAvailabilityError(
                f"Could not query latest-available timestamp for GEE collection '{collection_id}' "
                f"(within a {window_days}-day window): {type(e).__name__}: {e}"
            )

        if millis is not None:
            return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
        log_fn(f"    no image for '{collection_id}' within the last {window_days} days -- widening search window...")

    raise DataAvailabilityError(
        f"GEE collection '{collection_id}' returned no image within the widest search window "
        f"({SEARCH_WINDOWS_DAYS[-1]} days) -- cannot determine its latest availability."
    )


def _latest_safe_local_date(latest_utc_timestamp):
    """Converts a latest-available UTC instant into the latest Philippine
    (Asia/Manila) calendar date whose ENTIRE 00:00-23:59 local span is
    <= that instant. Using the local calendar date the instant merely
    FALLS ON would be wrong whenever the instant is earlier than 23:00
    local time on that date -- the later hours of that same local day
    would then have no data yet."""
    candidate_date = latest_utc_timestamp.astimezone(LOCAL_TIMEZONE).date()
    end_of_candidate_day_utc = (
        datetime.combine(candidate_date, datetime.min.time(), tzinfo=LOCAL_TIMEZONE)
        + timedelta(days=1) - timedelta(seconds=1)
    ).astimezone(timezone.utc)
    if end_of_candidate_day_utc > latest_utc_timestamp:
        candidate_date = candidate_date - timedelta(days=1)
    return candidate_date


def check_dynamic_data_availability(log_fn=print):
    """Queries every dynamic dataset the trained model requires for its
    own latest available timestamp, and computes the latest Philippine
    calendar date for which a prediction can actually be constructed.

    Returns a dict: {
        "per_dataset": {name: {"collection": str, "latest_utc": datetime}},
        "binding_dataset": str,          # whichever dataset is most restrictive
        "overall_latest_utc": datetime,  # min() across all datasets
        "latest_valid_date": date,       # the calendar bound to expose to the UI
        "checked_at_utc": datetime,
    }

    Raises DataAvailabilityError if ANY required collection's query fails
    or times out. Intended to be called ONCE per process lifetime, from
    a background thread (see backend/app.py) -- not at blocking Flask
    startup, and never per prediction request.
    """
    per_dataset = {}
    all_collections = [("daily", name, coll) for name, coll in DAILY_COLLECTIONS.items()] + \
                       [("hourly", name, coll) for name, coll in HOURLY_COLLECTIONS.items()]

    log_fn("Checking live GEE data availability for all dynamic predictors "
           "(bounded recent-window queries, not a full-collection scan)...")

    for cadence, name, coll_id in all_collections:
        log_fn(f"  Availability check: starting {name} ({coll_id})...")
        t0 = datetime.now()
        ts = _latest_timestamp(coll_id, log_fn)
        elapsed = (datetime.now() - t0).total_seconds()
        per_dataset[name] = {"collection": coll_id, "latest_utc": ts}
        log_fn(f"  Availability check: {name} returned in {elapsed:.1f}s -- latest available = {ts.isoformat()}")

    binding_dataset, binding_info = min(per_dataset.items(), key=lambda kv: kv[1]["latest_utc"])
    overall_latest_utc = binding_info["latest_utc"]
    latest_valid_date = _latest_safe_local_date(overall_latest_utc)

    log_fn(f"Binding constraint: '{binding_dataset}' ({per_dataset[binding_dataset]['collection']}), "
           f"latest UTC timestamp {overall_latest_utc.isoformat()}.")
    log_fn(f"Calculated latest valid prediction date (Philippine calendar day): {latest_valid_date.isoformat()}")

    return {
        "per_dataset": per_dataset,
        "binding_dataset": binding_dataset,
        "overall_latest_utc": overall_latest_utc,
        "latest_valid_date": latest_valid_date,
        "checked_at_utc": datetime.now(timezone.utc),
    }


# ============================================================
# Background-check state -- so the Flask app can start immediately
# without waiting for this check, following the exact same
# threading.Lock()-guarded shared-state pattern already used in
# job_state.py for prediction jobs, applied here to a single shared
# result instead of a dict of many jobs.
# ============================================================
_lock = threading.Lock()
_state = {"status": "pending", "result": None, "error": None, "refresh_error": None}


def get_availability_state():
    """Thread-safe snapshot of the current state: {"status": "pending"
    | "ready" | "failed", "result": <dict from check_dynamic_data_
    availability() or None>, "error": <str or None>}. Safe to call from
    any request handler at any time -- never triggers a GEE query
    itself."""
    with _lock:
        return dict(_state)


def start_availability_check_in_background(log_fn=print):
    """Starts check_dynamic_data_availability() in a daemon background
    thread and returns immediately -- the Flask app is never blocked
    waiting for this. Call ONCE, at application startup. Updates the
    shared state (read via get_availability_state()) when the check
    completes, however long that takes."""
    def _run():
        try:
            result = check_dynamic_data_availability(log_fn=log_fn)
            with _lock:
                _state["status"] = "ready"
                _state["result"] = result
                _state["error"] = None
        except Exception as e:  # noqa: BLE001 -- deliberately broad: this is the top-level
                                  # background-check boundary (same reasoning as job_state.py's
                                  # run_job_in_background). Confirmed by testing that narrowing
                                  # this to DataAvailabilityError only left the shared state stuck
                                  # at "pending" forever if an unanticipated exception type
                                  # escaped check_dynamic_data_availability -- every exception
                                  # type must transition the state to "failed", not just the
                                  # ones this module explicitly anticipated.
            log_fn(f"  FAILED to determine dynamic-data availability: {type(e).__name__}: {e}")
            log_fn("  Prediction requests will be refused until this is fixed and the backend is restarted "
                   "-- see the error above for the specific GEE collection/reason.")
            with _lock:
                _state["status"] = "failed"
                _state["result"] = None
                _state["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def refresh_availability_once(log_fn=print):
    """Re-runs the check and, on success, replaces the shared result
    (status -> "ready"). On failure the LAST GOOD result is kept and the
    error is recorded under "refresh_error" -- a transient GEE problem
    must not take an already-working bound away from the running app.
    Returns True on success."""
    try:
        result = check_dynamic_data_availability(log_fn=log_fn)
        with _lock:
            _state["status"] = "ready"
            _state["result"] = result
            _state["error"] = None
            _state["refresh_error"] = None
        return True
    except Exception as e:  # noqa: BLE001 -- same top-level boundary reasoning as above
        log_fn(f"  Availability refresh failed (keeping the previous result): {type(e).__name__}: {e}")
        with _lock:
            _state["refresh_error"] = f"{type(e).__name__}: {e}"
        return False


def start_availability_refresh_loop(interval_seconds, log_fn=print):
    """Periodic refresh in a daemon thread so a newer latest-valid date
    (mobile T*) is discovered without restarting Flask. The first run
    happens after one interval (the startup check already ran). Cheap:
    each run is the same ~8 bounded queries as the startup check."""
    def _loop():
        while True:
            time.sleep(interval_seconds)
            log_fn("Refreshing live GEE dynamic-data availability (periodic)...")
            refresh_availability_once(log_fn=log_fn)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread
