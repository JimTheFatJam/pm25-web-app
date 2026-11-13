"""
backend/prediction/job_state.py
============================================================================
Minimal in-memory job tracking for the FIRST diagnostic prediction test.

Deliberately NOT a production job queue: no persistence, no multi-worker
support, no external broker. A single Python dict, guarded by a lock, plus
one background thread per running job. This is intentionally the smallest
mechanism that gives the frontend REAL progress and REAL cancellation
(per explicit instruction: no fake percentage progress, and no
sophisticated job queue at this stage) -- appropriate for one developer
running one test at a time, not for concurrent production users. See the
deployment plan's Section 11 "requires a design decision" / "must remain
unimplemented" categories for what a production version of this would
need on top of this (persistence, multi-worker safety, etc).
"""

import threading
import time
import uuid
from datetime import datetime, timezone

STAGES = [
    "Validating request",
    "Preparing selected cells",
    "Loading static predictors",
    "Retrieving dynamic GEE predictors",
    "Building feature matrix",
    "Running PM2.5 model inference",
    "Preparing prediction response",
]
TOTAL_STAGES = len(STAGES)

_jobs = {}
_lock = threading.Lock()

# MEMORY LEAK FIX (this revision): _jobs previously never removed a
# completed/errored/cancelled job -- every prediction run in the
# process's entire lifetime stayed in memory permanently, each retaining
# its full log_lines list (dozens to hundreds of lines given the
# per-batch GEE logging) and its full result (the entire predictions
# array). Over many predictions in one long-running backend session
# (e.g. repeated testing), this grows without bound. Fixed by keeping
# only the most recent MAX_RETAINED_JOBS -- old completed jobs are
# evicted once this limit is exceeded, oldest first. A single-user
# diagnostic tool has no real need to keep every historical job
# in-process forever; the console log is the durable record.
MAX_RETAINED_JOBS = 20


def _evict_old_jobs_locked():
    """Must be called while holding _lock. Only evicts jobs that have
    reached a terminal state -- a still-running job is never evicted,
    regardless of how many jobs exist."""
    if len(_jobs) <= MAX_RETAINED_JOBS:
        return
    terminal_ids = [jid for jid, j in _jobs.items() if j["status"] != "running"]
    terminal_ids.sort(key=lambda jid: _jobs[jid]["started_at"])
    n_to_evict = len(_jobs) - MAX_RETAINED_JOBS
    for jid in terminal_ids[:n_to_evict]:
        del _jobs[jid]
        # Also drop any feature rows retained for SHAP explanations of
        # this job (explanation_store has its own lock and never calls
        # back into job_state, so calling it here under _lock is safe).
        try:
            from backend.prediction import explanation_store
            explanation_store.discard(jid)
        except Exception:  # noqa: BLE001 -- optional subsystem; eviction must not fail
            pass


class JobCancelled(Exception):
    """Raised inside the pipeline when a job's cancel flag is observed --
    caught in run_job_in_background to end the job cleanly, not as an error."""


def create_job():
    job_id = str(uuid.uuid4())
    with _lock:
        _evict_old_jobs_locked()
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",       # running | complete | error | cancelled
            "stage_index": 0,
            "stage_name": STAGES[0],
            "started_at": time.time(),
            "started_at_iso": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": 0.0,
            "log_lines": [],
            "cancel_requested": False,
            "result": None,
            "error": None,
        }
    return job_id


def get_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        snapshot = dict(job)
        snapshot["elapsed_seconds"] = round(time.time() - job["started_at"], 1)
        return snapshot


def request_cancel(job_id):
    with _lock:
        if job_id in _jobs and _jobs[job_id]["status"] == "running":
            _jobs[job_id]["cancel_requested"] = True
            return True
    return False


def _is_cancelled(job_id):
    with _lock:
        return _jobs.get(job_id, {}).get("cancel_requested", False)


def check_cancelled(job_id):
    """Call from inside the pipeline between stages/batches. Raises
    JobCancelled if the user has requested cancellation -- callers should
    let this propagate up to run_job_in_background."""
    if _is_cancelled(job_id):
        raise JobCancelled()


def set_stage(job_id, stage_index, extra_log=None):
    with _lock:
        if job_id not in _jobs:
            return
        _jobs[job_id]["stage_index"] = stage_index
        _jobs[job_id]["stage_name"] = STAGES[stage_index]
        line = f"[{stage_index + 1}/{TOTAL_STAGES}] {STAGES[stage_index]}"
        _jobs[job_id]["log_lines"].append(line)
    print(line)
    if extra_log:
        log(job_id, extra_log)


def log(job_id, message):
    """Adds a line to this job's log AND prints it to the console --
    per explicit instruction, detailed console logging is required for
    the first live tests, not just frontend-visible status."""
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["log_lines"].append(message)
    print(f"    {message}")


def set_complete(job_id, result):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["stage_index"] = TOTAL_STAGES
    log(job_id, "Prediction complete.")


def set_error(job_id, error_message, error_type=None):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = {"message": error_message, "type": error_type}
    log(job_id, f"ERROR ({error_type}): {error_message}")


def set_cancelled(job_id):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "cancelled"
    log(job_id, "Job cancelled by user.")


def run_job_in_background(job_id, pipeline_fn, *args, **kwargs):
    """Starts pipeline_fn(job_id, *args, **kwargs) in a background thread.
    Catches JobCancelled (clean stop, no error state) and any other
    exception (recorded as a real error with its actual type, per
    explicit instruction -- never a generic 'something went wrong')."""
    def _run():
        try:
            result = pipeline_fn(job_id, *args, **kwargs)
            set_complete(job_id, result)
        except JobCancelled:
            set_cancelled(job_id)
        except Exception as e:  # noqa: BLE001 -- deliberately broad: this is the top-level
                                  # job boundary, and every exception type must be captured
                                  # and reported to the user, not just anticipated ones.
            set_error(job_id, str(e), error_type=type(e).__name__)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
