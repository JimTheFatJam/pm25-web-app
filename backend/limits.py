"""
backend/limits.py
============================================================================
Abuse controls for the public deployment (https://pm25.dcism.org/).

WHAT IS ACTUALLY AT RISK
------------------------
Not CPU. The expensive, rate-limited, *billable* resource is Google Earth
Engine: one prediction job issues many reduceRegions calls against the
project in config.settings.GEE_PROJECT_ID. An unthrottled public endpoint
lets anyone burn that project's quota -- and the grid has ~62,000 cells, so
a script walking it would mint a new (cell, T*) cache key every single time
and never hit the mobile cache.

So the controls here are layered, cheapest first:

  1. per-IP rate limits          -- stop one client hammering the service
  2. global concurrency cap      -- bound simultaneous GEE work
  3. global daily budget         -- hard circuit breaker on total GEE jobs
  4. request-size caps           -- bound the work a single call can ask for

nginx also applies its own `limit_req` (see deploy/nginx-pm25.conf) as
defence in depth: that one is shared across everything, this one knows
which endpoint is expensive.

SINGLE PROCESS, BY DESIGN
-------------------------
These counters are in-process, which is correct here because the whole app
already is: job_state._jobs, explanation_store._stores, the mobile
PredictionCache and the availability state are all module-level. The app
MUST run as one process (gunicorn --workers 1 --threads N); with multiple
workers a status poll could land on a worker that never ran the job. See
DEPLOYMENT.md.

DISABLED BY DEFAULT
-------------------
Everything here is inert unless PM25_PUBLIC_MODE=1, so local development
and the integration tests are untouched.
"""

import os
import threading
import time
from collections import defaultdict, deque
from functools import wraps

from flask import jsonify, request

PUBLIC_MODE = os.environ.get("PM25_PUBLIC_MODE", "").strip() in ("1", "true", "yes", "on")


def _int_env(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# Per-IP limits: bucket -> (max_requests, window_seconds). Tuned for a public
# thesis demo: a human exploring the map never comes close, a script does.
RATE_LIMITS = {
    "predict": [(_int_env("PM25_LIMIT_PREDICT_HOUR", 5), 3600),
                (_int_env("PM25_LIMIT_PREDICT_DAY", 20), 86400)],
    "mobile":  [(_int_env("PM25_LIMIT_MOBILE_HOUR", 30), 3600),
                (_int_env("PM25_LIMIT_MOBILE_DAY", 200), 86400)],
    "read":    [(_int_env("PM25_LIMIT_READ_HOUR", 600), 3600)],
    # Job-status polling and cancel: in-memory only, no GEE, and a single
    # 3-minute job legitimately polls ~90 times.
    "poll":    [(_int_env("PM25_LIMIT_POLL_HOUR", 1200), 3600)],
}

MAX_CONCURRENT_PREDICTION_JOBS = _int_env("PM25_MAX_CONCURRENT_JOBS", 2)
DAILY_GEE_JOB_BUDGET = _int_env("PM25_DAILY_GEE_JOB_BUDGET", 300)
MAX_CELLS_PER_REQUEST = _int_env("PM25_MAX_CELLS_PER_REQUEST", 2000)

# Bound the per-IP table so the limiter itself cannot be used to exhaust
# memory (one entry is a small deque of timestamps).
MAX_TRACKED_CLIENTS = 20000


class SlidingWindowLimiter:
    """Per-key request timestamps, trimmed on read. Exact (not a token
    bucket) so the published limit is the real limit."""

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key, rules):
        """Returns (allowed, retry_after_seconds)."""
        now = time.time()
        longest = max(w for _, w in rules)
        with self._lock:
            if len(self._hits) > MAX_TRACKED_CLIENTS:
                self._evict_locked(now, longest)
            dq = self._hits[key]
            while dq and now - dq[0] > longest:
                dq.popleft()
            for limit, window in rules:
                in_window = sum(1 for t in dq if now - t <= window)
                if in_window >= limit:
                    oldest = next(t for t in dq if now - t <= window)
                    return False, max(1, int(window - (now - oldest)) + 1)
            dq.append(now)
            return True, 0

    def _evict_locked(self, now, longest):
        for k in [k for k, dq in self._hits.items() if not dq or now - dq[-1] > longest]:
            del self._hits[k]


class DailyBudget:
    """Global circuit breaker: total GEE-touching jobs per UTC day."""

    def __init__(self, limit):
        self.limit = limit
        self._day = None
        self._used = 0
        self._lock = threading.Lock()

    def _roll(self):
        day = int(time.time() // 86400)
        if day != self._day:
            self._day, self._used = day, 0

    def remaining(self):
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used)

    def consume(self, n=1):
        with self._lock:
            self._roll()
            if self._used + n > self.limit:
                return False
            self._used += n
            return True

    def stats(self):
        with self._lock:
            self._roll()
            return {"used": self._used, "limit": self.limit, "remaining": max(0, self.limit - self._used)}


_limiter = SlidingWindowLimiter()
_budget = DailyBudget(DAILY_GEE_JOB_BUDGET)
_jobs_in_flight = threading.Semaphore(MAX_CONCURRENT_PREDICTION_JOBS)
_in_flight_count = {"n": 0}
_in_flight_lock = threading.Lock()


def client_ip():
    """The real client address. nginx sets X-Forwarded-For and the app is
    wrapped in ProxyFix, so request.remote_addr is already correct -- this
    falls back explicitly in case the app is ever run without the proxy."""
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_limited(bucket):
    """Per-IP rate limit for a route. No-op unless PM25_PUBLIC_MODE=1."""
    rules = RATE_LIMITS[bucket]

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not PUBLIC_MODE:
                return fn(*args, **kwargs)
            allowed, retry_after = _limiter.check(f"{bucket}:{client_ip()}", rules)
            if not allowed:
                limit, window = rules[0]
                resp = jsonify({
                    "status": "error",
                    "error_code": "rate_limited",
                    "message": (f"Too many requests. This is a public research demo with a limited "
                                f"Earth Engine quota, so predictions are capped per visitor "
                                f"({limit} per {window // 3600} hour(s)). Try again in "
                                f"{retry_after // 60 + 1} minute(s)."),
                })
                resp.status_code = 429
                resp.headers["Retry-After"] = str(retry_after)
                return resp
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def budget_available():
    """False when the global daily GEE budget is exhausted."""
    return not PUBLIC_MODE or _budget.remaining() > 0


def consume_budget(n=1):
    """Record GEE-touching work. Always returns True when not public."""
    return True if not PUBLIC_MODE else _budget.consume(n)


def budget_error():
    resp = jsonify({
        "status": "error",
        "error_code": "daily_budget_exhausted",
        "message": ("This public demo has reached its Earth Engine budget for today. The limit "
                    "protects a research project's quota. Please try again tomorrow."),
    })
    resp.status_code = 503
    return resp


class JobSlot:
    """Bounds concurrent prediction jobs. Usable as a context manager for
    synchronous work (the mobile endpoint), or acquire()/release() for the
    web endpoint, whose job outlives the request in a background thread.
    `acquired` is False when the service is already at capacity."""

    def __init__(self, blocking=False):
        self.blocking = blocking
        self.acquired = False

    def acquire(self):
        if not PUBLIC_MODE:
            self.acquired = True
            return True
        self.acquired = _jobs_in_flight.acquire(blocking=self.blocking)
        if self.acquired:
            with _in_flight_lock:
                _in_flight_count["n"] += 1
        return self.acquired

    def release(self):
        if self.acquired and PUBLIC_MODE:
            _jobs_in_flight.release()
            with _in_flight_lock:
                _in_flight_count["n"] -= 1
        self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def busy_error():
    resp = jsonify({
        "status": "error",
        "error_code": "server_busy",
        "message": (f"The service is already running its maximum of {MAX_CONCURRENT_PREDICTION_JOBS} "
                    f"simultaneous predictions. Each one retrieves live satellite data and takes "
                    f"up to a few minutes. Please try again shortly."),
    })
    resp.status_code = 503
    return resp


def too_many_cells_error(n):
    resp = jsonify({
        "status": "error",
        "error_code": "request_too_large",
        "message": (f"This request covers {n:,} grid cells; the public demo allows at most "
                    f"{MAX_CELLS_PER_REQUEST:,}. Select a smaller area."),
    })
    resp.status_code = 413
    return resp


def status():
    """Diagnostics for /api/health."""
    with _in_flight_lock:
        running = _in_flight_count["n"]
    return {
        "public_mode": PUBLIC_MODE,
        "jobs_running": running,
        "max_concurrent_jobs": MAX_CONCURRENT_PREDICTION_JOBS,
        "daily_gee_budget": _budget.stats() if PUBLIC_MODE else None,
        "max_cells_per_request": MAX_CELLS_PER_REQUEST,
    }


def _reset_for_tests():
    global _limiter, _budget
    _limiter = SlidingWindowLimiter()
    _budget = DailyBudget(DAILY_GEE_JOB_BUDGET)
