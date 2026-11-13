"""
tests/test_limits.py -- the public-deployment abuse controls
(backend/limits.py). Pure logic, no Flask app, no GEE, no model: the
limiter, the global daily budget and the concurrency slot, plus the
guarantee that everything is inert when PM25_PUBLIC_MODE is unset.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import limits
from backend.limits import DailyBudget, SlidingWindowLimiter


def test_sliding_window_allows_up_to_the_limit_then_blocks():
    lim = SlidingWindowLimiter()
    rules = [(3, 3600)]
    for i in range(3):
        allowed, _ = lim.check("ip-a", rules)
        assert allowed, f"request {i + 1} of 3 should be allowed"
    allowed, retry_after = lim.check("ip-a", rules)
    assert not allowed
    assert 0 < retry_after <= 3601


def test_limits_are_per_key():
    lim = SlidingWindowLimiter()
    rules = [(1, 3600)]
    assert lim.check("ip-a", rules)[0]
    assert not lim.check("ip-a", rules)[0]
    assert lim.check("ip-b", rules)[0], "a second client must not inherit the first client's budget"


def test_shorter_window_blocks_before_the_longer_one():
    """The hourly rule must bite before the daily rule."""
    lim = SlidingWindowLimiter()
    rules = [(2, 3600), (10, 86400)]
    assert lim.check("ip", rules)[0]
    assert lim.check("ip", rules)[0]
    allowed, retry_after = lim.check("ip", rules)
    assert not allowed
    assert retry_after <= 3601, "retry hint should point at the hourly window, not the daily one"


def test_window_expiry_lets_requests_through_again(monkeypatch):
    lim = SlidingWindowLimiter()
    rules = [(1, 60)]
    now = [1_000_000.0]
    monkeypatch.setattr(limits.time, "time", lambda: now[0])
    assert lim.check("ip", rules)[0]
    assert not lim.check("ip", rules)[0]
    now[0] += 61
    assert lim.check("ip", rules)[0], "a request outside the window must be allowed"


def test_daily_budget_is_a_hard_ceiling_and_rolls_over(monkeypatch):
    now = [86400.0 * 100]
    monkeypatch.setattr(limits.time, "time", lambda: now[0])
    b = DailyBudget(2)
    assert b.consume() and b.consume()
    assert not b.consume(), "budget must refuse once exhausted"
    assert b.remaining() == 0
    now[0] += 86400  # next UTC day
    assert b.remaining() == 2 and b.consume()


def test_daily_budget_is_thread_safe():
    b = DailyBudget(50)
    granted = []
    lock = threading.Lock()

    def worker():
        ok = b.consume()
        with lock:
            granted.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(granted) == 50, "exactly the budget may be granted, never more"


def test_job_slot_bounds_concurrency_and_releases(monkeypatch):
    monkeypatch.setattr(limits, "PUBLIC_MODE", True)
    monkeypatch.setattr(limits, "_jobs_in_flight", threading.Semaphore(2))
    a, b = limits.JobSlot(), limits.JobSlot()
    assert a.acquire() and b.acquire()
    c = limits.JobSlot()
    assert not c.acquire(), "third concurrent job must be refused"
    a.release()
    assert c.acquire(), "a slot must become available again after release"
    b.release()
    c.release()


def test_job_slot_releases_on_exception(monkeypatch):
    monkeypatch.setattr(limits, "PUBLIC_MODE", True)
    monkeypatch.setattr(limits, "_jobs_in_flight", threading.Semaphore(1))
    try:
        with limits.JobSlot() as slot:
            assert slot.acquired
            raise RuntimeError("job blew up")
    except RuntimeError:
        pass
    with limits.JobSlot() as slot:
        assert slot.acquired, "a failed job must not leak its slot"


def test_everything_is_inert_outside_public_mode(monkeypatch):
    """Local development and the rest of the test suite must be unaffected."""
    monkeypatch.setattr(limits, "PUBLIC_MODE", False)
    assert limits.budget_available()
    for _ in range(1000):
        assert limits.consume_budget()
    slots = [limits.JobSlot() for _ in range(50)]
    assert all(s.acquire() for s in slots), "concurrency is unbounded when not public"
    for s in slots:
        s.release()


def test_configured_defaults_are_sane():
    """Guards against a typo making the public deployment wide open."""
    for bucket in ("predict", "mobile", "read", "poll"):
        rules = limits.RATE_LIMITS[bucket]
        assert rules and all(n > 0 and w > 0 for n, w in rules)
    predict_hourly = limits.RATE_LIMITS["predict"][0][0]
    assert predict_hourly <= 20, "per-IP prediction limit must stay small -- it spends GEE quota"
    assert limits.MAX_CONCURRENT_PREDICTION_JOBS >= 1
    assert limits.DAILY_GEE_JOB_BUDGET >= 1
    assert limits.MAX_CELLS_PER_REQUEST >= 1
