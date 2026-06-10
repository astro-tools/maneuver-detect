"""Tests for ``maneuver_detect.data.ratelimit`` — the injectable-clock rate limiter."""

from __future__ import annotations

import pytest

from maneuver_detect.data.ratelimit import RateLimiter


class _FakeClock:
    """A controllable monotonic clock whose ``sleep`` advances it and records the waits."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _limiter(clock: _FakeClock, min_interval_s: float) -> RateLimiter:
    return RateLimiter(min_interval_s, monotonic=clock.monotonic, sleep=clock.sleep)


def test_first_acquire_never_waits() -> None:
    clock = _FakeClock()
    _limiter(clock, 10.0).acquire()
    assert clock.sleeps == []


def test_second_acquire_waits_the_remaining_interval() -> None:
    clock = _FakeClock()
    limiter = _limiter(clock, 10.0)
    limiter.acquire()  # t=0, no wait
    clock.now = 3.0  # only 3s elapsed
    limiter.acquire()
    assert clock.sleeps == [pytest.approx(7.0)]  # waited the missing 7s


def test_no_wait_when_interval_already_elapsed() -> None:
    clock = _FakeClock()
    limiter = _limiter(clock, 10.0)
    limiter.acquire()
    clock.now = 100.0  # well past the interval
    limiter.acquire()
    assert clock.sleeps == []


def test_spacing_is_measured_between_releases() -> None:
    clock = _FakeClock()
    limiter = _limiter(clock, 5.0)
    limiter.acquire()  # release at t=0
    clock.now = 2.0
    limiter.acquire()  # waits 3 -> released at t=5
    clock.now = 6.0
    limiter.acquire()  # only 1s since last release -> waits 4
    assert clock.sleeps == [pytest.approx(3.0), pytest.approx(4.0)]


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_non_positive_interval_disables_pacing(interval: float) -> None:
    clock = _FakeClock()
    limiter = _limiter(clock, interval)
    limiter.acquire()
    clock.now = 0.0
    limiter.acquire()
    assert clock.sleeps == []
