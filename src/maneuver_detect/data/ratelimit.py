"""A minimal request-rate limiter for the catalogue fetchers.

Both sources ask callers to pace themselves — Space-Track enforces query-rate limits and CelesTrak
publishes a soft per-IP daily cap and a one-download-per-update discipline. :class:`RateLimiter`
enforces a minimum interval between successive requests: :meth:`acquire` blocks just long enough
that no two calls are closer together than ``min_interval_s``.

The clock and the sleep are injectable so the wait is deterministically testable without real
time passing. Intended for single-threaded (batch) use, matching the v0.1 data layer.
"""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = ["RateLimiter"]


class RateLimiter:
    """Enforce a minimum spacing between successive :meth:`acquire` calls.

    Args:
        min_interval_s: Minimum seconds between two acquisitions. ``0`` (or negative) disables
            pacing — every :meth:`acquire` returns immediately.
        monotonic: Clock source; defaults to :func:`time.monotonic`. Injected in tests.
        sleep: Blocking sleep; defaults to :func:`time.sleep`. Injected in tests.
    """

    def __init__(
        self,
        min_interval_s: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._last: float | None = None

    def acquire(self) -> None:
        """Block until at least ``min_interval_s`` has elapsed since the previous acquisition.

        The first call never waits. Each call records the time it *returns* as the new anchor, so
        the spacing is measured between releases — a request that itself takes a while does not
        earn extra delay on top.
        """
        if self._min_interval_s <= 0:
            return
        now = self._monotonic()
        if self._last is not None:
            wait = self._min_interval_s - (now - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._monotonic()
