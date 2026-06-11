"""Tests for ``maneuver_detect.datasets.tle_history`` — fetch -> clean -> assemble wiring.

The fetcher is stubbed, so no network or credentials are exercised; the test asserts the accessor
selects the requested source, threads the epoch window through, and returns the canonical
mean-element frame the detector consumes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import maneuver_detect.data as data
from maneuver_detect import datasets
from maneuver_detect.data.base import FetchResult
from maneuver_detect.data.elset import Elset
from maneuver_detect.data.history import MEAN_ELEMENT_COLUMNS


def _elset(norad_id: int, day: int) -> Elset:
    return Elset(
        norad_id=norad_id,
        epoch=datetime(2024, 1, day, tzinfo=timezone.utc),
        mean_motion=15.5,
        eccentricity=0.0006,
        inclination=51.64,
        raan=208.0,
        arg_perigee=130.0,
        mean_anomaly=325.0,
        bstar=0.0001,
        mean_motion_dot=0.00016,
        mean_motion_ddot=0.0,
        element_set_no=900,
        rev_at_epoch=12345,
        classification="U",
        object_id="1998-067A",
    )


class _StubFetcher:
    """Records the source it was built for and the fetch arguments; returns canned elsets."""

    def __init__(self, source: str, calls: list[dict[str, Any]]) -> None:
        self.source = source
        self._calls = calls

    def __enter__(self) -> _StubFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def fetch(
        self, norad_id: int, *, start: str | None = None, end: str | None = None
    ) -> FetchResult:
        self._calls.append(
            {"norad_id": norad_id, "start": start, "end": end, "source": self.source}
        )
        elsets = (_elset(norad_id, 1), _elset(norad_id, 2))
        return FetchResult(
            norad_id=norad_id,
            elsets=elsets,
            fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
            source=self.source,
        )


def test_tle_history_returns_canonical_frame(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(data, "get_fetcher", lambda source: _StubFetcher(source, calls))

    frame = datasets.tle_history(25544, start="2024-01-01", end="2024-01-31")

    assert list(frame.columns) == list(MEAN_ELEMENT_COLUMNS)
    assert len(frame) == 2
    assert frame["norad_id"].iloc[0] == 25544
    # The epoch window and the (default) source are threaded straight to the fetcher.
    assert calls == [
        {"norad_id": 25544, "start": "2024-01-01", "end": "2024-01-31", "source": "spacetrack"}
    ]


def test_tle_history_selects_the_requested_source(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(data, "get_fetcher", lambda source: _StubFetcher(source, calls))

    datasets.tle_history(25544, source="celestrak")

    assert calls[0]["source"] == "celestrak"
