"""Tests for ``maneuver_detect.leaderboard.service`` — the server-side scoring + board policy.

The properties the V7 proof asserts on the scorer, here on the service the Space drives in practice:
response is aggregate-only and leaks no held-out label, the submission channel rejects anything that
is not a predictions file, scoring is deterministic, the board ranks by headline recall, and the
per-user per-UTC-day rate limit bounds submission volume (a courtesy guard now the key is public).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from _leaderboard import build_fixture, honest_predictions, partial_predictions
from maneuver_detect.leaderboard import (
    InvalidSubmissionError,
    LeaderboardService,
    RateLimitError,
)


class _Clock:
    """A settable UTC clock so the per-day rate limit is exercised without sleeping."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def test_response_is_aggregate_only() -> None:
    service = LeaderboardService(build_fixture())
    result = service.submit("u", honest_predictions()).result
    assert set(result) == {
        "operating_point_fa_per_sat_year",
        "headline_recall_above_floor",
        "timing_only_floor_auc",
    }


def test_response_leaks_no_held_out_label_epoch() -> None:
    fixture = build_fixture()
    result = LeaderboardService(fixture).submit("u", honest_predictions()).result
    blob = json.dumps(result, default=str)
    leaked = [
        label.interval.epoch.isoformat()
        for label in fixture.labels
        if label.interval.epoch.isoformat() in blob
    ]
    assert not leaked


def test_submission_channel_rejects_non_predictions() -> None:
    service = LeaderboardService(build_fixture())
    for payload in (
        '[{"epoch": "2026-03-01T00:00:00+00:00", "confidence": 0.9}]',  # missing canonical fields
        '{"give_me": "the labels"}',  # not an array of records
        '"SELECT * FROM labels"',  # a query string, not predictions
        "not json at all",  # not even JSON
    ):
        with pytest.raises(InvalidSubmissionError):
            service.submit("u", payload)


def test_scoring_is_deterministic() -> None:
    service = LeaderboardService(build_fixture())
    first = service.submit("a", honest_predictions()).result
    second = service.submit("b", honest_predictions()).result
    assert first == second


def test_headline_recall_matches_the_recovered_population() -> None:
    result = LeaderboardService(build_fixture()).submit("u", honest_predictions())
    headline = result.result["headline_recall_above_floor"]
    assert isinstance(headline, dict)
    assert headline["LEO"] == 1.0  # all three above-floor LEO maneuvers recovered
    assert headline["GEO"] == 0.0  # no GEO detections submitted
    assert result.entry.headline_recall == pytest.approx(0.6)  # (3*1.0 + 2*0.0) / 5


def test_rate_limit_blocks_after_quota_and_resets_next_utc_day() -> None:
    clock = _Clock(datetime(2026, 3, 1, tzinfo=timezone.utc))
    service = LeaderboardService(build_fixture(), rate_limit_per_day=5, clock=clock)

    remaining = [service.submit("u", honest_predictions()).remaining_today for _ in range(5)]
    assert remaining == [4, 3, 2, 1, 0]
    with pytest.raises(RateLimitError):
        service.submit("u", honest_predictions())

    clock.now = datetime(2026, 3, 2, tzinfo=timezone.utc)
    assert service.submit("u", honest_predictions()).remaining_today == 4


def test_invalid_submission_does_not_consume_quota() -> None:
    clock = _Clock(datetime(2026, 3, 1, tzinfo=timezone.utc))
    service = LeaderboardService(build_fixture(), rate_limit_per_day=5, clock=clock)

    for _ in range(3):
        with pytest.raises(InvalidSubmissionError):
            service.submit("u", "not a predictions file")

    # The full quota is still available.
    for _ in range(5):
        service.submit("u", honest_predictions())
    with pytest.raises(RateLimitError):
        service.submit("u", honest_predictions())


def test_rate_limit_is_keyed_per_user() -> None:
    clock = _Clock(datetime(2026, 3, 1, tzinfo=timezone.utc))
    service = LeaderboardService(build_fixture(), rate_limit_per_day=2, clock=clock)

    service.submit("a", honest_predictions())
    service.submit("a", honest_predictions())
    with pytest.raises(RateLimitError):
        service.submit("a", honest_predictions())

    assert service.submit("b", honest_predictions()).remaining_today == 1


def test_board_ranks_by_headline_recall_with_seeds_first() -> None:
    service = LeaderboardService(build_fixture())
    service.add_seed("perfect", honest_predictions())
    service.submit("user-weak", partial_predictions())

    board = service.board()
    assert [entry.name for entry in board] == ["perfect", "user-weak"]
    assert board[0].is_seed is True
    assert board[0].headline_recall == pytest.approx(0.6)
    assert board[1].headline_recall == pytest.approx(0.2)  # one of three above-floor LEO labels


def test_seeds_bypass_the_rate_limit() -> None:
    service = LeaderboardService(build_fixture(), rate_limit_per_day=1)
    for index in range(10):
        service.add_seed(f"seed-{index}", honest_predictions())
    assert len(service.board()) == 10
