"""Tests for ``maneuver_detect.leaderboard.service`` — the server-side scoring + board policy.

The properties the V7 proof asserts on the scorer, here on the service the Space drives in practice:
response is aggregate-only and leaks no held-out label, the submission channel rejects anything that
is not a predictions file, scoring is deterministic, the board ranks by headline recall, and the
per-user per-UTC-day rate limit bounds submission volume (a courtesy guard now the key is public).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from _leaderboard import build_fixture, detection, honest_predictions, partial_predictions
from maneuver_detect.benchmark import predictions_to_json
from maneuver_detect.leaderboard import (
    InvalidSubmissionError,
    LeaderboardService,
    RateLimitError,
)
from maneuver_detect.schema import ManeuverType


def _iter_values(node: object) -> Iterator[object]:
    """Yield every nested value of a JSON-like result tree (dicts, lists, and scalars)."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_values(value)
    elif isinstance(node, list):
        yield node
        for item in node:
            yield from _iter_values(item)
    else:
        yield node


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


def test_response_carries_no_per_label_match_table() -> None:
    # Aggregate-only is structural, not just a key whitelist: no list of per-label / per-match rows
    # may appear anywhere in the response tree, so a reply hands an attacker no row of the key.
    result = LeaderboardService(build_fixture()).submit("u", honest_predictions()).result
    for value in _iter_values(result):
        if isinstance(value, list):
            assert not any(isinstance(item, dict) for item in value)  # no match/label table


def test_below_floor_recovery_does_not_change_the_public_above_floor_recall() -> None:
    # The public number is recall over the ABOVE-floor population (D7/D11). Recovering a below-floor
    # (undetectable) label must not inflate it — the held-out below-floor label is excluded from the
    # scored population, so hitting it leaves the aggregate response unchanged.
    honest = LeaderboardService(build_fixture()).submit("u", honest_predictions()).result
    with_below = predictions_to_json(
        [
            detection(90001, 5, ManeuverType.IN_TRACK),
            detection(90001, 12, ManeuverType.CROSS_TRACK),
            detection(90001, 22, ManeuverType.IN_TRACK),
            detection(90001, 27, ManeuverType.IN_TRACK),  # the below-floor LEO label at day 27
        ]
    )
    result = LeaderboardService(build_fixture()).submit("u", with_below).result
    assert result["headline_recall_above_floor"] == honest["headline_recall_above_floor"]


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
