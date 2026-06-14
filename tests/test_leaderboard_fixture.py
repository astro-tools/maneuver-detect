"""Tests for ``maneuver_detect.leaderboard.fixture`` — the held-out scoring fixture's I/O.

The fixture is built once offline and read back by the Space, so the load-bearing property is that
it round-trips losslessly: serialise, reload, and the reloaded fixture scores a submission
identically. The fixture also carries the D4 matching windows (real elset epochs, D2-restricted),
which is why it is private deploy-time data rather than a committed artifact — exercised here on a
synthetic fixture.
"""

from __future__ import annotations

from _leaderboard import build_fixture, honest_predictions
from maneuver_detect.leaderboard import LeaderboardService, fixture_to_json, load_fixture


def test_fixture_round_trips_byte_for_byte() -> None:
    fixture = build_fixture()
    text = fixture_to_json(fixture)
    assert fixture_to_json(load_fixture(text)) == text


def test_reloaded_fixture_scores_identically() -> None:
    fixture = build_fixture()
    reloaded = load_fixture(fixture_to_json(fixture))

    submission = honest_predictions()
    original = LeaderboardService(fixture).submit("u", submission)
    after_round_trip = LeaderboardService(reloaded).submit("u", submission)

    assert original.result == after_round_trip.result
    assert original.entry.headline_recall == after_round_trip.entry.headline_recall
    assert original.entry.per_class_recall == after_round_trip.entry.per_class_recall


def test_fixture_serialisation_is_canonical_and_stable() -> None:
    fixture = build_fixture()
    assert fixture_to_json(fixture) == fixture_to_json(fixture)
    assert fixture_to_json(fixture).endswith("}\n")
