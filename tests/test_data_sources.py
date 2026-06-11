"""Tests for the catalogue-source registry — ``FETCHERS`` / ``get_fetcher`` / ``DEFAULT_SOURCE``."""

from __future__ import annotations

import pytest

from maneuver_detect.data import (
    DEFAULT_SOURCE,
    FETCHERS,
    CelestrakFetcher,
    SpacetrackFetcher,
    get_fetcher,
)


def test_registry_keys_match_fetcher_source_names() -> None:
    # The registry is keyed by each fetcher's own canonical source name, not a separate string.
    assert {
        CelestrakFetcher.source: CelestrakFetcher,
        SpacetrackFetcher.source: SpacetrackFetcher,
    } == FETCHERS
    assert set(FETCHERS) == {"celestrak", "spacetrack"}


def test_default_source_is_the_history_archive() -> None:
    # The default must be the source with multi-epoch history, so a NORAD detection has a series.
    assert DEFAULT_SOURCE == SpacetrackFetcher.source == "spacetrack"


@pytest.mark.parametrize(
    ("source", "expected"),
    [("celestrak", CelestrakFetcher), ("spacetrack", SpacetrackFetcher)],
)
def test_get_fetcher_returns_the_selected_fetcher(source: str, expected: type) -> None:
    fetcher = get_fetcher(source)
    assert isinstance(fetcher, expected)
    fetcher.close()  # constructing does no network I/O; close releases nothing here


def test_get_fetcher_unknown_source_raises_listing_available() -> None:
    with pytest.raises(ValueError, match=r"unknown source 'nope'.*celestrak.*spacetrack"):
        get_fetcher("nope")
