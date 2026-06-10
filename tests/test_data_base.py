"""Tests for ``maneuver_detect.data.base`` — the range helpers and the cache singleton."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from maneuver_detect.data.base import in_range, normalise_range, parse_bound
from maneuver_detect.data.cache import Cache, default_cache


class TestParseBound:
    def test_none_passes_through(self) -> None:
        assert parse_bound(None) is None

    def test_naive_string_is_stamped_utc(self) -> None:
        parsed = parse_bound("2024-01-01")
        assert parsed is not None and parsed.tzinfo is not None
        assert parsed == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_space_separator_and_trailing_z(self) -> None:
        assert parse_bound("2024-01-01 12:00:00Z") == datetime(2024, 1, 1, 12, tzinfo=timezone.utc)

    def test_naive_datetime_is_stamped_utc(self) -> None:
        assert parse_bound(datetime(2024, 1, 1, 12)) == datetime(
            2024, 1, 1, 12, tzinfo=timezone.utc
        )

    def test_aware_datetime_is_converted_to_utc(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        parsed = parse_bound(datetime(2024, 1, 1, 7, tzinfo=eastern))
        assert parsed == datetime(2024, 1, 1, 12, tzinfo=timezone.utc)

    def test_unparseable_string_raises(self) -> None:
        with pytest.raises(ValueError, match="ISO-8601"):
            parse_bound("not a date")


class TestNormaliseRange:
    def test_both_none(self) -> None:
        assert normalise_range(None, None) == (None, None)

    def test_start_after_end_raises(self) -> None:
        with pytest.raises(ValueError, match="after end"):
            normalise_range("2024-06-01", "2024-01-01")

    def test_equal_bounds_are_allowed(self) -> None:
        lo, hi = normalise_range("2024-01-01", "2024-01-01")
        assert lo == hi


class TestInRange:
    _E = datetime(2024, 3, 1, tzinfo=timezone.utc)

    def test_unbounded_always_in(self) -> None:
        assert in_range(self._E, None, None) is True

    def test_before_lo_is_out(self) -> None:
        assert in_range(self._E, datetime(2024, 4, 1, tzinfo=timezone.utc), None) is False

    def test_after_hi_is_out(self) -> None:
        assert in_range(self._E, None, datetime(2024, 2, 1, tzinfo=timezone.utc)) is False

    def test_boundaries_are_inclusive(self) -> None:
        assert in_range(self._E, self._E, self._E) is True


def test_default_cache_is_a_singleton() -> None:
    first = default_cache()
    assert isinstance(first, Cache)
    assert default_cache() is first
