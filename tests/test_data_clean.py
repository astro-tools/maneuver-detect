"""Tests for ``maneuver_detect.data.clean`` — validity filtering and duplicate-epoch dedup."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from maneuver_detect.data.clean import clean_elsets, is_valid_elset
from maneuver_detect.data.elset import Elset

_T0 = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)


def _elset(epoch: datetime = _T0, **overrides: object) -> Elset:
    """An ISS-like elset; override any field for the case under test."""
    fields: dict[str, object] = {
        "norad_id": 25544,
        "epoch": epoch,
        "mean_motion": 15.4916,
        "eccentricity": 0.0005881,
        "inclination": 51.6361,
        "raan": 333.6061,
        "arg_perigee": 172.368,
        "mean_anomaly": 187.7399,
        "bstar": 1.045167e-4,
        "mean_motion_dot": 0.0001,
        "mean_motion_ddot": 0.0,
        "element_set_no": 100,
        "rev_at_epoch": 57070,
        "classification": "U",
        "object_id": "1998-067A",
    }
    fields.update(overrides)
    return Elset(**fields)  # type: ignore[arg-type]


def _elset_with(overrides: dict[str, object]) -> Elset:
    """Build an elset from a dynamic field->value override map (for parametrized cases)."""
    return _elset(**overrides)  # type: ignore[arg-type]


class TestValidity:
    def test_good_elset_is_valid(self) -> None:
        assert is_valid_elset(_elset()) is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("eccentricity", 1.2),  # hyperbolic
            ("eccentricity", -0.1),  # negative
            ("mean_motion", 0.0),  # non-positive
            ("mean_motion", -15.0),
            ("inclination", 200.0),  # out of [0, 180]
            ("inclination", -5.0),
            ("mean_motion", float("nan")),  # non-finite
            ("bstar", float("inf")),
        ],
    )
    def test_non_physical_elsets_rejected(self, field: str, value: float) -> None:
        assert is_valid_elset(_elset_with({field: value})) is False

    @pytest.mark.parametrize(
        ("desc", "overrides"),
        [
            ("sub-orbital mean motion", {"mean_motion": 20.0}),
            ("near-parabolic", {"eccentricity": 0.9999, "mean_motion": 15.0}),
        ],
    )
    def test_sgp4_insane_but_element_bounds_pass_rejected(
        self, desc: str, overrides: dict[str, object]
    ) -> None:
        # These pass the analytic bounds (finite, 0<=e<1, n>0, 0<=i<=180) but SGP4 cannot propagate
        # them at epoch — a decayed / sub-orbital fit. The SGP4 gate is what catches them.
        assert is_valid_elset(_elset_with(overrides)) is False

    def test_clean_drops_invalid_keeps_valid(self) -> None:
        good = _elset(_T0)
        bad = _elset(_T0 + timedelta(days=1), eccentricity=1.5)
        cleaned = clean_elsets([good, bad])
        assert [e.epoch for e in cleaned] == [good.epoch]


class TestDedup:
    def test_exact_duplicate_collapses_keeping_highest_revision(self) -> None:
        # Identical elements at one epoch, differing only in element_set_no (a redistribution):
        # collapse to one, keeping the highest revision.
        a = _elset(_T0, element_set_no=100)
        b = _elset(_T0, element_set_no=250)
        cleaned = clean_elsets([a, b])
        assert len(cleaned) == 1
        assert cleaned[0].element_set_no == 250

    def test_exact_duplicate_collapses_even_with_placeholder_elset_no(self) -> None:
        # CelesTrak's 999 placeholder carries no ordering — elements decide identity.
        a = _elset(_T0, element_set_no=999)
        b = _elset(_T0, element_set_no=999)
        assert len(clean_elsets([a, b])) == 1

    def test_same_epoch_refit_keeps_highest_element_set_no(self) -> None:
        # Differing elements at one epoch (a real re-fit): keep the later revision.
        old_fit = _elset(_T0, mean_motion=15.49, element_set_no=100)
        new_fit = _elset(_T0, mean_motion=15.50, element_set_no=200)
        cleaned = clean_elsets([old_fit, new_fit])
        assert len(cleaned) == 1
        assert cleaned[0].element_set_no == 200
        assert cleaned[0].mean_motion == pytest.approx(15.50)

    def test_dedup_is_deterministic_regardless_of_input_order(self) -> None:
        # Same-epoch re-fit with a non-discriminating element_set_no: the choice must not depend on
        # input order (byte-stable reconstruction, D8).
        x = _elset(_T0, mean_motion=15.49, element_set_no=999)
        y = _elset(_T0, mean_motion=15.50, element_set_no=999)
        forward = clean_elsets([x, y])
        backward = clean_elsets([y, x])
        assert len(forward) == 1
        assert forward[0].mean_motion == backward[0].mean_motion

    def test_distinct_epochs_are_all_kept_and_sorted(self) -> None:
        e0 = _elset(_T0)
        e1 = _elset(_T0 + timedelta(days=1))
        e2 = _elset(_T0 + timedelta(days=2))
        cleaned = clean_elsets([e2, e0, e1])  # shuffled
        assert [e.epoch for e in cleaned] == [e0.epoch, e1.epoch, e2.epoch]

    def test_empty_input_yields_empty_list(self) -> None:
        assert clean_elsets([]) == []
