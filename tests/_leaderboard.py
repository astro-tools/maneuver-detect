"""Synthetic leaderboard fixtures — a held-out scoring fixture and submissions, built in memory.

Mirrors the V7 proof's hidden benchmark: a LEO object with three above-floor maneuvers and one
below-floor (undetectable, not scored), and a GEO object with two. Labels are spaced ≥3 gaps apart
so each gets its own D4 ±1-adjacent-gap window. Catalogue ids 9000x are fictional, so nothing here
is a redistributed elset (the V1/D2 practice). No network, no series reconstruction — the fixture is
built straight from :class:`~maneuver_detect.benchmark.ScoredLabel`\\ s the tests reason about.
"""

from __future__ import annotations

import pandas as pd

from maneuver_detect.benchmark import ObjectExposure, ScoredLabel, predictions_to_json
from maneuver_detect.labels.labeller import LabelledInterval
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.leaderboard import ScoringFixture
from maneuver_detect.schema import Maneuver, ManeuverType

DAY = pd.Timedelta(days=1)
BASE = pd.Timestamp("2026-03-01T00:00:00", tz="UTC")
SPAN_DAYS = 30


def label(
    norad_id: int,
    day: int,
    orbit_class: OrbitClass,
    maneuver_type: ManeuverType,
    delta_v: float,
    *,
    above_floor: bool = True,
) -> ScoredLabel:
    """A held-out label on gap ``[day, day + 1)`` with the D4 ±1-adjacent-gap matching window."""
    gap_start = BASE + day * DAY
    gap_end = gap_start + DAY
    return ScoredLabel(
        LabelledInterval(
            norad_id=norad_id,
            epoch=gap_start + pd.Timedelta(hours=12),
            elset_epoch_before=gap_start,
            elset_epoch_after=gap_end,
            tol_start=gap_start - DAY,
            tol_end=gap_end + DAY,
            maneuver_type=maneuver_type,
            delta_v=delta_v,
            source="SYNTH",
            source_ref=f"{norad_id}:{day}",
            orbit_class=orbit_class,
        ),
        above_floor=above_floor,
    )


def build_fixture() -> ScoringFixture:
    """A two-object held-out fixture: three above-floor LEO labels (one below-floor) and two GEO."""
    labels = (
        label(90001, 5, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.5),
        label(90001, 12, OrbitClass.LEO, ManeuverType.CROSS_TRACK, 0.3),
        label(90001, 22, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.4),
        label(90001, 27, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.002, above_floor=False),
        label(90002, 8, OrbitClass.GEO, ManeuverType.CROSS_TRACK, 0.12),
        label(90002, 18, OrbitClass.GEO, ManeuverType.IN_TRACK, 0.10),
    )
    sat_years = SPAN_DAYS / 365.25
    exposure = (
        ObjectExposure(90001, OrbitClass.LEO, sat_years),
        ObjectExposure(90002, OrbitClass.GEO, sat_years),
    )
    return ScoringFixture(
        dataset_version="0.0-test",
        labels=labels,
        exposure=exposure,
        timing_floor={"LEO": 0.62, "GEO": 0.68},
    )


def detection(norad_id: int, day: int, maneuver_type: ManeuverType) -> Maneuver:
    """A single predicted maneuver at the midpoint of the gap ``[day, day + 1)``."""
    gap_start = BASE + day * DAY
    return Maneuver(
        epoch=gap_start + pd.Timedelta(hours=12),
        confidence=1.0,
        type=maneuver_type,
        delta_v_estimate=None,
        norad_id=norad_id,
        elset_epoch_before=gap_start,
        elset_epoch_after=gap_start + DAY,
    )


def honest_predictions() -> str:
    """A submission that recovers all three above-floor LEO maneuvers (LEO recall 1.0, GEO 0.0)."""
    return predictions_to_json(
        [
            detection(90001, 5, ManeuverType.IN_TRACK),
            detection(90001, 12, ManeuverType.CROSS_TRACK),
            detection(90001, 22, ManeuverType.IN_TRACK),
        ]
    )


def partial_predictions() -> str:
    """A weaker submission that recovers one of the three above-floor LEO maneuvers (recall 1/3)."""
    return predictions_to_json([detection(90001, 5, ManeuverType.IN_TRACK)])
