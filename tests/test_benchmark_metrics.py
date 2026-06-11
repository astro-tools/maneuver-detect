"""Tests for ``maneuver_detect.benchmark.metrics`` — P/R at a fixed FA/sat-year per class.

Validated on toy labelled sets with hand-counted true positives, false positives, and misses: that
recall climbs along the false-alarm-rate sweep, that the satellite-year denominator sets the
operating point, that below-floor labels are ignored rather than scored, that the type confusion
tabulates the above-floor matches at the operating point, and that a class with no above-floor
reports an undefined (``None``) recall rather than a divide-by-zero.
"""

from __future__ import annotations

import pandas as pd
import pytest

from maneuver_detect.benchmark.matching import ScoredLabel, match_detections
from maneuver_detect.benchmark.metrics import ObjectExposure, class_metrics
from maneuver_detect.labels.labeller import LabelledInterval
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import Maneuver, ManeuverType

pytestmark = pytest.mark.benchmark


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _label(
    norad_id: int,
    epoch: str,
    *,
    half_window_days: int = 1,
    orbit_class: OrbitClass = OrbitClass.LEO,
    maneuver_type: ManeuverType | None = None,
    above_floor: bool = True,
) -> ScoredLabel:
    centre = _ts(epoch)
    span = pd.Timedelta(days=half_window_days)
    interval = LabelledInterval(
        norad_id=norad_id,
        epoch=centre,
        elset_epoch_before=centre,
        elset_epoch_after=centre,
        tol_start=centre - span,
        tol_end=centre + span,
        maneuver_type=maneuver_type,
        delta_v=None,
        source="TEST",
        source_ref=f"{norad_id}:{epoch}",
        orbit_class=orbit_class,
    )
    return ScoredLabel(interval=interval, above_floor=above_floor)


def _detection(
    norad_id: int,
    epoch: str,
    confidence: float,
    maneuver_type: ManeuverType = ManeuverType.IN_TRACK,
) -> Maneuver:
    ts = _ts(epoch)
    return Maneuver(
        epoch=ts,
        confidence=confidence,
        type=maneuver_type,
        delta_v_estimate=None,
        norad_id=norad_id,
        elset_epoch_before=ts,
        elset_epoch_after=ts,
    )


# --- recall climbs along the FA/sat-year sweep ----------------------------------------------------


def test_recall_increases_as_the_false_alarm_budget_grows() -> None:
    # Two real maneuvers and a miss on one 2-sat-year object, with a false alarm more confident than
    # the second true positive — so reaching that true positive costs one false alarm.
    labels = [
        _label(100, "2024-01-03", maneuver_type=ManeuverType.IN_TRACK),
        _label(100, "2024-01-10", maneuver_type=ManeuverType.IN_TRACK),
        _label(100, "2024-01-20", maneuver_type=ManeuverType.IN_TRACK),  # the miss (no detection)
    ]
    detections = [
        _detection(100, "2024-01-03", 0.9),  # true positive on label 1
        _detection(100, "2024-01-15", 0.7),  # false alarm (no label nearby), more confident than...
        _detection(100, "2024-01-10", 0.6),  # ...the true positive on label 2
    ]
    exposure = [ObjectExposure(100, OrbitClass.LEO, 2.0)]
    leo = class_metrics(match_detections(detections, labels), exposure)[OrbitClass.LEO]

    assert leo.sat_years == 2.0
    assert leo.n_labels_above_floor == 3
    # 0.3 FA/sat-year => budget 0.6 FP: the false alarm is unaffordable, only the first TP admitted.
    assert leo.pr_curve[0].fa_per_sat_year == 0.3
    assert leo.pr_curve[0].recall == pytest.approx(1 / 3)
    assert leo.pr_curve[0].precision == pytest.approx(1.0)
    # 1 FA/sat-year => budget 2.0 FP: the false alarm is affordable, the second TP is now reached.
    assert leo.pr_curve[1].fa_per_sat_year == 1.0
    assert leo.pr_curve[1].recall == pytest.approx(2 / 3)
    assert leo.pr_curve[1].precision == pytest.approx(2 / 3)
    # 3 FA/sat-year: no further detections exist, so the operating point is unchanged.
    assert leo.pr_curve[2].recall == pytest.approx(2 / 3)
    # The headline is the 1 FA/sat-year recall; nothing below floor here, so full == above-floor.
    assert leo.recall == pytest.approx(2 / 3)
    assert leo.precision == pytest.approx(2 / 3)
    assert leo.full_population_recall == pytest.approx(2 / 3)


def test_satellite_years_set_the_operating_point() -> None:
    # The same one false alarm is affordable at 1 FA/sat-year only once there is at least a sat-year
    # of exposure: with half a sat-year the budget is 0.5 FP, so the alarm cannot be admitted.
    labels = [_label(100, "2024-01-10")]
    detections = [
        _detection(100, "2024-01-03", 0.9),  # false alarm, most confident
        _detection(100, "2024-01-10", 0.6),  # true positive, behind the alarm
    ]
    thin = class_metrics(
        match_detections(detections, labels), [ObjectExposure(100, OrbitClass.LEO, 0.5)]
    )
    fat = class_metrics(
        match_detections(detections, labels), [ObjectExposure(100, OrbitClass.LEO, 1.0)]
    )
    assert thin[OrbitClass.LEO].recall == pytest.approx(0.0)  # alarm unaffordable, TP unreachable
    assert fat[OrbitClass.LEO].recall == pytest.approx(1.0)  # alarm affordable, TP admitted


# --- below-floor labels are ignored, not scored ---------------------------------------------------


def test_below_floor_match_is_neither_true_positive_nor_false_alarm() -> None:
    labels = [
        _label(100, "2024-01-03", maneuver_type=ManeuverType.IN_TRACK, above_floor=True),
        _label(100, "2024-01-20", maneuver_type=ManeuverType.IN_TRACK, above_floor=False),
    ]
    detections = [
        _detection(100, "2024-01-03", 0.9),  # true positive (above floor)
        _detection(100, "2024-01-20", 0.8),  # matches the below-floor label -> ignored
    ]
    leo = class_metrics(
        match_detections(detections, labels), [ObjectExposure(100, OrbitClass.LEO, 1.0)]
    )[OrbitClass.LEO]
    assert leo.n_labels_above_floor == 1
    assert leo.n_labels_total == 2
    # The below-floor match does not inflate precision (it is not a TP) nor dent it (not an FP).
    assert leo.recall == pytest.approx(1.0)
    assert leo.precision == pytest.approx(1.0)
    # It does count as a recovery in the secondary full-population recall.
    assert leo.full_population_recall == pytest.approx(1.0)


def test_unmatched_below_floor_label_is_not_a_miss() -> None:
    labels = [
        _label(100, "2024-01-03", above_floor=True),
        _label(100, "2024-01-20", above_floor=False),  # undetected, but below floor -> not a miss
    ]
    detections = [_detection(100, "2024-01-03", 0.9)]
    leo = class_metrics(
        match_detections(detections, labels), [ObjectExposure(100, OrbitClass.LEO, 1.0)]
    )[OrbitClass.LEO]
    assert leo.recall == pytest.approx(1.0)  # 1/1 above-floor, the below-floor miss is excluded
    assert leo.full_population_recall == pytest.approx(0.5)  # 1/2 over the full population


# --- type confusion at the operating point --------------------------------------------------------


def test_confusion_tabulates_true_vs_predicted_type() -> None:
    labels = [
        _label(100, "2024-01-03", maneuver_type=ManeuverType.IN_TRACK),
        _label(100, "2024-01-10", maneuver_type=ManeuverType.CROSS_TRACK),
        _label(100, "2024-01-17", maneuver_type=None),  # epoch-only: no ground-truth type
    ]
    detections = [
        _detection(100, "2024-01-03", 0.9, ManeuverType.IN_TRACK),  # correct
        _detection(100, "2024-01-10", 0.8, ManeuverType.IN_TRACK),  # cross-track called in-track
        _detection(100, "2024-01-17", 0.7, ManeuverType.RADIAL),  # matched a typeless label
    ]
    leo = class_metrics(
        match_detections(detections, labels), [ObjectExposure(100, OrbitClass.LEO, 1.0)]
    )[OrbitClass.LEO]
    counts = leo.confusion.counts
    assert counts[ManeuverType.IN_TRACK][ManeuverType.IN_TRACK] == 1
    assert counts[ManeuverType.CROSS_TRACK][ManeuverType.IN_TRACK] == 1
    # The epoch-only label contributes no row — only typed above-floor matches are tabulated.
    assert leo.confusion.total() == 2


# --- classes with no above-floor labels report an undefined recall --------------------------------


def test_class_without_labels_has_none_recall_and_a_stable_shape() -> None:
    detections = [_detection(300, "2024-01-02", 0.5, ManeuverType.RADIAL)]  # a lone false alarm
    metrics = class_metrics(
        match_detections(detections, []), [ObjectExposure(300, OrbitClass.GEO, 0.5)]
    )

    geo = metrics[OrbitClass.GEO]
    assert geo.n_labels_above_floor == 0
    assert geo.recall is None
    assert geo.full_population_recall is None
    # 3 FA/sat-year over 0.5 sat-years affords the alarm, so precision is a defined 0.0 there.
    assert geo.pr_curve[2].precision == pytest.approx(0.0)
    assert geo.pr_curve[2].recall is None

    # Every class is present in the report even with no objects, at zero counts.
    assert set(metrics) == set(OrbitClass)
    leo = metrics[OrbitClass.LEO]
    assert leo.n_objects == 0
    assert leo.sat_years == 0.0
    assert leo.recall is None


# --- input validation -----------------------------------------------------------------------------


def test_detection_outside_the_scored_population_is_rejected() -> None:
    detections = [_detection(404, "2024-01-03", 0.9)]
    with pytest.raises(ValueError, match="absent from the scored population"):
        class_metrics(match_detections(detections, []), [ObjectExposure(100, OrbitClass.LEO, 1.0)])


def test_duplicate_exposure_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate exposure"):
        class_metrics(
            match_detections([], []),
            [ObjectExposure(100, OrbitClass.LEO, 1.0), ObjectExposure(100, OrbitClass.LEO, 2.0)],
        )


def test_negative_observation_years_is_rejected() -> None:
    with pytest.raises(ValueError, match="observation_years"):
        ObjectExposure(100, OrbitClass.LEO, -1.0)
