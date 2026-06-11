"""Tests for ``maneuver_detect.benchmark.matching`` — the D4 detection-matching rule.

The rule is unit-tested on hand-constructed near-boundary cases (a detection exactly on each edge
of the ±1-adjacent-gap window, and one nanosecond outside), on the one-to-one assignment under
contention, and end-to-end against a real series labelled by
:func:`~maneuver_detect.labels.labeller.label_series`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from maneuver_detect.benchmark.matching import ScoredLabel, match_detections
from maneuver_detect.labels.labeller import LabelledInterval, label_series
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.schema import Maneuver, ManeuverType

pytestmark = pytest.mark.benchmark

_NS = pd.Timedelta(nanoseconds=1)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _label(
    *,
    norad_id: int | None,
    epoch: str,
    tol_start: str,
    tol_end: str,
    gap_start: str | None = None,
    gap_end: str | None = None,
    maneuver_type: ManeuverType | None = None,
    delta_v: float | None = None,
    above_floor: bool = True,
) -> ScoredLabel:
    interval = LabelledInterval(
        norad_id=norad_id,
        epoch=_ts(epoch),
        elset_epoch_before=_ts(gap_start or tol_start),
        elset_epoch_after=_ts(gap_end or tol_end),
        tol_start=_ts(tol_start),
        tol_end=_ts(tol_end),
        maneuver_type=maneuver_type,
        delta_v=delta_v,
        source="TEST",
        source_ref=f"{norad_id}:{epoch}",
        orbit_class=OrbitClass.LEO,
    )
    return ScoredLabel(interval=interval, above_floor=above_floor)


def _detection(
    *,
    norad_id: int,
    epoch: str,
    confidence: float = 0.9,
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


# --- near-boundary cases (the ±1-adjacent-gap window edges) ------------------------------------


@pytest.fixture
def window_label() -> ScoredLabel:
    # Bracketing gap [Jan3, Jan4]; the matching window spans one gap either side: [Jan2, Jan5].
    return _label(
        norad_id=100,
        epoch="2024-01-03T12:00:00",
        gap_start="2024-01-03",
        gap_end="2024-01-04",
        tol_start="2024-01-02",
        tol_end="2024-01-05",
    )


def test_detection_in_bracketing_gap_matches(window_label: ScoredLabel) -> None:
    det = _detection(norad_id=100, epoch="2024-01-03T12:00:00")
    matching = match_detections([det], [window_label])
    assert matching.matches[0].label is window_label
    assert matching.unmatched_labels == ()


def test_detection_exactly_on_tol_start_matches(window_label: ScoredLabel) -> None:
    det = _detection(norad_id=100, epoch="2024-01-02T00:00:00")
    assert match_detections([det], [window_label]).matches[0].label is window_label


def test_detection_exactly_on_tol_end_matches(window_label: ScoredLabel) -> None:
    det = _detection(norad_id=100, epoch="2024-01-05T00:00:00")
    assert match_detections([det], [window_label]).matches[0].label is window_label


def test_detection_one_ns_before_window_is_a_false_positive(window_label: ScoredLabel) -> None:
    det = _detection(norad_id=100, epoch="2024-01-02T00:00:00")
    just_outside = Maneuver(
        epoch=det.epoch - _NS,
        confidence=det.confidence,
        type=det.type,
        delta_v_estimate=None,
        norad_id=det.norad_id,
        elset_epoch_before=det.epoch,
        elset_epoch_after=det.epoch,
    )
    matching = match_detections([just_outside], [window_label])
    assert matching.matches[0].label is None
    assert matching.unmatched_labels == (window_label,)


def test_detection_one_ns_after_window_is_a_false_positive(window_label: ScoredLabel) -> None:
    edge = _ts("2024-01-05T00:00:00")
    just_outside = Maneuver(
        epoch=edge + _NS,
        confidence=0.9,
        type=ManeuverType.IN_TRACK,
        delta_v_estimate=None,
        norad_id=100,
        elset_epoch_before=edge,
        elset_epoch_after=edge,
    )
    assert match_detections([just_outside], [window_label]).matches[0].label is None


# --- object isolation and the one-to-one rule -----------------------------------------------------


def test_detection_on_a_different_object_never_matches(window_label: ScoredLabel) -> None:
    # Same epoch, wrong NORAD — the tolerance is per-object, it never reaches across objects.
    det = _detection(norad_id=999, epoch="2024-01-03T12:00:00")
    matching = match_detections([det], [window_label])
    assert matching.matches[0].label is None
    assert matching.unmatched_labels == (window_label,)


def test_one_label_is_claimed_by_at_most_one_detection(window_label: ScoredLabel) -> None:
    strong = _detection(norad_id=100, epoch="2024-01-03T06:00:00", confidence=0.9)
    weak = _detection(norad_id=100, epoch="2024-01-03T18:00:00", confidence=0.4)
    matching = match_detections([weak, strong], [window_label])
    by_conf = {m.detection.confidence: m.label for m in matching.matches}
    # The more confident detection wins the label; the other is left a false positive.
    assert by_conf[0.9] is window_label
    assert by_conf[0.4] is None


def test_detection_claims_the_nearest_label_in_window() -> None:
    # One detection sits in the overlap of two labels' windows; it takes the nearer one by epoch.
    near = _label(
        norad_id=100, epoch="2024-01-03T00:00:00", tol_start="2024-01-01", tol_end="2024-01-05"
    )
    far = _label(
        norad_id=100, epoch="2024-01-05T00:00:00", tol_start="2024-01-01", tol_end="2024-01-07"
    )
    det = _detection(norad_id=100, epoch="2024-01-03T06:00:00")
    matching = match_detections([det], [near, far])
    assert matching.matches[0].label is near
    assert matching.unmatched_labels == (far,)


def test_two_detections_fill_two_labels_one_to_one() -> None:
    first = _label(
        norad_id=100, epoch="2024-01-03T00:00:00", tol_start="2024-01-01", tol_end="2024-01-05"
    )
    second = _label(
        norad_id=100, epoch="2024-01-05T00:00:00", tol_start="2024-01-01", tol_end="2024-01-07"
    )
    d_first = _detection(norad_id=100, epoch="2024-01-03T01:00:00", confidence=0.9)
    d_second = _detection(norad_id=100, epoch="2024-01-05T01:00:00", confidence=0.8)
    matching = match_detections([d_first, d_second], [first, second])
    claimed = {m.detection.confidence: m.label for m in matching.matches}
    assert claimed[0.9] is first
    assert claimed[0.8] is second
    assert matching.unmatched_labels == ()


# --- floor status and unmatchable labels carry through --------------------------------------------


def test_below_floor_label_is_carried_through_unchanged() -> None:
    below = _label(
        norad_id=100,
        epoch="2024-01-03T12:00:00",
        tol_start="2024-01-02",
        tol_end="2024-01-05",
        above_floor=False,
    )
    det = _detection(norad_id=100, epoch="2024-01-03T12:00:00")
    matched = match_detections([det], [below]).matches[0].label
    assert matched is below
    assert matched.above_floor is False


def test_labels_without_a_norad_are_ignored() -> None:
    orphan = _label(
        norad_id=None, epoch="2024-01-03T12:00:00", tol_start="2024-01-02", tol_end="2024-01-05"
    )
    det = _detection(norad_id=100, epoch="2024-01-03T12:00:00")
    matching = match_detections([det], [orphan])
    assert matching.matches[0].label is None
    assert matching.unmatched_labels == ()  # the orphan attaches to no scored object


def test_no_detections_leaves_every_label_unmatched(window_label: ScoredLabel) -> None:
    matching = match_detections([], [window_label])
    assert matching.matches == ()
    assert matching.unmatched_labels == (window_label,)


# --- determinism and end-to-end through the labeller ----------------------------------------------


def test_matching_is_order_independent_and_stable() -> None:
    labels = [
        _label(
            norad_id=100, epoch="2024-01-03T00:00:00", tol_start="2024-01-01", tol_end="2024-01-05"
        ),
        _label(
            norad_id=100, epoch="2024-01-10T00:00:00", tol_start="2024-01-08", tol_end="2024-01-12"
        ),
    ]
    dets = [
        _detection(norad_id=100, epoch="2024-01-03T06:00:00", confidence=0.9),
        _detection(norad_id=100, epoch="2024-01-10T06:00:00", confidence=0.8),
    ]
    forward = match_detections(dets, labels)
    reversed_inputs = match_detections(list(reversed(dets)), list(reversed(labels)))
    assert {m.detection.confidence: id(m.label) for m in forward.matches} == {
        m.detection.confidence: id(m.label) for m in reversed_inputs.matches
    }


def test_end_to_end_through_label_series() -> None:
    # A real daily series; a maneuver epoch in one gap, labelled, then a detection in it matches.
    epochs = pd.date_range("2024-01-01", periods=8, freq="1D", tz="UTC")
    series = pd.DataFrame({"epoch": epochs})
    raw = ManeuverLabel(
        norad_id=100,
        epoch=_ts("2024-01-04T12:00:00"),
        window_start=_ts("2024-01-04T12:00:00"),
        window_end=_ts("2024-01-04T12:00:00"),
        source="TEST",
        source_ref="m1",
        orbit_class=OrbitClass.LEO,
        maneuver_type=ManeuverType.IN_TRACK,
        delta_v=0.4,
    )
    result = label_series(series, [raw])
    labels = [ScoredLabel(interval=result.intervals[0])]

    det = _detection(norad_id=100, epoch="2024-01-04T18:00:00")
    matching = match_detections([det], labels)
    assert matching.matches[0].label is labels[0]

    # A detection three gaps away is outside the ±1-adjacent window — a false positive.
    far = _detection(norad_id=100, epoch="2024-01-08T00:00:00")
    assert match_detections([far], labels).matches[0].label is None
