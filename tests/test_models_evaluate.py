"""Tests for the temporal-split scoring helper — era scoping of labels, detections, and exposure."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from _synthetic import Burn, synthetic_series
from maneuver_detect.benchmark import SplitName, TemporalSplit
from maneuver_detect.detectors import ClassicalDetector
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.models.evaluate import score_on_temporal_split

# Cuts that put day ~100 of a 2024-01-01 daily series in the oldest era and day ~700 in the newest.
_CUT1 = datetime(2024, 8, 1, tzinfo=timezone.utc)
_CUT2 = datetime(2025, 8, 1, tzinfo=timezone.utc)
_GUARD = timedelta(days=7)


def _label(frame: pd.DataFrame, gap_index: int, dv: float) -> ManeuverLabel:
    epochs = list(frame["epoch"])
    midpoint = epochs[gap_index - 1] + (epochs[gap_index] - epochs[gap_index - 1]) / 2
    return ManeuverLabel(
        norad_id=int(frame["norad_id"].iloc[0]),
        epoch=midpoint.to_pydatetime(),
        window_start=epochs[gap_index - 1].to_pydatetime(),
        window_end=epochs[gap_index].to_pydatetime(),
        source="SYNTHETIC",
        source_ref=f"{int(frame['norad_id'].iloc[0])}-{gap_index}",
        orbit_class=OrbitClass.LEO,
        maneuver_type=None,
        delta_v=dv,
    )


def _split(**members: frozenset[int]) -> TemporalSplit:
    return TemporalSplit(
        dataset_version="test",
        seed=0,
        cut1=_CUT1,
        cut2=_CUT2,
        guard=_GUARD,
        train=members.get("train", frozenset()),
        val=members.get("val", frozenset()),
        test=members.get("test", frozenset()),
    )


def test_scores_only_the_partition_era_label() -> None:
    # One test object with a maneuver in the oldest era (day 100) and one in the newest (day 700).
    # The object's partition is the newest era, so only the day-700 maneuver is scored; the day-100
    # one — and any detection of it — must not count.
    frame = synthetic_series(
        norad_id=1,
        seed=0,
        n=900,
        burns=(Burn(100, "in_track_ms", 4.0), Burn(700, "in_track_ms", 4.0)),
    )
    labels = [_label(frame, 100, 4.0), _label(frame, 700, 4.0)]
    split = _split(test=frozenset({1}))

    report = score_on_temporal_split(
        ClassicalDetector(), {1: frame}, labels, split, partition=SplitName.TEST
    )
    leo = report.per_class[OrbitClass.LEO]

    assert leo.n_labels_total == 1  # only the newest-era label is in scope
    assert leo.recall == 1.0  # the classical detector recovers the above-floor in-track burn


def test_empty_partition_scores_nothing() -> None:
    frame = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(700, "in_track_ms", 4.0),))
    labels = [_label(frame, 700, 4.0)]
    split = _split(test=frozenset({1}))

    # The object is in TEST, so scoring the (empty) TRAIN partition counts no labels or detections.
    report = score_on_temporal_split(
        ClassicalDetector(), {1: frame}, labels, split, partition=SplitName.TRAIN
    )
    leo = report.per_class[OrbitClass.LEO]
    assert leo.n_labels_total == 0
    assert leo.n_detections == 0


@pytest.mark.filterwarnings("error::UserWarning")
def test_scoring_does_not_warn_on_nanosecond_epochs() -> None:
    # A detection epoch is a gap midpoint and carries nanoseconds; era classification must not warn
    # about discarding them (it compares the Timestamp directly, not a lossy native datetime).
    frame = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(700, "in_track_ms", 4.0),))
    labels = [_label(frame, 700, 4.0)]
    split = _split(test=frozenset({1}))
    score_on_temporal_split(ClassicalDetector(), {1: frame}, labels, split)


def test_label_outside_the_objects_era_is_dropped() -> None:
    # A test object whose only maneuver is in the oldest era: nothing is in its (newest) era, so the
    # partition scores zero labels — the era half of the leak-free guarantee.
    frame = synthetic_series(norad_id=2, seed=1, n=900, burns=(Burn(100, "in_track_ms", 4.0),))
    labels = [_label(frame, 100, 4.0)]
    split = _split(test=frozenset({2}))

    report = score_on_temporal_split(
        ClassicalDetector(), {2: frame}, labels, split, partition=SplitName.TEST
    )
    assert report.per_class[OrbitClass.LEO].n_labels_total == 0
