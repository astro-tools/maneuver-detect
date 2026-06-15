"""Tests for the temporal-split scoring helper — era scoping of labels, detections, and exposure."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from _synthetic import Burn, synthetic_series
from maneuver_detect.benchmark import ClassMetrics, Confusion, ScoreReport, SplitName, TemporalSplit
from maneuver_detect.detectors import ClassicalDetector
from maneuver_detect.detectors.base import Detector
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.models import evaluate
from maneuver_detect.models.evaluate import (
    calibration_samples_on_val,
    fit_temperature_on_val,
    macro_above_floor_recall,
    objective_recall,
    pooled_above_floor_recall,
    score_on_temporal_split,
    tune_threshold_on_val,
    tune_thresholds_per_class_on_val,
)
from maneuver_detect.schema import empty_frame

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


def test_tune_threshold_rejects_empty_candidates() -> None:
    # The candidate guard fires before any detector is built (the factory is never called).
    with pytest.raises(ValueError, match="non-empty"):
        tune_threshold_on_val(lambda _t: ClassicalDetector(), {}, [], _split(), candidates=())


# --- selection objective (A) and per-class threshold tuning (B) --------------------------------


def _class_metrics(orbit_class: OrbitClass, recall: float | None, n_above: int) -> ClassMetrics:
    """A ClassMetrics carrying only the fields the objective / tuner read (the rest are filler)."""
    return ClassMetrics(
        orbit_class=orbit_class,
        sat_years=1.0,
        n_objects=1,
        n_detections=0,
        n_labels_above_floor=n_above,
        n_labels_total=n_above,
        operating_point=1.0,
        ci_level=0.95,
        recall=recall,
        recall_ci=None,
        precision=None,
        precision_ci=None,
        full_population_recall=None,
        pr_curve=(),
        confusion=Confusion(counts={}),
    )


def _report(per_class: dict[OrbitClass, ClassMetrics]) -> ScoreReport:
    return ScoreReport(operating_point=1.0, sweep=(1.0,), ci_level=0.95, per_class=per_class)


class _StubDetector(Detector):
    """A detector that does nothing but carry its threshold, so a faked scorer can read it back."""

    name = "stub"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def detect(self, history: pd.DataFrame) -> pd.DataFrame:
        return empty_frame()


def test_pooled_weights_by_label_count_macro_weights_classes_equally() -> None:
    # GEO holds the majority of above-floor labels but zero recall; LEO is small but perfect.
    report = _report(
        {
            OrbitClass.LEO: _class_metrics(OrbitClass.LEO, recall=1.0, n_above=1),
            OrbitClass.GEO: _class_metrics(OrbitClass.GEO, recall=0.0, n_above=9),
        }
    )
    # pooled = (1*1 + 0*9) / 10 = 0.1 (GEO dominates); macro = (1.0 + 0.0) / 2 = 0.5.
    assert pooled_above_floor_recall(report) == pytest.approx(0.1)
    assert macro_above_floor_recall(report) == pytest.approx(0.5)
    assert objective_recall(report, "pooled") == pytest.approx(0.1)
    assert objective_recall(report, "macro") == pytest.approx(0.5)


def test_objectives_skip_classes_without_above_floor_labels() -> None:
    report = _report(
        {
            OrbitClass.LEO: _class_metrics(OrbitClass.LEO, recall=0.4, n_above=5),
            OrbitClass.GEO: _class_metrics(OrbitClass.GEO, recall=None, n_above=0),
        }
    )
    assert macro_above_floor_recall(report) == pytest.approx(0.4)
    assert pooled_above_floor_recall(report) == pytest.approx(0.4)
    # No scored class at all -> 0.0, not a division by zero.
    assert macro_above_floor_recall(_report({})) == 0.0


def test_objective_recall_rejects_an_unknown_objective() -> None:
    report = _report({OrbitClass.LEO: _class_metrics(OrbitClass.LEO, recall=1.0, n_above=1)})
    with pytest.raises(ValueError, match="unknown selection objective"):
        objective_recall(report, "weighted")  # type: ignore[arg-type]


def test_tune_threshold_scalar_honours_the_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    # @0.2 GEO is recovered but LEO is not; @0.8 the reverse. Pooled (GEO-weighted) prefers 0.2,
    # macro (class-balanced) prefers 0.8 — same sweep, opposite winners by objective.
    def fake_score(detector: _StubDetector, *args: object, **kwargs: object) -> ScoreReport:
        if detector.threshold == 0.2:
            return _report(
                {
                    OrbitClass.LEO: _class_metrics(OrbitClass.LEO, recall=0.0, n_above=1),
                    OrbitClass.GEO: _class_metrics(OrbitClass.GEO, recall=1.0, n_above=9),
                }
            )
        return _report(
            {
                OrbitClass.LEO: _class_metrics(OrbitClass.LEO, recall=1.0, n_above=1),
                OrbitClass.GEO: _class_metrics(OrbitClass.GEO, recall=0.3, n_above=9),
            }
        )

    monkeypatch.setattr(evaluate, "score_on_temporal_split", fake_score)
    pooled = tune_threshold_on_val(
        _StubDetector, {}, [], _split(), candidates=(0.2, 0.8), objective="pooled"
    )
    macro = tune_threshold_on_val(
        _StubDetector, {}, [], _split(), candidates=(0.2, 0.8), objective="macro"
    )
    assert pooled.threshold == 0.2  # pooled@0.2 = 0.9 > pooled@0.8 = 0.37
    assert macro.threshold == 0.8  # macro@0.8 = 0.65 > macro@0.2 = 0.5


def test_tune_thresholds_per_class_picks_each_class_its_own_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GEO recall is best at a low gate, LEO at a high gate; the tuner reads each class's best
    # threshold off the one shared sweep.
    def fake_score(detector: _StubDetector, *args: object, **kwargs: object) -> ScoreReport:
        gate = detector.threshold
        return _report(
            {
                OrbitClass.LEO: _class_metrics(OrbitClass.LEO, recall=gate, n_above=3),
                OrbitClass.GEO: _class_metrics(OrbitClass.GEO, recall=1.0 - gate, n_above=7),
            }
        )

    monkeypatch.setattr(evaluate, "score_on_temporal_split", fake_score)
    tuning = tune_thresholds_per_class_on_val(
        _StubDetector, {}, [], _split(), candidates=(0.2, 0.5, 0.8)
    )
    assert tuning.thresholds == {"LEO": 0.8, "GEO": 0.2}
    assert OrbitClass.MEO.value not in tuning.thresholds  # no above-floor MEO label -> omitted


def test_tune_thresholds_per_class_fallback_follows_the_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_score(detector: _StubDetector, *args: object, **kwargs: object) -> ScoreReport:
        if detector.threshold == 0.2:
            return _report(
                {
                    OrbitClass.LEO: _class_metrics(OrbitClass.LEO, recall=0.0, n_above=1),
                    OrbitClass.GEO: _class_metrics(OrbitClass.GEO, recall=1.0, n_above=9),
                }
            )
        return _report(
            {
                OrbitClass.LEO: _class_metrics(OrbitClass.LEO, recall=1.0, n_above=1),
                OrbitClass.GEO: _class_metrics(OrbitClass.GEO, recall=0.3, n_above=9),
            }
        )

    monkeypatch.setattr(evaluate, "score_on_temporal_split", fake_score)
    pooled = tune_thresholds_per_class_on_val(
        _StubDetector, {}, [], _split(), candidates=(0.2, 0.8), objective="pooled"
    )
    macro = tune_thresholds_per_class_on_val(
        _StubDetector, {}, [], _split(), candidates=(0.2, 0.8), objective="macro"
    )
    # The scalar fallback tracks the objective; the per-class GEO/LEO gates do not.
    assert pooled.fallback == 0.2
    assert macro.fallback == 0.8
    assert pooled.thresholds == macro.thresholds == {"GEO": 0.2, "LEO": 0.8}


def test_tune_thresholds_per_class_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tune_thresholds_per_class_on_val(
            lambda _t: ClassicalDetector(), {}, [], _split(), candidates=()
        )


# --- val-split calibration samples (uncertainty calibration, fit on val only) ------------------


def test_calibration_samples_recover_a_true_positive_on_val() -> None:
    # A val object with an above-floor burn in the val era: the classical detector recovers it, so
    # the LEO samples carry that true positive (outcome 1.0); other classes have no detections.
    frame = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(450, "in_track_ms", 4.0),))
    labels = [_label(frame, 450, 4.0)]
    split = _split(val=frozenset({1}))

    samples = calibration_samples_on_val(ClassicalDetector(), {1: frame}, labels, split)
    leo = samples[OrbitClass.LEO]
    assert len(leo) >= 1
    assert 1.0 in set(leo.outcomes.tolist())  # the recovered burn is a true positive
    assert len(leo.confidences) == len(leo.outcomes)
    assert len(samples[OrbitClass.GEO]) == 0  # no GEO objects -> no GEO samples


def test_calibration_samples_read_only_the_named_partition() -> None:
    # The object is in TEST, not VAL, so calibrating on the VAL partition reads nothing from it —
    # the no-leakage guarantee: a calibrator fit on val never sees the test object's detections.
    frame = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(700, "in_track_ms", 4.0),))
    labels = [_label(frame, 700, 4.0)]
    split = _split(test=frozenset({1}))

    samples = calibration_samples_on_val(ClassicalDetector(), {1: frame}, labels, split)
    assert all(len(s) == 0 for s in samples.values())


def test_fit_temperature_on_val_returns_a_positive_temperature() -> None:
    frame = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(450, "in_track_ms", 4.0),))
    labels = [_label(frame, 450, 4.0)]
    split = _split(val=frozenset({1}))

    temperature = fit_temperature_on_val(ClassicalDetector(), {1: frame}, labels, split)
    assert temperature.temperature > 0.0


def test_fit_temperature_on_val_raises_without_samples() -> None:
    # No val members -> no matched detections -> nothing to calibrate on.
    frame = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(700, "in_track_ms", 4.0),))
    labels = [_label(frame, 700, 4.0)]
    split = _split(test=frozenset({1}))
    with pytest.raises(ValueError, match="no matched detections"):
        fit_temperature_on_val(ClassicalDetector(), {1: frame}, labels, split)
