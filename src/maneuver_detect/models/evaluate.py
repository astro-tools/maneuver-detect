"""Score a Detector over a leak-free temporal split — the held-out, era-scoped evaluation.

Mirrors the classical real-eval scoring (``tests/test_real_eval._evaluate``) but for a
:class:`~maneuver_detect.benchmark.TemporalSplit`: each partition's labels are era-scoped via
:meth:`TemporalSplit.assign` (so a held-out object is scored only on its novel era), detections are
restricted to that era, the per-object per-type detectability floor gates the above-floor
population, and exposure is the era-only span. The function is **detector-agnostic** — the same call
scores the classical or a learned detector — so the v0.2 model-card / leaderboard numbers come from
one tested path rather than an ad-hoc script.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from maneuver_detect.benchmark import (
    DEFAULT_CI_LEVEL,
    DEFAULT_OPERATING_POINT,
    DEFAULT_SWEEP,
    ObjectExposure,
    ScoredLabel,
    ScoreReport,
    SplitName,
    TemporalSplit,
    score,
)
from maneuver_detect.benchmark.matching import match_detections
from maneuver_detect.calibration import CalibrationSamples, TemperatureScaling
from maneuver_detect.detectors.base import Detector
from maneuver_detect.detectors.classical import ClassicalDetector
from maneuver_detect.labels.labeller import label_series
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.physics import orbit_class_of
from maneuver_detect.schema import Maneuver, ManeuverType, from_frame

__all__ = [
    "DEFAULT_THRESHOLD_SWEEP",
    "PerClassThresholdTuning",
    "SelectionObjective",
    "ThresholdTuning",
    "calibration_samples_on_val",
    "detections_for_partition",
    "fit_temperature_on_val",
    "macro_above_floor_recall",
    "objective_recall",
    "pooled_above_floor_recall",
    "score_on_temporal_split",
    "scoring_inputs_for_partition",
    "tune_threshold_on_val",
    "tune_thresholds_per_class_on_val",
]

_SECONDS_PER_YEAR = 365.25 * 86400.0

#: The default per-gap detection thresholds the threshold tuners search over.
DEFAULT_THRESHOLD_SWEEP: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: The class-balance of the checkpoint-selection / threshold-tuning objective: ``"pooled"`` weights
#: each orbit class by its above-floor label count (the overall recall — GEO, the majority class,
#: dominates), ``"macro"`` weights every class equally (so the smaller LEO/MEO classes are not
#: traded away for GEO, and an epoch or threshold that keeps the GEO signal can still win on its own
#: merits). See :func:`objective_recall`.
SelectionObjective = Literal["pooled", "macro"]

# Which era each partition draws from (oldest -> newest), matching TemporalSplit's construction.
_PARTITION_ERA: dict[SplitName, int] = {
    SplitName.TRAIN: 0,
    SplitName.VAL: 1,
    SplitName.TEST: 2,
}


def _in_era(epochs: pd.Series, split: TemporalSplit, era: int) -> pd.Series:
    """Boolean mask of the epochs that fall in ``era`` (guard bands excluded).

    A ``pd.Timestamp`` is a ``datetime`` subclass, so it goes to :meth:`TemporalSplit.era_of`
    directly — converting to a native ``datetime`` would discard sub-microsecond nanoseconds (a
    gap-midpoint detection epoch carries them) and warn, for no gain at day-grained cuts.
    """
    return epochs.map(lambda ts: split.era_of(pd.Timestamp(ts)) == era)


def scoring_inputs_for_partition(
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    partition: SplitName = SplitName.TEST,
) -> tuple[list[ScoredLabel], list[ObjectExposure]]:
    """Build the ``(scored_labels, exposure)`` the benchmark scores ``partition`` over.

    The detector-free half of :func:`score_on_temporal_split`: each partition object's labels are
    era-scoped via :meth:`TemporalSplit.assign`, mapped onto their bracketing inter-elset gaps, and
    gated by the per-object per-type detectability floor; exposure is the era-only observation span.
    A label with no Δv (an epoch-only source) cannot be floor-tested and is counted as above-floor.
    The leaderboard's held-out scoring fixture is built from the same function the local scorer
    uses, so the hosted board and a local :func:`score_on_temporal_split` run agree by construction.
    """
    era = _PARTITION_ERA[partition]
    partition_labels = split.assign(labels)[partition]
    labels_by_norad: dict[int, list[ManeuverLabel]] = {}
    for label in partition_labels:
        if label.norad_id is not None:
            labels_by_norad.setdefault(label.norad_id, []).append(label)

    floor_calibrator = ClassicalDetector()
    scored_labels: list[ScoredLabel] = []
    exposure: list[ObjectExposure] = []
    for norad_id in sorted(split.members(partition)):
        series = series_by_norad.get(norad_id)
        if series is None or series.empty:
            continue
        era_series = series[_in_era(series["epoch"], split, era)]
        if era_series.empty:
            continue

        obj_labels = labels_by_norad.get(norad_id, [])
        floors = floor_calibrator.floor_for(series)
        for label in obj_labels:
            kind = label.maneuver_type if label.maneuver_type is not None else ManeuverType.IN_TRACK
            above_floor = label.delta_v is None or label.delta_v >= floors[kind]
            for interval in label_series(series, [label]).intervals:
                scored_labels.append(ScoredLabel(interval=interval, above_floor=above_floor))

        span_seconds = (era_series["epoch"].max() - era_series["epoch"].min()).total_seconds()
        orbit_class = (
            obj_labels[0].orbit_class
            if obj_labels
            else orbit_class_of(float(series["semi_major_axis"].median()))
        )
        exposure.append(
            ObjectExposure(
                norad_id=norad_id,
                orbit_class=orbit_class,
                observation_years=span_seconds / _SECONDS_PER_YEAR,
            )
        )
    return scored_labels, exposure


def detections_for_partition(
    detector: Detector,
    series_by_norad: Mapping[int, pd.DataFrame],
    split: TemporalSplit,
    *,
    partition: SplitName = SplitName.TEST,
) -> list[Maneuver]:
    """The in-era detections ``detector`` produces on a partition — the detector half of the score.

    Each partition object is detected on its full series and the detections are restricted to the
    partition's novel era. Exposed so the leaderboard's seed predictions are generated by the same
    path :func:`score_on_temporal_split` evaluates, rather than an ad-hoc loop that could drift.
    """
    era = _PARTITION_ERA[partition]
    detections: list[Maneuver] = []
    for norad_id in sorted(split.members(partition)):
        series = series_by_norad.get(norad_id)
        if series is None or series.empty:
            continue
        era_series = series[_in_era(series["epoch"], split, era)]
        if era_series.empty:
            continue

        # The object's series spans every era; keep only the in-era detections. The detection epoch
        # is a gap midpoint (carries nanoseconds), so it is passed as a Timestamp — a datetime
        # subclass — rather than converted (which would warn on the nanosecond discard).
        for maneuver in from_frame(detector.detect(series)):
            if split.era_of(maneuver.epoch) == era:
                detections.append(maneuver)
    return detections


def score_on_temporal_split(
    detector: Detector,
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    partition: SplitName = SplitName.TEST,
    operating_point: float = DEFAULT_OPERATING_POINT,
    sweep: tuple[float, ...] = DEFAULT_SWEEP,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> ScoreReport:
    """Score ``detector`` on the ``partition`` (default test) of a leak-free temporal ``split``.

    ``series_by_norad`` maps each object to its full reconstructed mean-element series; ``labels``
    is the full label set. Only the partition's objects are scored, each on its novel era: the
    labels are era-scoped via :meth:`TemporalSplit.assign`, detections outside the era are dropped,
    exposure is the era-only observation span, and the per-object per-type detectability floor (the
    classical calibration, as the benchmark uses) gates the above-floor population. A label with no
    Δv (an epoch-only source) cannot be floor-tested and is counted as above-floor.
    """
    scored_labels, exposure = scoring_inputs_for_partition(
        series_by_norad, labels, split, partition=partition
    )
    detections = detections_for_partition(detector, series_by_norad, split, partition=partition)

    return score(
        detections,
        scored_labels,
        exposure,
        operating_point=operating_point,
        sweep=sweep,
        ci_level=ci_level,
    )


def pooled_above_floor_recall(report: ScoreReport) -> float:
    """Above-floor recall pooled across classes, weighted by each class's above-floor label count.

    A single scalar to maximise: the label-count-weighted mean of the per-class recalls, i.e. the
    overall above-floor recall. Classes with no above-floor labels (or an undefined recall) are
    skipped; an empty population scores ``0.0``. This is the selection objective the val-benchmark
    checkpoint selection and the threshold tuner both maximise.
    """
    hit = 0.0
    total = 0
    for metrics in report.per_class.values():
        if metrics.recall is not None and metrics.n_labels_above_floor > 0:
            hit += metrics.recall * metrics.n_labels_above_floor
            total += metrics.n_labels_above_floor
    return hit / total if total else 0.0


def macro_above_floor_recall(report: ScoreReport) -> float:
    """Above-floor recall averaged across classes with equal weight per orbit class.

    The unweighted mean of the per-class recalls, so each class counts the same regardless of how
    many above-floor labels it carries. Because GEO holds the large majority of the v0.2 labels, the
    label-count-weighted :func:`pooled_above_floor_recall` lets a model trade the smaller LEO/MEO
    classes away; this objective does not. Classes with no above-floor labels (or an undefined
    recall) are skipped; with no scored class the score is ``0.0``.
    """
    recalls = [
        metrics.recall
        for metrics in report.per_class.values()
        if metrics.recall is not None and metrics.n_labels_above_floor > 0
    ]
    return sum(recalls) / len(recalls) if recalls else 0.0


def objective_recall(report: ScoreReport, objective: SelectionObjective = "pooled") -> float:
    """The selection objective the checkpoint selection and the threshold tuners maximise.

    Dispatches on ``objective``: ``"pooled"`` is :func:`pooled_above_floor_recall` (label-count
    weighted, the unchanged default), ``"macro"`` is :func:`macro_above_floor_recall` (equal weight
    per orbit class). One scalar either way, so the same selection code serves both balances.

    Raises:
        ValueError: if ``objective`` is neither ``"pooled"`` nor ``"macro"``.
    """
    if objective == "pooled":
        return pooled_above_floor_recall(report)
    if objective == "macro":
        return macro_above_floor_recall(report)
    raise ValueError(f"unknown selection objective {objective!r}; expected 'pooled' or 'macro'")


@dataclass(frozen=True)
class ThresholdTuning:
    """The outcome of a single-threshold val-split search.

    Attributes:
        threshold: The per-gap detection threshold with the best objective above-floor recall on the
            scored partition (the lowest such threshold on a tie, favouring recall).
        recall: That best objective above-floor recall.
        by_threshold: The objective recall at every candidate threshold, for provenance.
    """

    threshold: float
    recall: float
    by_threshold: dict[float, float]


def _sweep_reports(
    make_detector: Callable[[float], Detector],
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    candidates: Sequence[float],
    partition: SplitName,
    operating_point: float,
    sweep: tuple[float, ...],
) -> dict[float, ScoreReport]:
    """Score ``make_detector(candidate)`` on ``partition`` for each candidate threshold.

    The shared sweep behind both threshold tuners: one benchmark :class:`ScoreReport` per candidate
    — era-scoped labels, in-era detections, per-type floor — so the global and the per-class tuner
    read the same per-class metrics from one pass rather than re-scoring.

    Raises:
        ValueError: if ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    reports: dict[float, ScoreReport] = {}
    for candidate in candidates:
        reports[candidate] = score_on_temporal_split(
            make_detector(candidate),
            series_by_norad,
            labels,
            split,
            partition=partition,
            operating_point=operating_point,
            sweep=sweep,
        )
    return reports


def _best_threshold(by_value: Mapping[float, float]) -> float:
    """The threshold with the highest value, ties to the lower one (favouring recall)."""
    return max(by_value, key=lambda t: (by_value[t], -t))


def tune_threshold_on_val(
    make_detector: Callable[[float], Detector],
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    candidates: Sequence[float] = DEFAULT_THRESHOLD_SWEEP,
    partition: SplitName = SplitName.VAL,
    operating_point: float = DEFAULT_OPERATING_POINT,
    sweep: tuple[float, ...] = DEFAULT_SWEEP,
    objective: SelectionObjective = "pooled",
) -> ThresholdTuning:
    """Pick the per-gap threshold that maximises the ``objective`` recall on a held-out partition.

    A trained model fixes its weights but not its decision threshold; the right threshold is a
    selection on held-out data, not a guess. For each ``candidate`` this builds a detector at that
    threshold (``make_detector(threshold)``) and scores the partition (the val split by default)
    through the same benchmark the leaderboard uses — era-scoped labels, in-era detections, per-type
    floor, recall reported at ``operating_point`` false-alarms/satellite-year — then keeps the
    threshold with the best :func:`objective_recall` (``"pooled"`` by default, ``"macro"`` to weight
    every class equally). Because recall is measured *at* a fixed false-alarm budget, a flood of
    low-threshold detections does not trivially win, so the search trades recall against precision
    the way the published metric does. Re-freeze the chosen threshold into the bundle
    (``dataclasses.replace(bundle, threshold=...)``) before scoring test.

    Raises:
        ValueError: if ``candidates`` is empty or ``objective`` is unknown.
    """
    reports = _sweep_reports(
        make_detector,
        series_by_norad,
        labels,
        split,
        candidates=candidates,
        partition=partition,
        operating_point=operating_point,
        sweep=sweep,
    )
    by_threshold = {t: objective_recall(report, objective) for t, report in reports.items()}
    best = _best_threshold(by_threshold)
    return ThresholdTuning(threshold=best, recall=by_threshold[best], by_threshold=by_threshold)


@dataclass(frozen=True)
class PerClassThresholdTuning:
    """The outcome of a per-orbit-class val-split threshold search.

    Each orbit class gets its own per-gap detection threshold — GEO can take a lower gate than
    LEO/MEO — selected independently as the threshold maximising *that class's* above-floor recall.
    A scalar fallback (the best ``objective`` threshold across the whole population) covers classes
    with no above-floor signal on the val split, so the result is always usable on every class.

    Attributes:
        thresholds: ``OrbitClass`` value → the per-gap threshold maximising that class's above-floor
            recall (lowest on a tie, favouring recall). Classes with no above-floor val label — and
            so no defined recall to optimise — are omitted; the detector falls back to ``fallback``.
        fallback: The single best-:func:`objective_recall` threshold over the whole population (what
            :func:`tune_threshold_on_val` would return) — the bundle's scalar ``threshold``, applied
            to any class absent from ``thresholds``.
        recall: The objective above-floor recall at ``fallback`` (provenance).
        by_threshold: The objective recall at every candidate, for provenance / inspection.
        by_class: ``OrbitClass`` value → that class's recall at every candidate, for provenance.
    """

    thresholds: dict[str, float]
    fallback: float
    recall: float
    by_threshold: dict[float, float]
    by_class: dict[str, dict[float, float]]


def tune_thresholds_per_class_on_val(
    make_detector: Callable[[float], Detector],
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    candidates: Sequence[float] = DEFAULT_THRESHOLD_SWEEP,
    partition: SplitName = SplitName.VAL,
    operating_point: float = DEFAULT_OPERATING_POINT,
    sweep: tuple[float, ...] = DEFAULT_SWEEP,
    objective: SelectionObjective = "pooled",
) -> PerClassThresholdTuning:
    """Select a per-orbit-class detection threshold (plus a scalar fallback) on a held-out split.

    The per-class generalisation of :func:`tune_threshold_on_val`. The candidate sweep is scored
    once; because the benchmark computes each class's recall-at-fixed-false-alarm independently for
    a given detector threshold, the best threshold *for each class* is read off the same reports —
    for each :class:`~maneuver_detect.labels.record.OrbitClass`, the candidate maximising that
    class's above-floor recall (lowest on a tie, favouring recall). Per-class composition at
    inference is
    therefore exact: a GEO object gated at the GEO threshold behaves as it did at that point of the
    sweep, regardless of the gates the other classes take. ``objective`` selects only the scalar
    ``fallback`` (the whole-population best, for classes with no above-floor val signal); the
    per-class choices are each on their own class's recall. Freeze the result into the bundle with
    ``replace(bundle, threshold=tuning.fallback, class_thresholds=tuning.thresholds)``.

    Raises:
        ValueError: if ``candidates`` is empty or ``objective`` is unknown.
    """
    reports = _sweep_reports(
        make_detector,
        series_by_norad,
        labels,
        split,
        candidates=candidates,
        partition=partition,
        operating_point=operating_point,
        sweep=sweep,
    )
    by_threshold = {t: objective_recall(report, objective) for t, report in reports.items()}
    fallback = _best_threshold(by_threshold)

    thresholds: dict[str, float] = {}
    by_class: dict[str, dict[float, float]] = {}
    for orbit_class in OrbitClass:
        per_candidate: dict[float, float] = {}
        for candidate, report in reports.items():
            metrics = report.per_class.get(orbit_class)
            if metrics is not None and metrics.recall is not None and metrics.n_labels_above_floor:
                per_candidate[candidate] = metrics.recall
        if not per_candidate:
            continue  # no above-floor val label for this class — it falls back to the scalar
        by_class[orbit_class.value] = per_candidate
        thresholds[orbit_class.value] = _best_threshold(per_candidate)

    return PerClassThresholdTuning(
        thresholds=thresholds,
        fallback=fallback,
        recall=by_threshold[fallback],
        by_threshold=by_threshold,
        by_class=by_class,
    )


def calibration_samples_on_val(
    detector: Detector,
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    partition: SplitName = SplitName.VAL,
) -> dict[OrbitClass, CalibrationSamples]:
    """The per-orbit-class ``(confidence, outcome)`` pairs to calibrate ``detector`` on a split.

    Runs ``detector`` on the partition (the val split by default) and matches its detections to the
    era-scoped labels through the **same** :func:`~maneuver_detect.benchmark.match_detections` the
    scorer uses, then records, per orbit class, each detection's emitted confidence and its verdict:
    ``1.0`` for an above-floor true positive, ``0.0`` for a false alarm. Below-floor matches are
    excluded, exactly as the benchmark's precision excludes them. Because only the named partition
    is read, fitting a calibrator on the result never touches the test labels (no leakage). Every
    :class:`~maneuver_detect.labels.record.OrbitClass` is present; a class with no detections maps
    to empty arrays.
    """
    detections = detections_for_partition(detector, series_by_norad, split, partition=partition)
    scored_labels, exposure = scoring_inputs_for_partition(
        series_by_norad, labels, split, partition=partition
    )
    class_by_norad = {obj.norad_id: obj.orbit_class for obj in exposure}
    matching = match_detections(detections, scored_labels)

    confidences: dict[OrbitClass, list[float]] = {c: [] for c in OrbitClass}
    outcomes: dict[OrbitClass, list[float]] = {c: [] for c in OrbitClass}
    for match in matching.matches:
        det = match.detection
        orbit_class = class_by_norad.get(det.norad_id)
        if orbit_class is None:
            continue
        if match.label is None:
            outcome = 0.0  # false alarm
        elif match.label.above_floor:
            outcome = 1.0  # above-floor true positive
        else:
            continue  # below-floor match: ignored, as the benchmark's precision ignores it
        confidences[orbit_class].append(det.confidence)
        outcomes[orbit_class].append(outcome)

    return {
        orbit_class: CalibrationSamples(
            confidences=np.asarray(confidences[orbit_class], dtype=np.float64),
            outcomes=np.asarray(outcomes[orbit_class], dtype=np.float64),
        )
        for orbit_class in OrbitClass
    }


def fit_temperature_on_val(
    detector: Detector,
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    partition: SplitName = SplitName.VAL,
) -> TemperatureScaling:
    """Fit one temperature on the detector's pooled val-split ``(confidence, outcome)`` pairs.

    A convenience over :func:`calibration_samples_on_val`: pools every class's samples and fits a
    single :class:`~maneuver_detect.calibration.TemperatureScaling`. Raises :class:`ValueError` if
    the partition yields no matched detections to calibrate on.
    """
    samples = calibration_samples_on_val(
        detector, series_by_norad, labels, split, partition=partition
    )
    confidences = np.concatenate([s.confidences for s in samples.values()])
    outcomes = np.concatenate([s.outcomes for s in samples.values()])
    if confidences.size == 0:
        raise ValueError("no matched detections on the val split to calibrate on")
    return TemperatureScaling.fit(confidences, outcomes)
