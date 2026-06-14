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
from maneuver_detect.detectors.base import Detector
from maneuver_detect.detectors.classical import ClassicalDetector
from maneuver_detect.labels.labeller import label_series
from maneuver_detect.labels.record import ManeuverLabel
from maneuver_detect.physics import orbit_class_of
from maneuver_detect.schema import Maneuver, ManeuverType, from_frame

__all__ = [
    "DEFAULT_THRESHOLD_SWEEP",
    "ThresholdTuning",
    "detections_for_partition",
    "pooled_above_floor_recall",
    "score_on_temporal_split",
    "scoring_inputs_for_partition",
    "tune_threshold_on_val",
]

_SECONDS_PER_YEAR = 365.25 * 86400.0

#: The default per-gap detection thresholds :func:`tune_threshold_on_val` searches over.
DEFAULT_THRESHOLD_SWEEP: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

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


@dataclass(frozen=True)
class ThresholdTuning:
    """The outcome of a val-split threshold search.

    Attributes:
        threshold: The per-gap detection threshold with the best pooled above-floor recall on the
            scored partition (the lowest such threshold on a tie, favouring recall).
        recall: That best pooled above-floor recall.
        by_threshold: The pooled recall at every candidate threshold, for provenance / inspection.
    """

    threshold: float
    recall: float
    by_threshold: dict[float, float]


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
) -> ThresholdTuning:
    """Pick the per-gap threshold that maximises pooled above-floor recall on a held-out partition.

    A trained model fixes its weights but not its decision threshold; the right threshold is a
    selection on held-out data, not a guess. For each ``candidate`` this builds a detector at that
    threshold (``make_detector(threshold)``) and scores the partition (the val split by default)
    through the same benchmark the leaderboard uses — era-scoped labels, in-era detections, per-type
    floor, recall reported at ``operating_point`` false-alarms/satellite-year — then keeps the
    threshold with the best :func:`pooled_above_floor_recall`. Because recall is measured *at* a
    fixed false-alarm budget, a flood of low-threshold detections does not trivially win, so the
    search trades recall against precision the way the published metric does. Re-freeze the chosen
    threshold into the bundle (``dataclasses.replace(bundle, threshold=...)``) before scoring test.

    Raises:
        ValueError: if ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    by_threshold: dict[float, float] = {}
    for candidate in candidates:
        report = score_on_temporal_split(
            make_detector(candidate),
            series_by_norad,
            labels,
            split,
            partition=partition,
            operating_point=operating_point,
            sweep=sweep,
        )
        by_threshold[candidate] = pooled_above_floor_recall(report)

    # Best recall, breaking ties towards the lower threshold (favouring recall over precision).
    best = max(by_threshold, key=lambda t: (by_threshold[t], -t))
    return ThresholdTuning(threshold=best, recall=by_threshold[best], by_threshold=by_threshold)
