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

from collections.abc import Mapping, Sequence

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

__all__ = ["score_on_temporal_split"]

_SECONDS_PER_YEAR = 365.25 * 86400.0

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
    era = _PARTITION_ERA[partition]
    partition_labels = split.assign(labels)[partition]
    labels_by_norad: dict[int, list[ManeuverLabel]] = {}
    for label in partition_labels:
        if label.norad_id is not None:
            labels_by_norad.setdefault(label.norad_id, []).append(label)

    floor_calibrator = ClassicalDetector()
    detections: list[Maneuver] = []
    scored_labels: list[ScoredLabel] = []
    exposure: list[ObjectExposure] = []
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

    return score(
        detections,
        scored_labels,
        exposure,
        operating_point=operating_point,
        sweep=sweep,
        ci_level=ci_level,
    )
