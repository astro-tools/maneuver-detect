"""The benchmark metric — precision and recall at a fixed false-alarm rate per orbit class.

The false-alarm rate is measured in **false alarms per satellite-year** (D4): the count of false
positives over the satellite-years of observation in a class. Sweeping a confidence threshold down
trades recall for false alarms, so the metric fixes an operating point — the threshold at which the
class's false-alarm rate first reaches the target — and reports recall and precision there. The
headline is **recall at 1 false alarm per satellite-year over the above-floor population, per
class**; a full-population recall (counting below-floor recoveries) is reported as a secondary lower
bound, and the protocol asks for the operating point as a curve over a 0.3 / 1 / 3 sweep.

Below-floor labels are *ignored*, not scored: a detection that matches one is neither a true
positive nor a false alarm (it correctly flagged a real but undetectable maneuver), and an unmatched
below-floor label is not a miss. Only above-floor labels enter the recall denominator and the
precision and false-alarm counts.

Per-class **type confusion** (in-track / cross-track / radial) is tabulated over the above-floor
true positives included at the headline operating point, restricted to labels that carry a known
type (epoch-only labels contribute no ground-truth type). All arithmetic is integer counts and IEEE
divisions over inputs fixed by the caller, so a class's metrics are byte-stable across runs and
platforms.

Each per-class recall and precision is reported with a **Wilson score confidence interval** (default
95%, configurable) — a sampling interval on the metric *estimate*, so a recall computed from a
handful of test objects is read with its uncertainty rather than as a point fact (a 1-of-1 recall is
not certainty). The interval is closed-form — no resampling — so the report stays byte-stable. This
is the uncertainty of the *estimate*, distinct from calibrating a detector's per-detection
``confidence`` output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist

from maneuver_detect.benchmark.matching import Matching
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import ManeuverType

__all__ = [
    "DEFAULT_CI_LEVEL",
    "DEFAULT_OPERATING_POINT",
    "DEFAULT_SWEEP",
    "ClassMetrics",
    "Confusion",
    "ObjectExposure",
    "PRPoint",
    "class_metrics",
]

#: The headline operating point — false alarms per satellite-year (D4).
DEFAULT_OPERATING_POINT = 1.0

#: The false-alarm-per-satellite-year sweep the P/R curve is reported over (D7).
DEFAULT_SWEEP: tuple[float, ...] = (0.3, 1.0, 3.0)

#: The default confidence level for the per-class sampling intervals on recall and precision — a
#: Wilson score interval that reads a few-object estimate honestly (1-of-1 recall is not certainty).
DEFAULT_CI_LEVEL = 0.95


@dataclass(frozen=True)
class ObjectExposure:
    """One scored object's class and observation span — the unit the false-alarm rate is over.

    The scored population is the set of objects the benchmark observed: each contributes its
    observation span to its class's satellite-year total (the false-alarm-rate denominator) and
    fixes the orbit class a detection on that object is attributed to.

    Attributes:
        norad_id: NORAD catalogue id of the object.
        orbit_class: The object's orbit class.
        observation_years: The span of the object's mean-element series, in years.
    """

    norad_id: int
    orbit_class: OrbitClass
    observation_years: float

    def __post_init__(self) -> None:
        if self.observation_years < 0.0:
            raise ValueError(
                f"observation_years must be non-negative, got {self.observation_years!r}"
            )


@dataclass(frozen=True)
class PRPoint:
    """One point of the precision/recall curve at a target false-alarm rate.

    Attributes:
        fa_per_sat_year: The target false alarms per satellite-year this point is evaluated at.
        recall: Recall over the above-floor population, or ``None`` when the class has no
            above-floor labels.
        precision: Precision over the above-floor population, or ``None`` when no detection is
            admitted at this operating point (an empty true-positive-plus-false-positive set).
        recall_ci: The ``(low, high)`` Wilson confidence interval for ``recall``, or ``None`` when
            ``recall`` is undefined.
        precision_ci: The ``(low, high)`` Wilson confidence interval for ``precision``, or ``None``
            when ``precision`` is undefined.
    """

    fa_per_sat_year: float
    recall: float | None
    precision: float | None
    recall_ci: tuple[float, float] | None
    precision_ci: tuple[float, float] | None


@dataclass(frozen=True)
class Confusion:
    """Type confusion over above-floor true positives — true label type vs. predicted type.

    Attributes:
        counts: ``counts[true_type][predicted_type]`` over the above-floor matches included at the
            operating point whose label carries a known type. Every :class:`ManeuverType` pair is
            present (zero included), so the matrix shape is stable.
    """

    counts: dict[ManeuverType, dict[ManeuverType, int]]

    def total(self) -> int:
        """The number of typed matches tabulated."""
        return sum(n for row in self.counts.values() for n in row.values())


@dataclass(frozen=True)
class ClassMetrics:
    """The benchmark metrics for one orbit class at the headline operating point.

    Attributes:
        orbit_class: The class scored.
        sat_years: Satellite-years of observation in the class (the false-alarm-rate denominator).
        n_objects: Objects in the class in the scored population.
        n_detections: Detections attributed to the class.
        n_labels_above_floor: Above-floor labels in the class — the recall denominator.
        n_labels_total: All matchable labels in the class (the full-population denominator).
        operating_point: The headline false-alarm-per-satellite-year target (D4).
        ci_level: The confidence level of ``recall_ci`` / ``precision_ci`` (and the sweep CIs).
        recall: Recall over the above-floor population at ``operating_point`` (the headline), or
            ``None`` when the class has no above-floor labels.
        recall_ci: The ``(low, high)`` Wilson interval for the headline ``recall`` at ``ci_level``,
            or ``None`` when ``recall`` is undefined — the sampling uncertainty of the estimate, so
            a recall from few objects is read with its width rather than as a point fact.
        precision: Precision over the above-floor population at ``operating_point``, or ``None``
            when no detection is admitted there.
        precision_ci: The ``(low, high)`` Wilson interval for the headline ``precision`` at
            ``ci_level``, or ``None`` when ``precision`` is undefined.
        full_population_recall: Recall counting below-floor recoveries, over all labels — a
            secondary lower bound, or ``None`` when the class has no labels.
        pr_curve: ``(fa_per_sat_year, recall, precision, recall_ci, precision_ci)`` over the sweep.
        confusion: Type confusion over the above-floor true positives at ``operating_point``.
    """

    orbit_class: OrbitClass
    sat_years: float
    n_objects: int
    n_detections: int
    n_labels_above_floor: int
    n_labels_total: int
    operating_point: float
    ci_level: float
    recall: float | None
    recall_ci: tuple[float, float] | None
    precision: float | None
    precision_ci: tuple[float, float] | None
    full_population_recall: float | None
    pr_curve: tuple[PRPoint, ...]
    confusion: Confusion


# One detection attributed to a class, with the outcome the metric scores it as.
class _Outcome(Enum):
    TP_ABOVE = "tp_above"  # matched an above-floor label — a true positive
    TP_BELOW = "tp_below"  # matched a below-floor label — ignored (a full-population recovery)
    FP = "fp"  # matched nothing — a false alarm


@dataclass(frozen=True)
class _ClassDetection:
    confidence: float
    outcome: _Outcome
    predicted_type: ManeuverType
    true_type: ManeuverType | None  # the matched label's type, when above-floor and typed


def _empty_confusion() -> dict[ManeuverType, dict[ManeuverType, int]]:
    return {true: dict.fromkeys(ManeuverType, 0) for true in ManeuverType}


def _exposure_map(
    exposure: list[ObjectExposure] | tuple[ObjectExposure, ...],
) -> dict[int, ObjectExposure]:
    by_norad: dict[int, ObjectExposure] = {}
    for obj in exposure:
        if obj.norad_id in by_norad:
            raise ValueError(f"duplicate exposure for NORAD {obj.norad_id}")
        by_norad[obj.norad_id] = obj
    return by_norad


def class_metrics(
    matching: Matching,
    exposure: list[ObjectExposure] | tuple[ObjectExposure, ...],
    *,
    operating_point: float = DEFAULT_OPERATING_POINT,
    sweep: tuple[float, ...] = DEFAULT_SWEEP,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> dict[OrbitClass, ClassMetrics]:
    """Score a :class:`~maneuver_detect.benchmark.matching.Matching` per orbit class.

    ``exposure`` is the scored population — every detection and every matchable label must belong to
    an object it lists (a :class:`ValueError` is raised otherwise), since the object fixes both the
    orbit class and the satellite-year denominator. ``ci_level`` (in ``(0, 1)``) sets the confidence
    level of the per-class Wilson intervals on recall and precision. Returns one
    :class:`ClassMetrics` per :class:`OrbitClass`, present even at zero, so the report shape is
    stable regardless of which classes the data covers.
    """
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level!r}")
    z = _z_for(ci_level)
    by_norad = _exposure_map(exposure)

    def class_of(norad_id: int, what: str) -> OrbitClass:
        obj = by_norad.get(norad_id)
        if obj is None:
            raise ValueError(
                f"{what} references NORAD {norad_id}, absent from the scored population"
            )
        return obj.orbit_class

    # Bucket detections (with outcomes) and label counts by class.
    detections: dict[OrbitClass, list[_ClassDetection]] = {c: [] for c in OrbitClass}
    for match in matching.matches:
        det = match.detection
        orbit_class = class_of(det.norad_id, "detection")
        if match.label is None:
            outcome, true_type = _Outcome.FP, None
        elif match.label.above_floor:
            outcome = _Outcome.TP_ABOVE
            true_type = match.label.interval.maneuver_type
        else:
            outcome, true_type = _Outcome.TP_BELOW, None
        detections[orbit_class].append(
            _ClassDetection(det.confidence, outcome, ManeuverType(det.type), true_type)
        )

    above_floor_labels: dict[OrbitClass, int] = dict.fromkeys(OrbitClass, 0)
    total_labels: dict[OrbitClass, int] = dict.fromkeys(OrbitClass, 0)
    for matched in (m.label for m in matching.matches if m.label is not None):
        assert matched.interval.norad_id is not None
        c = class_of(matched.interval.norad_id, "label")
        total_labels[c] += 1
        if matched.above_floor:
            above_floor_labels[c] += 1
    for label in matching.unmatched_labels:
        assert label.interval.norad_id is not None
        c = class_of(label.interval.norad_id, "label")
        total_labels[c] += 1
        if label.above_floor:
            above_floor_labels[c] += 1

    sat_years: dict[OrbitClass, float] = dict.fromkeys(OrbitClass, 0.0)
    n_objects: dict[OrbitClass, int] = dict.fromkeys(OrbitClass, 0)
    for obj in by_norad.values():
        sat_years[obj.orbit_class] += obj.observation_years
        n_objects[obj.orbit_class] += 1

    return {
        orbit_class: _score_class(
            orbit_class=orbit_class,
            detections=detections[orbit_class],
            sat_years=sat_years[orbit_class],
            n_objects=n_objects[orbit_class],
            n_above_floor=above_floor_labels[orbit_class],
            n_total=total_labels[orbit_class],
            operating_point=operating_point,
            sweep=sweep,
            ci_level=ci_level,
            z=z,
        )
        for orbit_class in OrbitClass
    }


def _score_class(
    *,
    orbit_class: OrbitClass,
    detections: list[_ClassDetection],
    sat_years: float,
    n_objects: int,
    n_above_floor: int,
    n_total: int,
    operating_point: float,
    sweep: tuple[float, ...],
    ci_level: float,
    z: float,
) -> ClassMetrics:
    """Compute one class's metrics from its detections (with outcomes) and label counts."""
    # Descending confidence; ties broken so the operating-point cut is deterministic.
    ranked = sorted(
        detections, key=lambda d: (-d.confidence, d.outcome.value, d.predicted_type.value)
    )

    def point(rate: float) -> PRPoint:
        walked = _walk(ranked, rate * sat_years)
        tp_above, fp = walked.precision_counts
        return PRPoint(
            fa_per_sat_year=rate,
            recall=_recall(tp_above, n_above_floor),
            precision=_precision(tp_above, fp),
            recall_ci=_wilson_interval(tp_above, n_above_floor, z),
            precision_ci=_wilson_interval(tp_above, tp_above + fp, z),
        )

    pr_curve = tuple(point(rate) for rate in sweep)

    cut = _walk(ranked, operating_point * sat_years)
    tp_above, fp = cut.precision_counts
    recall = _recall(tp_above, n_above_floor)
    precision = _precision(tp_above, fp)
    full_recall = (cut.tp_above + cut.tp_below) / n_total if n_total else None

    confusion = _empty_confusion()
    for det in cut.included_typed:
        assert det.true_type is not None
        confusion[det.true_type][det.predicted_type] += 1

    return ClassMetrics(
        orbit_class=orbit_class,
        sat_years=sat_years,
        n_objects=n_objects,
        n_detections=len(detections),
        n_labels_above_floor=n_above_floor,
        n_labels_total=n_total,
        operating_point=operating_point,
        ci_level=ci_level,
        recall=recall,
        recall_ci=_wilson_interval(tp_above, n_above_floor, z),
        precision=precision,
        precision_ci=_wilson_interval(tp_above, tp_above + fp, z),
        full_population_recall=full_recall,
        pr_curve=pr_curve,
        confusion=Confusion(counts=confusion),
    )


@dataclass(frozen=True)
class _Cut:
    """The counts admitted up to an operating point (the highest threshold within the FP budget)."""

    tp_above: int
    tp_below: int
    fp: int
    included_typed: tuple[_ClassDetection, ...]

    @property
    def precision_counts(self) -> tuple[int, int]:
        return self.tp_above, self.fp


def _walk(ranked: list[_ClassDetection], target_fp: float) -> _Cut:
    """Admit detections in descending confidence until the next false alarm would exceed the budget.

    Ignored (below-floor) matches neither consume the false-alarm budget nor break the walk; they
    are carried as full-population recoveries. The walk stops the instant admitting a false alarm
    would push the count past the budget — the cut is the highest-confidence point within it.
    """
    tp_above = tp_below = fp = 0
    included_typed: list[_ClassDetection] = []
    for det in ranked:
        if det.outcome is _Outcome.FP:
            if fp + 1 > target_fp:
                break
            fp += 1
        elif det.outcome is _Outcome.TP_ABOVE:
            tp_above += 1
            if det.true_type is not None:
                included_typed.append(det)
        else:  # TP_BELOW — ignored for scoring, counted only for full-population recall
            tp_below += 1
    return _Cut(tp_above=tp_above, tp_below=tp_below, fp=fp, included_typed=tuple(included_typed))


def _recall(tp: int, n_above_floor: int) -> float | None:
    return tp / n_above_floor if n_above_floor else None


def _precision(tp: int, fp: int) -> float | None:
    return tp / (tp + fp) if (tp + fp) else None


def _z_for(ci_level: float) -> float:
    """The two-sided standard-normal critical value for ``ci_level`` (e.g. 0.95 → ≈1.96)."""
    return NormalDist().inv_cdf(0.5 + ci_level / 2.0)


def _wilson_interval(successes: int, trials: int, z: float) -> tuple[float, float] | None:
    """The Wilson score interval for ``successes`` / ``trials`` at critical value ``z``.

    A small-sample-honest interval on a binomial proportion: it stays within ``[0, 1]`` and stays
    informative at the extremes — ``successes == trials`` yields an upper bound of exactly 1.0 with
    a lower bound below it (so a 1-of-1 estimate does not read as certainty), and ``successes == 0``
    yields a lower bound of exactly 0.0. Closed-form, so the report stays byte-stable (D8). Returns
    ``None`` when there are no trials — mirroring the ``None`` the point estimate takes there.
    """
    if trials <= 0:
        return None
    phat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denom
    half = (z / denom) * ((phat * (1.0 - phat) / trials + z2 / (4.0 * trials * trials)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))
