"""The deterministic scorer — a predictions file plus held-out labels in, a score report out.

This ties the detection-matching rule (:mod:`~maneuver_detect.benchmark.matching`) to the metric
(:mod:`~maneuver_detect.benchmark.metrics`): match the predictions to the labels under the D4
tolerance, then score per class. The report serialises to canonical JSON — sorted keys, ISO-8601 UTC
epochs, shortest-round-trip floats — so the same predictions reproduce the same numbers
byte-for-byte across runs and platforms, the guarantee the benchmark is frozen on (D8).

The predictions file is the leaderboard-facing artifact: a JSON array of canonical maneuver records
(:mod:`maneuver_detect.schema`). :func:`read_predictions` parses one into the in-memory schema and
:func:`predictions_to_json` writes it back canonically.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from maneuver_detect.benchmark.matching import ScoredLabel, match_detections
from maneuver_detect.benchmark.metrics import (
    DEFAULT_CI_LEVEL,
    DEFAULT_OPERATING_POINT,
    DEFAULT_SWEEP,
    ClassMetrics,
    Confusion,
    ObjectExposure,
    PRPoint,
    class_metrics,
)
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import COLUMNS, Maneuver, ManeuverType, from_frame

__all__ = [
    "ScoreReport",
    "predictions_to_json",
    "read_predictions",
    "score",
]


@dataclass(frozen=True)
class ScoreReport:
    """The benchmark score — per-class metrics at the headline operating point, plus the sweep.

    Attributes:
        operating_point: The headline false-alarm-per-satellite-year target (D4).
        sweep: The false-alarm-per-satellite-year sweep the P/R curve covers.
        ci_level: The confidence level of the per-class recall / precision intervals.
        per_class: One :class:`~maneuver_detect.benchmark.metrics.ClassMetrics` per
            :class:`~maneuver_detect.labels.record.OrbitClass`, present even at zero.
    """

    operating_point: float
    sweep: tuple[float, ...]
    ci_level: float
    per_class: dict[OrbitClass, ClassMetrics]

    def headline(self) -> dict[OrbitClass, float | None]:
        """The headline — recall over the above-floor population at the operating point."""
        return {orbit_class: metrics.recall for orbit_class, metrics in self.per_class.items()}

    def to_json(self) -> str:
        """Serialise to canonical JSON — byte-stable across runs and platforms (D8)."""
        payload = {
            "operating_point": self.operating_point,
            "sweep": list(self.sweep),
            "ci_level": self.ci_level,
            "per_class": {
                orbit_class.value: _class_payload(self.per_class[orbit_class])
                for orbit_class in OrbitClass
                if orbit_class in self.per_class
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def summary(self) -> str:
        """A human-readable per-class summary of the headline recall and precision, with CIs."""
        lines = [f"benchmark score @ {self.operating_point:g} FA/sat-year ({self.ci_level:.0%} CI)"]
        for orbit_class in OrbitClass:
            metrics = self.per_class.get(orbit_class)
            if metrics is None:
                continue
            lines.append(
                f"  {orbit_class.value}: "
                f"recall={_fmt(metrics.recall)}{_fmt_ci(metrics.recall_ci)} "
                f"precision={_fmt(metrics.precision)}{_fmt_ci(metrics.precision_ci)} "
                f"(above-floor labels={metrics.n_labels_above_floor}, "
                f"sat-years={metrics.sat_years:g})"
            )
        return "\n".join(lines)


def score(
    predictions: pd.DataFrame | Sequence[Maneuver],
    labels: Sequence[ScoredLabel],
    exposure: Sequence[ObjectExposure],
    *,
    operating_point: float = DEFAULT_OPERATING_POINT,
    sweep: tuple[float, ...] = DEFAULT_SWEEP,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> ScoreReport:
    """Score ``predictions`` against held-out ``labels`` over the ``exposure`` population.

    ``predictions`` is the canonical maneuver frame (or a sequence of :class:`Maneuver`); ``labels``
    are the held-out labels tagged with their detectability-floor status; ``exposure`` is the scored
    population (every prediction and label must belong to an object it lists). ``ci_level`` (in
    ``(0, 1)``) sets the confidence level of the per-class recall / precision intervals. Returns a
    deterministic :class:`ScoreReport` — the same inputs always yield the same numbers (D8).
    """
    detections = list(
        from_frame(predictions) if isinstance(predictions, pd.DataFrame) else predictions
    )
    matching = match_detections(detections, list(labels))
    per_class = class_metrics(
        matching, list(exposure), operating_point=operating_point, sweep=sweep, ci_level=ci_level
    )
    return ScoreReport(
        operating_point=operating_point,
        sweep=tuple(sweep),
        ci_level=ci_level,
        per_class=per_class,
    )


def read_predictions(text: str) -> list[Maneuver]:
    """Parse a predictions file (a JSON array of canonical maneuver records) into the schema.

    The inverse of :func:`predictions_to_json`. Each record must carry **exactly** the canonical
    columns (:data:`~maneuver_detect.schema.COLUMNS`) and nothing else; a ``null``
    ``delta_v_estimate`` becomes ``None``. The schema is fixed both ways: a record missing a
    canonical field *or* carrying any field beyond them is rejected with :class:`ValueError`, so a
    submission cannot smuggle a query or any other non-prediction payload past the reader (the D12
    fixed-schema integrity surface). A non-array payload, or a record that is not a JSON object, is
    rejected the same way.
    """
    records = json.loads(text)
    if not isinstance(records, list):
        raise ValueError("predictions file must be a JSON array of maneuver records")
    allowed = set(COLUMNS)
    maneuvers: list[Maneuver] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                f"prediction record must be a JSON object, got {type(record).__name__}"
            )
        missing = [column for column in COLUMNS if column not in record]
        if missing:
            raise ValueError(f"prediction record is missing canonical fields: {missing}")
        unknown = sorted(set(record) - allowed)
        if unknown:
            raise ValueError(f"prediction record carries unknown fields: {unknown}")
        delta_v = record["delta_v_estimate"]
        maneuvers.append(
            Maneuver(
                epoch=pd.Timestamp(record["epoch"]),
                confidence=float(record["confidence"]),
                type=ManeuverType(record["type"]),
                delta_v_estimate=None if delta_v is None else float(delta_v),
                norad_id=int(record["norad_id"]),
                elset_epoch_before=pd.Timestamp(record["elset_epoch_before"]),
                elset_epoch_after=pd.Timestamp(record["elset_epoch_after"]),
            )
        )
    return maneuvers


def predictions_to_json(maneuvers: Sequence[Maneuver]) -> str:
    """Serialise ``maneuvers`` to a canonical predictions file (sorted keys, ISO-8601 epochs)."""
    payload = [
        {
            "epoch": maneuver.epoch.isoformat(),
            "confidence": maneuver.confidence,
            "type": maneuver.type.value,
            "delta_v_estimate": maneuver.delta_v_estimate,
            "norad_id": maneuver.norad_id,
            "elset_epoch_before": maneuver.elset_epoch_before.isoformat(),
            "elset_epoch_after": maneuver.elset_epoch_after.isoformat(),
        }
        for maneuver in maneuvers
    ]
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _class_payload(metrics: ClassMetrics) -> dict[str, object]:
    return {
        "orbit_class": metrics.orbit_class.value,
        "sat_years": metrics.sat_years,
        "n_objects": metrics.n_objects,
        "n_detections": metrics.n_detections,
        "n_labels_above_floor": metrics.n_labels_above_floor,
        "n_labels_total": metrics.n_labels_total,
        "operating_point": metrics.operating_point,
        "ci_level": metrics.ci_level,
        "recall": metrics.recall,
        "recall_ci": _ci_payload(metrics.recall_ci),
        "precision": metrics.precision,
        "precision_ci": _ci_payload(metrics.precision_ci),
        "full_population_recall": metrics.full_population_recall,
        # The per-class operating point (D7) — the confidence cut admitted within the false-alarm
        # budget at the headline rate, the threshold an uncertainty-calibration pass publishes. An
        # additive v0.3-boundary field: ``None`` when no detection is admitted (the v0.2 report kept
        # it in-memory only; persisting it is the v0.3 protocol bump).
        "operating_point_confidence": metrics.operating_point_confidence,
        "pr_curve": [_pr_payload(point) for point in metrics.pr_curve],
        "confusion": _confusion_payload(metrics.confusion),
    }


def _pr_payload(point: PRPoint) -> dict[str, object]:
    return {
        "fa_per_sat_year": point.fa_per_sat_year,
        "recall": point.recall,
        "recall_ci": _ci_payload(point.recall_ci),
        "precision": point.precision,
        "precision_ci": _ci_payload(point.precision_ci),
    }


def _ci_payload(ci: tuple[float, float] | None) -> list[float] | None:
    """Serialise a ``(low, high)`` confidence interval to a JSON list, or ``None`` if undefined."""
    return None if ci is None else [ci[0], ci[1]]


def _confusion_payload(confusion: Confusion) -> dict[str, dict[str, int]]:
    return {
        true_type.value: {
            predicted_type.value: confusion.counts[true_type][predicted_type]
            for predicted_type in ManeuverType
        }
        for true_type in ManeuverType
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_ci(ci: tuple[float, float] | None) -> str:
    return "" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
