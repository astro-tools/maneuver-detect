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
        per_class: One :class:`~maneuver_detect.benchmark.metrics.ClassMetrics` per
            :class:`~maneuver_detect.labels.record.OrbitClass`, present even at zero.
    """

    operating_point: float
    sweep: tuple[float, ...]
    per_class: dict[OrbitClass, ClassMetrics]

    def headline(self) -> dict[OrbitClass, float | None]:
        """The headline — recall over the above-floor population at the operating point."""
        return {orbit_class: metrics.recall for orbit_class, metrics in self.per_class.items()}

    def to_json(self) -> str:
        """Serialise to canonical JSON — byte-stable across runs and platforms (D8)."""
        payload = {
            "operating_point": self.operating_point,
            "sweep": list(self.sweep),
            "per_class": {
                orbit_class.value: _class_payload(self.per_class[orbit_class])
                for orbit_class in OrbitClass
                if orbit_class in self.per_class
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def summary(self) -> str:
        """A human-readable per-class summary of the headline recall and precision."""
        lines = [f"benchmark score @ {self.operating_point:g} FA/sat-year"]
        for orbit_class in OrbitClass:
            metrics = self.per_class.get(orbit_class)
            if metrics is None:
                continue
            lines.append(
                f"  {orbit_class.value}: recall={_fmt(metrics.recall)} "
                f"precision={_fmt(metrics.precision)} "
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
) -> ScoreReport:
    """Score ``predictions`` against held-out ``labels`` over the ``exposure`` population.

    ``predictions`` is the canonical maneuver frame (or a sequence of :class:`Maneuver`); ``labels``
    are the held-out labels tagged with their detectability-floor status; ``exposure`` is the scored
    population (every prediction and label must belong to an object it lists). Returns a
    deterministic :class:`ScoreReport` — the same inputs always yield the same numbers (D8).
    """
    detections = list(
        from_frame(predictions) if isinstance(predictions, pd.DataFrame) else predictions
    )
    matching = match_detections(detections, list(labels))
    per_class = class_metrics(
        matching, list(exposure), operating_point=operating_point, sweep=sweep
    )
    return ScoreReport(operating_point=operating_point, sweep=tuple(sweep), per_class=per_class)


def read_predictions(text: str) -> list[Maneuver]:
    """Parse a predictions file (a JSON array of canonical maneuver records) into the schema.

    The inverse of :func:`predictions_to_json`. Each record carries the canonical columns
    (:data:`~maneuver_detect.schema.COLUMNS`); a ``null`` ``delta_v_estimate`` becomes ``None``.
    Raises :class:`ValueError` on a record missing a canonical field.
    """
    records = json.loads(text)
    maneuvers: list[Maneuver] = []
    for record in records:
        missing = [column for column in COLUMNS if column not in record]
        if missing:
            raise ValueError(f"prediction record is missing canonical fields: {missing}")
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
        "recall": metrics.recall,
        "precision": metrics.precision,
        "full_population_recall": metrics.full_population_recall,
        "pr_curve": [_pr_payload(point) for point in metrics.pr_curve],
        "confusion": _confusion_payload(metrics.confusion),
    }


def _pr_payload(point: PRPoint) -> dict[str, object]:
    return {
        "fa_per_sat_year": point.fa_per_sat_year,
        "recall": point.recall,
        "precision": point.precision,
    }


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
