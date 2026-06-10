"""The canonical maneuver schema — the frozen library contract.

A detected maneuver is one row of the canonical DataFrame that :func:`maneuver_detect.detect`
returns and the benchmark scores against. This module is the single source of truth for that
schema: the per-maneuver :class:`Maneuver` record, the canonical column set and dtypes
(:data:`COLUMNS`), and the lossless :func:`to_frame` / :func:`from_frame` serialisation the
detectors and the scorer share.

The columns, in order, are ``epoch`` (UTC detection epoch), ``confidence`` (calibrated, ``[0, 1]``),
``type`` (in-track / cross-track / radial), ``delta_v_estimate`` (m/s, ``NaN`` when not reported),
and the provenance ``norad_id``, ``elset_epoch_before``, ``elset_epoch_after`` (the bounding elset
epochs of the inter-elset gap the detection brackets).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import pandas as pd

__all__ = [
    "COLUMNS",
    "Maneuver",
    "ManeuverType",
    "empty_frame",
    "from_frame",
    "to_frame",
    "validate_frame",
]


class ManeuverType(str, Enum):
    """The maneuver type, attributed from the dominant element change (D5)."""

    IN_TRACK = "in-track"
    CROSS_TRACK = "cross-track"
    RADIAL = "radial"


@dataclass(frozen=True)
class Maneuver:
    """A single detected maneuver — one row of the canonical DataFrame.

    Attributes:
        epoch: Detection epoch (timezone-aware UTC).
        confidence: Calibrated detection confidence in ``[0, 1]``.
        type: The maneuver type (:class:`ManeuverType`).
        delta_v_estimate: Estimated ``|Δv|`` in m/s, or ``None`` when not reported — below the
            detectability floor, or for a radial-dominated maneuver (D5).
        norad_id: NORAD catalogue id of the object.
        elset_epoch_before: Epoch of the elset bounding the start of the inter-elset gap that
            brackets the maneuver (timezone-aware UTC).
        elset_epoch_after: Epoch of the elset bounding the end of that gap (timezone-aware UTC).
    """

    epoch: pd.Timestamp
    confidence: float
    type: ManeuverType
    delta_v_estimate: float | None
    norad_id: int
    elset_epoch_before: pd.Timestamp
    elset_epoch_after: pd.Timestamp

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        if self.delta_v_estimate is not None and self.delta_v_estimate < 0.0:
            raise ValueError(
                f"delta_v_estimate must be non-negative, got {self.delta_v_estimate!r}"
            )
        for field_name, ts in (
            ("epoch", self.epoch),
            ("elset_epoch_before", self.elset_epoch_before),
            ("elset_epoch_after", self.elset_epoch_after),
        ):
            if ts.tzinfo is None:
                raise ValueError(f"{field_name} must be timezone-aware (UTC)")


COLUMNS: tuple[str, ...] = (
    "epoch",
    "confidence",
    "type",
    "delta_v_estimate",
    "norad_id",
    "elset_epoch_before",
    "elset_epoch_after",
)

_DATETIME_DTYPE = "datetime64[ns, UTC]"


def to_frame(maneuvers: Sequence[Maneuver]) -> pd.DataFrame:
    """Serialise ``maneuvers`` to the canonical DataFrame (canonical column order and dtypes).

    An empty sequence yields an empty frame that still carries the full schema, so a detector
    that finds nothing returns the same shape as one that finds something.
    """
    data = {
        "epoch": pd.Series([m.epoch for m in maneuvers], dtype=_DATETIME_DTYPE),
        "confidence": pd.Series([m.confidence for m in maneuvers], dtype="float64"),
        "type": pd.Series([m.type.value for m in maneuvers], dtype="string"),
        "delta_v_estimate": pd.Series(
            [math.nan if m.delta_v_estimate is None else m.delta_v_estimate for m in maneuvers],
            dtype="float64",
        ),
        "norad_id": pd.Series([m.norad_id for m in maneuvers], dtype="int64"),
        "elset_epoch_before": pd.Series(
            [m.elset_epoch_before for m in maneuvers], dtype=_DATETIME_DTYPE
        ),
        "elset_epoch_after": pd.Series(
            [m.elset_epoch_after for m in maneuvers], dtype=_DATETIME_DTYPE
        ),
    }
    return pd.DataFrame(data, columns=list(COLUMNS))


def empty_frame() -> pd.DataFrame:
    """Return an empty canonical frame carrying the full schema and dtypes."""
    return to_frame([])


def from_frame(frame: pd.DataFrame) -> list[Maneuver]:
    """Deserialise a canonical DataFrame back to :class:`Maneuver` records.

    The inverse of :func:`to_frame`: ``NaN`` ``delta_v_estimate`` values become ``None``. Raises
    :class:`ValueError` if ``frame`` is missing canonical columns.
    """
    validate_frame(frame)
    maneuvers: list[Maneuver] = []
    for record in frame.to_dict(orient="records"):
        delta_v = record["delta_v_estimate"]
        maneuvers.append(
            Maneuver(
                epoch=pd.Timestamp(record["epoch"]),
                confidence=float(record["confidence"]),
                type=ManeuverType(record["type"]),
                delta_v_estimate=None if pd.isna(delta_v) else float(delta_v),
                norad_id=int(record["norad_id"]),
                elset_epoch_before=pd.Timestamp(record["elset_epoch_before"]),
                elset_epoch_after=pd.Timestamp(record["elset_epoch_after"]),
            )
        )
    return maneuvers


def validate_frame(frame: pd.DataFrame) -> None:
    """Validate that ``frame`` carries the canonical columns; raise :class:`ValueError` if not."""
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing canonical columns: {missing}")
