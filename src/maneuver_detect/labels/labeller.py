"""The epoch-to-elset-gap labeller and the per-class coverage report.

A maneuver is observable only as a discontinuity *between* two consecutive elsets, so a label is
the **inter-elset gap** that brackets the maneuver epoch, not a point in time (D4).
:func:`label_series` maps each :class:`~maneuver_detect.labels.record.ManeuverLabel` onto the gap of
a per-object mean-element series (the :data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS`
frame) that contains its epoch, and records the matching window — that gap plus one adjacent gap on
each side, ≈ ±2 days — that the benchmark's matching rule uses. A maneuver whose epoch falls outside
the series' span is reported as unmatched.

:func:`label_coverage` summarises a set of labels by orbit class — how many events, how many carry a
ΔV magnitude, how many are catalogue-resolved, and which sources contribute — which is how the
class scope (LEO Δv-labelled, MEO epoch-only, GEO best-effort) is reported.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.schema import ManeuverType

__all__ = [
    "INTERVAL_COLUMNS",
    "ClassCoverage",
    "CoverageReport",
    "LabelledInterval",
    "LabellingResult",
    "intervals_to_frame",
    "label_coverage",
    "label_series",
]


@dataclass(frozen=True)
class LabelledInterval:
    """One maneuver label mapped onto the inter-elset gap that brackets its epoch.

    Attributes:
        norad_id: NORAD id of the object (``None`` when un-crosswalked).
        epoch: The maneuver epoch that was mapped (timezone-aware UTC).
        elset_epoch_before: Epoch of the elset bounding the start of the bracketing gap.
        elset_epoch_after: Epoch of the elset bounding the end of the bracketing gap.
        tol_start: Earliest elset epoch a detection may land on and still match, under the D4
            tolerance (the bracketing gap plus one adjacent gap before it).
        tol_end: Latest elset epoch a detection may land on and still match (one gap after).
        maneuver_type: The label's maneuver type, carried through (``None`` for epoch-only sources).
        delta_v: The label's ``|Δv|`` in m/s, carried through (``None`` for epoch-only sources).
        source: The label source, carried through.
        source_ref: The label's provenance reference, carried through.
        orbit_class: The object's orbit class, carried through.
    """

    norad_id: int | None
    epoch: pd.Timestamp
    elset_epoch_before: pd.Timestamp
    elset_epoch_after: pd.Timestamp
    tol_start: pd.Timestamp
    tol_end: pd.Timestamp
    maneuver_type: ManeuverType | None
    delta_v: float | None
    source: str
    source_ref: str
    orbit_class: OrbitClass


@dataclass(frozen=True)
class LabellingResult:
    """The outcome of labelling one object's series: the matched intervals and the unmatched labels.

    Attributes:
        intervals: One :class:`LabelledInterval` per label whose epoch fell within the series' span.
        unmatched: The labels whose epoch fell before the first or at/after the last elset, so no
            bracketing gap exists in this series.
    """

    intervals: list[LabelledInterval]
    unmatched: list[ManeuverLabel]


def label_series(series: pd.DataFrame, labels: Sequence[ManeuverLabel]) -> LabellingResult:
    """Map each label onto the inter-elset gap of ``series`` that brackets its epoch.

    ``series`` is one object's mean-element series (it must carry an ``epoch`` column, e.g. the
    :func:`~maneuver_detect.data.history.assemble` output); ``labels`` are assumed to belong to that
    object. For each label, the bracketing gap ``[t_i, t_{i+1})`` containing its epoch is found and
    a :class:`LabelledInterval` is produced, carrying the bounding elset epochs and the D4 matching
    window (that gap ±1 adjacent gap). Labels outside the series' span are ``unmatched``.
    """
    if "epoch" not in series.columns:
        raise ValueError("series must carry an 'epoch' column")

    epochs: list[pd.Timestamp] = sorted(pd.Timestamp(value) for value in series["epoch"])
    n = len(epochs)

    intervals: list[LabelledInterval] = []
    unmatched: list[ManeuverLabel] = []
    for label in labels:
        epoch = pd.Timestamp(label.epoch)
        # First elset strictly after the epoch; the gap [epochs[p-1], epochs[p]) then brackets it.
        p = bisect.bisect_right(epochs, epoch)
        if p == 0 or p == n:
            unmatched.append(label)
            continue
        i = p - 1
        intervals.append(
            LabelledInterval(
                norad_id=label.norad_id,
                epoch=epoch,
                elset_epoch_before=epochs[i],
                elset_epoch_after=epochs[i + 1],
                tol_start=epochs[max(0, i - 1)],
                tol_end=epochs[min(n - 1, i + 2)],
                maneuver_type=label.maneuver_type,
                delta_v=label.delta_v,
                source=label.source,
                source_ref=label.source_ref,
                orbit_class=label.orbit_class,
            )
        )
    return LabellingResult(intervals=intervals, unmatched=unmatched)


#: Columns of the :func:`intervals_to_frame` view, in order.
INTERVAL_COLUMNS: tuple[str, ...] = (
    "norad_id",
    "epoch",
    "elset_epoch_before",
    "elset_epoch_after",
    "tol_start",
    "tol_end",
    "maneuver_type",
    "delta_v",
    "orbit_class",
    "source",
    "source_ref",
)

_DATETIME_DTYPE = "datetime64[ns, UTC]"


def intervals_to_frame(intervals: Sequence[LabelledInterval]) -> pd.DataFrame:
    """Serialise labelled intervals to a DataFrame (:data:`INTERVAL_COLUMNS`) for downstream use."""
    data = {
        "norad_id": pd.array([iv.norad_id for iv in intervals], dtype="Int64"),
        "epoch": pd.Series([iv.epoch for iv in intervals], dtype=_DATETIME_DTYPE),
        "elset_epoch_before": pd.Series(
            [iv.elset_epoch_before for iv in intervals], dtype=_DATETIME_DTYPE
        ),
        "elset_epoch_after": pd.Series(
            [iv.elset_epoch_after for iv in intervals], dtype=_DATETIME_DTYPE
        ),
        "tol_start": pd.Series([iv.tol_start for iv in intervals], dtype=_DATETIME_DTYPE),
        "tol_end": pd.Series([iv.tol_end for iv in intervals], dtype=_DATETIME_DTYPE),
        "maneuver_type": pd.array(
            [None if iv.maneuver_type is None else iv.maneuver_type.value for iv in intervals],
            dtype="string",
        ),
        "delta_v": pd.Series(
            [float("nan") if iv.delta_v is None else iv.delta_v for iv in intervals],
            dtype="float64",
        ),
        "orbit_class": pd.Series([iv.orbit_class.value for iv in intervals], dtype="string"),
        "source": pd.Series([iv.source for iv in intervals], dtype="string"),
        "source_ref": pd.Series([iv.source_ref for iv in intervals], dtype="string"),
    }
    return pd.DataFrame(data, columns=list(INTERVAL_COLUMNS))


@dataclass(frozen=True)
class ClassCoverage:
    """Per-orbit-class label coverage.

    Attributes:
        orbit_class: The orbit class summarised.
        n_events: Number of labelled maneuver events in this class.
        n_with_delta_v: How many carry a ΔV magnitude (the Δv-labelled subset).
        n_with_norad: How many are resolved to a NORAD id.
        sources: The distinct sources contributing to this class, sorted.
    """

    orbit_class: OrbitClass
    n_events: int
    n_with_delta_v: int
    n_with_norad: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class CoverageReport:
    """Per-class coverage across a label set, plus the total event count.

    Attributes:
        per_class: One :class:`ClassCoverage` per :class:`OrbitClass`, present even at zero count so
            the report shape is stable.
        total: Total number of labelled events across all classes.
    """

    per_class: dict[OrbitClass, ClassCoverage]
    total: int

    def summary(self) -> str:
        """A human-readable one-line-per-class coverage summary."""
        lines = [f"{self.total} labelled maneuver events"]
        for orbit_class in OrbitClass:
            cov = self.per_class[orbit_class]
            sources = ", ".join(cov.sources) if cov.sources else "—"
            lines.append(
                f"  {orbit_class.value}: {cov.n_events} events "
                f"({cov.n_with_delta_v} with Δv, {cov.n_with_norad} catalogue-resolved) "
                f"[{sources}]"
            )
        return "\n".join(lines)


def label_coverage(labels: Sequence[ManeuverLabel]) -> CoverageReport:
    """Summarise ``labels`` by orbit class — events, Δv-availability, catalogue resolution, sources.

    Every :class:`OrbitClass` appears in the report (zero counts included), so the per-class scope —
    LEO carrying ΔV, MEO epoch-only, GEO best-effort — is reported in a stable shape regardless of
    which classes the input happens to contain.
    """
    per_class: dict[OrbitClass, ClassCoverage] = {}
    for orbit_class in OrbitClass:
        in_class = [label for label in labels if label.orbit_class is orbit_class]
        sources = tuple(sorted({label.source for label in in_class}))
        per_class[orbit_class] = ClassCoverage(
            orbit_class=orbit_class,
            n_events=len(in_class),
            n_with_delta_v=sum(1 for label in in_class if label.delta_v is not None),
            n_with_norad=sum(1 for label in in_class if label.norad_id is not None),
            sources=sources,
        )
    return CoverageReport(per_class=per_class, total=len(labels))
