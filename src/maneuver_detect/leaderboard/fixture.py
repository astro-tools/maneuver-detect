"""The held-out scoring fixture the leaderboard Space loads — labels, exposure, and timing floor.

The benchmark scorer matches a submission against held-out
:class:`~maneuver_detect.benchmark.ScoredLabel`\\ s over an
:class:`~maneuver_detect.benchmark.ObjectExposure` population. Both are *derived* from the
reconstructed mean-element series — the per-object floor that flags a label above- or
below-detectability, the inter-elset gap each label's matching window spans, and the era-only
observation span — none of which the recipe-first dataset ships. This module serialises that derived
state once, offline, so the Space can run the shipped scorer on the frozen test split without
reconstructing the series at request time.

The fixture carries the D4 matching windows (``tol_start`` / ``tol_end``), which are real elset
epochs — derived Space-Track data the dataset deliberately does not redistribute (D2). It is
therefore **not** a committed repo artifact: it is built by ``leaderboard/build_fixture.py`` from a
credentialed reconstruction and supplied to the Space as private deploy-time data. The held-out
labels it encodes are the same ones the public dataset already publishes (the v0.2 answer key is
public — the D12 amendment), so the board is a reproducibility / convenience board, not a
hidden-label competition.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from maneuver_detect.benchmark import ObjectExposure, ScoredLabel
from maneuver_detect.labels.labeller import LabelledInterval
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import ManeuverType

__all__ = [
    "ScoringFixture",
    "fixture_to_json",
    "load_fixture",
]


@dataclass(frozen=True)
class ScoringFixture:
    """The frozen test-split scoring inputs plus the published timing-only floor.

    Attributes:
        dataset_version: The dataset version the fixture was built from (e.g. ``"0.2.0"``).
        labels: The held-out :class:`~maneuver_detect.benchmark.ScoredLabel`\\ s the scorer matches
            a submission against — the test split, gated by the per-object detectability floor.
        exposure: The scored population (one :class:`~maneuver_detect.benchmark.ObjectExposure` per
            object), the satellite-year denominator the false-alarm rate is over.
        timing_floor: The published D11 timing-only "cheating floor" — per-class rank-AUC a Δt-only
            model reaches, shown alongside every score so a result is read in context. A benchmark
            constant, never derived from a submission.
    """

    dataset_version: str
    labels: tuple[ScoredLabel, ...]
    exposure: tuple[ObjectExposure, ...]
    timing_floor: Mapping[str, float]


def fixture_to_json(fixture: ScoringFixture) -> str:
    """Serialise ``fixture`` to canonical JSON — sorted keys, ISO-8601 UTC epochs."""
    payload = {
        "dataset_version": fixture.dataset_version,
        "labels": [_label_payload(label) for label in fixture.labels],
        "exposure": [_exposure_payload(obj) for obj in fixture.exposure],
        "timing_floor": dict(fixture.timing_floor),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def load_fixture(text: str) -> ScoringFixture:
    """Parse a scoring fixture file (the inverse of :func:`fixture_to_json`)."""
    payload = json.loads(text)
    return ScoringFixture(
        dataset_version=str(payload["dataset_version"]),
        labels=tuple(_read_label(record) for record in payload["labels"]),
        exposure=tuple(_read_exposure(record) for record in payload["exposure"]),
        timing_floor={str(k): float(v) for k, v in payload["timing_floor"].items()},
    )


def _label_payload(label: ScoredLabel) -> dict[str, object]:
    interval = label.interval
    return {
        "above_floor": label.above_floor,
        "interval": {
            "norad_id": interval.norad_id,
            "epoch": interval.epoch.isoformat(),
            "elset_epoch_before": interval.elset_epoch_before.isoformat(),
            "elset_epoch_after": interval.elset_epoch_after.isoformat(),
            "tol_start": interval.tol_start.isoformat(),
            "tol_end": interval.tol_end.isoformat(),
            "maneuver_type": None
            if interval.maneuver_type is None
            else interval.maneuver_type.value,
            "delta_v": interval.delta_v,
            "source": interval.source,
            "source_ref": interval.source_ref,
            "orbit_class": interval.orbit_class.value,
        },
    }


def _read_label(record: Any) -> ScoredLabel:
    interval = record["interval"]
    maneuver_type = interval["maneuver_type"]
    delta_v = interval["delta_v"]
    norad_id = interval["norad_id"]
    return ScoredLabel(
        interval=LabelledInterval(
            norad_id=None if norad_id is None else int(norad_id),
            epoch=pd.Timestamp(interval["epoch"]),
            elset_epoch_before=pd.Timestamp(interval["elset_epoch_before"]),
            elset_epoch_after=pd.Timestamp(interval["elset_epoch_after"]),
            tol_start=pd.Timestamp(interval["tol_start"]),
            tol_end=pd.Timestamp(interval["tol_end"]),
            maneuver_type=None if maneuver_type is None else ManeuverType(maneuver_type),
            delta_v=None if delta_v is None else float(delta_v),
            source=str(interval["source"]),
            source_ref=str(interval["source_ref"]),
            orbit_class=OrbitClass(interval["orbit_class"]),
        ),
        above_floor=bool(record["above_floor"]),
    )


def _exposure_payload(obj: ObjectExposure) -> dict[str, object]:
    return {
        "norad_id": obj.norad_id,
        "orbit_class": obj.orbit_class.value,
        "observation_years": obj.observation_years,
    }


def _read_exposure(record: Any) -> ObjectExposure:
    return ObjectExposure(
        norad_id=int(record["norad_id"]),
        orbit_class=OrbitClass(record["orbit_class"]),
        observation_years=float(record["observation_years"]),
    )
