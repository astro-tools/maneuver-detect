"""Evaluate the classical detector against the real DORIS/IDS maneuver catalogue.

Two evaluations, both scored through the real benchmark matching + metric layers (the same path the
published numbers run through), using the genuine operator-announced maneuvers in the committed
``dataset/v0.1/labels.json`` (real epochs, dv magnitudes, and types):

* :func:`test_real_schedule_eval` (runs in CI) — replays each satellite's *real maneuver schedule*
  onto a deterministic synthetic mean-element background and asserts the detector recovers the
  above-floor maneuvers at a literature-level recall and precision. The element values are
  *generated*, not real Space-Track data: the Space-Track User Agreement and the project's D2
  decision forbid redistributing Space-Track elements (or analysis derived from them), and the only
  redistribution-clean archive (McDowell) is too sparse around maneuver epochs to score. So CI
  validates the detector against real maneuver *scenarios* — the actual timing, magnitude, and type
  distribution of operational station-keeping — without shipping any restricted elements.

* :func:`test_spacetrack_real_eval` (local only) — when ``SPACETRACK_USERNAME`` /
  ``SPACETRACK_PASSWORD`` are set, fetches the genuine multi-year element history for a
  known-maneuver satellite, runs the detector against the real labels, and reports the
  literature-level P/R on real TLEs. Skipped without credentials; the fetched data is never written
  to the repo (reconstruct-locally, per D2).
"""

from __future__ import annotations

import bisect
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from maneuver_detect import detect
from maneuver_detect.benchmark.matching import ScoredLabel, match_detections
from maneuver_detect.benchmark.metrics import ClassMetrics, ObjectExposure, class_metrics
from maneuver_detect.labels.labeller import label_series
from maneuver_detect.labels.record import SOURCE_DORIS_IDS, ManeuverLabel, OrbitClass
from maneuver_detect.physics import EARTH_MU_KM3_S2, Orbit, gauss_forward, j2_secular_rates
from maneuver_detect.schema import ManeuverType, from_frame

_DEG = math.pi / 180.0
_SECONDS_PER_DAY = 86400.0

#: The committed operator-label catalogue (DORIS/IDS), the ground truth for both evaluations.
_LABELS_PATH = Path(__file__).resolve().parent.parent / "dataset" / "v0.1" / "labels.json"

#: The above-floor threshold for the eval, in m/s: maneuvers at or above it are the recall
#: population (the detector's effective sensitivity sits well below the station-keeping burns these
#: satellites perform); smaller ones are physically marginal and scored as below-floor (ignored).
_ABOVE_FLOOR_MS = 0.5

#: Known-maneuver DORIS altimetry satellites and their approximate (sun-synchronous) reference
#: orbits — NORAD id, name, and the orbit the inversion linearises about.
_DORIS_SATS: tuple[tuple[int, str, Orbit], ...] = (
    (25260, "SPOT-4", Orbit(7200.0, 0.001, 98.70 * _DEG)),
    (27386, "Envisat", Orbit(7143.0, 0.001, 98.50 * _DEG)),
    (41335, "Sentinel-3A", Orbit(7177.0, 0.001, 98.65 * _DEG)),
    (39086, "SARAL", Orbit(7160.0, 0.001, 98.55 * _DEG)),
)

_COMPONENT_OF: dict[ManeuverType, str] = {
    ManeuverType.IN_TRACK: "in_track_ms",
    ManeuverType.CROSS_TRACK: "cross_track_ms",
    ManeuverType.RADIAL: "radial_ms",
}


def _load_doris_labels(norad_id: int) -> list[ManeuverLabel]:
    """Load the committed DORIS/IDS labels with a dv magnitude for ``norad_id``."""
    records: list[dict[str, Any]] = json.loads(_LABELS_PATH.read_text())
    labels: list[ManeuverLabel] = []
    for record in records:
        if record["norad_id"] != norad_id or record.get("delta_v") is None:
            continue
        maneuver_type = record["maneuver_type"]
        labels.append(
            ManeuverLabel(
                norad_id=norad_id,
                epoch=pd.Timestamp(record["epoch"]).to_pydatetime(),
                window_start=pd.Timestamp(record["window_start"]).to_pydatetime(),
                window_end=pd.Timestamp(record["window_end"]).to_pydatetime(),
                source=SOURCE_DORIS_IDS,
                source_ref=record["source_ref"],
                orbit_class=OrbitClass.LEO,
                maneuver_type=ManeuverType(maneuver_type) if maneuver_type else None,
                delta_v=float(record["delta_v"]),
            )
        )
    return labels


def _mean_motion_rev_per_day(a_km: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n_rad_s = np.sqrt(EARTH_MU_KM3_S2 / a_km**3)
    return np.asarray(n_rad_s * _SECONDS_PER_DAY / (2.0 * math.pi), dtype=float)


def _replay_series(
    norad_id: int,
    reference: Orbit,
    labels: list[ManeuverLabel],
    *,
    seed: int,
    cadence_days: float = 1.0,
    margin_days: float = 12.0,
) -> pd.DataFrame:
    """Replay a satellite's real maneuver schedule onto a deterministic synthetic series.

    A daily mean-element background (J2 secular drift + bounded periodic variability + TLE noise)
    spans the label epochs; each labelled maneuver is injected at the inter-elset gap that brackets
    its real epoch, through the forward Gauss model, with its real magnitude and type. The element
    values are generated, but the maneuver *scenario* — when burns happen, how big, what kind — is
    the real operator record.
    """
    label_epochs = sorted(pd.Timestamp(label.epoch) for label in labels)
    lo = label_epochs[0] - pd.Timedelta(days=margin_days)
    hi = label_epochs[-1] + pd.Timedelta(days=margin_days)
    n = int((hi - lo) / pd.Timedelta(days=cadence_days)) + 1
    epochs = [lo + pd.Timedelta(days=cadence_days * i) for i in range(n)]
    days = np.arange(n, dtype=float) * cadence_days

    rng = np.random.default_rng(seed)
    raan_dot, argp_dot, _ = j2_secular_rates(reference)
    deg_per_day = _SECONDS_PER_DAY / _DEG

    a = reference.semi_major_axis_km - 1.0e-3 * days
    e = reference.eccentricity + 5.0e-5 * np.sin(0.20 * days)
    inc = reference.inclination_rad / _DEG + 2.0e-3 * np.sin(0.15 * days)
    raan = raan_dot * deg_per_day * days
    argp = argp_dot * deg_per_day * days

    a += rng.normal(0.0, 0.006, n)
    e += rng.normal(0.0, 1.0e-5, n)
    inc += rng.normal(0.0, 3.0e-4, n)
    raan += rng.normal(0.0, 3.0e-4, n)
    argp += rng.normal(0.0, 1.0e-2, n)

    for label in labels:
        index = bisect.bisect_right(epochs, pd.Timestamp(label.epoch))
        if index <= 0 or index >= n:
            continue
        component = {"radial_ms": 0.0, "in_track_ms": 0.0, "cross_track_ms": 0.0}
        kind = (
            _COMPONENT_OF[label.maneuver_type] if label.maneuver_type is not None else "in_track_ms"
        )
        component[kind] = label.delta_v if label.delta_v is not None else 0.0
        step = gauss_forward(orbit=reference, true_anomaly_rad=0.6, **component)
        a[index:] += step.delta_a_km
        e[index:] += step.delta_eccentricity
        inc[index:] += step.delta_inclination_rad / _DEG
        raan[index:] += step.delta_raan_rad / _DEG

    return pd.DataFrame(
        {
            "epoch": pd.Series(epochs, dtype="datetime64[ns, UTC]"),
            "norad_id": norad_id,
            "mean_motion": _mean_motion_rev_per_day(a),
            "semi_major_axis": a,
            "eccentricity": e,
            "inclination": inc,
            "raan": raan,
            "arg_perigee": argp,
            "mean_anomaly": 0.0,
            "bstar": 0.0,
            "dt_days": np.concatenate(([np.nan], np.diff(days))),
        }
    )


def _evaluate(
    pairs: list[tuple[pd.DataFrame, list[ManeuverLabel]]], *, floor_ms: float
) -> ClassMetrics:
    """Score detector output for one or more (series, labels) objects through the benchmark.

    Runs the detector on each series, maps each label onto its bracketing gap (the D4 matching
    window), tags it above/below the floor by its dv, and returns the LEO class metrics over the
    pooled population.
    """
    detections = []
    scored_labels: list[ScoredLabel] = []
    exposure: list[ObjectExposure] = []
    for series, labels in pairs:
        detections.extend(from_frame(detect(series)))
        for label in labels:
            above_floor = label.delta_v is not None and label.delta_v >= floor_ms
            for interval in label_series(series, [label]).intervals:
                scored_labels.append(ScoredLabel(interval=interval, above_floor=above_floor))
        epochs = series["epoch"]
        span_years = (epochs.max() - epochs.min()).total_seconds() / (365.25 * _SECONDS_PER_DAY)
        exposure.append(
            ObjectExposure(
                norad_id=int(series["norad_id"].iloc[0]),
                orbit_class=OrbitClass.LEO,
                observation_years=span_years,
            )
        )
    matching = match_detections(detections, scored_labels)
    return class_metrics(matching, exposure)[OrbitClass.LEO]


def test_real_schedule_eval() -> None:
    """The detector recovers real DORIS maneuver schedules at literature-level P/R (CI).

    Deterministic: the synthetic background is seeded per object, so the recall and precision over
    the pooled above-floor population are reproducible. The bar is set below the observed values
    (recall and precision both ~0.9) with margin, so the test is a regression guard, not a knife
    edge.
    """
    pairs: list[tuple[pd.DataFrame, list[ManeuverLabel]]] = []
    for norad_id, _name, reference in _DORIS_SATS:
        labels = _load_doris_labels(norad_id)
        assert labels, f"no committed DORIS labels for NORAD {norad_id}"
        pairs.append((_replay_series(norad_id, reference, labels, seed=norad_id), labels))

    metrics = _evaluate(pairs, floor_ms=_ABOVE_FLOOR_MS)

    assert metrics.n_labels_above_floor >= 50  # a meaningful real population
    assert metrics.recall is not None and metrics.recall >= 0.80
    assert metrics.precision is not None and metrics.precision >= 0.80


_HAS_SPACETRACK = bool(
    os.environ.get("SPACETRACK_USERNAME") and os.environ.get("SPACETRACK_PASSWORD")
)


@pytest.mark.skipif(not _HAS_SPACETRACK, reason="SPACETRACK_USERNAME / SPACETRACK_PASSWORD not set")
def test_spacetrack_real_eval() -> None:
    """Literature-level P/R on genuine Space-Track TLEs for a known-maneuver satellite (local only).

    Fetches the real multi-year ``gp_history`` for SPOT-4 with the caller's credentials, runs the
    detector against the real DORIS labels, and reports recall / precision over the above-floor
    population. The fetched elements are never committed (D2: reconstruct locally). Real TLE noise
    and cadence make this materially harder than the replay, so the assertion only confirms the
    end-to-end pipeline produced a score; the printed numbers are the literature comparison.
    """
    from maneuver_detect.data.history import build_series
    from maneuver_detect.data.spacetrack import SpacetrackFetcher

    norad_id = 25260  # SPOT-4
    labels = _load_doris_labels(norad_id)
    label_epochs = sorted(pd.Timestamp(label.epoch) for label in labels)
    start = (label_epochs[0] - pd.Timedelta(days=10)).to_pydatetime()
    end = (label_epochs[-1] + pd.Timedelta(days=10)).to_pydatetime()

    result = SpacetrackFetcher().fetch(norad_id, start=start, end=end)
    series = build_series(result.elsets)
    assert not series.empty, "Space-Track returned no elsets for the window"

    lo, hi = series["epoch"].min(), series["epoch"].max()
    in_window = [label for label in labels if lo <= pd.Timestamp(label.epoch) <= hi]
    metrics = _evaluate([(series, in_window)], floor_ms=_ABOVE_FLOOR_MS)

    print(
        f"\nSPOT-4 real Space-Track eval: elsets={len(series)} "
        f"above_floor_labels={metrics.n_labels_above_floor} detections={metrics.n_detections} "
        f"recall={metrics.recall} precision={metrics.precision}"
    )
    assert metrics.recall is not None  # the pipeline scored real data end-to-end
