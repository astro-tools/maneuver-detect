"""Tests for the classical reference detector.

The detector cannot be validated against the real labelled satellites in CI (their multi-year
series are reconstructed from Space-Track, which is not redistributable, and need live
credentials), so correctness here is established on *synthetic* mean-element series with known
ground truth: a realistic secular-drift + perturbation + TLE-noise background into which maneuvers
of a known Δv, type, and epoch are injected through the forward Gauss model the inversion inverts
(:func:`maneuver_detect.physics.gauss_forward`). That lets every property be checked exactly —
detection at the right gap, the recovered type and Δv, the floor gate, and, crucially, the
**negative control**: no detector firing on drag / SRP / luni-solar variability and TLE noise over
maneuver-free intervals. A population-level precision/recall check is scored through the real
benchmark matching and metric layers, the same path the published numbers run through. The
literature-level numbers on the real satellites are reproduced by the credentialed evaluation under
``tests/eval`` (see :mod:`tests.eval.run_real_eval`), not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from maneuver_detect import detect
from maneuver_detect.benchmark.matching import ScoredLabel, match_detections
from maneuver_detect.benchmark.metrics import ObjectExposure, class_metrics
from maneuver_detect.detectors import ClassicalDetector, available_models, get_detector
from maneuver_detect.labels.labeller import label_series
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.physics import (
    EARTH_MU_KM3_S2,
    Inversion,
    Orbit,
    gauss_forward,
    j2_secular_rates,
)
from maneuver_detect.schema import COLUMNS, ManeuverType, from_frame, validate_frame

_DEG = math.pi / 180.0
_SECONDS_PER_DAY = 86400.0

# Representative LEO / GEO reference orbits for the synthetic background.
_LEO = Orbit(semi_major_axis_km=6778.0, eccentricity=0.001, inclination_rad=66.0 * _DEG)
_GEO = Orbit(semi_major_axis_km=42164.0, eccentricity=0.0002, inclination_rad=2.0 * _DEG)

# Per-element TLE-noise scales (1-sigma), in each element's native units, in the range LEO altimetry
# TLEs sit at; the magnitudes are what the detector's robust scale calibrates against.
_NOISE = {
    "semi_major_axis": 0.006,  # km
    "eccentricity": 1.0e-5,
    "inclination": 3.0e-4,  # deg
    "raan": 3.0e-4,  # deg
    "arg_perigee": 1.0e-2,  # deg
}


@dataclass(frozen=True)
class Burn:
    """One injected maneuver: the gap it brackets, its RSW kind, magnitude, and burn anomaly."""

    gap_index: int  # index of the first elset after the burn; the burn brackets [gap-1, gap]
    component: str  # "in_track_ms" | "cross_track_ms" | "radial_ms"
    delta_v_ms: float
    true_anomaly_rad: float = 0.7

    @property
    def maneuver_type(self) -> ManeuverType:
        return {
            "in_track_ms": ManeuverType.IN_TRACK,
            "cross_track_ms": ManeuverType.CROSS_TRACK,
            "radial_ms": ManeuverType.RADIAL,
        }[self.component]


def _mean_motion_rev_per_day(a_km: float) -> float:
    n_rad_s = math.sqrt(EARTH_MU_KM3_S2 / a_km**3)
    return n_rad_s * _SECONDS_PER_DAY / (2.0 * math.pi)


def synthetic_series(
    *,
    norad_id: int,
    seed: int,
    n: int = 120,
    cadence_days: float = 1.0,
    reference: Orbit = _LEO,
    burns: tuple[Burn, ...] = (),
    noise_scale: float = 1.0,
    confound_scale: float = 1.0,
) -> pd.DataFrame:
    """Build a synthetic mean-element series with realistic confounds and injected burns.

    The background carries the J2 secular nodal regression and apsidal rotation
    (:func:`maneuver_detect.physics.j2_secular_rates`), a slow drag decay of the semi-major axis,
    bounded periodic SRP / luni-solar wobble on the eccentricity and inclination, and per-element
    Gaussian TLE noise. Each :class:`Burn` adds the forward-Gauss element step from its Δv to every
    sample at or after its gap. The frame carries the canonical
    :data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS`.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    epochs = [t0 + pd.Timedelta(days=cadence_days * i) for i in range(n)]
    days = np.arange(n, dtype=float) * cadence_days

    raan_dot, argp_dot, _ = j2_secular_rates(reference)
    deg_per_day = _SECONDS_PER_DAY / _DEG

    a = reference.semi_major_axis_km - 2.0e-3 * confound_scale * days  # mild drag decay (km/day)
    e = reference.eccentricity + 1.0e-4 * confound_scale * np.sin(0.30 * days)  # SRP/luni-solar
    inc = reference.inclination_rad / _DEG + 5.0e-3 * confound_scale * np.sin(0.20 * days)
    raan = 30.0 + raan_dot * deg_per_day * days
    argp = 90.0 + argp_dot * deg_per_day * days

    a += rng.normal(0.0, _NOISE["semi_major_axis"] * noise_scale, n)
    e += rng.normal(0.0, _NOISE["eccentricity"] * noise_scale, n)
    inc += rng.normal(0.0, _NOISE["inclination"] * noise_scale, n)
    raan += rng.normal(0.0, _NOISE["raan"] * noise_scale, n)
    argp += rng.normal(0.0, _NOISE["arg_perigee"] * noise_scale, n)

    for burn in burns:
        kwargs = {"radial_ms": 0.0, "in_track_ms": 0.0, "cross_track_ms": 0.0}
        kwargs[burn.component] = burn.delta_v_ms
        step = gauss_forward(orbit=reference, true_anomaly_rad=burn.true_anomaly_rad, **kwargs)
        idx = burn.gap_index
        a[idx:] += step.delta_a_km
        e[idx:] += step.delta_eccentricity
        inc[idx:] += step.delta_inclination_rad / _DEG
        raan[idx:] += step.delta_raan_rad / _DEG

    mean_motion = np.array([_mean_motion_rev_per_day(value) for value in a], dtype=float)
    dt_days = np.concatenate(([np.nan], np.diff(days)))
    return pd.DataFrame(
        {
            "epoch": pd.Series(epochs, dtype="datetime64[ns, UTC]"),
            "norad_id": norad_id,
            "mean_motion": mean_motion,
            "semi_major_axis": a,
            "eccentricity": e,
            "inclination": inc,
            "raan": raan,
            "arg_perigee": argp,
            "mean_anomaly": 0.0,
            "bstar": 0.0,
            "dt_days": dt_days,
        }
    )


def _gap_midpoint(frame: pd.DataFrame, gap_index: int) -> pd.Timestamp:
    epochs = list(frame["epoch"])
    before, after = epochs[gap_index - 1], epochs[gap_index]
    return pd.Timestamp(before + (after - before) / 2)


# --------------------------------------------------------------------------- registration & schema
def test_registered_as_classical() -> None:
    assert "classical" in available_models()
    assert isinstance(get_detector("classical"), ClassicalDetector)


def test_detect_default_model_dispatches_to_classical() -> None:
    frame = synthetic_series(norad_id=25544, seed=0, burns=(Burn(60, "in_track_ms", 2.0),))
    # The public detect() with no model argument runs the classical detector.
    default = detect(frame)
    explicit = detect(frame, model="classical")
    assert from_frame(default) == from_frame(explicit)
    assert len(default) == 1


def test_empty_history_returns_empty_canonical_frame() -> None:
    out = ClassicalDetector().detect(synthetic_series(norad_id=1, seed=0).iloc[0:0])
    assert list(out.columns) == list(COLUMNS)
    assert out.empty


def test_too_short_history_returns_empty_frame() -> None:
    # Fewer than 2*window+1 samples: no gap can be scored.
    out = ClassicalDetector(window=4).detect(synthetic_series(norad_id=1, seed=0, n=8))
    assert out.empty
    assert list(out.columns) == list(COLUMNS)


def test_output_is_canonical_schema() -> None:
    out = detect(synthetic_series(norad_id=25544, seed=1, burns=(Burn(60, "cross_track_ms", 3.0),)))
    validate_frame(out)
    assert list(out.columns) == list(COLUMNS)
    # Round-trips losslessly through the schema.
    assert len(from_frame(out)) == len(out)
    assert str(out["epoch"].dtype) == "datetime64[ns, UTC]"
    assert out["type"].iloc[0] in {t.value for t in ManeuverType}


def test_missing_columns_raises() -> None:
    frame = synthetic_series(norad_id=1, seed=0).drop(columns=["raan"])
    with pytest.raises(ValueError, match="missing required columns"):
        ClassicalDetector().detect(frame)


# --------------------------------------------------------------------------- positive detection
def test_detects_in_track_burn_with_recovered_delta_v() -> None:
    burn = Burn(60, "in_track_ms", 2.0)
    frame = synthetic_series(norad_id=25544, seed=0, burns=(burn,))
    out = detect(frame)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["type"] == ManeuverType.IN_TRACK.value
    # Detection epoch falls in the bracketing gap, with the right provenance.
    assert row["elset_epoch_before"] == list(frame["epoch"])[burn.gap_index - 1]
    assert row["elset_epoch_after"] == list(frame["epoch"])[burn.gap_index]
    assert row["elset_epoch_before"] <= row["epoch"] <= row["elset_epoch_after"]
    # Δv recovered within the D5 ±25% band, above the floor.
    assert row["delta_v_estimate"] == pytest.approx(2.0, rel=0.25)
    assert row["confidence"] > 0.8


def test_detects_cross_track_burn_with_recovered_delta_v() -> None:
    burn = Burn(60, "cross_track_ms", 3.0)
    frame = synthetic_series(norad_id=39634, seed=2, burns=(burn,))
    out = detect(frame)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["type"] == ManeuverType.CROSS_TRACK.value
    assert row["delta_v_estimate"] == pytest.approx(3.0, rel=0.25)


def test_detects_geo_station_keeping_burn() -> None:
    burn = Burn(60, "in_track_ms", 1.5)
    frame = synthetic_series(norad_id=41866, seed=4, reference=_GEO, burns=(burn,))
    out = detect(frame)
    assert len(out) == 1
    assert out.iloc[0]["type"] == ManeuverType.IN_TRACK.value


def test_two_separated_burns_both_detected() -> None:
    burns = (Burn(40, "in_track_ms", 2.5), Burn(90, "cross_track_ms", 3.0))
    frame = synthetic_series(norad_id=25544, seed=5, burns=burns)
    out = detect(frame)
    assert len(out) == 2
    assert sorted(out["type"]) == sorted(
        [ManeuverType.IN_TRACK.value, ManeuverType.CROSS_TRACK.value]
    )


def test_confidence_increases_with_burn_magnitude() -> None:
    def conf(dv: float) -> float:
        out = detect(synthetic_series(norad_id=1, seed=7, burns=(Burn(60, "in_track_ms", dv),)))
        return float(out.iloc[0]["confidence"])

    assert conf(0.5) < conf(1.5) < conf(5.0)


# --------------------------------------------------------------------------- negative control
@pytest.mark.parametrize("seed", range(12))
def test_negative_control_no_false_alarms(seed: int) -> None:
    # Drag + SRP/luni-solar wobble + secular node regression + TLE noise, but no maneuver: the
    # detector must not fire. This is the false-positive control of the DoD.
    frame = synthetic_series(norad_id=10000 + seed, seed=seed, n=150)
    assert detect(frame).empty


@pytest.mark.parametrize("seed", range(4))
def test_negative_control_robust_to_elevated_noise(seed: int) -> None:
    # Triple the nominal TLE noise: the self-calibrated threshold still holds the line.
    frame = synthetic_series(norad_id=20000 + seed, seed=seed, n=150, noise_scale=3.0)
    assert detect(frame).empty


def test_no_false_alarm_when_node_wraps_through_360() -> None:
    # A fast nodal regression that wraps past 360°: unwrapping must keep it from reading as a jump.
    frame = synthetic_series(norad_id=25544, seed=3, n=150, cadence_days=1.0)
    # Force a large node rate by overwriting raan with a steep, wrapping ramp plus noise.
    rng = np.random.default_rng(99)
    days = np.arange(len(frame), dtype=float)
    frame["raan"] = (30.0 + 50.0 * days + rng.normal(0.0, _NOISE["raan"], len(frame))) % 360.0
    assert detect(frame).empty


# --------------------------------------------------------------------------- floor gate (D5)
def test_below_floor_burn_reports_no_delta_v() -> None:
    # A 0.006 m/s cross-track burn (below the 0.01 m/s LEO floor) on a near-noise-free,
    # confound-free series: the step is still a clear detection, but no Δv is reported below the
    # floor (D5).
    burn = Burn(60, "cross_track_ms", 0.006)
    frame = synthetic_series(
        norad_id=25544, seed=8, burns=(burn,), noise_scale=0.005, confound_scale=0.0
    )
    out = detect(frame)
    assert len(out) == 1
    assert bool(out["delta_v_estimate"].isna().iloc[0])


def test_radial_dominant_detection_is_low_confidence() -> None:
    # The radial down-weighting (D5) applies to a radial-dominated inversion.
    detector = ClassicalDetector(radial_confidence_factor=0.6)
    radial = Inversion(
        delta_v_ms=1.0,
        radial_ms=0.9,
        in_track_ms=0.1,
        cross_track_ms=0.1,
        maneuver_type=ManeuverType.RADIAL,
    )
    in_track = Inversion(
        delta_v_ms=1.0,
        radial_ms=0.1,
        in_track_ms=0.9,
        cross_track_ms=0.1,
        maneuver_type=ManeuverType.IN_TRACK,
    )
    assert detector._confidence(20.0, radial) == pytest.approx(
        0.6 * detector._confidence(20.0, in_track)
    )


# --------------------------------------------------------------------------- multi-object
def test_multi_object_history_detected_per_object() -> None:
    first = synthetic_series(norad_id=25544, seed=0, burns=(Burn(60, "in_track_ms", 2.5),))
    second = synthetic_series(norad_id=39634, seed=1, burns=(Burn(70, "cross_track_ms", 3.0),))
    out = detect(pd.concat([first, second], ignore_index=True))
    assert sorted(out["norad_id"].unique().tolist()) == [25544, 39634]
    assert len(out) == 2
    # Each detection carries its own object's id and bracketing gap.
    by_id = {int(row["norad_id"]): row for _, row in out.iterrows()}
    assert by_id[25544]["type"] == ManeuverType.IN_TRACK.value
    assert by_id[39634]["type"] == ManeuverType.CROSS_TRACK.value


# --------------------------------------------------------------------------- population P/R
def test_population_precision_recall_through_benchmark() -> None:
    """Score a synthetic population through the real benchmark matching + metric layers.

    A mix of maneuvering and maneuver-free LEO objects is run through the detector and scored
    exactly as the published benchmark scores predictions (D4/D7): the matching rule assigns
    detections to labelled gaps one-to-one, and the metric computes recall over the above-floor
    population and precision. The detector must clear a literature-reasonable bar at a low
    false-alarm budget.
    """
    burn_plan = [
        (
            Burn(25, "in_track_ms", 1.2),
            Burn(60, "cross_track_ms", 2.5),
            Burn(95, "in_track_ms", 3.0),
        ),
        (Burn(30, "in_track_ms", 0.8), Burn(75, "cross_track_ms", 2.0)),
        (Burn(40, "cross_track_ms", 3.0), Burn(90, "in_track_ms", 2.0)),
        (
            Burn(35, "in_track_ms", 1.5),
            Burn(70, "in_track_ms", 2.5),
            Burn(100, "cross_track_ms", 3.0),
        ),
        (Burn(50, "cross_track_ms", 2.5),),
        (Burn(45, "in_track_ms", 2.0), Burn(95, "in_track_ms", 1.0)),
    ]

    detections = []
    scored_labels: list[ScoredLabel] = []
    exposure: list[ObjectExposure] = []

    for index, burns in enumerate(burn_plan):
        norad_id = 30000 + index
        frame = synthetic_series(norad_id=norad_id, seed=200 + index, n=130, burns=burns)
        detections.extend(from_frame(detect(frame)))

        labels = [
            ManeuverLabel(
                norad_id=norad_id,
                epoch=_gap_midpoint(frame, burn.gap_index),
                window_start=list(frame["epoch"])[burn.gap_index - 1],
                window_end=list(frame["epoch"])[burn.gap_index],
                source="SYNTHETIC",
                source_ref=f"{norad_id}-{burn.gap_index}",
                orbit_class=OrbitClass.LEO,
                maneuver_type=burn.maneuver_type,
                delta_v=burn.delta_v_ms,
            )
            for burn in burns
        ]
        intervals = label_series(frame, labels).intervals
        scored_labels.extend(
            ScoredLabel(interval=interval, above_floor=True) for interval in intervals
        )
        exposure.append(_exposure_of(frame, OrbitClass.LEO))

    # Maneuver-free objects: they contribute satellite-years and must not generate detections.
    for index in range(4):
        norad_id = 40000 + index
        frame = synthetic_series(norad_id=norad_id, seed=300 + index, n=130)
        detections.extend(from_frame(detect(frame)))
        exposure.append(_exposure_of(frame, OrbitClass.LEO))

    matching = match_detections(detections, scored_labels)
    metrics = class_metrics(matching, exposure)[OrbitClass.LEO]

    assert metrics.recall is not None and metrics.precision is not None
    assert metrics.recall >= 0.85
    assert metrics.precision >= 0.9


def _exposure_of(frame: pd.DataFrame, orbit_class: OrbitClass) -> ObjectExposure:
    epochs = list(frame["epoch"])
    span_years = (epochs[-1] - epochs[0]).total_seconds() / (365.25 * _SECONDS_PER_DAY)
    return ObjectExposure(
        norad_id=int(frame["norad_id"].iloc[0]),
        orbit_class=orbit_class,
        observation_years=span_years,
    )


# --------------------------------------------------------------------------- constructor validation
@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 1},
        {"threshold": 0.0},
        {"threshold": -1.0},
        {"smoothing_level": 1.5},
        {"smoothing_trend": -0.1},
        {"radial_confidence_factor": 2.0},
    ],
)
def test_constructor_rejects_bad_parameters(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ClassicalDetector(**kwargs)
