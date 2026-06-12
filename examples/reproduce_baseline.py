"""Reproduce the classical baseline through the benchmark scorer, fully offline.

The real benchmark numbers come from the labelled satellites, whose multi-year histories are
reconstructed from Space-Track and so cannot run in a self-contained script. This example instead
demonstrates the exact same pipeline on a *synthetic* labelled population: a realistic mean-element
background (J2 nodal regression, drag decay, periodic perturbations, and TLE noise) into which
maneuvers of a known delta-v, type, and epoch are injected through the forward Gauss model the
detector inverts. The classical detector runs over each series, and the detections are scored
through the same matching rule and metric the published benchmark uses.

Run it with no arguments; it needs no credentials or network:

    python examples/reproduce_baseline.py

The output stays ASCII (the delta-v column prints as ``delta_v_estimate``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maneuver_detect import detect
from maneuver_detect.benchmark import (
    ObjectExposure,
    ScoredLabel,
    class_metrics,
    match_detections,
)
from maneuver_detect.labels.labeller import label_series
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.physics import EARTH_MU_KM3_S2, Orbit, gauss_forward, j2_secular_rates
from maneuver_detect.schema import ManeuverType, from_frame

_DEG = math.pi / 180.0
_SECONDS_PER_DAY = 86400.0

# A representative LEO altimetry reference orbit for the synthetic background.
_LEO = Orbit(semi_major_axis_km=6778.0, eccentricity=0.001, inclination_rad=66.0 * _DEG)

# Per-element 1-sigma TLE noise, in each element's native units.
_NOISE = {
    "semi_major_axis": 0.006,  # km
    "eccentricity": 1.0e-5,
    "inclination": 3.0e-4,  # deg
    "raan": 3.0e-4,  # deg
    "arg_perigee": 1.0e-2,  # deg
}


@dataclass(frozen=True)
class Burn:
    """One injected maneuver: the gap it brackets, its RSW component, and its magnitude."""

    gap_index: int  # index of the first elset after the burn; brackets [gap_index - 1, gap_index]
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


# Each tuple is one satellite's maneuver schedule; the empty tuples are maneuver-free objects that
# contribute satellite-years of exposure (and so a false-alarm budget) without any labels.
_POPULATION: tuple[tuple[Burn, ...], ...] = (
    (Burn(25, "in_track_ms", 1.2), Burn(60, "cross_track_ms", 2.5), Burn(95, "in_track_ms", 3.0)),
    (Burn(30, "in_track_ms", 0.8), Burn(75, "cross_track_ms", 2.0)),
    (Burn(40, "cross_track_ms", 3.0), Burn(90, "in_track_ms", 2.0)),
    (Burn(45, "in_track_ms", 2.0), Burn(95, "in_track_ms", 1.0)),
    (),
    (),
)


def _mean_motion_rev_per_day(a_km: float) -> float:
    n_rad_s = math.sqrt(EARTH_MU_KM3_S2 / a_km**3)
    return n_rad_s * _SECONDS_PER_DAY / (2.0 * math.pi)


def synthetic_series(
    norad_id: int, seed: int, burns: tuple[Burn, ...], n: int = 130
) -> pd.DataFrame:
    """Build a synthetic mean-element series with realistic confounds and the injected burns."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    epochs = [t0 + pd.Timedelta(days=float(i)) for i in range(n)]
    days = np.arange(n, dtype=float)

    raan_dot, argp_dot, _ = j2_secular_rates(_LEO)
    deg_per_day = _SECONDS_PER_DAY / _DEG

    a = _LEO.semi_major_axis_km - 2.0e-3 * days  # mild drag decay (km/day)
    e = _LEO.eccentricity + 1.0e-4 * np.sin(0.30 * days)  # SRP / luni-solar wobble
    inc = _LEO.inclination_rad / _DEG + 5.0e-3 * np.sin(0.20 * days)
    raan = 30.0 + raan_dot * deg_per_day * days
    argp = 90.0 + argp_dot * deg_per_day * days

    a += rng.normal(0.0, _NOISE["semi_major_axis"], n)
    e += rng.normal(0.0, _NOISE["eccentricity"], n)
    inc += rng.normal(0.0, _NOISE["inclination"], n)
    raan += rng.normal(0.0, _NOISE["raan"], n)
    argp += rng.normal(0.0, _NOISE["arg_perigee"], n)

    for burn in burns:
        kwargs = {"radial_ms": 0.0, "in_track_ms": 0.0, "cross_track_ms": 0.0}
        kwargs[burn.component] = burn.delta_v_ms
        step = gauss_forward(orbit=_LEO, true_anomaly_rad=burn.true_anomaly_rad, **kwargs)
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


def _exposure_of(frame: pd.DataFrame) -> ObjectExposure:
    epochs = list(frame["epoch"])
    span_years = (epochs[-1] - epochs[0]).total_seconds() / (365.25 * _SECONDS_PER_DAY)
    return ObjectExposure(
        norad_id=int(frame["norad_id"].iloc[0]),
        orbit_class=OrbitClass.LEO,
        observation_years=span_years,
    )


def main() -> int:
    detections = []
    scored_labels: list[ScoredLabel] = []
    exposure: list[ObjectExposure] = []

    for index, burns in enumerate(_POPULATION):
        norad_id = 30000 + index
        frame = synthetic_series(norad_id=norad_id, seed=200 + index, burns=burns)
        detections.extend(from_frame(detect(frame)))
        exposure.append(_exposure_of(frame))

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

    matching = match_detections(detections, scored_labels)
    metrics = class_metrics(matching, exposure)[OrbitClass.LEO]

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    n_objects = len(_POPULATION)
    n_labels = len(scored_labels)
    print(f"Scored {n_objects} synthetic LEO objects ({n_labels} above-floor maneuver labels).")
    print(f"  operating point : {metrics.operating_point:g} false-alarm(s)/satellite-year")
    print(f"  recall          : {fmt(metrics.recall)}")
    print(f"  precision       : {fmt(metrics.precision)}")
    print(f"  detections      : {len(detections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
