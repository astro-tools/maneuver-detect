"""Synthetic labelled element series shared by the learned-baseline tests.

The learned baselines cannot be exercised against the real labelled satellites in CI (their
multi-year series are reconstructed from Space-Track, which is not redistributable and needs live
credentials), so the harness and detector mechanics are checked on *synthetic* mean-element series
with known maneuver gaps — the same forward-Gauss construction the classical-detector tests use,
exposed here as :class:`Burn` / :func:`synthetic_series` plus the :class:`ObjectSeries` adapter the
data module consumes. These tests assert mechanics (tensor shapes, target alignment, determinism,
checkpoint round-trip, schema validity), not detection accuracy — the literature-level numbers come
from the offline credentialed run, not from CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maneuver_detect.models.datamodule import ObjectSeries
from maneuver_detect.physics import EARTH_MU_KM3_S2, Orbit, gauss_forward, j2_secular_rates
from maneuver_detect.schema import ManeuverType

_DEG = math.pi / 180.0
_SECONDS_PER_DAY = 86400.0

_LEO = Orbit(semi_major_axis_km=6778.0, eccentricity=0.001, inclination_rad=66.0 * _DEG)

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
    reference: Orbit = _LEO,
    burns: tuple[Burn, ...] = (),
) -> pd.DataFrame:
    """A synthetic mean-element series with J2 drift, drag/SRP confounds, TLE noise, and the burns.

    The frame carries the canonical
    :data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS`; each :class:`Burn` adds its
    forward-Gauss element step to every sample at or after its gap.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    epochs = [t0 + pd.Timedelta(days=float(i)) for i in range(n)]
    days = np.arange(n, dtype=float)

    raan_dot, argp_dot, _ = j2_secular_rates(reference)
    deg_per_day = _SECONDS_PER_DAY / _DEG

    a = reference.semi_major_axis_km - 2.0e-3 * days
    e = reference.eccentricity + 1.0e-4 * np.sin(0.30 * days)
    inc = reference.inclination_rad / _DEG + 5.0e-3 * np.sin(0.20 * days)
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


def maneuver_epochs(frame: pd.DataFrame, burns: tuple[Burn, ...]) -> tuple[pd.Timestamp, ...]:
    """The ``elset_epoch_after`` (token epoch the target attaches to) of each burn's gap."""
    epochs = list(frame["epoch"])
    return tuple(pd.Timestamp(epochs[burn.gap_index]) for burn in burns)


def object_series(
    *, norad_id: int, seed: int, burns: tuple[Burn, ...] = (), n: int = 120
) -> ObjectSeries:
    """A synthetic :class:`ObjectSeries` (series + its maneuver-gap epochs) for the data module."""
    frame = synthetic_series(norad_id=norad_id, seed=seed, burns=burns, n=n)
    return ObjectSeries(
        norad_id=norad_id, series=frame, maneuver_epochs=maneuver_epochs(frame, burns)
    )
