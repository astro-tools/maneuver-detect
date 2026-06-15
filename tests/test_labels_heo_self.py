"""Tests for ``maneuver_detect.labels.heo_self`` — the self-labelled HEO deriver.

Synthetic HEO series exercise the two channels: a semi-major-axis (energy) step → in-track, and an
eccentricity (shape) step → radial, plus the quiet-series, determinism, and short-series guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maneuver_detect.labels.heo_self import derive_heo_labels
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import ManeuverType


def _series(semi_major_axis: np.ndarray, eccentricity: np.ndarray) -> pd.DataFrame:
    n = len(semi_major_axis)
    epochs = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "epoch": epochs,
            "semi_major_axis": semi_major_axis,
            "eccentricity": eccentricity,
        }
    )


def test_detects_semi_major_axis_step() -> None:
    n = 40
    sma = np.full(n, 65000.0)
    sma[n // 2 :] += 5.0  # a 5 km energy step (apogee/perigee maintenance) — well above the floor
    (label,) = derive_heo_labels(_series(sma, np.full(n, 0.7)))
    assert label.maneuver_type is ManeuverType.IN_TRACK
    assert label.orbit_class is OrbitClass.HEO
    assert label.delta_v is None  # self-labelled, epoch-only


def test_detects_eccentricity_step() -> None:
    n = 40
    ecc = np.full(n, 0.7)
    ecc[n // 2 :] += 0.001  # a shape change with no energy change -> radial
    labels = derive_heo_labels(_series(np.full(n, 65000.0), ecc))
    assert len(labels) == 1
    assert labels[0].maneuver_type is ManeuverType.RADIAL


def test_quiet_series_yields_no_labels() -> None:
    n = 40
    rng = np.random.default_rng(0)
    sma = 65000.0 + rng.normal(0.0, 0.01, n)  # sub-floor element noise only
    ecc = 0.7 + rng.normal(0.0, 1e-7, n)
    assert derive_heo_labels(_series(sma, ecc)) == []


def test_is_deterministic() -> None:
    n = 40
    sma = np.full(n, 65000.0)
    sma[n // 2 :] += 5.0
    series = _series(sma, np.full(n, 0.7))
    assert derive_heo_labels(series) == derive_heo_labels(series)


def test_short_series_yields_no_labels() -> None:
    assert derive_heo_labels(_series(np.full(2, 65000.0), np.full(2, 0.7))) == []
