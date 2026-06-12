"""Tests for ``maneuver_detect.labels.longitude_shift`` — the self-labelled GEO derivation.

Synthetic GEO series exercise the two channels: an east-west drift-rate reversal (in-track) and a
downward inclination step (cross-track), plus the quiet-series and determinism guarantees.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from maneuver_detect.labels.longitude_shift import _N_SYNC_REV_PER_DAY, derive_geo_labels
from maneuver_detect.labels.record import SOURCE_SELF_GEO, OrbitClass
from maneuver_detect.schema import ManeuverType

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _series(
    mean_motion: np.ndarray, inclination: np.ndarray, *, norad_id: int = 41866
) -> pd.DataFrame:
    n = len(mean_motion)
    return pd.DataFrame(
        {
            "epoch": pd.to_datetime([_T0 + timedelta(days=i) for i in range(n)], utc=True),
            "norad_id": norad_id,
            "mean_motion": mean_motion,
            "inclination": inclination,
        }
    )


def test_detects_east_west_drift_reversal() -> None:
    n = 12
    mean_motion = np.full(n, _N_SYNC_REV_PER_DAY)
    mean_motion[:6] += 0.02 / 360.0  # one drift direction
    mean_motion[6:] -= 0.02 / 360.0  # reversed at the gap-5 vertex
    (label,) = derive_geo_labels(_series(mean_motion, np.full(n, 0.05)))
    assert label.source == SOURCE_SELF_GEO
    assert label.orbit_class is OrbitClass.GEO
    assert label.maneuver_type is ManeuverType.IN_TRACK
    assert label.delta_v is None
    assert label.norad_id == 41866


def test_detects_north_south_inclination_step() -> None:
    n = 12
    inclination = 0.05 + 0.0002 * np.arange(n)  # secular rise
    inclination[6:] -= 0.04  # a downward N-S burn at gap 5
    labels = derive_geo_labels(_series(np.full(n, _N_SYNC_REV_PER_DAY), inclination))
    assert [label.maneuver_type for label in labels] == [ManeuverType.CROSS_TRACK]


def test_quiet_series_yields_no_labels() -> None:
    n = 20
    rng = np.random.default_rng(0)
    mean_motion = _N_SYNC_REV_PER_DAY + rng.normal(0, 1e-8, n)  # noise well below the floor
    inclination = np.full(n, 0.05) + rng.normal(0, 1e-5, n)
    assert derive_geo_labels(_series(mean_motion, inclination)) == []


def test_is_deterministic() -> None:
    n = 12
    mean_motion = np.full(n, _N_SYNC_REV_PER_DAY)
    mean_motion[:6] += 0.02 / 360.0
    mean_motion[6:] -= 0.02 / 360.0
    series = _series(mean_motion, np.full(n, 0.05))
    first = derive_geo_labels(series)
    second = derive_geo_labels(series)
    assert [label.epoch for label in first] == [label.epoch for label in second]


def test_short_series_yields_no_labels() -> None:
    assert derive_geo_labels(_series(np.full(2, _N_SYNC_REV_PER_DAY), np.full(2, 0.05))) == []
