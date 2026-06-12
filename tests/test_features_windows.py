"""Tests for sliding-window extraction.

The windows must tile the series so the last token is always covered, zero-pad and invalidate any
overrun, window an optional per-token target in lockstep with the features, and never let the target
touch the features (the leak-free boundary).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from maneuver_detect.features import (
    ClassNormaliser,
    FeatureMatrix,
    build_channels,
    make_windows,
)
from maneuver_detect.features.channels import N_CHANNELS
from maneuver_detect.labels.record import OrbitClass


def _matrix(*, n: int = 200, norad_id: int = 7) -> FeatureMatrix:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    epochs = [start + timedelta(days=i) for i in range(n)]
    frame = pd.DataFrame(
        {
            "epoch": pd.Series(epochs, dtype="datetime64[ns, UTC]"),
            "norad_id": np.full(n, norad_id, dtype=int),
            "semi_major_axis": 6778.0 - 0.002 * np.arange(n),
            "eccentricity": np.full(n, 0.001),
            "inclination": np.full(n, 66.0),
            "raan": (20.0 + 5.0 * np.arange(n)) % 360.0,
            "arg_perigee": np.full(n, 90.0),
        }
    )
    rc = build_channels(frame)
    return ClassNormaliser.fit([rc]).transform(rc)


def _empty_matrix() -> FeatureMatrix:
    return FeatureMatrix(
        norad_id=7,
        orbit_class=OrbitClass.LEO,
        epochs=np.array([], dtype="datetime64[ns]"),
        features=np.zeros((0, N_CHANNELS), dtype=np.float32),
        validity=np.zeros(0, dtype=np.bool_),
    )


def test_window_shapes_and_count() -> None:
    wt = make_windows(_matrix(n=200), window=64, stride=32)
    expected = len(range(0, 200, 32))
    assert wt.features.shape == (expected, 64, N_CHANNELS)
    assert wt.validity.shape == (expected, 64)
    assert wt.n_windows == expected
    assert wt.target is None


def test_last_token_is_covered_with_valid_context() -> None:
    matrix = _matrix(n=200)
    wt = make_windows(matrix, window=64, stride=32)
    last_start = max(range(0, 200, 32))
    real = 200 - last_start
    assert bool(wt.validity[-1, real - 1])  # token 199 is a real, valid position
    assert np.array_equal(wt.features[-1, real - 1], matrix.features[199])


def test_overrun_is_zero_padded_and_invalid() -> None:
    wt = make_windows(_matrix(n=200), window=64, stride=32)
    last_start = max(range(0, 200, 32))
    real = 200 - last_start
    assert not wt.validity[-1, real:].any()
    assert np.all(wt.features[-1, real:] == 0.0)


def test_target_is_windowed_in_lockstep() -> None:
    matrix = _matrix(n=200)
    target = np.arange(200, dtype=float)
    wt = make_windows(matrix, window=64, stride=32, target=target)
    assert wt.target is not None
    # first window: positions map straight onto target[0:64]
    assert np.array_equal(wt.target[0], target[0:64].astype(np.float32))
    # last window: real positions carry the target, padding is zero
    last_start = max(range(0, 200, 32))
    real = 200 - last_start
    assert np.array_equal(wt.target[-1, :real], target[last_start:].astype(np.float32))
    assert np.all(wt.target[-1, real:] == 0.0)


def test_short_series_yields_a_single_padded_window() -> None:
    wt = make_windows(_matrix(n=20), window=64, stride=32)
    assert wt.n_windows == 1
    assert int(wt.validity.sum()) == 20


def test_empty_matrix_yields_no_windows() -> None:
    wt = make_windows(_empty_matrix(), window=64, stride=32)
    assert wt.n_windows == 0
    assert wt.features.shape == (0, 64, N_CHANNELS)


def test_target_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="does not match"):
        make_windows(_matrix(n=50), target=np.zeros(49))


def test_invalid_window_and_stride_raise() -> None:
    matrix = _matrix(n=50)
    with pytest.raises(ValueError, match="window must be at least 1"):
        make_windows(matrix, window=0)
    with pytest.raises(ValueError, match="stride must be in"):
        make_windows(matrix, window=64, stride=0)
    with pytest.raises(ValueError, match="stride must be in"):
        make_windows(matrix, window=64, stride=65)
