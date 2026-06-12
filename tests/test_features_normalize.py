"""Tests for the per-class robust normaliser.

The normaliser standardises the element channels per orbit class on the statistics of the series it
is fit on — the train-split-only contract the caller honours. The checks: the pooled training data
standardises to median 0 / IQR 1 by construction, the bounded timing and mask columns pass through
untouched, a class the normaliser never saw is refused, and the fitted statistics round-trip through
serialisation.
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
    encode_history,
)
from maneuver_detect.features.channels import N_CHANNELS, N_ELEMENT_CHANNELS, RawChannels
from maneuver_detect.labels.record import OrbitClass


def _frame(*, norad_id: int, a0: float, n: int = 100, seed: int = 0) -> pd.DataFrame:
    """A synthetic single-object series with mild noise (so the IQR is non-degenerate)."""
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    epochs = [start + timedelta(days=i) for i in range(n)]
    a = a0 - 0.002 * np.arange(n) + rng.normal(0.0, 0.01, n)
    return pd.DataFrame(
        {
            "epoch": pd.Series(epochs, dtype="datetime64[ns, UTC]"),
            "norad_id": np.full(n, norad_id, dtype=int),
            "semi_major_axis": a,
            "eccentricity": np.full(n, 0.001) + rng.normal(0.0, 1e-5, n),
            "inclination": np.full(n, 66.0) + rng.normal(0.0, 3e-4, n),
            "raan": (20.0 + 5.0 * np.arange(n) + rng.normal(0.0, 3e-4, n)) % 360.0,
            "arg_perigee": np.full(n, 90.0) + rng.normal(0.0, 1e-2, n),
        }
    )


def _leo_geo_history() -> pd.DataFrame:
    return pd.concat(
        [
            _frame(norad_id=1, a0=6778.0, seed=1),
            _frame(norad_id=2, a0=6900.0, seed=2),
            _frame(norad_id=3, a0=42164.0, seed=3),
        ],
        ignore_index=True,
    )


def test_fit_covers_the_classes_present() -> None:
    norm = ClassNormaliser.fit(encode_history(_leo_geo_history()))
    assert norm.classes == frozenset({OrbitClass.LEO, OrbitClass.GEO})


def test_pooled_training_data_standardises_to_median0_iqr1() -> None:
    channels = [build_channels(_frame(norad_id=i, a0=6778.0, seed=i)) for i in range(1, 4)]
    norm = ClassNormaliser.fit(channels)
    pooled = np.vstack([norm.transform(c).features[:, :N_ELEMENT_CHANNELS] for c in channels])
    q25, q50, q75 = np.percentile(pooled, [25.0, 50.0, 75.0], axis=0)
    assert np.allclose(q50, 0.0, atol=1e-5)
    assert np.allclose(q75 - q25, 1.0, atol=1e-5)


def test_timing_and_mask_columns_pass_through_unchanged() -> None:
    rc = build_channels(_frame(norad_id=1, a0=6778.0))
    norm = ClassNormaliser.fit([rc])
    fm = norm.transform(rc)
    raw_tail = rc.matrix[:, N_ELEMENT_CHANNELS:].astype(np.float32)
    assert np.array_equal(fm.features[:, N_ELEMENT_CHANNELS:], raw_tail)


def test_transform_yields_float32_and_all_valid() -> None:
    rc = build_channels(_frame(norad_id=1, a0=6778.0))
    fm = ClassNormaliser.fit([rc]).transform(rc)
    assert isinstance(fm, FeatureMatrix)
    assert fm.features.dtype == np.float32
    assert bool(fm.validity.all())
    assert fm.validity.shape == (rc.n_tokens,)


def test_unfit_class_is_refused() -> None:
    leo = build_channels(_frame(norad_id=1, a0=6778.0))
    geo = build_channels(_frame(norad_id=2, a0=42164.0))
    norm = ClassNormaliser.fit([leo])
    with pytest.raises(ValueError, match="no statistics for class GEO"):
        norm.transform(geo)


def test_statistics_round_trip_through_serialisation() -> None:
    channels = encode_history(_leo_geo_history())
    norm = ClassNormaliser.fit(channels)
    restored = ClassNormaliser.from_dict(norm.to_dict())
    for rc in channels:
        assert np.array_equal(norm.transform(rc).features, restored.transform(rc).features)


def test_fit_on_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty set of channels"):
        ClassNormaliser.fit([])


def test_zero_token_channels_are_skipped_in_fit() -> None:
    empty = RawChannels(
        norad_id=9,
        orbit_class=OrbitClass.LEO,
        epochs=np.array([], dtype="datetime64[ns]"),
        matrix=np.zeros((0, N_CHANNELS)),
    )
    real = build_channels(_frame(norad_id=1, a0=6778.0))
    norm = ClassNormaliser.fit([empty, real])
    assert norm.classes == frozenset({OrbitClass.LEO})
