"""TimesFM forecast-residual checks — the second ``[foundation]`` entry (CPU, downloads weights).

Marked ``foundation`` and ``importorskip``ped on ``timesfm``. The predictive-scale derivation is
unit-tested without loading any weights; the end-to-end forecast loads the real (large) TimesFM
checkpoint and is skipped if it cannot be fetched, so a heavy or missing download never fails CI.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("timesfm")
pytestmark = pytest.mark.foundation

from maneuver_detect.detectors._timesfm import TimesFmForecaster  # noqa: E402
from maneuver_detect.detectors.foundation import Forecast  # noqa: E402
from maneuver_detect.models.foundation import FOUNDATION_DEFAULTS  # noqa: E402


def test_predictive_spread_uses_quantiles_then_falls_back_to_mad() -> None:
    # The scale logic is independent of the loaded model, so exercise it on a bare instance.
    forecaster = object.__new__(TimesFmForecaster)
    series = np.arange(6, dtype=np.float64)
    predicted = series[1:] - 0.1

    # Non-degenerate quantiles -> half the outer spread, per forecast.
    quantiles = np.stack([np.full(5, -1.0), np.full(5, 0.0), np.full(5, 2.0)], axis=-1)[:, None, :]
    spread = forecaster._predictive_spread(quantiles, series, predicted)
    assert np.allclose(spread, 1.5)  # (2 - (-1)) / 2

    # Degenerate (zero-width) quantiles -> a single robust residual scale (MAD) for every forecast.
    flat = np.zeros((5, 1, 3))
    fallback = forecaster._predictive_spread(flat, series, predicted)
    assert fallback.shape == predicted.shape
    assert np.all(fallback > 0.0)


def test_real_timesfm_forecaster_end_to_end() -> None:
    checkpoint = FOUNDATION_DEFAULTS["timesfm"].checkpoint_id
    try:
        forecaster = TimesFmForecaster(checkpoint_id=checkpoint, revision="main", context_length=32)
    except Exception as exc:
        pytest.skip(f"could not load TimesFM checkpoint {checkpoint!r}: {exc}")

    series = np.linspace(7000.0, 7001.0, 40, dtype=np.float64)
    forecast = forecaster.forecast(series)
    assert isinstance(forecast, Forecast)
    assert forecast.mean.shape == series.shape == forecast.scale.shape
    assert np.isnan(forecast.mean[0])
    assert np.all(np.isfinite(forecast.mean[1:]))
    assert np.all(forecast.scale[1:] > 0.0)
