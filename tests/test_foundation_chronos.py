"""Real-Chronos forecast-residual checks — the ``[foundation]`` extra path (CPU, downloads weights).

Marked ``foundation`` and ``importorskip``ped on ``chronos``, so they run only in the dedicated CI
job that installs the extra (and elsewhere only if it happens to be installed). They download the
smallest Chronos checkpoint from the Hub and exercise the real forecast-residual pipeline end to end
on a synthetic series — mechanics (shapes, schema, dispatch, a fine-tune step), not detection
accuracy (the literature-level numbers come from the offline credentialed run).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("chronos")
pytestmark = pytest.mark.foundation

import maneuver_detect  # noqa: E402
from _synthetic import Burn, synthetic_series  # noqa: E402
from maneuver_detect.detectors.foundation import (  # noqa: E402
    CHRONOS_CHECKPOINT_ENV,
    ChronosResidualDetector,
    Forecast,
)
from maneuver_detect.models.foundation import (  # noqa: E402
    FoundationBundle,
    save_foundation_bundle,
    zero_shot_bundle,
)
from maneuver_detect.schema import COLUMNS, validate_frame  # noqa: E402

_CHECKPOINT = "amazon/chronos-bolt-tiny"  # the smallest published Chronos checkpoint
_THRESHOLDS = {"LEO": 4.0, "MEO": 4.0, "GEO": 4.0}


def _bundle(context_length: int = 16) -> FoundationBundle:
    return zero_shot_bundle(
        "chronos",
        checkpoint_id=_CHECKPOINT,
        context_length=context_length,
        class_thresholds=_THRESHOLDS,
    )


@pytest.fixture(scope="module")
def chronos_forecaster() -> object:
    from maneuver_detect.detectors._chronos import ChronosForecaster

    return ChronosForecaster(
        checkpoint_id=_CHECKPOINT, revision="main", context_length=16, finetune_state=None
    )


def test_forecaster_returns_aligned_mean_and_finite_scale(chronos_forecaster: object) -> None:
    series = np.linspace(7000.0, 7001.0, 40, dtype=np.float64)
    forecast = chronos_forecaster.forecast(series)  # type: ignore[attr-defined]
    assert isinstance(forecast, Forecast)
    assert forecast.mean.shape == series.shape == forecast.scale.shape
    assert np.isnan(forecast.mean[0])
    assert np.all(np.isfinite(forecast.mean[1:]))
    assert np.all(forecast.scale[1:] > 0.0)  # the predictive interval gives a positive scale


def test_detect_end_to_end_returns_canonical_schema() -> None:
    frame = synthetic_series(norad_id=1, seed=3, n=60, burns=(Burn(30, "in_track_ms", 5.0),))
    out = ChronosResidualDetector(_bundle()).detect(frame)
    validate_frame(out)
    assert list(out.columns) == list(COLUMNS)
    if not out.empty:
        assert (out["confidence"].between(0.0, 1.0)).all()


def test_dispatch_through_env_var_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "chronos-residual.pt"
    save_foundation_bundle(_bundle(), path)
    monkeypatch.setenv(CHRONOS_CHECKPOINT_ENV, str(path))
    frame = synthetic_series(norad_id=1, seed=5, n=60, burns=(Burn(30, "in_track_ms", 5.0),))
    out = maneuver_detect.detect(frame, model="chronos-residual")
    validate_frame(out)


def test_light_finetune_produces_loadable_state() -> None:
    from maneuver_detect.detectors._chronos import ChronosForecaster, finetune_chronos_model

    frame = synthetic_series(norad_id=1, seed=2, n=160, burns=(Burn(80, "in_track_ms", 5.0),))
    state, cost = finetune_chronos_model(
        checkpoint_id=_CHECKPOINT,
        revision="main",
        context_length=16,
        series=[frame],
        max_steps=1,
    )
    assert cost["steps"] == 1
    assert cost["n_windows"] > 0
    # The fine-tune state loads back onto a forecaster (the inference side of the fine-tuned model).
    forecaster = ChronosForecaster(
        checkpoint_id=_CHECKPOINT, revision="main", context_length=16, finetune_state=state
    )
    forecast = forecaster.forecast(np.linspace(7000.0, 7001.0, 30, dtype=np.float64))
    assert np.all(np.isfinite(forecast.mean[1:]))
