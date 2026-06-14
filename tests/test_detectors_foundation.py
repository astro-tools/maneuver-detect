"""Tests for the foundation forecast-residual detectors — the forecaster-agnostic pipeline.

These exercise the whole detector mechanics (residual scoring, per-class thresholding, NMS,
canonical emission, dispatch, gating) with a deterministic **stand-in** forecaster, so they run in
the default suite without the optional ``[foundation]`` extra. The real Chronos / TimesFM backends
are covered by the ``foundation``-marked tests, which need the extra installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _synthetic import Burn, synthetic_series
from maneuver_detect.detectors import available_models, get_detector
from maneuver_detect.detectors.foundation import (
    ChronosResidualDetector,
    DriftContinuationForecaster,
    Forecast,
    TimesFmResidualDetector,
)
from maneuver_detect.models.foundation import FoundationBundle
from maneuver_detect.schema import COLUMNS, validate_frame

_THRESHOLDS = {"LEO": 6.0, "MEO": 6.0, "GEO": 6.0}


def _stand_in_detector(threshold: float | None = None) -> ChronosResidualDetector:
    """A Chronos detector backed by the deterministic drift-continuation stand-in forecaster."""
    return ChronosResidualDetector(
        forecaster=DriftContinuationForecaster(),
        class_thresholds=_THRESHOLDS,
        threshold=threshold,
    )


def test_both_foundation_detectors_are_registered() -> None:
    assert {"chronos-residual", "timesfm-residual"} <= set(available_models())
    assert isinstance(get_detector("chronos-residual"), ChronosResidualDetector)
    assert isinstance(get_detector("timesfm-residual"), TimesFmResidualDetector)


def test_detects_an_injected_in_track_burn() -> None:
    frame = synthetic_series(norad_id=1, seed=3, n=120, burns=(Burn(45, "in_track_ms", 4.0),))
    out = _stand_in_detector().detect(frame)

    validate_frame(out)
    assert list(out.columns) == list(COLUMNS)
    assert not out.empty
    # The semi-major-axis step lands as one residual spike on the burn's gap (token 45 brackets the
    # gap [44, 45]); the strongest detection is that gap, within the D4 +/-1 tolerance.
    strongest = out.loc[out["confidence"].idxmax()]
    assert frame["epoch"].iloc[44] <= strongest["elset_epoch_after"] <= frame["epoch"].iloc[46]
    assert 0.0 <= strongest["confidence"] <= 1.0
    assert strongest["delta_v_estimate"] > 0.0  # a 4 m/s in-track burn clears the LEO floor


def test_threshold_gates_detection_count() -> None:
    frame = synthetic_series(norad_id=1, seed=4, n=120, burns=(Burn(60, "in_track_ms", 4.0),))
    low = _stand_in_detector(threshold=2.0).detect(frame)
    high = _stand_in_detector(threshold=50.0).detect(frame)
    assert len(high) <= len(low)


def test_empty_history_returns_empty_canonical_frame() -> None:
    out = _stand_in_detector().detect(synthetic_series(norad_id=1, seed=0).iloc[0:0])
    assert list(out.columns) == list(COLUMNS)
    assert out.empty


def test_too_short_series_yields_no_detections() -> None:
    out = _stand_in_detector().detect(synthetic_series(norad_id=1, seed=0, n=1))
    assert out.empty


def test_multi_object_history_is_grouped_and_sorted() -> None:
    frame = pd.concat(
        [
            synthetic_series(norad_id=7, seed=5, n=120, burns=(Burn(45, "in_track_ms", 4.0),)),
            synthetic_series(norad_id=3, seed=6, n=120, burns=(Burn(50, "cross_track_ms", 4.0),)),
        ],
        ignore_index=True,
    )
    out = _stand_in_detector().detect(frame)
    validate_frame(out)
    ordered = out.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(out, ordered)
    assert set(out["norad_id"]).issubset({3, 7})


def test_unconfigured_detector_falls_back_to_hub_and_surfaces_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no forecaster, no bundle, and no env-var path, detect() pulls the Hub-published bundle on
    # first use; a download failure surfaces as a clear HubError (not a raw transport error). This
    # path fails at the download, before any chronos import, so it runs without the extra.
    from maneuver_detect.detectors.foundation import CHRONOS_CHECKPOINT_ENV
    from maneuver_detect.hub import HubError

    def _offline(**kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.delenv(CHRONOS_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _offline)
    detector = ChronosResidualDetector()
    assert detector.is_loaded is False
    with pytest.raises(HubError, match="could not fetch"):
        detector.detect(synthetic_series(norad_id=1, seed=0))


def test_backend_mismatch_is_rejected() -> None:
    # A bundle for the other backend must not load into this detector — the backend check fires
    # before any forecaster is built, so this needs neither chronos nor timesfm.
    timesfm_bundle = FoundationBundle(
        backend="timesfm",
        checkpoint_id="google/timesfm-2.5-200m-pytorch",
        revision="main",
        context_length=64,
        class_thresholds=_THRESHOLDS,
    )
    with pytest.raises(ValueError, match="expects a 'chronos' bundle"):
        ChronosResidualDetector(timesfm_bundle)


def test_unconfigured_detector_without_hub_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicitly forecaster-less, non-Hub detector (the env path set but empty) cannot detect.
    detector = _stand_in_detector()
    detector._forecaster = None  # simulate a detector that resolved nothing
    detector._hub_pending = False
    with pytest.raises(ValueError, match="needs a calibrated bundle"):
        detector.detect(synthetic_series(norad_id=1, seed=0))


def test_rolling_contexts_grow_and_are_capped() -> None:
    from maneuver_detect.detectors.foundation import _rolling_contexts

    series = np.arange(6, dtype=np.float64)
    contexts = _rolling_contexts(series, context_length=3)
    assert len(contexts) == 5  # one context per token from index 1
    assert contexts[0].tolist() == [0.0]  # token 1 sees only token 0
    assert contexts[-1].tolist() == [2.0, 3.0, 4.0]  # token 5 is capped to the last 3 before it


def test_build_forecaster_dispatch_and_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import maneuver_detect.detectors.foundation as foundation
    from maneuver_detect.models.foundation import (
        FoundationBundle,
        build_forecaster_for,
        zero_shot_bundle,
    )

    sentinel = DriftContinuationForecaster()
    monkeypatch.setattr(foundation, "_build_chronos_forecaster", lambda bundle: sentinel)
    monkeypatch.setattr(foundation, "_build_timesfm_forecaster", lambda bundle: sentinel)

    assert foundation.build_forecaster(zero_shot_bundle("chronos")) is sentinel
    assert foundation.build_forecaster(zero_shot_bundle("timesfm")) is sentinel
    # The driver's thin re-export resolves through the same dispatch.
    assert build_forecaster_for(zero_shot_bundle("chronos")) is sentinel

    bogus = FoundationBundle(
        backend="nope", checkpoint_id="x", revision="main", context_length=8, class_thresholds={}
    )
    with pytest.raises(ValueError, match="unknown foundation backend"):
        foundation.build_forecaster(bogus)


def test_loads_bundle_thresholds_and_context(monkeypatch: pytest.MonkeyPatch) -> None:
    # The bundle-load path (backend match, forecaster built, thresholds + context adopted) with the
    # real forecaster build stubbed, so it runs without the extra.
    import maneuver_detect.detectors.foundation as foundation
    from maneuver_detect.models.foundation import zero_shot_bundle

    monkeypatch.setattr(
        foundation, "build_forecaster", lambda bundle: DriftContinuationForecaster()
    )
    bundle = zero_shot_bundle("chronos", class_thresholds={"LEO": 5.0, "MEO": 5.0, "GEO": 5.0})
    detector = ChronosResidualDetector(bundle)
    assert detector.is_loaded
    assert detector._class_thresholds == {"LEO": 5.0, "MEO": 5.0, "GEO": 5.0}
    out = detector.detect(
        synthetic_series(norad_id=1, seed=3, n=120, burns=(Burn(45, "in_track_ms", 4.0),))
    )
    validate_frame(out)


def test_hub_load_success_with_stubbed_forecaster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The no-argument detector resolves its bundle from the Hub on first detect(); with the download
    # pointed at a local file and the forecaster build stubbed, the whole resolve-then-detect path
    # runs without the extra.
    import maneuver_detect.detectors.foundation as foundation
    from maneuver_detect import hub
    from maneuver_detect.detectors.foundation import CHRONOS_CHECKPOINT_ENV
    from maneuver_detect.models.foundation import save_foundation_bundle, zero_shot_bundle

    bundle = zero_shot_bundle("chronos", class_thresholds={"LEO": 6.0, "MEO": 6.0, "GEO": 6.0})
    path = tmp_path / "chronos-residual.pt"
    save_foundation_bundle(bundle, path)
    monkeypatch.delenv(CHRONOS_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(hub, "checkpoint_path", lambda name, **kw: path)
    monkeypatch.setattr(foundation, "build_forecaster", lambda b: DriftContinuationForecaster())

    detector = ChronosResidualDetector()
    out = detector.detect(
        synthetic_series(norad_id=1, seed=3, n=120, burns=(Burn(45, "in_track_ms", 4.0),))
    )
    validate_frame(out)
    assert detector.is_loaded


def test_drift_continuation_forecaster_shapes_and_quiet_residual() -> None:
    # The stand-in's contract: a forecast value + scale per token, token 0 unused, a quiet series
    # standardised to small residuals.
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0.0, 1.0, 50)).astype(np.float64)
    forecast = DriftContinuationForecaster().forecast(series)
    assert isinstance(forecast, Forecast)
    assert forecast.mean.shape == series.shape == forecast.scale.shape
    assert np.isnan(forecast.mean[0])
    z = np.abs(series[1:] - forecast.mean[1:]) / forecast.scale[1:]
    assert np.median(z) < 2.0  # a quiet random walk sits well below a maneuver-scale residual
