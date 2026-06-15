"""Tests for the BiLSTM detector — checkpoint gating, dispatch, grouping, and the gap reducer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import maneuver_detect
from _synthetic import Burn, object_series, synthetic_series
from maneuver_detect.detectors import available_models, get_detector
from maneuver_detect.detectors.bilstm import CHECKPOINT_ENV, BiLstmDetector, _detected_gaps
from maneuver_detect.models.bilstm import BiLstmConfig
from maneuver_detect.models.checkpoint import ModelBundle, save_bundle
from maneuver_detect.models.train import train_bilstm
from maneuver_detect.schema import COLUMNS, validate_frame

_CONFIG = BiLstmConfig(hidden_size=8, num_layers=1, dropout=0.0)


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    objects = [
        object_series(norad_id=1, seed=1, burns=(Burn(40, "in_track_ms", 4.0),), n=90),
        object_series(norad_id=2, seed=2, burns=(Burn(55, "cross_track_ms", 4.0),), n=90),
    ]
    return train_bilstm(
        objects,
        config=_CONFIG,
        max_epochs=1,
        seed=0,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
    )


def test_registered_as_bilstm_base() -> None:
    assert "bilstm-base" in available_models()
    assert isinstance(get_detector("bilstm-base"), BiLstmDetector)


def test_detect_without_local_checkpoint_fetches_from_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no explicit bundle and no env-var path, detect() pulls the Hub-published checkpoint on
    # first use; a download failure surfaces as a clear HubError rather than a raw transport error.
    from maneuver_detect.hub import HubError

    def _offline(**kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _offline)
    detector = BiLstmDetector()
    assert detector.is_loaded is False
    with pytest.raises(HubError, match="could not fetch"):
        detector.detect(synthetic_series(norad_id=1, seed=0))


def test_empty_history_returns_empty_canonical_frame(bundle: ModelBundle) -> None:
    out = BiLstmDetector(bundle).detect(synthetic_series(norad_id=1, seed=0).iloc[0:0])
    assert list(out.columns) == list(COLUMNS)
    assert out.empty


def test_dispatch_through_env_var_checkpoint(
    bundle: ModelBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bilstm.pt"
    save_bundle(bundle, path)
    monkeypatch.setenv(CHECKPOINT_ENV, str(path))

    frame = synthetic_series(norad_id=1, seed=5, burns=(Burn(45, "in_track_ms", 4.0),), n=90)
    out = maneuver_detect.detect(frame, model="bilstm-base")
    validate_frame(out)
    assert list(out.columns) == list(COLUMNS)


def test_multi_object_history_is_grouped_and_sorted(bundle: ModelBundle) -> None:
    frame = pd.concat(
        [
            synthetic_series(norad_id=7, seed=5, burns=(Burn(45, "in_track_ms", 4.0),), n=90),
            synthetic_series(norad_id=3, seed=6, burns=(Burn(50, "cross_track_ms", 4.0),), n=90),
        ],
        ignore_index=True,
    )
    out = BiLstmDetector(bundle, threshold=0.5).detect(frame)
    validate_frame(out)
    if not out.empty:
        # Rows are returned sorted by (norad_id, epoch).
        ordered = out.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(out, ordered)
        assert set(out["norad_id"]).issubset({3, 7})


def test_threshold_gates_detection_count(bundle: ModelBundle) -> None:
    frame = synthetic_series(norad_id=1, seed=5, burns=(Burn(45, "in_track_ms", 4.0),), n=90)
    low = BiLstmDetector(bundle, threshold=0.1).detect(frame)
    high = BiLstmDetector(bundle, threshold=0.99).detect(frame)
    assert len(high) <= len(low)


def test_per_class_threshold_from_bundle_gates_detection_count(bundle: ModelBundle) -> None:
    # The per-class machinery lives in the shared _LearnedDetector, so the BiLSTM adopts the
    # bundle's per-class gates and applies the LEO gate to the (LEO) object like the scalar does.
    frame = synthetic_series(norad_id=1, seed=5, burns=(Burn(45, "in_track_ms", 4.0),), n=90)
    low = BiLstmDetector(replace(bundle, class_thresholds={"LEO": 0.1})).detect(frame)
    high = BiLstmDetector(replace(bundle, class_thresholds={"LEO": 0.99})).detect(frame)
    assert BiLstmDetector(replace(bundle, class_thresholds={"LEO": 0.1}))._class_thresholds == {
        "LEO": 0.1
    }
    assert len(high) <= len(low)


def test_detected_gaps_reduces_runs_to_their_peak() -> None:
    probs = np.array([0.0, 0.1, 0.6, 0.7, 0.2, 0.9, 0.1], dtype=np.float64)
    assert _detected_gaps(probs, 0.5) == [3, 5]


def test_detected_gaps_ignores_the_first_token() -> None:
    # Token 0 has no preceding gap and is never a detection, even above threshold.
    probs = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    assert _detected_gaps(probs, 0.5) == []
