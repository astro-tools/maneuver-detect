"""Tests for the transformer detector — checkpoint gating, dispatch, grouping, and threshold."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import maneuver_detect
from _synthetic import Burn, object_series, synthetic_series
from maneuver_detect.detectors import available_models, get_detector
from maneuver_detect.detectors.transformer import CHECKPOINT_ENV, TransformerDetector
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.models.checkpoint import ModelBundle, save_bundle
from maneuver_detect.models.train import train_transformer
from maneuver_detect.models.transformer import TransformerConfig
from maneuver_detect.schema import COLUMNS, validate_frame

_CONFIG = TransformerConfig(
    d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0, max_len=64
)


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    objects = [
        object_series(norad_id=1, seed=1, burns=(Burn(40, "in_track_ms", 4.0),), n=90),
        object_series(norad_id=2, seed=2, burns=(Burn(55, "cross_track_ms", 4.0),), n=90),
    ]
    return train_transformer(
        objects,
        config=_CONFIG,
        max_epochs=1,
        seed=0,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
    )


def test_registered_as_transformer_base() -> None:
    assert "transformer-base" in available_models()
    assert isinstance(get_detector("transformer-base"), TransformerDetector)


def test_detect_without_local_checkpoint_fetches_from_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no explicit bundle and no env-var path, detect() pulls the Hub-published checkpoint on
    # first use; a download failure surfaces as a clear HubError rather than a raw transport error.
    from maneuver_detect.hub import HubError

    def _offline(**kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _offline)
    detector = TransformerDetector()
    assert detector.is_loaded is False
    with pytest.raises(HubError, match="could not fetch"):
        detector.detect(synthetic_series(norad_id=1, seed=0))


def test_empty_history_returns_empty_canonical_frame(bundle: ModelBundle) -> None:
    out = TransformerDetector(bundle).detect(synthetic_series(norad_id=1, seed=0).iloc[0:0])
    assert list(out.columns) == list(COLUMNS)
    assert out.empty


def test_dispatch_through_env_var_checkpoint(
    bundle: ModelBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "transformer.pt"
    save_bundle(bundle, path)
    monkeypatch.setenv(CHECKPOINT_ENV, str(path))

    frame = synthetic_series(norad_id=1, seed=5, burns=(Burn(45, "in_track_ms", 4.0),), n=90)
    out = maneuver_detect.detect(frame, model="transformer-base")
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
    out = TransformerDetector(bundle, threshold=0.5).detect(frame)
    validate_frame(out)
    if not out.empty:
        # Rows are returned sorted by (norad_id, epoch).
        ordered = out.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(out, ordered)
        assert set(out["norad_id"]).issubset({3, 7})


def test_threshold_gates_detection_count(bundle: ModelBundle) -> None:
    frame = synthetic_series(norad_id=1, seed=5, burns=(Burn(45, "in_track_ms", 4.0),), n=90)
    low = TransformerDetector(bundle, threshold=0.1).detect(frame)
    high = TransformerDetector(bundle, threshold=0.99).detect(frame)
    assert len(high) <= len(low)


def test_per_class_threshold_overrides_resolve(bundle: ModelBundle) -> None:
    # The shared _LearnedDetector resolves the gate per orbit class: explicit per-class map, with a
    # fallback to the bundle's scalar threshold for any class without its own entry.
    detector = TransformerDetector(bundle, class_thresholds={"LEO": 0.1, "GEO": 0.9})
    assert detector._threshold_for(OrbitClass.LEO) == 0.1
    assert detector._threshold_for(OrbitClass.GEO) == 0.9
    assert detector._threshold_for(OrbitClass.MEO) == bundle.threshold  # no gate -> scalar fallback
    # A scalar threshold override takes precedence over every per-class gate.
    override = TransformerDetector(bundle, threshold=0.42, class_thresholds={"LEO": 0.1})
    assert override._threshold_for(OrbitClass.LEO) == 0.42


def test_per_class_thresholds_adopted_from_bundle(bundle: ModelBundle) -> None:
    tuned = replace(bundle, class_thresholds={"LEO": 0.2, "GEO": 0.7})
    assert TransformerDetector(tuned)._class_thresholds == {"LEO": 0.2, "GEO": 0.7}
    # A constructor-supplied map wins over the bundle's.
    explicit = TransformerDetector(tuned, class_thresholds={"LEO": 0.05})
    assert explicit._class_thresholds == {"LEO": 0.05}


def test_per_class_threshold_gates_detection_count(bundle: ModelBundle) -> None:
    # The synthetic object is LEO, so its LEO per-class gate controls the detections — a high LEO
    # gate yields no more than a low one (the per-class analogue of the scalar gate).
    frame = synthetic_series(norad_id=1, seed=5, burns=(Burn(45, "in_track_ms", 4.0),), n=90)
    low = TransformerDetector(bundle, class_thresholds={"LEO": 0.1}).detect(frame)
    high = TransformerDetector(bundle, class_thresholds={"LEO": 0.99}).detect(frame)
    assert len(high) <= len(low)
