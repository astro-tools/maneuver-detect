"""Tests for the training harness — a seeded run, checkpoint round-trip, determinism, and scoring.

These assert the harness *mechanics* on synthetic data fast on CPU: a run produces a checkpoint
bundle, the bundle round-trips and rebuilds an identical network, a fixed seed reproduces the
weights byte-for-byte (D8), and a trained model runs through the detector and the benchmark scorer
end to end. Detection *accuracy* is not asserted here — that is the offline credentialed run's job,
recorded on the model card.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from _synthetic import Burn, object_series, synthetic_series
from maneuver_detect.benchmark import ObjectExposure, ScoredLabel, score
from maneuver_detect.benchmark.matching import match_detections
from maneuver_detect.detectors.bilstm import BiLstmDetector
from maneuver_detect.labels.labeller import label_series
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.models.bilstm import BiLstmConfig
from maneuver_detect.models.checkpoint import ModelBundle, build_network, load_bundle, save_bundle
from maneuver_detect.models.datamodule import ObjectSeries
from maneuver_detect.models.train import train_bilstm
from maneuver_detect.schema import COLUMNS, from_frame, validate_frame

_CONFIG = BiLstmConfig(hidden_size=8, num_layers=1, dropout=0.0)


def _train_objects() -> list[ObjectSeries]:
    return [
        object_series(norad_id=1, seed=1, burns=(Burn(40, "in_track_ms", 4.0),), n=90),
        object_series(norad_id=2, seed=2, burns=(Burn(55, "cross_track_ms", 4.0),), n=90),
    ]


def _train(seed: int = 0) -> ModelBundle:
    return train_bilstm(
        _train_objects(),
        config=_CONFIG,
        max_epochs=1,
        seed=seed,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
    )


def test_train_returns_a_consistent_bundle() -> None:
    bundle = _train()
    assert bundle.network_config["network"] == "bilstm"
    assert bundle.window == 32 and bundle.stride == 16
    assert bundle.metadata["seed"] == 0
    # The frozen normaliser covers the train classes and the hyper-parameters round-trip.
    assert "LEO" in bundle.normaliser["medians"]
    assert bundle.train_hparams["pos_weight"] > 1.0


def test_training_with_validation_runs_and_merges_metadata() -> None:
    # A val set exercises the validation loop (validation_step + the real val dataloader); the
    # extra metadata is merged into the bundle's provenance alongside the auto-added seed.
    val = [object_series(norad_id=9, seed=9, burns=(Burn(45, "in_track_ms", 4.0),), n=90)]
    bundle = train_bilstm(
        _train_objects(),
        val,
        config=_CONFIG,
        max_epochs=1,
        seed=0,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
        metadata={"dataset_version": "synthetic-test"},
    )
    assert bundle.metadata["seed"] == 0
    assert bundle.metadata["dataset_version"] == "synthetic-test"


def test_train_with_progress_runs() -> None:
    # The opt-in progress bar / model summary path runs (default is off for quiet CI).
    bundle = train_bilstm(
        _train_objects(),
        config=_CONFIG,
        max_epochs=1,
        seed=0,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
        progress=True,
    )
    assert bundle.network_config["network"] == "bilstm"


def test_early_stopping_restores_best_and_records_metadata() -> None:
    # With a validation set, early stopping monitors val_loss, restores the best epoch's weights,
    # and records the best val_loss on the bundle (a few epochs is enough to exercise the callback).
    val = [object_series(norad_id=9, seed=9, burns=(Burn(45, "in_track_ms", 4.0),), n=90)]
    bundle = train_bilstm(
        _train_objects(),
        val,
        config=_CONFIG,
        max_epochs=3,
        seed=0,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
        early_stopping=True,
        patience=2,
    )
    assert isinstance(bundle.metadata["best_val_loss"], float)


def test_early_stopping_requires_a_validation_set() -> None:
    with pytest.raises(ValueError, match="early_stopping requires a validation set"):
        train_bilstm(
            _train_objects(),
            config=_CONFIG,
            max_epochs=1,
            accelerator="cpu",
            early_stopping=True,
        )


def test_train_accepts_warn_determinism() -> None:
    # The GPU escape hatch (cuDNN LSTM has no deterministic backward) runs on CPU too.
    bundle = train_bilstm(
        _train_objects(),
        config=_CONFIG,
        max_epochs=1,
        seed=0,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
        deterministic="warn",
    )
    assert bundle.network_config["network"] == "bilstm"


def test_checkpoint_round_trips_and_rebuilds_identical_network(tmp_path: Path) -> None:
    bundle = _train()
    path = tmp_path / "bilstm.pt"
    save_bundle(bundle, path)
    reloaded = load_bundle(path)

    original = build_network(bundle)
    restored = build_network(reloaded)
    sample = torch.zeros(1, 32, _CONFIG.n_channels, dtype=torch.float32)
    with torch.no_grad():
        assert torch.equal(original(sample), restored(sample))


def test_training_is_deterministic_for_a_fixed_seed() -> None:
    first = _train(seed=7)
    second = _train(seed=7)
    assert first.state_dict.keys() == second.state_dict.keys()
    for key in first.state_dict:
        assert torch.equal(first.state_dict[key], second.state_dict[key]), key


def test_detect_returns_canonical_schema() -> None:
    bundle = _train()
    frame = synthetic_series(norad_id=1, seed=11, burns=(Burn(45, "in_track_ms", 4.0),), n=90)

    out = BiLstmDetector(bundle).detect(frame)
    validate_frame(out)
    assert list(out.columns) == list(COLUMNS)


def test_trained_model_scores_through_the_benchmark() -> None:
    bundle = _train()
    frame = synthetic_series(norad_id=1, seed=21, burns=(Burn(45, "in_track_ms", 4.0),), n=90)
    detector = BiLstmDetector(bundle)

    detections = from_frame(detector.detect(frame))

    epochs = list(frame["epoch"])
    label = ManeuverLabel(
        norad_id=1,
        epoch=(epochs[44] + (epochs[45] - epochs[44]) / 2).to_pydatetime(),
        window_start=epochs[44].to_pydatetime(),
        window_end=epochs[45].to_pydatetime(),
        source="SYNTHETIC",
        source_ref="1-45",
        orbit_class=OrbitClass.LEO,
        maneuver_type=None,
        delta_v=4.0,
    )
    intervals = label_series(frame, [label]).intervals
    scored_labels = [ScoredLabel(interval=iv, above_floor=True) for iv in intervals]
    exposure = [
        ObjectExposure(
            norad_id=1,
            orbit_class=OrbitClass.LEO,
            observation_years=(epochs[-1] - epochs[0]).total_seconds() / (365.25 * 86400.0),
        )
    ]

    report = score(detector.detect(frame), scored_labels, exposure)
    assert OrbitClass.LEO in report.per_class
    # The matching runs without error over the model's detections (mechanics, not accuracy).
    match_detections(detections, scored_labels)
