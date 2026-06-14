"""Tests for the training harness — a seeded run, checkpoint round-trip, determinism, and scoring.

These assert the harness *mechanics* on synthetic data fast on CPU: a run produces a checkpoint
bundle, the bundle round-trips and rebuilds an identical network, a fixed seed reproduces the
weights byte-for-byte (D8), and a trained model runs through the detector and the benchmark scorer
end to end. Detection *accuracy* is not asserted here — that is the offline credentialed run's job,
recorded on the model card.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import torch

from _synthetic import Burn, object_series, synthetic_series
from maneuver_detect.benchmark import ObjectExposure, ScoredLabel, SplitName, TemporalSplit, score
from maneuver_detect.benchmark.matching import match_detections
from maneuver_detect.detectors.bilstm import BiLstmDetector
from maneuver_detect.detectors.transformer import TransformerDetector
from maneuver_detect.labels.labeller import label_series
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.models.bilstm import BiLstmConfig
from maneuver_detect.models.checkpoint import ModelBundle, build_network, load_bundle, save_bundle
from maneuver_detect.models.datamodule import ObjectSeries
from maneuver_detect.models.evaluate import (
    score_on_temporal_split,
    scoring_inputs_for_partition,
    tune_threshold_on_val,
)
from maneuver_detect.models.train import ValBenchmark, train_bilstm, train_transformer
from maneuver_detect.models.transformer import TransformerConfig
from maneuver_detect.schema import COLUMNS, from_frame, validate_frame

_CONFIG = BiLstmConfig(hidden_size=8, num_layers=1, dropout=0.0)
_TF_CONFIG = TransformerConfig(
    d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0, max_len=64
)


def _train_objects() -> list[ObjectSeries]:
    return [
        object_series(norad_id=1, seed=1, burns=(Burn(40, "in_track_ms", 4.0),), n=90),
        object_series(norad_id=2, seed=2, burns=(Burn(55, "cross_track_ms", 4.0),), n=90),
    ]


def _temporal_split(**members: frozenset[int]) -> TemporalSplit:
    # Era cuts that put day ~450 of a 2024-01-01 daily series in the middle (val) era.
    return TemporalSplit(
        dataset_version="test",
        seed=0,
        cut1=datetime(2024, 8, 1, tzinfo=timezone.utc),
        cut2=datetime(2025, 8, 1, tzinfo=timezone.utc),
        guard=timedelta(days=7),
        train=members.get("train", frozenset()),
        val=members.get("val", frozenset()),
        test=members.get("test", frozenset()),
    )


def _val_label(frame: pd.DataFrame, gap_index: int, dv: float) -> ManeuverLabel:
    epochs = list(frame["epoch"])
    midpoint = epochs[gap_index - 1] + (epochs[gap_index] - epochs[gap_index - 1]) / 2
    return ManeuverLabel(
        norad_id=int(frame["norad_id"].iloc[0]),
        epoch=midpoint.to_pydatetime(),
        window_start=epochs[gap_index - 1].to_pydatetime(),
        window_end=epochs[gap_index].to_pydatetime(),
        source="SYNTHETIC",
        source_ref=f"{int(frame['norad_id'].iloc[0])}-{gap_index}",
        orbit_class=OrbitClass.LEO,
        maneuver_type=None,
        delta_v=dv,
    )


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


def test_val_benchmark_selection_runs_and_records_recall() -> None:
    # A val object with a maneuver in the middle era: selection scores it through the benchmark each
    # epoch, restores the best-recall weights, and records best_val_recall on the bundle.
    val_frame = synthetic_series(norad_id=5, seed=5, n=900, burns=(Burn(450, "in_track_ms", 4.0),))
    spec = ValBenchmark(
        series_by_norad={5: val_frame},
        labels=[_val_label(val_frame, 450, 4.0)],
        split=_temporal_split(val=frozenset({5})),
    )
    bundle = train_bilstm(
        _train_objects(),
        config=_CONFIG,
        max_epochs=2,
        seed=0,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
        val_benchmark=spec,
        patience=10,
    )
    assert isinstance(bundle.metadata["best_val_recall"], float)


def test_temporal_split_scoring_is_keyed_to_the_named_partition() -> None:
    # The anti-leak invariant behind val-based selection and threshold tuning (D8: tuned on VAL,
    # never TEST): scoring is keyed to the named partition, so an object placed in VAL is scored
    # only when VAL is named. If the selection path scored TEST instead, here it would score the
    # empty test partition and see nothing — so the two partitions give different scoring inputs.
    val_frame = synthetic_series(norad_id=5, seed=5, n=900, burns=(Burn(450, "in_track_ms", 4.0),))
    split = _temporal_split(val=frozenset({5}))  # the object is in VAL; TEST is empty
    series = {5: val_frame}
    labels = [_val_label(val_frame, 450, 4.0)]

    val_labels, val_exposure = scoring_inputs_for_partition(
        series, labels, split, partition=SplitName.VAL
    )
    test_labels, test_exposure = scoring_inputs_for_partition(
        series, labels, split, partition=SplitName.TEST
    )
    assert val_labels and val_exposure  # the VAL object is scored on VAL
    assert not test_labels and not test_exposure  # nothing is scored on the empty TEST partition

    report = score_on_temporal_split(
        BiLstmDetector(_train()), series, labels, split, partition=SplitName.VAL
    )
    assert OrbitClass.LEO in report.per_class
    assert report.per_class[OrbitClass.LEO].n_labels_above_floor >= 1


def test_early_stopping_and_val_benchmark_are_mutually_exclusive() -> None:
    spec = ValBenchmark(series_by_norad={}, labels=[], split=_temporal_split())
    with pytest.raises(ValueError, match="either early_stopping"):
        train_bilstm(
            _train_objects(),
            config=_CONFIG,
            max_epochs=1,
            accelerator="cpu",
            early_stopping=True,
            val_benchmark=spec,
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


def test_reloaded_bundle_reproduces_the_full_inference_pipeline(tmp_path: Path) -> None:
    # The bundle exists to carry more than weights (D11.3): the frozen train-split normaliser, the
    # window geometry, and the decision threshold, so a reload reproduces the exact inference
    # pipeline. Pin every inference-bearing field and the end-to-end detections — a dropped or
    # re-fit normaliser, or a lost threshold, would change the detections and be caught here.
    bundle = _train()
    path = tmp_path / "bilstm.pt"
    save_bundle(bundle, path)
    reloaded = load_bundle(path)

    assert reloaded.window == bundle.window
    assert reloaded.stride == bundle.stride
    assert reloaded.threshold == pytest.approx(bundle.threshold)
    assert reloaded.normaliser == bundle.normaliser
    assert reloaded.network_config == bundle.network_config

    frame = synthetic_series(norad_id=1, seed=11, burns=(Burn(45, "in_track_ms", 4.0),), n=90)
    from_memory = BiLstmDetector(bundle).detect(frame)
    from_disk = BiLstmDetector(reloaded).detect(frame)
    pd.testing.assert_frame_equal(from_memory, from_disk)


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


# --- The transformer baseline trains through the same shared harness ------------------------------


def _train_transformer(seed: int = 0, **kwargs: object) -> ModelBundle:
    return train_transformer(
        _train_objects(),
        config=_TF_CONFIG,
        max_epochs=1,
        seed=seed,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
        **kwargs,  # type: ignore[arg-type]
    )


def test_train_transformer_returns_a_consistent_bundle() -> None:
    bundle = _train_transformer()
    assert bundle.network_config["network"] == "transformer"
    assert bundle.window == 32 and bundle.stride == 16
    assert bundle.metadata["seed"] == 0
    # The transformer-friendly defaults are recorded as provenance.
    assert bundle.metadata["optimizer"] == "adamw"
    assert bundle.metadata["scheduler"] == "warmup_cosine"
    assert bundle.metadata["loss"] == "bce"
    assert "LEO" in bundle.normaliser["medians"]


def test_transformer_training_is_deterministic_for_a_fixed_seed() -> None:
    first = _train_transformer(seed=7)
    second = _train_transformer(seed=7)
    assert first.state_dict.keys() == second.state_dict.keys()
    for key in first.state_dict:
        assert torch.equal(first.state_dict[key], second.state_dict[key]), key


def test_transformer_checkpoint_round_trips_and_rebuilds_identical_network(tmp_path: Path) -> None:
    bundle = _train_transformer()
    path = tmp_path / "transformer.pt"
    save_bundle(bundle, path)
    reloaded = load_bundle(path)

    original = build_network(bundle)
    restored = build_network(reloaded)
    sample = torch.zeros(1, 32, _TF_CONFIG.n_channels, dtype=torch.float32)
    sample[:, :, -2] = 1.0  # elset_valid: mark tokens real so the window is not all-padding (NaN)
    with torch.no_grad():
        assert torch.equal(original(sample), restored(sample))


def test_transformer_detect_returns_canonical_schema() -> None:
    bundle = _train_transformer()
    frame = synthetic_series(norad_id=1, seed=11, burns=(Burn(45, "in_track_ms", 4.0),), n=90)

    out = TransformerDetector(bundle).detect(frame)
    validate_frame(out)
    assert list(out.columns) == list(COLUMNS)


def test_transformer_focal_loss_path_trains() -> None:
    bundle = _train_transformer(loss="focal")
    assert bundle.metadata["loss"] == "focal"
    assert bundle.train_hparams["focal_gamma"] == 2.0  # the default focal exponent


def test_transformer_plain_optimizer_and_scheduler_paths_train() -> None:
    # The Adam + no-schedule path also runs for the transformer, covering the non-default knobs.
    bundle = _train_transformer(optimizer="adam", scheduler="none")
    assert bundle.metadata["optimizer"] == "adam"
    assert bundle.metadata["scheduler"] == "none"


def test_tune_threshold_on_val_selects_a_candidate() -> None:
    bundle = _train_transformer()
    val_frame = synthetic_series(norad_id=5, seed=5, n=900, burns=(Burn(450, "in_track_ms", 4.0),))
    tuning = tune_threshold_on_val(
        lambda t: TransformerDetector(bundle, threshold=t),
        {5: val_frame},
        [_val_label(val_frame, 450, 4.0)],
        _temporal_split(val=frozenset({5})),
        candidates=(0.2, 0.5, 0.8),
    )
    assert tuning.threshold in (0.2, 0.5, 0.8)
    assert 0.0 <= tuning.recall <= 1.0
    assert set(tuning.by_threshold) == {0.2, 0.5, 0.8}
