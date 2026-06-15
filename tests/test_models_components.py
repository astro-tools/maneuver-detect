"""Unit tests for the model-stack components — config validation, serialisation, and the loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from maneuver_detect.errors import ManeuverDetectError
from maneuver_detect.models.bilstm import BiLstmConfig
from maneuver_detect.models.checkpoint import ModelBundle, build_network, load_bundle, save_bundle
from maneuver_detect.models.module import TrainHyperParams
from maneuver_detect.models.transformer import TransformerConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_channels": 0},
        {"hidden_size": 0},
        {"num_layers": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
    ],
)
def test_bilstm_config_rejects_bad_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BiLstmConfig(**kwargs)  # type: ignore[arg-type]  # deliberately invalid values


def test_bilstm_config_round_trips_and_ignores_network_tag() -> None:
    config = BiLstmConfig(hidden_size=24, num_layers=2, dropout=0.2)
    restored = BiLstmConfig.from_dict({"network": "bilstm", **config.to_dict()})
    assert restored == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_channels": 0},
        {"d_model": 0},
        {"d_model": 7},  # odd
        {"d_model": 6, "nhead": 4},  # 6 not divisible by 4
        {"nhead": 0},
        {"num_layers": 0},
        {"dim_feedforward": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"max_len": 0},
    ],
)
def test_transformer_config_rejects_bad_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TransformerConfig(**kwargs)  # type: ignore[arg-type]  # deliberately invalid values


def test_transformer_config_round_trips_and_ignores_network_tag() -> None:
    config = TransformerConfig(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2, max_len=256
    )
    restored = TransformerConfig.from_dict({"network": "transformer", **config.to_dict()})
    assert restored == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 0.0},
        {"pos_weight": 0.0},
        {"weight_decay": -1.0},
        {"warmup_frac": 1.0},
        {"warmup_frac": -0.1},
        {"focal_gamma": -1.0},
    ],
)
def test_train_hyperparams_rejects_bad_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TrainHyperParams(**kwargs)


def test_train_hyperparams_round_trips() -> None:
    hparams = TrainHyperParams(
        lr=2e-3, pos_weight=5.0, weight_decay=1e-4, warmup_frac=0.1, focal_gamma=2.0
    )
    assert TrainHyperParams.from_dict(hparams.to_dict()) == hparams


def test_train_hyperparams_from_dict_tolerates_missing_knobs() -> None:
    # An older bundle's hparams dict predates warmup_frac / focal_gamma; they default to 0.0.
    restored = TrainHyperParams.from_dict({"lr": 1e-3, "pos_weight": 10.0, "weight_decay": 0.0})
    assert restored.warmup_frac == 0.0
    assert restored.focal_gamma == 0.0


def test_build_network_rejects_unknown_kind() -> None:
    bundle = ModelBundle(
        network_config={"network": "does-not-exist"},
        state_dict={},
        normaliser={},
        train_hparams={},
        window=64,
        stride=32,
        threshold=0.5,
    )
    with pytest.raises(ValueError, match="unknown network kind"):
        build_network(bundle)


def test_model_bundle_metadata_defaults_to_empty() -> None:
    bundle = ModelBundle(
        network_config={"network": "bilstm"},
        state_dict={},
        normaliser={},
        train_hparams={},
        window=64,
        stride=32,
        threshold=0.5,
    )
    assert bundle.metadata == {}
    assert bundle.class_thresholds == {}


def test_save_load_round_trips_class_thresholds(tmp_path: Path) -> None:
    bundle = ModelBundle(
        network_config={"network": "bilstm"},
        state_dict={},
        normaliser={},
        train_hparams={},
        window=64,
        stride=32,
        threshold=0.5,
        class_thresholds={"GEO": 0.3, "LEO": 0.6},
    )
    path = tmp_path / "tuned.pt"
    save_bundle(bundle, path)
    assert load_bundle(path).class_thresholds == {"GEO": 0.3, "LEO": 0.6}


def test_load_bundle_without_class_thresholds_is_back_compatible(tmp_path: Path) -> None:
    # A checkpoint saved before per-class tuning has no class_thresholds key; it must still load,
    # with an empty map (so the scalar threshold gates every class).
    payload = {
        "network_config": {"network": "bilstm"},
        "state_dict": {},
        "normaliser": {},
        "train_hparams": {},
        "window": 64,
        "stride": 32,
        "threshold": 0.5,
        # "class_thresholds" deliberately omitted (a pre-tuning checkpoint).
    }
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    assert load_bundle(path).class_thresholds == {}


def test_save_load_round_trips_calibration(tmp_path: Path) -> None:
    from maneuver_detect.calibration import BundledCalibration, ReliabilityBin, ReliabilityCurve

    calibration = BundledCalibration(
        temperature=1.7,
        conformal_q=0.4,
        conformal_alpha=0.1,
        reliability={
            "LEO": ReliabilityCurve(
                bins=(
                    ReliabilityBin(
                        lo=0.0, hi=0.5, count=2, mean_confidence=0.3, empirical_precision=0.25
                    ),
                )
            ),
            # A sparse class: an empty bin round-trips its None mean/precision.
            "IGSO": ReliabilityCurve(
                bins=(
                    ReliabilityBin(
                        lo=0.0, hi=0.5, count=0, mean_confidence=None, empirical_precision=None
                    ),
                )
            ),
        },
        ece={"LEO": 0.12, "IGSO": 0.0},
    )
    bundle = ModelBundle(
        network_config={"network": "bilstm"},
        state_dict={},
        normaliser={},
        train_hparams={},
        window=64,
        stride=32,
        threshold=0.5,
        calibration=calibration,
    )
    path = tmp_path / "calibrated.pt"
    save_bundle(bundle, path)
    assert load_bundle(path).calibration == calibration


def test_load_bundle_without_calibration_is_back_compatible(tmp_path: Path) -> None:
    # A checkpoint saved before the calibration slot has no calibration key; it must still load,
    # with calibration None (so the detector emits raw, uncalibrated confidence).
    payload = {
        "network_config": {"network": "bilstm"},
        "state_dict": {},
        "normaliser": {},
        "train_hparams": {},
        "window": 64,
        "stride": 32,
        "threshold": 0.5,
        # "calibration" deliberately omitted (a pre-calibration checkpoint).
    }
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    assert load_bundle(path).calibration is None


def test_load_bundle_reports_a_missing_required_key_clearly(tmp_path: Path) -> None:
    # A truncated or version-mismatched Hub artifact (here, the train-split normaliser dropped) must
    # surface to the detect() caller as a clear typed error naming the missing field, not a bare
    # KeyError from a raw subscript.
    payload = {
        "network_config": {"network": "bilstm"},
        "state_dict": {},
        "train_hparams": {},
        "window": 64,
        "stride": 32,
        "threshold": 0.5,
        # "normaliser" deliberately omitted.
    }
    path = tmp_path / "broken.pt"
    torch.save(payload, path)
    with pytest.raises(ManeuverDetectError, match=r"missing required bundle keys.*normaliser"):
        load_bundle(path)


def test_load_bundle_rejects_a_non_bundle_file(tmp_path: Path) -> None:
    # A bare tensor / state_dict saved by mistake is not a bundle — reject it with a clear message.
    path = tmp_path / "tensor.pt"
    torch.save(torch.zeros(3), path)
    with pytest.raises(ManeuverDetectError, match="is not a model bundle"):
        load_bundle(path)
