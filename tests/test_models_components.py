"""Unit tests for the model-stack components — config validation, serialisation, and the loader."""

from __future__ import annotations

import pytest

from maneuver_detect.models.bilstm import BiLstmConfig
from maneuver_detect.models.checkpoint import ModelBundle, build_network
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
