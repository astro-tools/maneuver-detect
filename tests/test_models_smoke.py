"""CPU model-loading smoke tests — the modelling stack imports and runs forward on a CPU runner.

Mirrors the v0.1 model-smoke CI job: the torch / Lightning stack imports cleanly and the BiLSTM
baseline constructs and runs a forward pass on CPU, with no GPU, no data, and no checkpoint. The
end-to-end training + inference path is exercised on synthetic data in ``test_models_train`` /
``test_detectors_bilstm``.
"""

from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_modelling_stack_imports() -> None:
    import lightning  # noqa: F401
    import torch  # noqa: F401

    from maneuver_detect import models

    assert models.NETWORK_KIND == "bilstm"
    assert models.TRANSFORMER_NETWORK_KIND == "transformer"


@pytest.mark.smoke
def test_bilstm_constructs_and_runs_forward_on_cpu() -> None:
    import torch

    from maneuver_detect.features.channels import N_CHANNELS
    from maneuver_detect.models import BiLstmConfig, BiLstmNetwork

    network = BiLstmNetwork(BiLstmConfig(hidden_size=8, num_layers=1))
    network.eval()
    batch, window = 2, 16
    with torch.no_grad():
        logits = network(torch.zeros(batch, window, N_CHANNELS, dtype=torch.float32))
    assert logits.shape == (batch, window)


@pytest.mark.smoke
def test_bilstm_detector_registered_without_checkpoint() -> None:
    # Registration must not need a checkpoint or import torch eagerly.
    from maneuver_detect import available_models
    from maneuver_detect.detectors.bilstm import BiLstmDetector

    assert "bilstm-base" in available_models()
    assert BiLstmDetector().is_loaded is False


@pytest.mark.smoke
def test_transformer_constructs_and_runs_forward_on_cpu() -> None:
    import torch

    from maneuver_detect.features.channels import N_CHANNELS
    from maneuver_detect.models import TransformerConfig, TransformerNetwork

    network = TransformerNetwork(
        TransformerConfig(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_len=64)
    )
    network.eval()
    batch, window = 2, 16
    features = torch.zeros(batch, window, N_CHANNELS, dtype=torch.float32)
    features[:, :, -2] = 1.0  # elset_valid: mark tokens real so the window is not all-padding
    with torch.no_grad():
        logits = network(features)
    assert logits.shape == (batch, window)
    assert bool(torch.isfinite(logits).all())


@pytest.mark.smoke
def test_transformer_detector_registered_without_checkpoint() -> None:
    from maneuver_detect import available_models
    from maneuver_detect.detectors.transformer import TransformerDetector

    assert "transformer-base" in available_models()
    assert TransformerDetector().is_loaded is False
