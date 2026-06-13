"""The learned-baseline model stack — architectures, the shared training loop, and checkpoints.

The v0.2 learned baselines — the BiLSTM (:mod:`~maneuver_detect.models.bilstm`) and the transformer
(:mod:`~maneuver_detect.models.transformer`) — share one harness: the shared Lightning training/eval
loop (:mod:`~maneuver_detect.models.module`), the labelled-series data module
(:mod:`~maneuver_detect.models.datamodule`), and the checkpoint bundle
(:mod:`~maneuver_detect.models.checkpoint`) that freezes weights with the train-split normaliser so
inference reproduces training-time standardisation. The two training entrypoints
(:func:`~maneuver_detect.models.train.train_bilstm` and ``train_transformer``) assemble them into
one reproducible call.

Only the inference-light pieces (the network and the checkpoint bundle — ``torch`` but not
Lightning) are re-exported here, so a detector loading a checkpoint never imports the training
stack. The data
module, the Lightning module, and the training entrypoint are imported from their submodules
(``maneuver_detect.models.train`` etc.) when a *training* run actually needs them.
"""

from __future__ import annotations

from maneuver_detect.models.bilstm import (
    NETWORK_KIND,
    BiLstmConfig,
    BiLstmNetwork,
    build_bilstm,
)
from maneuver_detect.models.checkpoint import (
    ModelBundle,
    build_network,
    load_bundle,
    save_bundle,
)
from maneuver_detect.models.transformer import NETWORK_KIND as TRANSFORMER_NETWORK_KIND
from maneuver_detect.models.transformer import (
    TransformerConfig,
    TransformerNetwork,
    build_transformer,
)

__all__ = [
    "NETWORK_KIND",
    "TRANSFORMER_NETWORK_KIND",
    "BiLstmConfig",
    "BiLstmNetwork",
    "ModelBundle",
    "TransformerConfig",
    "TransformerNetwork",
    "build_bilstm",
    "build_network",
    "build_transformer",
    "load_bundle",
    "save_bundle",
]
