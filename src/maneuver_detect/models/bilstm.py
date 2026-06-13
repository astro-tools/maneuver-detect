"""The BiLSTM baseline network — a bidirectional LSTM over the V5-encoded element series.

The first learned baseline: a stack of bidirectional LSTM layers over the per-token channel matrix
the feature layer emits (:mod:`maneuver_detect.features`), with a per-token linear head producing
one maneuver logit per token. One token is one elset, so the per-token output is the per-gap
maneuver score the benchmark scores (D4) — the transition *into* a token is the inter-elset gap
that token's logit decides.

The network is intentionally small (~1-3 M parameters, the V7 budget tier): a sequence model the
benchmark can train in hours on a single commodity GPU and run on CPU at inference. It is a plain
:class:`torch.nn.Module` — the training loop, loss, and optimisation live in the shared
:class:`~maneuver_detect.models.module.SequenceDetectorModule`, so this file is only the
architecture and its serialisable :class:`BiLstmConfig`. The transformer baseline arrives as a
sibling network
behind the same module and checkpoint contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from maneuver_detect.features.channels import N_CHANNELS

__all__ = ["NETWORK_KIND", "BiLstmConfig", "BiLstmNetwork", "build_bilstm"]

#: The network-kind tag stored in a checkpoint bundle so the loader rebuilds the right architecture.
NETWORK_KIND = "bilstm"


@dataclass(frozen=True)
class BiLstmConfig:
    """The BiLSTM architecture hyper-parameters — serialised verbatim into the checkpoint bundle.

    Attributes:
        n_channels: Input channel count ``C`` per token; defaults to the feature layer's frozen
            :data:`~maneuver_detect.features.channels.N_CHANNELS`.
        hidden_size: Hidden units per LSTM direction (the layer is bidirectional, so each layer
            outputs ``2 * hidden_size``).
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between stacked LSTM layers (ignored when
            ``num_layers == 1``, where PyTorch applies none).
    """

    n_channels: int = N_CHANNELS
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.n_channels < 1:
            raise ValueError(f"n_channels must be at least 1, got {self.n_channels}")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be at least 1, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def to_dict(self) -> dict[str, float | int]:
        """Serialise to a plain dict (the network block of a checkpoint bundle)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BiLstmConfig:
        """Reconstruct a config from :meth:`to_dict` output, ignoring any ``network`` tag."""
        return cls(
            n_channels=int(data["n_channels"]),
            hidden_size=int(data["hidden_size"]),
            num_layers=int(data["num_layers"]),
            dropout=float(data["dropout"]),
        )


class BiLstmNetwork(nn.Module):
    """A bidirectional LSTM with a per-token maneuver-logit head.

    ``forward`` maps a ``(batch, window, n_channels)`` feature tensor to ``(batch, window)``
    per-token logits — one maneuver score per token (per gap). Padding tokens are processed like any
    other; the training loss and the inference reducer mask them out using the validity tensor, so
    the network
    itself needs no mask input.
    """

    def __init__(self, config: BiLstmConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.n_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * config.hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Per-token logits ``(batch, window)`` from features ``(batch, window, channels)``."""
        outputs, _ = self.lstm(features)
        logits: torch.Tensor = self.head(outputs).squeeze(-1)
        return logits


def build_bilstm(config: Mapping[str, Any]) -> BiLstmNetwork:
    """Build a :class:`BiLstmNetwork` from a serialised config dict (the bundle network factory)."""
    return BiLstmNetwork(BiLstmConfig.from_dict(config))
