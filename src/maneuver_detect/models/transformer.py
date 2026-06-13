"""The transformer baseline network — a compact encoder over the V5-encoded element series.

The second learned baseline: a small (~10 M-parameter) pre-norm transformer encoder over the
per-token channel matrix the feature layer emits (:mod:`maneuver_detect.features`), with a per-token
linear head producing one maneuver logit per token. As with the BiLSTM, one token is one elset, so
the per-token output is the per-gap maneuver score the benchmark scores (D4) — the transition *into*
a token is the inter-elset gap that token's logit decides.

Two details matter for this architecture. **Masking**: a sliding window that runs off the end of a
series is zero-padded (:mod:`maneuver_detect.features.windows`), and the ``elset_valid`` channel —
``1`` for every real token, ``0`` on the zero-padding — lets the encoder build its self-attention
key-padding mask from the features alone, so padding tokens never contaminate a real token's
representation and the network keeps the bare ``forward(features) -> logits`` contract the shared
training loop calls (no separate mask input). **Position**: the irregular sampling *cadence* already
rides in the input via the ``time2vec`` timing channels; a fixed sinusoidal positional encoding adds
the within-window ordinal position the attention otherwise lacks.

The network is the V7 ~10 M-parameter budget tier (the default config below is ≈9.5 M): an encoder
the benchmark can train in hours on a single commodity GPU and run on CPU at inference. It is a
plain :class:`torch.nn.Module` — the training loop, loss, and optimisation live in the shared
:class:`~maneuver_detect.models.module.SequenceDetectorModule` — so this file is only the
architecture and its serialisable :class:`TransformerConfig`, behind the same checkpoint contract as
the BiLSTM.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from maneuver_detect.features.channels import CHANNEL_NAMES, N_CHANNELS

__all__ = ["NETWORK_KIND", "TransformerConfig", "TransformerNetwork", "build_transformer"]

#: The network-kind tag stored in a checkpoint bundle so the loader rebuilds the right architecture.
NETWORK_KIND = "transformer"

#: Column index of the ``elset_valid`` channel — ``1`` on a real token, ``0`` on zero-padding — from
#: which the encoder derives its self-attention key-padding mask.
_ELSET_VALID_INDEX = CHANNEL_NAMES.index("elset_valid")


@dataclass(frozen=True)
class TransformerConfig:
    """The transformer architecture hyper-parameters — serialised verbatim into the bundle.

    Attributes:
        n_channels: Input channel count ``C`` per token; defaults to the feature layer's frozen
            :data:`~maneuver_detect.features.channels.N_CHANNELS`.
        d_model: The encoder's model (embedding) width. Must be even and divisible by ``nhead``.
        nhead: Number of self-attention heads.
        num_layers: Number of stacked encoder layers.
        dim_feedforward: Width of each layer's position-wise feed-forward block.
        dropout: Dropout probability inside the encoder layers and on the input embedding.
        max_len: Length of the precomputed sinusoidal positional encoding; must be at least the
            training window length.
    """

    n_channels: int = N_CHANNELS
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 12
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 1024

    def __post_init__(self) -> None:
        if self.n_channels < 1:
            raise ValueError(f"n_channels must be at least 1, got {self.n_channels}")
        if self.d_model < 1:
            raise ValueError(f"d_model must be at least 1, got {self.d_model}")
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {self.d_model}")
        if self.nhead < 1:
            raise ValueError(f"nhead must be at least 1, got {self.nhead}")
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")
        if self.dim_feedforward < 1:
            raise ValueError(f"dim_feedforward must be at least 1, got {self.dim_feedforward}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.max_len < 1:
            raise ValueError(f"max_len must be at least 1, got {self.max_len}")

    def to_dict(self) -> dict[str, float | int]:
        """Serialise to a plain dict (the network block of a checkpoint bundle)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TransformerConfig:
        """Reconstruct a config from :meth:`to_dict` output, ignoring any ``network`` tag."""
        return cls(
            n_channels=int(data["n_channels"]),
            d_model=int(data["d_model"]),
            nhead=int(data["nhead"]),
            num_layers=int(data["num_layers"]),
            dim_feedforward=int(data["dim_feedforward"]),
            dropout=float(data["dropout"]),
            max_len=int(data["max_len"]),
        )


def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """The classic fixed sinusoidal positional encoding, ``(max_len, d_model)``."""
    encoding = torch.zeros(max_len, d_model)
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


class TransformerNetwork(nn.Module):
    """A pre-norm transformer encoder with a per-token maneuver-logit head.

    ``forward`` maps a ``(batch, window, n_channels)`` feature tensor to ``(batch, window)``
    per-token logits. The self-attention key-padding mask is derived from the ``elset_valid``
    channel, so end-of-series zero-padding is ignored by attention and the network needs no mask
    input — the same contract the BiLSTM honours, so the shared training loop is unchanged.
    """

    #: Declared so mypy reads the registered buffer as a tensor (not ``Tensor | Module``).
    positional_encoding: torch.Tensor

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.n_channels, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
            # The mask is always present and a window always has a real token, so the nested-tensor
            # fast path buys nothing and only risks non-determinism — keep the plain padded path.
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(config.d_model, 1)
        # Non-persistent: the sinusoidal table is recomputable, so it stays out of the state_dict
        # and the checkpoint carries weights only.
        self.register_buffer(
            "positional_encoding",
            _sinusoidal_encoding(config.max_len, config.d_model),
            persistent=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Per-token logits ``(batch, window)`` from features ``(batch, window, channels)``."""
        window = features.shape[1]
        # True where a token is padding (ignored by attention); real tokens carry elset_valid == 1.
        padding_mask = features[:, :, _ELSET_VALID_INDEX] < 0.5
        embedded = self.input_projection(features) + self.positional_encoding[:window].unsqueeze(0)
        embedded = self.input_dropout(embedded)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        logits: torch.Tensor = self.head(encoded).squeeze(-1)
        return logits


def build_transformer(config: Mapping[str, Any]) -> TransformerNetwork:
    """Build a :class:`TransformerNetwork` from a serialised config dict (the bundle factory)."""
    return TransformerNetwork(TransformerConfig.from_dict(config))
