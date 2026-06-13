"""The shared Lightning training/eval loop — one ``LightningModule`` both learned baselines reuse.

:class:`SequenceDetectorModule` wraps any per-token sequence network (the BiLSTM here, the
transformer next) and supplies the parts that do not change between architectures: a **masked**
binary-cross-entropy loss over the valid tokens of each window (padding is ignored), a
positive-class weight for the heavy maneuver/quiet-gap imbalance, the Adam optimiser, and the
train/validation logging. The network is the only moving part — pass a different one and the loop is
unchanged — which is what makes the harness shared (D11/D12: one training stack, two baselines).

The module is deliberately thin and architecture-agnostic: it does no feature engineering (that is
the frozen feature layer) and it does not own checkpoint serialisation (that is
:mod:`maneuver_detect.models.checkpoint`, which freezes the network weights together with the
train-split normaliser so inference reproduces training-time standardisation). Inference runs the
bare network on CPU; this module is only needed to *train* one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import lightning as L
import torch
from torch import nn

__all__ = ["SequenceDetectorModule", "TrainHyperParams"]


@dataclass(frozen=True)
class TrainHyperParams:
    """Optimisation hyper-parameters for :class:`SequenceDetectorModule`.

    Attributes:
        lr: Adam learning rate.
        pos_weight: Weight on the positive (maneuver) class in the masked BCE loss — maneuver gaps
            are a small fraction of all gaps, so the positive term is up-weighted to keep the model
            from collapsing to the all-quiet prediction. The training entrypoint defaults it to the
            train-split negative/positive ratio.
        weight_decay: Adam L2 weight decay.
    """

    lr: float = 1e-3
    pos_weight: float = 10.0
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.pos_weight <= 0.0:
            raise ValueError(f"pos_weight must be positive, got {self.pos_weight}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")

    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain dict (frozen alongside the weights in a checkpoint bundle)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> TrainHyperParams:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            lr=float(data["lr"]),
            pos_weight=float(data["pos_weight"]),
            weight_decay=float(data["weight_decay"]),
        )


class SequenceDetectorModule(L.LightningModule):
    """Train any per-token sequence network with a masked, class-weighted BCE loss.

    A batch is ``(features, validity, target)`` — features ``(B, W, C)``, validity ``(B, W)`` bool,
    target ``(B, W)`` float in ``{0, 1}`` — the tensors the
    :class:`~maneuver_detect.models.datamodule.ElementSeriesDataModule` yields. The loss is
    binary-cross-entropy-with-logits over the valid tokens only, averaged over the valid count so
    the padding ratio of a window cannot bias it.
    """

    def __init__(self, network: nn.Module, hparams: TrainHyperParams) -> None:
        super().__init__()
        self.network = network
        self.train_hparams = hparams
        self._pos_weight = float(hparams.pos_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Per-token logits ``(B, W)`` from features ``(B, W, C)`` — delegates to the network."""
        logits: torch.Tensor = self.network(features)
        return logits

    def _masked_loss(
        self, logits: torch.Tensor, target: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        """Class-weighted BCE-with-logits over valid tokens only, averaged over the valid count."""
        # Built on the logits' device/dtype so the weight follows the module to a GPU.
        pos_weight = torch.as_tensor(self._pos_weight, dtype=logits.dtype, device=logits.device)
        per_token = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction="none"
        )
        mask = validity.to(per_token.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (per_token * mask).sum() / denom

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, validity, target = batch
        loss = self._masked_loss(self(features), target, validity)
        self.log("train_loss", loss, prog_bar=True, batch_size=features.shape[0])
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, validity, target = batch
        loss = self._masked_loss(self(features), target, validity)
        self.log("val_loss", loss, prog_bar=True, batch_size=features.shape[0])
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.train_hparams.lr,
            weight_decay=self.train_hparams.weight_decay,
        )
