"""The shared Lightning training/eval loop — one ``LightningModule`` both learned baselines reuse.

:class:`SequenceDetectorModule` wraps any per-token sequence network (the BiLSTM, the transformer)
and supplies the parts that do not change between architectures: a **masked** binary-cross-entropy
loss over the valid tokens of each window (padding is ignored), a positive-class weight for the
heavy maneuver/quiet-gap imbalance, the optimiser, and the train/validation logging. The network is
the only moving part — pass a different one and the loop is unchanged — which is what makes the
harness shared (D11/D12: one training stack, two baselines).

Three optimisation choices are exposed so an architecture can pick what suits it without forking the
loop, **all defaulting to the original behaviour** (so a run that does not opt in is bit-for-bit
unchanged): the optimiser (``adam`` or decoupled-weight-decay ``adamw``), the learning-rate schedule
(``none`` or a linear-warmup-then-cosine-decay schedule that stabilises transformer training), and
the loss (class-weighted ``bce`` or ``focal``, which down-weights easy quiet-gap negatives under the
imbalance). The numeric knobs the latter two need — the warmup fraction and the focal exponent —
travel on :class:`TrainHyperParams`.

The module is deliberately thin and architecture-agnostic: it does no feature engineering (that is
the frozen feature layer) and it does not own checkpoint serialisation (that is
:mod:`maneuver_detect.models.checkpoint`, which freezes the network weights together with the
train-split normaliser so inference reproduces training-time standardisation). Inference runs the
bare network on CPU; this module is only needed to *train* one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import lightning as L
import torch
from torch import nn

__all__ = [
    "LossName",
    "OptimizerName",
    "SchedulerName",
    "SequenceDetectorModule",
    "TrainHyperParams",
]

#: The supported optimisers — plain Adam (the original default) or AdamW (decoupled weight decay).
OptimizerName = Literal["adam", "adamw"]

#: The supported learning-rate schedules — a fixed rate (the original default) or a linear warmup
#: into a cosine decay (the transformer-friendly schedule).
SchedulerName = Literal["none", "warmup_cosine"]

#: The supported losses — class-weighted BCE (the original default) or focal BCE.
LossName = Literal["bce", "focal"]


@dataclass(frozen=True)
class TrainHyperParams:
    """Optimisation hyper-parameters for :class:`SequenceDetectorModule`.

    Attributes:
        lr: The (peak) learning rate.
        pos_weight: Weight on the positive (maneuver) class in the masked loss — maneuver gaps
            are a small fraction of all gaps, so the positive term is up-weighted to keep the model
            from collapsing to the all-quiet prediction. The training entrypoint defaults it to the
            train-split negative/positive ratio.
        weight_decay: L2 weight decay (decoupled when the optimiser is ``adamw``).
        warmup_frac: Fraction of the total training steps spent linearly warming the learning rate
            from zero to ``lr`` before the cosine decay — used only by the ``warmup_cosine``
            schedule; ``0.0`` (the default) means no warmup.
        focal_gamma: The focal-loss focusing exponent — used only by the ``focal`` loss; ``0.0``
            (the default) makes focal weighting a no-op.
    """

    lr: float = 1e-3
    pos_weight: float = 10.0
    weight_decay: float = 0.0
    warmup_frac: float = 0.0
    focal_gamma: float = 0.0

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.pos_weight <= 0.0:
            raise ValueError(f"pos_weight must be positive, got {self.pos_weight}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not 0.0 <= self.warmup_frac < 1.0:
            raise ValueError(f"warmup_frac must be in [0, 1), got {self.warmup_frac}")
        if self.focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must be non-negative, got {self.focal_gamma}")

    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain dict (frozen alongside the weights in a checkpoint bundle)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> TrainHyperParams:
        """Reconstruct from :meth:`to_dict` output, tolerating dicts without the newer knobs."""
        return cls(
            lr=float(data["lr"]),
            pos_weight=float(data["pos_weight"]),
            weight_decay=float(data["weight_decay"]),
            warmup_frac=float(data.get("warmup_frac", 0.0)),
            focal_gamma=float(data.get("focal_gamma", 0.0)),
        )


class SequenceDetectorModule(L.LightningModule):
    """Train any per-token sequence network with a masked, class-weighted loss.

    A batch is ``(features, validity, target)`` — features ``(B, W, C)``, validity ``(B, W)`` bool,
    target ``(B, W)`` float in ``{0, 1}`` — the tensors the
    :class:`~maneuver_detect.models.datamodule.ElementSeriesDataModule` yields. The loss is computed
    over the valid tokens only, averaged over the valid count so the padding ratio of a window
    cannot bias it. ``optimizer`` / ``scheduler`` / ``loss`` select the optimisation strategy; their
    defaults reproduce the original plain-Adam, fixed-rate, class-weighted-BCE behaviour exactly.
    """

    def __init__(
        self,
        network: nn.Module,
        hparams: TrainHyperParams,
        *,
        optimizer: OptimizerName = "adam",
        scheduler: SchedulerName = "none",
        loss: LossName = "bce",
    ) -> None:
        super().__init__()
        self.network = network
        self.train_hparams = hparams
        self._pos_weight = float(hparams.pos_weight)
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss = loss

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Per-token logits ``(B, W)`` from features ``(B, W, C)`` — delegates to the network."""
        logits: torch.Tensor = self.network(features)
        return logits

    def _masked_loss(
        self, logits: torch.Tensor, target: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        """The configured loss over valid tokens only, averaged over the valid count."""
        mask = validity.to(logits.dtype)
        denom = mask.sum().clamp_min(1.0)
        per_token = self._per_token_loss(logits, target)
        return (per_token * mask).sum() / denom

    def _per_token_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-token (unreduced) loss — class-weighted BCE, or focal BCE when selected."""
        # Built on the logits' device/dtype so the weight follows the module to a GPU.
        pos_weight = torch.as_tensor(self._pos_weight, dtype=logits.dtype, device=logits.device)
        if self._loss == "bce":
            return nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight, reduction="none"
            )
        # Focal BCE: class-weight the positives by pos_weight (the alpha role) and down-weight
        # confidently-correct tokens by (1 - p_t) ** gamma so easy quiet gaps stop dominating.
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = target * prob + (1.0 - target) * (1.0 - prob)
        focal = (1.0 - p_t).clamp_min(0.0) ** self.train_hparams.focal_gamma
        class_weight = target * pos_weight + (1.0 - target)
        return focal * class_weight * bce

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

    def configure_optimizers(self) -> Any:
        optimizer = self._build_optimizer()
        if self._scheduler == "none":
            return optimizer
        scheduler = self._build_warmup_cosine(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr = self.train_hparams.lr
        weight_decay = self.train_hparams.weight_decay
        if self._optimizer == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _build_warmup_cosine(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """A linear warmup into a cosine decay over the trainer's estimated total steps."""
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = min(int(self.train_hparams.warmup_frac * total_steps), total_steps - 1)

        def lr_scale(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / (warmup_steps + 1)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
