"""The reproducible training entrypoints — seeded, single-GPU-budget, checkpoint out.

Both learned baselines train through one shared, network-agnostic core,
:func:`_train_sequence_detector`: seed everything (D8), build the
:class:`~maneuver_detect.models.datamodule.ElementSeriesDataModule`
over the labelled element series, fit a network through the shared
:class:`~maneuver_detect.models.module.SequenceDetectorModule`, and return a
:class:`~maneuver_detect.models.checkpoint.ModelBundle` that freezes the weights together with the
train-split normaliser so inference reproduces training-time standardisation. :func:`train_bilstm`
and :func:`train_transformer` are thin wrappers that supply their architecture, its serialised
config, and the detector the val-benchmark selection scores with; everything else is identical.

Checkpoint selection has three modes. By default the **last** epoch's weights are frozen. With
``early_stopping`` the lowest-``val_loss`` epoch is kept — but the loss bottoms out fast by going
conservative, which can undertrain the model for the benchmark. With ``val_benchmark`` the harness
instead scores the **val partition through the real benchmark each epoch** and keeps the best-recall
weights, so selection is aligned with the published metric rather than the surrogate.

Training honours the V7 budget — a single ``<= 24 GB`` GPU, hours of wall-clock — but the call is
device-agnostic (``accelerator="auto"``): it runs on CPU for the fast synthetic-data tests and on a
GPU for the real, credentialled reconstruction. The function is deterministic for a fixed seed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch import nn

from maneuver_detect.models.checkpoint import ModelBundle
from maneuver_detect.models.datamodule import ElementSeriesDataModule, ObjectSeries
from maneuver_detect.models.module import (
    LossName,
    OptimizerName,
    SchedulerName,
    SequenceDetectorModule,
    TrainHyperParams,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from maneuver_detect.benchmark import TemporalSplit
    from maneuver_detect.detectors.base import Detector
    from maneuver_detect.features.normalize import ClassNormaliser
    from maneuver_detect.labels.record import ManeuverLabel
    from maneuver_detect.models.bilstm import BiLstmConfig
    from maneuver_detect.models.evaluate import SelectionObjective
    from maneuver_detect.models.transformer import TransformerConfig

__all__ = ["ValBenchmark", "train_bilstm", "train_transformer"]

#: Builds a detector from a freshly-frozen bundle, for the val-benchmark checkpoint selection. Each
#: architecture passes its own (deferred so the core never imports the torch-heavy detector stack).
DetectorFactory = Callable[[ModelBundle], "Detector"]


@dataclass(frozen=True)
class ValBenchmark:
    """Inputs to select the checkpoint by VAL-split benchmark recall instead of ``val_loss``.

    Attributes:
        series_by_norad: Each object's full reconstructed mean-element series (the val objects must
            be present; extra objects are ignored by the scorer).
        labels: The full label set — the scorer era-scopes it to the val partition.
        split: The leak-free temporal split whose VAL partition is scored each epoch.
        objective: The class-balance of the per-epoch selection score —
            :data:`~maneuver_detect.models.evaluate.SelectionObjective`. ``"pooled"`` (default,
            label-count-weighted) keeps the unchanged behaviour; ``"macro"`` weights every orbit
            class equally, so a later epoch that keeps training the GEO signal can win even though
            GEO holds the minority of labels.
    """

    series_by_norad: Mapping[int, pd.DataFrame]
    labels: Sequence[ManeuverLabel]
    split: TemporalSplit
    objective: SelectionObjective = "pooled"


class _RestoreBestWeights(Callback):
    """Keep the lowest-``val_loss`` network weights and restore them at the end of training.

    Lightning's default keeps the *last* epoch's weights; on a small dataset that is the most
    over-trained state. This callback snapshots the network ``state_dict`` (on CPU) whenever the
    validation loss improves, so the trainer can restore the best epoch before freezing the bundle.
    The sanity-check pass (an untrained validation run before epoch 0) is skipped.
    """

    def __init__(self) -> None:
        self.best_score: float | None = None
        self._best_state: dict[str, torch.Tensor] | None = None

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("val_loss")
        if metric is None:
            return
        score = float(metric)
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self._best_state = {
                key: value.detach().cpu().clone()
                for key, value in pl_module.network.state_dict().items()
            }

    def restore(self, network: nn.Module) -> None:
        """Load the best-seen weights into ``network`` (a no-op if validation never ran)."""
        if self._best_state is not None:
            network.load_state_dict(self._best_state)


class _ValBenchmarkSelection(Callback):
    """Keep the weights with the best VAL-split benchmark recall; optionally early-stop on it.

    Each training epoch a detector is built from the current weights (and the train-split
    normaliser) and scored on the val partition through the same benchmark the leaderboard uses; the
    highest-:func:`~maneuver_detect.models.evaluate.objective_recall` weights (under the spec's
    ``objective`` — ``"pooled"`` or ``"macro"``) are snapshotted and restored before the bundle is
    frozen. Selection is therefore on the metric we report, not the loss surrogate. ``patience``
    early-stops when the val recall has not improved for that many epochs (a *recall* plateau,
    unlike the loss one). The detector is built through the architecture's ``detector_factory`` so
    the same callback serves either baseline.
    """

    def __init__(
        self,
        spec: ValBenchmark,
        *,
        detector_factory: DetectorFactory,
        network_config: dict[str, Any],
        normaliser: ClassNormaliser,
        window: int,
        stride: int,
        threshold: float,
        patience: int,
    ) -> None:
        self._spec = spec
        self._detector_factory = detector_factory
        self._network_config = network_config
        self._normaliser = normaliser
        self._window = window
        self._stride = stride
        self._threshold = threshold
        self._patience = patience
        self.best_score: float | None = None
        self._best_state: dict[str, torch.Tensor] | None = None
        self._stale = 0

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        # Deferred so the inference / benchmark stack is not pulled in unless this mode is used.
        from maneuver_detect.benchmark import SplitName
        from maneuver_detect.models.evaluate import (
            objective_recall,
            score_on_temporal_split,
        )

        state = {
            key: value.detach().cpu().clone()
            for key, value in pl_module.network.state_dict().items()
        }
        bundle = ModelBundle(
            network_config=self._network_config,
            state_dict=state,
            normaliser=self._normaliser.to_dict(),
            train_hparams={},
            window=self._window,
            stride=self._stride,
            threshold=self._threshold,
        )
        report = score_on_temporal_split(
            self._detector_factory(bundle),
            self._spec.series_by_norad,
            self._spec.labels,
            self._spec.split,
            partition=SplitName.VAL,
        )
        score = objective_recall(report, self._spec.objective)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self._best_state = state
            self._stale = 0
        else:
            self._stale += 1
            if self._stale >= self._patience:
                trainer.should_stop = True

    def restore(self, network: nn.Module) -> None:
        """Load the best-recall weights into ``network`` (a no-op if no epoch was scored)."""
        if self._best_state is not None:
            network.load_state_dict(self._best_state)


def _train_sequence_detector(
    train_objects: Sequence[ObjectSeries],
    val_objects: Sequence[ObjectSeries],
    *,
    network_factory: Callable[[], nn.Module],
    network_config: dict[str, Any],
    detector_factory: DetectorFactory,
    hparams: TrainHyperParams | None,
    max_epochs: int,
    seed: int,
    window: int | None,
    stride: int | None,
    batch_size: int,
    threshold: float,
    accelerator: str,
    deterministic: bool | Literal["warn"],
    progress: bool,
    early_stopping: bool,
    val_benchmark: ValBenchmark | None,
    patience: int,
    optimizer: OptimizerName,
    scheduler: SchedulerName,
    loss: LossName,
    warmup_frac: float,
    focal_gamma: float,
    metadata: dict[str, Any] | None,
) -> ModelBundle:
    """Train a network through the shared harness and return its frozen :class:`ModelBundle`.

    The architecture-agnostic core both wrappers delegate to. ``network_factory`` builds the
    per-token sequence network — called **after** the seed is set, so weight initialisation is
    reproducible — and ``network_config`` is its serialised config (carrying the ``network`` tag the
    loader rebuilds it from); ``detector_factory`` builds a detector from a frozen bundle for the
    val-benchmark selection. The remaining arguments are the wrappers' own, documented there.
    """
    from maneuver_detect.features.windows import STRIDE, WINDOW

    resolved_window = WINDOW if window is None else window
    resolved_stride = STRIDE if stride is None else stride

    if early_stopping and val_benchmark is not None:
        raise ValueError("pass either early_stopping (BCE val_loss) or val_benchmark, not both")

    L.seed_everything(seed, workers=True)

    datamodule = ElementSeriesDataModule(
        train_objects,
        val_objects,
        window=resolved_window,
        stride=resolved_stride,
        batch_size=batch_size,
    )
    datamodule.setup()
    assert datamodule.normaliser is not None  # setup() fits it

    hparams = hparams or TrainHyperParams(
        pos_weight=datamodule.positive_weight(),
        warmup_frac=warmup_frac,
        focal_gamma=focal_gamma,
    )
    network = network_factory()
    module = SequenceDetectorModule(
        network, hparams, optimizer=optimizer, scheduler=scheduler, loss=loss
    )

    validate = bool(val_objects)
    callbacks: list[Callback] = []
    loss_best: _RestoreBestWeights | None = None
    bench_best: _ValBenchmarkSelection | None = None
    if early_stopping:
        if not validate:
            raise ValueError("early_stopping requires a validation set (val_objects)")
        loss_best = _RestoreBestWeights()
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=patience), loss_best]
    elif val_benchmark is not None:
        bench_best = _ValBenchmarkSelection(
            val_benchmark,
            detector_factory=detector_factory,
            network_config=network_config,
            normaliser=datamodule.normaliser,
            window=resolved_window,
            stride=resolved_stride,
            threshold=threshold,
            patience=patience,
        )
        callbacks = [bench_best]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        deterministic=deterministic,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=progress,
        enable_model_summary=progress,
        callbacks=callbacks,
        # No validation set: skip validation entirely (the data module returns an empty val loader).
        limit_val_batches=1.0 if validate else 0,
        num_sanity_val_steps=2 if validate else 0,
    )
    trainer.fit(module, datamodule=datamodule)

    provenance: dict[str, Any] = {
        "seed": seed,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss": loss,
    }
    if metadata:
        provenance.update(metadata)
    if loss_best is not None:
        loss_best.restore(network)  # freeze the best epoch's weights, not the last
        if loss_best.best_score is not None:
            provenance["best_val_loss"] = loss_best.best_score
    if bench_best is not None:
        bench_best.restore(network)  # freeze the best val-benchmark-recall weights
        if bench_best.best_score is not None:
            provenance["best_val_recall"] = bench_best.best_score

    return ModelBundle(
        network_config=network_config,
        state_dict=network.state_dict(),
        normaliser=datamodule.normaliser.to_dict(),
        train_hparams=hparams.to_dict(),
        window=resolved_window,
        stride=resolved_stride,
        threshold=threshold,
        metadata=provenance,
    )


def train_bilstm(
    train_objects: Sequence[ObjectSeries],
    val_objects: Sequence[ObjectSeries] = (),
    *,
    config: BiLstmConfig | None = None,
    hparams: TrainHyperParams | None = None,
    max_epochs: int = 50,
    seed: int = 0,
    window: int | None = None,
    stride: int | None = None,
    batch_size: int = 64,
    threshold: float = 0.5,
    accelerator: str = "auto",
    deterministic: bool | Literal["warn"] = True,
    progress: bool = False,
    early_stopping: bool = False,
    val_benchmark: ValBenchmark | None = None,
    patience: int = 10,
    metadata: dict[str, Any] | None = None,
) -> ModelBundle:
    """Train a BiLSTM on ``train_objects`` and return its frozen :class:`ModelBundle`.

    The run is seeded (``seed``). ``config`` sets the architecture (defaults to
    :class:`~maneuver_detect.models.bilstm.BiLstmConfig`); ``hparams`` the optimisation (defaults to
    the train-split-derived positive-class weight). ``window`` / ``stride`` default to the frozen
    feature-layer geometry. ``accelerator`` is passed to the Lightning ``Trainer`` — ``"auto"``
    picks a GPU when present, CPU otherwise. ``metadata`` is merged into the bundle's provenance
    (the seed is added automatically).

    ``deterministic`` is forwarded to the ``Trainer``: the default ``True`` gives bit-exact
    reproducibility (CPU, and CUDA with ``CUBLAS_WORKSPACE_CONFIG`` set), but cuDNN's LSTM has no
    deterministic backward, so a GPU run that hits that error should pass ``"warn"`` to fall back to
    seed-level reproducibility (the seed is still recorded on the bundle).

    ``progress`` enables the Lightning progress bar (which surfaces the per-step train loss) and the
    model summary — useful for a long interactive run. It defaults to ``False`` so tests, CI, and
    headless runs stay quiet.

    Checkpoint selection (mutually exclusive, both default off → the last epoch is kept):

    * ``early_stopping`` monitors ``val_loss``, early-stops on a ``patience`` plateau, and restores
      the best epoch (records ``best_val_loss``). Needs a validation set (``val_objects``).
    * ``val_benchmark`` scores the val partition through the real benchmark each epoch and keeps the
      best-recall weights, early-stopping on a recall plateau (records ``best_val_recall``). This
      aligns selection with the published metric instead of the loss surrogate.
    """
    from maneuver_detect.detectors.bilstm import BiLstmDetector
    from maneuver_detect.models.bilstm import NETWORK_KIND, BiLstmConfig, BiLstmNetwork

    resolved_config = config or BiLstmConfig()
    network_config: dict[str, Any] = {"network": NETWORK_KIND, **resolved_config.to_dict()}

    return _train_sequence_detector(
        train_objects,
        val_objects,
        network_factory=lambda: BiLstmNetwork(resolved_config),
        network_config=network_config,
        detector_factory=BiLstmDetector,
        hparams=hparams,
        max_epochs=max_epochs,
        seed=seed,
        window=window,
        stride=stride,
        batch_size=batch_size,
        threshold=threshold,
        accelerator=accelerator,
        deterministic=deterministic,
        progress=progress,
        early_stopping=early_stopping,
        val_benchmark=val_benchmark,
        patience=patience,
        optimizer="adam",
        scheduler="none",
        loss="bce",
        warmup_frac=0.0,
        focal_gamma=0.0,
        metadata=metadata,
    )


def train_transformer(
    train_objects: Sequence[ObjectSeries],
    val_objects: Sequence[ObjectSeries] = (),
    *,
    config: TransformerConfig | None = None,
    hparams: TrainHyperParams | None = None,
    max_epochs: int = 50,
    seed: int = 0,
    window: int | None = None,
    stride: int | None = None,
    batch_size: int = 64,
    threshold: float = 0.5,
    accelerator: str = "auto",
    deterministic: bool | Literal["warn"] = True,
    progress: bool = False,
    early_stopping: bool = False,
    val_benchmark: ValBenchmark | None = None,
    patience: int = 10,
    optimizer: OptimizerName = "adamw",
    scheduler: SchedulerName = "warmup_cosine",
    loss: LossName = "bce",
    warmup_frac: float = 0.1,
    focal_gamma: float = 2.0,
    metadata: dict[str, Any] | None = None,
) -> ModelBundle:
    """Train the transformer baseline and return its frozen :class:`ModelBundle`.

    The same seeded, single-GPU-budget call as :func:`train_bilstm` over the shared harness, with
    transformer-tuned optimisation defaults: ``adamw`` (decoupled weight decay), a
    ``warmup_cosine`` learning-rate schedule (``warmup_frac`` of the steps spent warming up before
    the cosine decay — transformers train unstably without warmup), and class-weighted ``bce`` by
    default. ``loss="focal"`` switches to the focal objective, whose focusing exponent is
    ``focal_gamma``; the exponent is carried regardless so flipping ``loss`` needs no other change.

    ``config`` sets the architecture (defaults to
    :class:`~maneuver_detect.models.transformer.TransformerConfig`, ≈9.5 M parameters); ``hparams``
    overrides the optimisation hyper-parameters (defaults to the
    train-split-derived positive-class weight plus ``warmup_frac`` / ``focal_gamma``). ``window`` /
    ``stride`` default to the frozen feature-layer geometry; the config's ``max_len`` must cover the
    window.

    ``deterministic`` is forwarded to the ``Trainer``: the default ``True`` gives reproducible
    training (the encoder, unlike the LSTM, has a deterministic backward). Should a CUDA attention
    kernel ever refuse under :func:`torch.use_deterministic_algorithms`, pass ``"warn"`` to fall
    back to seed-level reproducibility. ``progress`` enables the Lightning progress bar; checkpoint
    selection (``early_stopping`` / ``val_benchmark``) works exactly as in :func:`train_bilstm`.
    """
    from maneuver_detect.detectors.transformer import TransformerDetector
    from maneuver_detect.models.transformer import (
        NETWORK_KIND,
        TransformerConfig,
        TransformerNetwork,
    )

    resolved_config = config or TransformerConfig()
    network_config: dict[str, Any] = {"network": NETWORK_KIND, **resolved_config.to_dict()}

    return _train_sequence_detector(
        train_objects,
        val_objects,
        network_factory=lambda: TransformerNetwork(resolved_config),
        network_config=network_config,
        detector_factory=TransformerDetector,
        hparams=hparams,
        max_epochs=max_epochs,
        seed=seed,
        window=window,
        stride=stride,
        batch_size=batch_size,
        threshold=threshold,
        accelerator=accelerator,
        deterministic=deterministic,
        progress=progress,
        early_stopping=early_stopping,
        val_benchmark=val_benchmark,
        patience=patience,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        warmup_frac=warmup_frac,
        focal_gamma=focal_gamma,
        metadata=metadata,
    )
