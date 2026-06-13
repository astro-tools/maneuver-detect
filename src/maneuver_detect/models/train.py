"""The reproducible BiLSTM training entrypoint — seeded, single-GPU-budget, checkpoint out.

:func:`train_bilstm` is the shared harness assembled into one call: seed everything (D8), build the
:class:`~maneuver_detect.models.datamodule.ElementSeriesDataModule` over the labelled element
series, fit a :class:`~maneuver_detect.models.bilstm.BiLstmNetwork` through the shared
:class:`~maneuver_detect.models.module.SequenceDetectorModule`, and return a
:class:`~maneuver_detect.models.checkpoint.ModelBundle` that freezes the weights together with the
train-split normaliser so inference reproduces training-time standardisation.

Checkpoint selection has three modes. By default the **last** epoch's weights are frozen. With
``early_stopping`` the lowest-``val_loss`` (BCE) epoch is kept — but BCE bottoms out fast by going
conservative, which can undertrain the model for the benchmark. With ``val_benchmark`` the harness
instead scores the **val partition through the real benchmark each epoch** and keeps the best-recall
weights, so selection is aligned with the published metric rather than the surrogate.

Training honours the V7 budget — a single ``<= 24 GB`` GPU, hours of wall-clock — but the call is
device-agnostic (``accelerator="auto"``): it runs on CPU for the fast synthetic-data tests and on a
GPU for the real, credentialled reconstruction. The function is deterministic for a fixed seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch import nn

from maneuver_detect.models.bilstm import NETWORK_KIND, BiLstmConfig, BiLstmNetwork
from maneuver_detect.models.checkpoint import ModelBundle
from maneuver_detect.models.datamodule import ElementSeriesDataModule, ObjectSeries
from maneuver_detect.models.module import SequenceDetectorModule, TrainHyperParams

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from maneuver_detect.benchmark import TemporalSplit
    from maneuver_detect.features.normalize import ClassNormaliser
    from maneuver_detect.labels.record import ManeuverLabel

__all__ = ["ValBenchmark", "train_bilstm"]


@dataclass(frozen=True)
class ValBenchmark:
    """Inputs to select the checkpoint by VAL-split benchmark recall instead of BCE ``val_loss``.

    Attributes:
        series_by_norad: Each object's full reconstructed mean-element series (the val objects must
            be present; extra objects are ignored by the scorer).
        labels: The full label set — the scorer era-scopes it to the val partition.
        split: The leak-free temporal split whose VAL partition is scored each epoch.
    """

    series_by_norad: Mapping[int, pd.DataFrame]
    labels: Sequence[ManeuverLabel]
    split: TemporalSplit


class _RestoreBestWeights(Callback):
    """Keep the lowest-``val_loss`` network weights and restore them at the end of training.

    Lightning's default keeps the *last* epoch's weights; on a small dataset that is the most
    over-trained state. This callback snapshots the network ``state_dict`` (on CPU) whenever the
    validation loss improves, so :func:`train_bilstm` can restore the best epoch before freezing the
    bundle. The sanity-check pass (an untrained validation run before epoch 0) is skipped.
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
    highest-recall weights are snapshotted and restored before the bundle is frozen. Selection is
    therefore on the metric we report, not the BCE surrogate. ``patience`` early-stops when the val
    recall has not improved for that many epochs (a *recall* plateau, unlike the BCE one).
    """

    def __init__(
        self,
        spec: ValBenchmark,
        *,
        network_config: dict[str, Any],
        normaliser: ClassNormaliser,
        window: int,
        stride: int,
        threshold: float,
        patience: int,
    ) -> None:
        self._spec = spec
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
        from maneuver_detect.detectors.bilstm import BiLstmDetector
        from maneuver_detect.models.evaluate import score_on_temporal_split

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
            BiLstmDetector(bundle),
            self._spec.series_by_norad,
            self._spec.labels,
            self._spec.split,
            partition=SplitName.VAL,
        )
        score = _weighted_val_recall(report)
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


def _weighted_val_recall(report: Any) -> float:
    """Above-floor recall pooled across classes (weighted by above-floor label count).

    A single scalar to maximise: the label-count-weighted mean of the per-class recalls, i.e. the
    overall above-floor recall. Classes with no above-floor labels (or an undefined recall) are
    skipped; an empty population scores ``0.0``.
    """
    hit = 0.0
    total = 0
    for metrics in report.per_class.values():
        if metrics.recall is not None and metrics.n_labels_above_floor > 0:
            hit += metrics.recall * metrics.n_labels_above_floor
            total += metrics.n_labels_above_floor
    return hit / total if total else 0.0


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
    :class:`BiLstmConfig`); ``hparams`` the optimisation (defaults to the train-split-derived
    positive-class weight). ``window`` / ``stride`` default to the frozen feature-layer geometry.
    ``accelerator`` is passed to the Lightning ``Trainer`` — ``"auto"`` picks a GPU when present,
    CPU otherwise. ``metadata`` is merged into the bundle's provenance (the seed is added
    automatically).

    ``deterministic`` is forwarded to the ``Trainer``: the default ``True`` gives bit-exact
    reproducibility (CPU, and CUDA with ``CUBLAS_WORKSPACE_CONFIG`` set), but cuDNN's LSTM has no
    deterministic backward, so a GPU run that hits that error should pass ``"warn"`` to fall back to
    seed-level reproducibility (the seed is still recorded on the bundle).

    ``progress`` enables the Lightning progress bar (which surfaces the per-step train loss) and the
    model summary — useful for a long interactive run. It defaults to ``False`` so tests, CI, and
    headless runs stay quiet.

    Checkpoint selection (mutually exclusive, both default off → the last epoch is kept):

    * ``early_stopping`` monitors BCE ``val_loss``, early-stops on a ``patience`` plateau, and
      restores the best epoch (records ``best_val_loss``). Needs a validation set (``val_objects``).
    * ``val_benchmark`` scores the val partition through the real benchmark each epoch and keeps the
      best-recall weights, early-stopping on a recall plateau (records ``best_val_recall``). This
      aligns selection with the published metric instead of the BCE surrogate.
    """
    from maneuver_detect.features.windows import STRIDE, WINDOW

    resolved_window = WINDOW if window is None else window
    resolved_stride = STRIDE if stride is None else stride
    config = config or BiLstmConfig()

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

    hparams = hparams or TrainHyperParams(pos_weight=datamodule.positive_weight())
    network = BiLstmNetwork(config)
    module = SequenceDetectorModule(network, hparams)
    network_config: dict[str, Any] = {"network": NETWORK_KIND, **config.to_dict()}

    validate = bool(val_objects)
    callbacks: list[Callback] = []
    bce_best: _RestoreBestWeights | None = None
    bench_best: _ValBenchmarkSelection | None = None
    if early_stopping:
        if not validate:
            raise ValueError("early_stopping requires a validation set (val_objects)")
        bce_best = _RestoreBestWeights()
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=patience), bce_best]
    elif val_benchmark is not None:
        bench_best = _ValBenchmarkSelection(
            val_benchmark,
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

    provenance: dict[str, Any] = {"seed": seed}
    if metadata:
        provenance.update(metadata)
    if bce_best is not None:
        bce_best.restore(network)  # freeze the best epoch's weights, not the last
        if bce_best.best_score is not None:
            provenance["best_val_loss"] = bce_best.best_score
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
