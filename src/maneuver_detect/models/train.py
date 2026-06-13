"""The reproducible BiLSTM training entrypoint — seeded, single-GPU-budget, checkpoint out.

:func:`train_bilstm` is the shared harness assembled into one call: seed everything (D8), build the
:class:`~maneuver_detect.models.datamodule.ElementSeriesDataModule` over the labelled element
series, fit a :class:`~maneuver_detect.models.bilstm.BiLstmNetwork` through the shared
:class:`~maneuver_detect.models.module.SequenceDetectorModule`, and return a
:class:`~maneuver_detect.models.checkpoint.ModelBundle` that freezes the weights together with the
train-split normaliser so inference reproduces training-time standardisation.

Training honours the V7 budget — a single ``<= 24 GB`` GPU, hours of wall-clock — but the call is
device-agnostic (``accelerator="auto"``): it runs on CPU for the fast synthetic-data tests and on a
GPU for the real, credentialled reconstruction. The function is deterministic for a fixed seed, so a
recorded run reproduces; the measured wall-clock / memory / benchmark numbers belong on the model
card, written by the offline run that has the real data and the GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import lightning as L

from maneuver_detect.models.bilstm import NETWORK_KIND, BiLstmConfig, BiLstmNetwork
from maneuver_detect.models.checkpoint import ModelBundle
from maneuver_detect.models.datamodule import ElementSeriesDataModule, ObjectSeries
from maneuver_detect.models.module import SequenceDetectorModule, TrainHyperParams

__all__ = ["train_bilstm"]


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
    metadata: dict[str, Any] | None = None,
) -> ModelBundle:
    """Train a BiLSTM on ``train_objects`` and return its frozen :class:`ModelBundle`.

    The run is seeded (``seed``) and deterministic. ``config`` sets the architecture (defaults to
    :class:`BiLstmConfig`); ``hparams`` the optimisation (defaults to the train-split-derived
    positive-class weight). ``window`` / ``stride`` default to the frozen feature-layer geometry.
    ``accelerator`` is passed to the Lightning ``Trainer`` — ``"auto"`` picks a GPU when present,
    CPU otherwise. ``metadata`` is merged into the bundle's provenance (the seed is added
    automatically).
    """
    from maneuver_detect.features.windows import STRIDE, WINDOW

    resolved_window = WINDOW if window is None else window
    resolved_stride = STRIDE if stride is None else stride
    config = config or BiLstmConfig()

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

    validate = bool(val_objects)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        # No validation set: skip validation entirely (the data module returns an empty val loader).
        limit_val_batches=1.0 if validate else 0,
        num_sanity_val_steps=2 if validate else 0,
    )
    trainer.fit(module, datamodule=datamodule)

    provenance: dict[str, Any] = {"seed": seed}
    if metadata:
        provenance.update(metadata)
    return ModelBundle(
        network_config={"network": NETWORK_KIND, **config.to_dict()},
        state_dict=network.state_dict(),
        normaliser=datamodule.normaliser.to_dict(),
        train_hparams=hparams.to_dict(),
        window=resolved_window,
        stride=resolved_stride,
        threshold=threshold,
        metadata=provenance,
    )
