"""The checkpoint bundle — network weights frozen together with the train-split normaliser.

A learned detector reproduces its training-time predictions only if it standardises inference inputs
with the **same** per-class statistics the model trained under (D11.3). A bare ``state_dict`` is
therefore not enough: the :class:`ModelBundle` ships the network weights, the frozen
:class:`~maneuver_detect.features.normalize.ClassNormaliser` statistics, the windowing parameters,
the detection threshold, and provenance metadata as one file, so loading it elsewhere — a CPU
runner, or from the Hub in a later release — rebuilds the exact inference pipeline.

The bundle is a flat dict saved with :func:`torch.save`. :func:`save_bundle` / :func:`load_bundle`
round-trip it; :func:`build_network` rebuilds the architecture from the stored network config (keyed
by its ``network`` tag) and loads the weights in eval mode on CPU. Inference needs only the bare
network and the normaliser — never Lightning — so the detector stays light.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from maneuver_detect.models.bilstm import NETWORK_KIND as BILSTM_KIND
from maneuver_detect.models.bilstm import build_bilstm

__all__ = ["ModelBundle", "build_network", "load_bundle", "save_bundle"]

# The network factories keyed by the bundle's ``network`` tag. The transformer baseline registers
# its own factory here so the same loader rebuilds either architecture.
_NETWORK_FACTORIES: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    BILSTM_KIND: build_bilstm,
}


@dataclass(frozen=True)
class ModelBundle:
    """A trained model and everything inference needs to reproduce it.

    Attributes:
        network_config: The architecture config, carrying a ``network`` tag that selects the rebuild
            factory (e.g. ``{"network": "bilstm", "n_channels": ..., "hidden_size": ...}``).
        state_dict: The network ``state_dict`` (the bare network's weights, not the module's).
        normaliser: The train-split :meth:`ClassNormaliser.to_dict` statistics.
        train_hparams: The optimisation hyper-parameters used (provenance).
        window: The window length the model trained on (used at inference).
        stride: The window stride (used at inference to reduce overlapping predictions).
        threshold: The per-gap maneuver-probability threshold the detector defaults to.
        metadata: Free-form provenance (seed, dataset version, measured training cost, scores).
    """

    network_config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    normaliser: dict[str, dict[str, list[float]]]
    train_hparams: dict[str, float]
    window: int
    stride: int
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    """Serialise ``bundle`` to ``path`` with :func:`torch.save`."""
    payload = {
        "network_config": bundle.network_config,
        "state_dict": bundle.state_dict,
        "normaliser": bundle.normaliser,
        "train_hparams": bundle.train_hparams,
        "window": bundle.window,
        "stride": bundle.stride,
        "threshold": bundle.threshold,
        "metadata": bundle.metadata,
    }
    torch.save(payload, Path(path))


def load_bundle(path: str | Path, *, map_location: str = "cpu") -> ModelBundle:
    """Load a :class:`ModelBundle` saved by :func:`save_bundle` (CPU by default)."""
    payload: dict[str, Any] = torch.load(Path(path), map_location=map_location, weights_only=False)
    return ModelBundle(
        network_config=payload["network_config"],
        state_dict=payload["state_dict"],
        normaliser=payload["normaliser"],
        train_hparams=payload["train_hparams"],
        window=int(payload["window"]),
        stride=int(payload["stride"]),
        threshold=float(payload["threshold"]),
        metadata=payload.get("metadata", {}),
    )


def build_network(bundle: ModelBundle) -> nn.Module:
    """Rebuild the bundle's network in eval mode on CPU, with its weights loaded.

    The ``network`` tag of :attr:`ModelBundle.network_config` selects the architecture factory; an
    unknown tag raises :class:`ValueError` listing the registered kinds.
    """
    kind = str(bundle.network_config.get("network", ""))
    factory = _NETWORK_FACTORIES.get(kind)
    if factory is None:
        raise ValueError(
            f"unknown network kind {kind!r}; registered kinds: {sorted(_NETWORK_FACTORIES)}"
        )
    network = factory(bundle.network_config)
    network.load_state_dict(bundle.state_dict)
    network.eval()
    return network
