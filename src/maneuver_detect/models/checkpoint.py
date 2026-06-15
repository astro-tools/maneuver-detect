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
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from maneuver_detect.calibration import BundledCalibration

from maneuver_detect.errors import ManeuverDetectError
from maneuver_detect.models.bilstm import NETWORK_KIND as BILSTM_KIND
from maneuver_detect.models.bilstm import build_bilstm
from maneuver_detect.models.transformer import NETWORK_KIND as TRANSFORMER_KIND
from maneuver_detect.models.transformer import build_transformer

__all__ = ["ModelBundle", "build_network", "load_bundle", "save_bundle"]

# Everything inference needs from a saved bundle. ``metadata`` is optional (provenance only); these
# are not, so a bundle missing any of them is a malformed / truncated / version-mismatched artifact.
_REQUIRED_KEYS = (
    "network_config",
    "state_dict",
    "normaliser",
    "train_hparams",
    "window",
    "stride",
    "threshold",
)

# The network factories keyed by the bundle's ``network`` tag, so the same loader rebuilds either
# architecture from its stored config.
_NETWORK_FACTORIES: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    BILSTM_KIND: build_bilstm,
    TRANSFORMER_KIND: build_transformer,
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
        threshold: The per-gap maneuver-probability threshold the detector defaults to — the gate
            for any orbit class absent from ``class_thresholds``.
        class_thresholds: Optional per-orbit-class detection thresholds (``OrbitClass`` value →
            gate), so GEO can take a lower gate than LEO/MEO. Empty by default — a scalar
            ``threshold`` then gates every class — so a checkpoint saved before per-class tuning
            loads and behaves unchanged.
        metadata: Free-form provenance (seed, dataset version, measured training cost, scores).
        calibration: The fitted, val-only uncertainty calibration baked into the bundle (a
            :class:`~maneuver_detect.calibration.BundledCalibration`), applied to the detector's
            emitted ``confidence`` at inference. ``None`` by default — a checkpoint saved before
            calibration loads and emits raw confidence unchanged.
    """

    network_config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    normaliser: dict[str, dict[str, list[float]]]
    train_hparams: dict[str, float]
    window: int
    stride: int
    threshold: float
    class_thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    calibration: BundledCalibration | None = None


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
        "class_thresholds": bundle.class_thresholds,
        "metadata": bundle.metadata,
        "calibration": None if bundle.calibration is None else bundle.calibration.to_dict(),
    }
    torch.save(payload, Path(path))


def load_bundle(path: str | Path, *, map_location: str = "cpu") -> ModelBundle:
    """Load a :class:`ModelBundle` saved by :func:`save_bundle` (CPU by default).

    Raises :class:`~maneuver_detect.errors.ManeuverDetectError` if the file is not a bundle dict, or
    is missing any key inference needs — so a truncated or version-mismatched Hub artifact surfaces
    as a clear error naming the path and the missing fields, not a bare ``KeyError``.
    """
    payload: dict[str, Any] = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ManeuverDetectError(
            f"checkpoint at {Path(path)} is not a model bundle "
            f"(expected a dict, got {type(payload).__name__})"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise ManeuverDetectError(
            f"checkpoint at {Path(path)} is missing required bundle keys: {missing}"
        )
    return ModelBundle(
        network_config=payload["network_config"],
        state_dict=payload["state_dict"],
        normaliser=payload["normaliser"],
        train_hparams=payload["train_hparams"],
        window=int(payload["window"]),
        stride=int(payload["stride"]),
        threshold=float(payload["threshold"]),
        # Not a required key: a checkpoint saved before per-class tuning has none, and the scalar
        # threshold then gates every class (the back-compatible fallback).
        class_thresholds={
            str(key): float(value) for key, value in payload.get("class_thresholds", {}).items()
        },
        metadata=payload.get("metadata", {}),
        # Not a required key either: a checkpoint saved before calibration has none, and the
        # detector emits raw confidence (the back-compatible fallback).
        calibration=_load_calibration(payload.get("calibration")),
    )


def _load_calibration(data: Any) -> BundledCalibration | None:
    """Reconstruct a bundle's :class:`BundledCalibration` from its payload dict (``None`` if absent)."""
    if data is None:
        return None
    from maneuver_detect.calibration import BundledCalibration

    return BundledCalibration.from_dict(data)


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
