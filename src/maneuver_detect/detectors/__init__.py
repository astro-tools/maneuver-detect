"""Maneuver detectors — one module per detector, behind a common interface and registry.

Every detector consumes a per-object mean-element series and returns the canonical maneuver
schema. The classical reference detector (Holt-Winters smoothing + rule-based jump detection +
the Δv inversion) is the baseline every learned model must beat; the learned baselines arrive on
top of the same interface. Detectors register themselves under a name with
:func:`register_detector`, and :func:`maneuver_detect.detect` dispatches on that name.
"""

from __future__ import annotations

from maneuver_detect.detectors.base import Detector

__all__ = [
    "BiLstmDetector",
    "ClassicalDetector",
    "Detector",
    "available_models",
    "get_detector",
    "register_detector",
]

_REGISTRY: dict[str, type[Detector]] = {}


def register_detector(cls: type[Detector]) -> type[Detector]:
    """Register a :class:`Detector` subclass under its :attr:`~Detector.name` for dispatch.

    Usable as a class decorator. Raises :class:`ValueError` if a *different* detector is already
    registered under the same name.
    """
    name = cls.name
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"a different detector is already registered under {name!r}")
    _REGISTRY[name] = cls
    return cls


def get_detector(model: str) -> Detector:
    """Instantiate the registered detector named ``model``.

    Raises :class:`ValueError` if no detector is registered under that name, listing the names
    that are available.
    """
    try:
        cls = _REGISTRY[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; available models: {available_models()}"
        ) from None
    return cls()


def available_models() -> list[str]:
    """Return the sorted names of all registered detectors."""
    return sorted(_REGISTRY)


# Register the built-in detectors. The imports sit at the foot of the module — after the registry
# helpers — so importing the package both exposes the registry API and registers the detectors,
# making them available to ``detect()`` without the caller importing their modules. The learned
# detector defers its torch / model imports to construction time, so registering it here keeps
# ``import maneuver_detect`` (and the classical path) free of the modelling stack.
from maneuver_detect.detectors.bilstm import BiLstmDetector  # noqa: E402
from maneuver_detect.detectors.classical import ClassicalDetector  # noqa: E402

register_detector(ClassicalDetector)
register_detector(BiLstmDetector)
