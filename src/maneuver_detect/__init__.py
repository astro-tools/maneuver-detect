"""maneuver-detect — detect orbital maneuvers from public TLE history.

The public surface is :func:`detect` and the :mod:`~maneuver_detect.datasets` accessor: hand
``detect`` a per-object mean-element TLE history and it returns the canonical maneuver DataFrame
(see :mod:`maneuver_detect.schema`) — each row a detected maneuver with a detection epoch, a
calibrated confidence, a maneuver type, and a Δv estimate. Detectors implement the
:class:`~maneuver_detect.detectors.Detector` interface and register under a name; ``detect``
dispatches on that name.
"""

from __future__ import annotations

import pandas as pd

from maneuver_detect import datasets
from maneuver_detect.detectors import Detector, available_models, get_detector
from maneuver_detect.schema import Maneuver, ManeuverType

__version__ = "0.1.0"

__all__ = [
    "Detector",
    "Maneuver",
    "ManeuverType",
    "__version__",
    "available_models",
    "datasets",
    "detect",
]


def detect(history: pd.DataFrame, model: str = "classical") -> pd.DataFrame:
    """Detect maneuvers in a per-object mean-element TLE ``history``.

    Dispatches to the named detector and returns the canonical maneuver DataFrame (``epoch``,
    ``confidence``, ``type``, ``delta_v_estimate``, plus provenance — see
    :mod:`maneuver_detect.schema`). The classical reference detector is the default; learned
    models are selected by name. Raises :class:`ValueError` for an unknown ``model`` — see
    :func:`~maneuver_detect.detectors.available_models` for the registered names.
    """
    return get_detector(model).detect(history)
