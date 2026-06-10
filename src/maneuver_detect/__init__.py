"""maneuver-detect — detect orbital maneuvers from public TLE history.

The public surface is :func:`detect` and the :mod:`~maneuver_detect.datasets` accessor: hand
``detect`` a per-object mean-element TLE history and it returns a DataFrame of detected
maneuvers, each with a detection epoch, a calibrated confidence, a maneuver type, and a Δv
estimate. The canonical maneuver schema and the pluggable detector interface are defined by the
core layer; this module pins the import-time surface the rest of the package builds on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from maneuver_detect import datasets

if TYPE_CHECKING:
    import pandas as pd

__version__ = "0.1.0"

__all__ = ["__version__", "datasets", "detect"]


def detect(history: object, model: str = "classical") -> pd.DataFrame:
    """Detect maneuvers in a per-object mean-element TLE ``history``.

    Dispatches to the named detector and returns the canonical maneuver DataFrame
    (``epoch``, ``confidence``, ``type``, ``delta_v_estimate``, plus provenance). The
    classical reference detector is the default; learned models are selected by name.

    The dispatch and the detector implementations live in the detector layer.
    """
    raise NotImplementedError("The detector layer is not implemented yet.")
