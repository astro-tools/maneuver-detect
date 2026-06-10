"""The detector interface every maneuver detector implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

__all__ = ["Detector"]


class Detector(ABC):
    """Abstract base class for maneuver detectors.

    A detector consumes a per-object mean-element TLE history and returns the canonical maneuver
    DataFrame (see :mod:`maneuver_detect.schema`), so the classical reference detector and future
    learned detectors are interchangeable. Subclasses set :attr:`name` — the key
    :func:`maneuver_detect.detect` dispatches on — and implement :meth:`detect`.
    """

    name: ClassVar[str]

    @abstractmethod
    def detect(self, history: pd.DataFrame) -> pd.DataFrame:
        """Detect maneuvers in ``history`` and return the canonical maneuver DataFrame."""
        raise NotImplementedError
