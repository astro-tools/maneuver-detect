"""Dataset accessors — the entry point to TLE histories and the curated benchmark dataset.

``tle_history`` fetches, cleans, and assembles a per-object mean-element series from the public
catalog; the benchmark release adds the labelled train / val / test splits on top. The data and
benchmark layers fill these in. This module pins the accessor surface that
:func:`maneuver_detect.detect` and the CLI build on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["tle_history"]


def tle_history(norad_id: int, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Return the cleaned mean-element TLE history for ``norad_id`` as a DataFrame.

    ``start`` and ``end`` bound the epoch range (ISO-8601); when omitted, the full available
    history is returned. Fetching, caching, and cleaning live in the data layer.
    """
    raise NotImplementedError("The data layer is not implemented yet.")
