"""Dataset accessors and the reconstructable v0.1 dataset.

``tle_history`` is the per-object accessor — the cleaned mean-element series for one NORAD id. The
recipe / manifest / reconstruction surface assembles the full v0.1 **labelled** dataset from a
pinned :class:`Recipe` (D2): each series is re-fetched and re-derived locally, then verified
byte-for-byte against a content-hash :class:`Manifest` (D8). The raw catalogue data is never shipped
— only the recipe parameters, the open labels, and the per-series digests. The benchmark release
adds the labelled train / val / test splits on top of this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from maneuver_detect.data import DEFAULT_SOURCE
from maneuver_detect.datasets.catalogue import (
    DATASET_VERSION,
    GPS_CONSTELLATION,
    GpsSatellite,
    gps_svn_to_norad,
    v01_recipe,
)
from maneuver_detect.datasets.manifest import Manifest, SeriesDigest, series_sha256
from maneuver_detect.datasets.recipe import Recipe, RecipeEntry
from maneuver_detect.datasets.reconstruct import (
    LabelledDataset,
    ObjectDataset,
    reconstruct,
    verify,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "DATASET_VERSION",
    "GPS_CONSTELLATION",
    "GpsSatellite",
    "LabelledDataset",
    "Manifest",
    "ObjectDataset",
    "Recipe",
    "RecipeEntry",
    "SeriesDigest",
    "gps_svn_to_norad",
    "reconstruct",
    "series_sha256",
    "tle_history",
    "v01_recipe",
    "verify",
]


def tle_history(
    norad_id: int,
    *,
    start: str | None = None,
    end: str | None = None,
    source: str = DEFAULT_SOURCE,
) -> pd.DataFrame:
    """Return the cleaned mean-element TLE history for ``norad_id`` as a DataFrame.

    ``start`` and ``end`` bound the epoch range (ISO-8601); when omitted, the full available
    history is returned. ``source`` selects the catalogue (``"spacetrack"`` for the credentialled
    ``gp_history`` archive — the default and the only source with multi-epoch history — or
    ``"celestrak"`` for the no-auth current GP elset); an unknown ``source`` raises
    :class:`ValueError`. Fetching, caching, and cleaning live in the data layer: the returned frame
    carries the canonical :data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS`, the same shape
    the detector consumes.

    Raises :class:`~maneuver_detect.errors.MissingCredentialError` when the Space-Track source is
    used without credentials, and :class:`~maneuver_detect.errors.DataSourceError` when the source
    is unreachable with nothing cached to fall back on.
    """
    from maneuver_detect.data import build_series, get_fetcher

    with get_fetcher(source) as fetcher:
        result = fetcher.fetch(norad_id, start=start, end=end)
    return build_series(result.elsets)
