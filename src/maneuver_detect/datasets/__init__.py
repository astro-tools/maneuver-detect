"""Dataset accessors and the reconstructable labelled dataset.

``tle_history`` is the per-object accessor — the cleaned mean-element series for one NORAD id. The
recipe / manifest / reconstruction surface assembles the full **labelled** dataset from a
pinned :class:`Recipe` (D2): each series is re-fetched and re-derived locally, then verified
byte-for-byte against a content-hash :class:`Manifest` (D8). The raw catalogue data is never shipped
— only the recipe parameters, the open labels, and the per-series digests. The benchmark release
adds the labelled train / val / test splits on top of this.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from maneuver_detect.data import DEFAULT_SOURCE
from maneuver_detect.datasets.catalogue import (
    DATASET_VERSION,
    GPS_CONSTELLATION,
    GpsSatellite,
    gps_svn_to_norad,
    recipe,
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

    from maneuver_detect.labels.record import ManeuverLabel

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
    "fetch_dataset",
    "gps_svn_to_norad",
    "load_labels",
    "load_manifest",
    "load_recipe",
    "recipe",
    "reconstruct",
    "series_sha256",
    "tle_history",
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


def fetch_dataset(*, revision: str | None = None) -> Path:
    """Download the published dataset from the Hub and return the local snapshot directory.

    Pulls the recipe, labels, manifest, and splits (and the dataset card) from the Hugging Face Hub
    dataset repo, cached on disk — so the distributable dataset is available on first use without
    cloning the repository. ``revision`` defaults to the lockstep release tag (see
    :func:`maneuver_detect.hub.hub_revision`); the raw element series is **not** shipped (D2) — use
    :func:`reconstruct` against the downloaded recipe to rebuild it locally. Raises
    :class:`~maneuver_detect.hub.HubError` if the download fails.
    """
    from maneuver_detect import hub

    return hub.fetch_dataset(revision=revision)


def load_recipe(*, revision: str | None = None) -> Recipe:
    """Download and parse the published reconstruction recipe from the Hub."""
    from maneuver_detect import hub

    text = hub.dataset_path("recipe.json", revision=revision).read_text(encoding="utf-8")
    return Recipe.from_json(text)


def load_manifest(*, revision: str | None = None) -> Manifest:
    """Download and parse the published content-hash manifest from the Hub."""
    from maneuver_detect import hub

    text = hub.dataset_path("manifest.json", revision=revision).read_text(encoding="utf-8")
    return Manifest.from_json(text)


def load_labels(*, revision: str | None = None) -> list[ManeuverLabel]:
    """Download and parse the published maneuver labels from the Hub."""
    from maneuver_detect import hub
    from maneuver_detect.datasets.build import labels_from_json

    text = hub.dataset_path("labels.json", revision=revision).read_text(encoding="utf-8")
    return labels_from_json(text)
