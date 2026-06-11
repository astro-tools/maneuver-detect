"""The data layer — public-catalog fetch, clean, and per-object mean-element assembly.

Catalogue fetchers with on-disk caching and rate-limit discipline, elset cleaning (epoch
dedup, bad-elset rejection, gap handling), and per-NORAD mean-element time-series assembly.
Built behind a stable internal interface so the historical-TLE pipeline can later be extracted.

A :class:`Fetcher` returns a :class:`FetchResult` of parsed :class:`Elset` records for a NORAD id
and epoch window. :class:`CelestrakFetcher` is the no-auth current-GP source;
:class:`SpacetrackFetcher` is the credentialled ``gp_history`` archive. Both share the XDG
:class:`Cache` and the rate-limit / stale-fallback discipline. :func:`clean_elsets` filters and
dedups a fetched series, and :func:`build_series` assembles it into the canonical mean-element
DataFrame (:data:`MEAN_ELEMENT_COLUMNS`) the features and detectors consume.
"""

from __future__ import annotations

from maneuver_detect.data.base import Bound, Fetcher, FetchResult
from maneuver_detect.data.cache import Cache
from maneuver_detect.data.celestrak import CelestrakFetcher
from maneuver_detect.data.clean import clean_elsets, is_valid_elset
from maneuver_detect.data.elset import Elset, from_omm, from_tle, read_tle_file
from maneuver_detect.data.history import MEAN_ELEMENT_COLUMNS, assemble, build_series
from maneuver_detect.data.ratelimit import RateLimiter
from maneuver_detect.data.spacetrack import SpacetrackFetcher

__all__ = [
    "DEFAULT_SOURCE",
    "FETCHERS",
    "MEAN_ELEMENT_COLUMNS",
    "Bound",
    "Cache",
    "CelestrakFetcher",
    "Elset",
    "FetchResult",
    "Fetcher",
    "RateLimiter",
    "SpacetrackFetcher",
    "assemble",
    "build_series",
    "clean_elsets",
    "from_omm",
    "from_tle",
    "get_fetcher",
    "is_valid_elset",
    "read_tle_file",
]

#: The selectable catalogue sources, keyed by each fetcher's canonical ``source`` name. CelesTrak
#: serves the current GP elset without credentials; Space-Track serves the credentialled
#: ``gp_history`` archive — the only source with the multi-epoch history maneuver detection needs.
FETCHERS: dict[str, type[Fetcher]] = {
    CelestrakFetcher.source: CelestrakFetcher,
    SpacetrackFetcher.source: SpacetrackFetcher,
}

#: The default source for a live fetch — the historical archive, so a NORAD-id detection has a
#: real series to work on rather than a single current elset.
DEFAULT_SOURCE = SpacetrackFetcher.source


def get_fetcher(source: str) -> Fetcher:
    """Instantiate the catalogue fetcher registered under ``source``.

    ``source`` is one of the keys of :data:`FETCHERS` (``"celestrak"`` / ``"spacetrack"``); an
    unknown name raises :class:`ValueError` listing the available sources. The fetcher is built
    with its production defaults (shared cache, default rate floor; Space-Track resolves
    credentials from the environment on its first network call).
    """
    try:
        fetcher_cls = FETCHERS[source]
    except KeyError:
        raise ValueError(
            f"unknown source {source!r}; available sources: {sorted(FETCHERS)}"
        ) from None
    return fetcher_cls()
