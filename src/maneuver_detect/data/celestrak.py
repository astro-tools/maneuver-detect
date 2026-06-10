"""CelesTrak fetcher — the no-auth, current-GP catalogue source.

CelesTrak's ``gp.php`` endpoint serves the **current** General Perturbations element set for an
object, not a historical archive: a fetch returns at most the latest elset, which the data layer
treats as a one-point (or empty) history filtered to the requested window. Multi-year history for
training comes from Space-Track (:mod:`~maneuver_detect.data.spacetrack`); CelesTrak is the
no-credential way to get an object's freshest elset and to extend a Space-Track series go-forward.

Discipline for CelesTrak's soft ~100 MB/day per-IP cap and one-download-per-update policy is two
things working together: the on-disk cache (a repeat fetch inside the TTL never touches the
network) and an ``If-Modified-Since`` conditional GET when a cached-but-expired entry exists — a
``304 Not Modified`` refreshes the entry's timestamp without re-downloading the body. A rate
limiter paces whatever requests remain.

Failure handling: an outage with a cached value serves the cached value flagged ``stale=True``;
an outage with no cache raises :class:`~maneuver_detect.errors.DataSourceError`.
"""

from __future__ import annotations

from datetime import datetime
from email.utils import format_datetime
from typing import ClassVar

import httpx

from maneuver_detect.data.base import FetchResult, _CatalogueFetcher, normalise_range
from maneuver_detect.errors import DataSourceError

__all__ = ["CelestrakFetcher"]

_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"


class CelestrakFetcher(_CatalogueFetcher):
    """Fetch current-GP elsets from CelesTrak, with caching and conditional-GET revalidation.

    Constructing the fetcher does no network I/O; the HTTP client is built lazily on the first
    fetch. The ``cache`` / ``client`` / ``rate_limiter`` parameters are injection points for tests
    (a ``tmp_path`` cache, a :class:`httpx.MockTransport` client, a fake-clock limiter);
    production callers pass none and get the shared singletons and a light default rate floor.
    """

    source: ClassVar[str] = "celestrak"
    # CelesTrak's discipline is served by the cache; a light request floor is belt-and-braces.
    default_min_interval_s: ClassVar[float] = 1.0

    def fetch(
        self,
        norad_id: int,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> FetchResult:
        """Fetch the current GP elset for ``norad_id``, filtered to ``[start, end]``.

        CelesTrak holds no history, so a closed window entirely in the past yields no elsets — use
        Space-Track for that range. Cache hit → network (conditional GET) → stale fallback.
        """
        lo, hi = normalise_range(start, end)
        key = f"catnr:{norad_id}"

        hit = self._cache.get(self.source, key, ttl_s=self._ttl_s)
        if hit is not None:
            elsets = self._parse_payload(hit.value, norad_id)
            return self._build_result(norad_id, elsets, hit.fetched_at, lo, hi, stale=False)

        stale_hit = self._cache.get_stale(self.source, key)
        self._rate_limiter.acquire()
        client = self._ensure_client()
        request_headers: dict[str, str] = {}
        if stale_hit is not None:
            request_headers["If-Modified-Since"] = format_datetime(
                stale_hit.fetched_at, usegmt=True
            )

        try:
            response = client.get(
                _GP_URL,
                params={"CATNR": str(norad_id), "FORMAT": "json"},
                headers=request_headers,
            )
            if response.status_code == 304 and stale_hit is not None:
                # Unchanged since the cached fetch — refresh the timestamp, serve cached as fresh,
                # and never re-download the body (the whole point of the conditional GET).
                self._cache.put(self.source, key, stale_hit.value)
                elsets = self._parse_payload(stale_hit.value, norad_id)
                fetched_at = self._fetched_at_after_put(key)
                return self._build_result(norad_id, elsets, fetched_at, lo, hi, stale=False)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            served = self._serve_stale(norad_id, stale_hit, lo, hi)
            if served is not None:
                return served
            raise DataSourceError(
                f"CelesTrak unreachable for NORAD {norad_id}: {exc}", source=self.source
            ) from exc

        # Validate before caching so a malformed payload never poisons the on-disk cache.
        elsets = self._parse_payload(payload, norad_id)
        self._cache.put(self.source, key, payload)
        fetched_at = self._fetched_at_after_put(key)
        return self._build_result(norad_id, elsets, fetched_at, lo, hi, stale=False)
