"""The stable internal interface every catalogue fetcher implements, plus shared machinery.

A :class:`Fetcher` takes a NORAD id and an optional epoch window and returns a :class:`FetchResult`
— a sequence of parsed :class:`~maneuver_detect.data.elset.Elset` records ordered by epoch, plus
the freshness metadata (when it was fetched, and whether it was served stale from cache during an
outage). The cleaning and assembly layer consumes this interface, not the concrete fetchers, so a
source can be swapped or a future ``orbit-formats``-backed reader dropped in without touching it.

:class:`_CatalogueFetcher` carries the parts CelesTrak and Space-Track share — cache lookup, OMM
parsing, epoch-window filtering, the stale-fallback rebuild, and the HTTP client lifecycle — so
each concrete fetcher only writes its own ``fetch`` (the part that genuinely differs: CelesTrak's
conditional GET vs. Space-Track's login + ``gp_history`` query).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from types import TracebackType
from typing import ClassVar, TypeVar

import httpx

from maneuver_detect.__about__ import __version__
from maneuver_detect.data.cache import DEFAULT_TTLS, Cache, CacheHit, default_cache
from maneuver_detect.data.elset import Elset, from_omm
from maneuver_detect.data.ratelimit import RateLimiter
from maneuver_detect.errors import DataSourceError

__all__ = ["Bound", "FetchResult", "Fetcher", "in_range", "normalise_range", "parse_bound"]

_logger = logging.getLogger(__name__)

# A date-range bound accepted by the public surface: an ISO-8601 string, a datetime, or None.
Bound = str | datetime | None

_F = TypeVar("_F", bound="Fetcher")

_HTTP_TIMEOUT = 30.0
# A real project name + contact URL keeps Cloudflare (fronting celestrak.org) from downgrading the
# bot-score the way it does for opaque defaults like ``python-httpx/x.y``; harmless on Space-Track.
_USER_AGENT = f"maneuver-detect/{__version__} (+https://github.com/astro-tools/maneuver-detect)"
_HTTP_HEADERS = {"User-Agent": _USER_AGENT}


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class FetchResult:
    """The result of a fetch: the elsets in the window plus freshness metadata.

    Attributes:
        norad_id: The object the elsets belong to.
        elsets: The parsed element sets in the requested window, ordered by epoch ascending.
        fetched_at: When the served data was fetched from the source (timezone-aware UTC). For a
            stale result this is the *original* fetch time, not now.
        source: The source name (``"celestrak"`` / ``"spacetrack"``).
        stale: ``True`` when the source was unreachable and a cached value was served instead —
            the explicit freshness flag a caller checks before trusting recency.
    """

    norad_id: int
    elsets: tuple[Elset, ...]
    fetched_at: datetime
    source: str
    stale: bool = False


class Fetcher(ABC):
    """Abstract catalogue fetcher: NORAD id + epoch window in, :class:`FetchResult` out.

    Concrete fetchers own a cache, a rate limiter, and (lazily) an HTTP client; constructing one
    does no network I/O, and neither does importing its module. Use as a context manager so the
    HTTP client is closed on exit.
    """

    source: ClassVar[str]

    @abstractmethod
    def fetch(
        self,
        norad_id: int,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> FetchResult:
        """Fetch the elset history for ``norad_id`` within ``[start, end]`` (both optional)."""
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027  (intentional concrete no-op default, not abstract)
        """Release any HTTP resources the fetcher owns. Idempotent; the base is a no-op."""

    def __enter__(self: _F) -> _F:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def parse_bound(value: str | datetime | None) -> datetime | None:
    """Normalise an epoch-range bound to a timezone-aware UTC datetime (or ``None``).

    Accepts an ISO-8601 string (``T`` or space separator, optional trailing ``Z``) or a datetime;
    a naive value is assumed UTC. Raises :class:`ValueError` on an unparseable string.

    Bounds are **exact instants**: a date-only value resolves to ``00:00:00`` UTC (the start of that
    day). Because :func:`in_range` is inclusive on both ends, a date-only ``end`` therefore excludes
    everything after midnight on that day — to cover a whole final day, pass an explicit end-of-day
    time (``"2024-06-30T23:59:59Z"``) or the following day's date.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip().replace(" ", "T")
        if text.endswith("Z"):
            text = text[:-1]
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError(f"could not parse epoch bound as ISO-8601: {value!r}") from None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalise_range(
    start: str | datetime | None, end: str | datetime | None
) -> tuple[datetime | None, datetime | None]:
    """Parse ``start`` / ``end`` to UTC datetimes, asserting ``start <= end`` when both given."""
    lo = parse_bound(start)
    hi = parse_bound(end)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"start {lo.isoformat()} is after end {hi.isoformat()}")
    return lo, hi


def in_range(epoch: datetime, lo: datetime | None, hi: datetime | None) -> bool:
    """Return whether ``epoch`` is within the inclusive ``[lo, hi]`` window (open-ended sides)."""
    if lo is not None and epoch < lo:
        return False
    return not (hi is not None and epoch > hi)


class _CatalogueFetcher(Fetcher):
    """Shared caching / OMM-parsing / stale-fallback machinery for the HTTP catalogue fetchers.

    Subclasses set :attr:`source` and :attr:`default_min_interval_s` and implement :meth:`fetch`,
    reusing :meth:`_parse_payload`, :meth:`_build_result`, :meth:`_serve_stale`, and the cache /
    rate-limiter / HTTP-client members this base wires up.
    """

    default_min_interval_s: ClassVar[float]

    def __init__(
        self,
        *,
        cache: Cache | None = None,
        client: httpx.Client | None = None,
        rate_limiter: RateLimiter | None = None,
        ttl_s: float | None = None,
    ) -> None:
        self._cache = cache if cache is not None else default_cache()
        self._client = client
        self._owns_client = client is None
        self._rate_limiter = (
            rate_limiter if rate_limiter is not None else RateLimiter(self.default_min_interval_s)
        )
        self._ttl_s = ttl_s if ttl_s is not None else DEFAULT_TTLS[self.source]

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=_HTTP_TIMEOUT, headers=_HTTP_HEADERS)
        return self._client

    def _fetched_at_after_put(self, key: str) -> datetime:
        """Read back the timestamp ``put`` stamped, falling back to now on a disabled cache.

        Uses ``get_stale`` (not ``get``): the just-written entry is the answer regardless of the
        TTL, so the reported ``fetched_at`` reflects the real cache write even for a tiny TTL.
        """
        refreshed = self._cache.get_stale(self.source, key)
        return refreshed.fetched_at if refreshed is not None else _utcnow()

    def _serve_stale(
        self,
        norad_id: int,
        stale_hit: CacheHit | None,
        lo: datetime | None,
        hi: datetime | None,
    ) -> FetchResult | None:
        """Serve a cached value during an outage, or ``None`` if absent / it won't rebuild."""
        if stale_hit is None:
            return None
        try:
            elsets = self._parse_payload(stale_hit.value, norad_id)
        except DataSourceError as exc:
            # "Outage beats hard error" only holds if the stale value still parses; if the schema
            # tightened since it was written, fall through to the typed unreachable error.
            _logger.warning(
                "%s stale cache entry for NORAD %s is unusable: %s", self.source, norad_id, exc
            )
            return None
        return self._build_result(norad_id, elsets, stale_hit.fetched_at, lo, hi, stale=True)

    def _build_result(
        self,
        norad_id: int,
        elsets: list[Elset],
        fetched_at: datetime,
        lo: datetime | None,
        hi: datetime | None,
        *,
        stale: bool,
    ) -> FetchResult:
        windowed = tuple(
            sorted((e for e in elsets if in_range(e.epoch, lo, hi)), key=lambda e: e.epoch)
        )
        return FetchResult(
            norad_id=norad_id,
            elsets=windowed,
            fetched_at=fetched_at,
            source=self.source,
            stale=stale,
        )

    def _parse_payload(self, payload: object, norad_id: int) -> list[Elset]:
        """Parse an OMM-array payload to elsets, raising :class:`DataSourceError` on bad shape."""
        if not isinstance(payload, list):
            raise DataSourceError(
                f"{self.source} returned a non-list payload of type {type(payload).__name__} "
                f"for NORAD {norad_id}",
                source=self.source,
            )
        elsets: list[Elset] = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise DataSourceError(
                    f"{self.source} record at index {i} is not an object for NORAD {norad_id}",
                    source=self.source,
                )
            try:
                elsets.append(from_omm(item))
            except ValueError as exc:
                raise DataSourceError(
                    f"{self.source} returned a malformed OMM record at index {i} "
                    f"for NORAD {norad_id}: {exc}",
                    source=self.source,
                ) from exc
        return elsets
