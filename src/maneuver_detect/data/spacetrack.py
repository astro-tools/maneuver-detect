"""Space-Track fetcher — the credentialled historical-archive source.

Space-Track's ``gp_history`` class is the multi-year back-element archive the dataset's training
history is reconstructed from (per the recipe-first distribution model, D2). This fetcher queries
it by NORAD id and epoch window, authenticating with the user's own Space-Track account — read
from the environment (:func:`~maneuver_detect.data.credentials.require_spacetrack_credential`),
never shipped or proxied. A query attempted without credentials raises a typed
:class:`~maneuver_detect.errors.MissingCredentialError` before any network call.

The session cookie is reused across fetches on one fetcher instance, so a dataset build pulling
many objects logs in once rather than per object — Space-Track's API Rules of Behaviour ask for
exactly this stewardship, alongside the cache (its "save it locally; do not query for the same
data repeatedly" guidance) and the rate limiter (its query-rate limits). A silently-expired
session (401/403) triggers one re-login and retry.

Failure handling mirrors CelesTrak: an outage with a cached value serves it flagged
``stale=True``; an outage with no cache, or a refused credential, raises
:class:`~maneuver_detect.errors.DataSourceError`.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

import httpx

from maneuver_detect.data.base import FetchResult, _CatalogueFetcher, normalise_range
from maneuver_detect.data.cache import Cache
from maneuver_detect.data.credentials import require_spacetrack_credential
from maneuver_detect.data.ratelimit import RateLimiter
from maneuver_detect.errors import DataSourceError

__all__ = ["SpacetrackFetcher"]

_BASE_URL = "https://www.space-track.org"
_LOGIN_URL = f"{_BASE_URL}/ajaxauth/login"
_SESSION_COOKIE_NAME = "chocolatechip"


def _st_date(value: datetime) -> str:
    """Format a UTC datetime for a Space-Track ``EPOCH`` predicate (``YYYY-MM-DD HH:MM:SS``)."""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _epoch_predicate(lo: datetime | None, hi: datetime | None) -> str:
    """Build the Space-Track ``EPOCH`` predicate value for a window (``""`` when unbounded)."""
    if lo is not None and hi is not None:
        return f"{_st_date(lo)}--{_st_date(hi)}"
    if lo is not None:
        return f">{_st_date(lo)}"
    if hi is not None:
        return f"<{_st_date(hi)}"
    return ""


class SpacetrackFetcher(_CatalogueFetcher):
    """Fetch elsets from Space-Track's ``gp_history`` class, with caching and login reuse.

    Constructing the fetcher does no network I/O and resolves no credentials; both happen lazily on
    the first fetch that misses the cache. The ``credential`` / ``cache`` / ``client`` /
    ``rate_limiter`` parameters are injection points for tests; production callers pass none, and
    credentials are read from the environment on demand.
    """

    source: ClassVar[str] = "spacetrack"
    # ~277 requests/hour: under Space-Track's sustained 300/hour limit (and its 30/min burst limit).
    default_min_interval_s: ClassVar[float] = 13.0

    def __init__(
        self,
        *,
        credential: dict[str, str] | None = None,
        cache: Cache | None = None,
        client: httpx.Client | None = None,
        rate_limiter: RateLimiter | None = None,
        ttl_s: float | None = None,
    ) -> None:
        super().__init__(cache=cache, client=client, rate_limiter=rate_limiter, ttl_s=ttl_s)
        self._credential = credential

    def fetch(
        self,
        norad_id: int,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> FetchResult:
        """Fetch the ``gp_history`` elsets for ``norad_id`` within ``[start, end]``.

        A cache hit needs no credentials; only a cache miss reaches the network, and that is where
        credentials are required (and a missing one raises :class:`MissingCredentialError`).
        """
        lo, hi = normalise_range(start, end)
        key = self._cache_key(norad_id, lo, hi)

        hit = self._cache.get(self.source, key, ttl_s=self._ttl_s)
        if hit is not None:
            elsets = self._parse_payload(hit.value, norad_id)
            return self._build_result(norad_id, elsets, hit.fetched_at, lo, hi, stale=False)

        credential = self._resolve_credential()  # raises MissingCredentialError before any network
        stale_hit = self._cache.get_stale(self.source, key)
        self._rate_limiter.acquire()
        client = self._ensure_client()

        try:
            self._ensure_logged_in(client, credential)
            response = self._query(client, norad_id, lo, hi)
            if response.status_code in (401, 403):
                # Session expired silently — drop the stale cookie, re-login, retry once.
                self._clear_session_cookie(client)
                self._ensure_logged_in(client, credential)
                response = self._query(client, norad_id, lo, hi)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            served = self._serve_stale(norad_id, stale_hit, lo, hi)
            if served is not None:
                return served
            raise DataSourceError(
                f"Space-Track unreachable for NORAD {norad_id}: {exc}", source=self.source
            ) from exc

        # Validate before caching so a malformed payload never poisons the on-disk cache.
        elsets = self._parse_payload(payload, norad_id)
        self._cache.put(self.source, key, payload)
        fetched_at = self._fetched_at_after_put(key)
        return self._build_result(norad_id, elsets, fetched_at, lo, hi, stale=False)

    def _resolve_credential(self) -> dict[str, str]:
        if self._credential is not None:
            return self._credential
        return require_spacetrack_credential()

    def _cache_key(self, norad_id: int, lo: datetime | None, hi: datetime | None) -> str:
        # Window is part of the key: different epoch ranges are different gp_history result sets.
        lo_key = lo.isoformat() if lo is not None else "*"
        hi_key = hi.isoformat() if hi is not None else "*"
        return f"catnr:{norad_id}|from:{lo_key}|to:{hi_key}"

    def _query(
        self, client: httpx.Client, norad_id: int, lo: datetime | None, hi: datetime | None
    ) -> httpx.Response:
        predicate = [f"NORAD_CAT_ID/{norad_id}"]
        epoch = _epoch_predicate(lo, hi)
        if epoch:
            predicate.append(f"EPOCH/{epoch}")
        predicate.append("orderby/EPOCH asc")
        path = "/".join(predicate)
        url = f"{_BASE_URL}/basicspacedata/query/class/gp_history/{path}/format/json"
        return client.get(url)

    def _ensure_logged_in(self, client: httpx.Client, credential: dict[str, str]) -> None:
        if self._has_session_cookie(client):
            return
        self._login(client, credential)

    def _login(self, client: httpx.Client, credential: dict[str, str]) -> None:
        """POST credentials to ``/ajaxauth/login`` and confirm the session cookie landed.

        Space-Track answers a bad credential with ``200`` + a JSON body, not a ``401``, so the only
        reliable success signal is the presence of the session cookie. Transport errors propagate
        as :class:`httpx.HTTPError` for the caller's outage handling; a refused credential raises
        :class:`DataSourceError`.
        """
        form = {"identity": credential["username"], "password": credential["password"]}
        response = client.post(_LOGIN_URL, data=form)
        response.raise_for_status()
        if not self._has_session_cookie(client):
            raise DataSourceError(
                "Space-Track login failed; the credential is likely invalid "
                "(no session cookie returned)",
                source=self.source,
            )

    @staticmethod
    def _has_session_cookie(client: httpx.Client) -> bool:
        return _SESSION_COOKIE_NAME in {cookie.name for cookie in client.cookies.jar}

    @staticmethod
    def _clear_session_cookie(client: httpx.Client) -> None:
        client.cookies.delete(_SESSION_COOKIE_NAME)
