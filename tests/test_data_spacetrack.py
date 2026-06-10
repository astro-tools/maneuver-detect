"""Tests for ``maneuver_detect.data.spacetrack`` — the credentialled gp_history fetcher.

Driven against ``httpx.MockTransport``. The mock dispatches ``/ajaxauth/login`` (returning the
session cookie) from the ``gp_history`` query path, so login reuse and the 401 re-login retry are
exercised without a real account or the network.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from maneuver_detect.data.cache import Cache
from maneuver_detect.data.ratelimit import RateLimiter
from maneuver_detect.data.spacetrack import SpacetrackFetcher
from maneuver_detect.errors import DataSourceError, MissingCredentialError

_SAMPLE_OMM: dict[str, Any] = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2024-01-01T12:00:00.000000",
    "MEAN_MOTION": 15.5,
    "ECCENTRICITY": 0.0001,
    "INCLINATION": 51.64,
    "RA_OF_ASC_NODE": 90.0,
    "ARG_OF_PERICENTER": 80.0,
    "MEAN_ANOMALY": 270.0,
    "NORAD_CAT_ID": 25544,
    "BSTAR": 0.00018,
    "MEAN_MOTION_DOT": 0.0001,
    "MEAN_MOTION_DDOT": 0.0,
}

_CRED = {"username": "alice@example.com", "password": "s3cret"}

_LOGIN_PATH = "/ajaxauth/login"
_QueryFn = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(directory=tmp_path)


@pytest.fixture(autouse=True)
def _no_env_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real environment's Space-Track creds out of every test by default."""
    monkeypatch.delenv("SPACETRACK_USERNAME", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)


def _no_pacing() -> RateLimiter:
    return RateLimiter(0.0)


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _handler(on_query: _QueryFn, *, login_ok: bool = True) -> _QueryFn:
    """A MockTransport handler: serve the login cookie, delegate the query to ``on_query``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == _LOGIN_PATH:
            headers = {"Set-Cookie": "chocolatechip=abc123; Path=/"} if login_ok else {}
            return httpx.Response(200, headers=headers)
        return on_query(request)

    return handler


def test_construction_does_no_network_and_resolves_no_credentials() -> None:
    # Constructing must not read the environment or build a client (no MissingCredentialError here).
    fetcher = SpacetrackFetcher()
    assert fetcher._client is None


class TestHappyPath:
    def test_login_then_query_returns_parsed_elsets(self, cache: Cache) -> None:
        urls: list[str] = []

        def on_query(request: httpx.Request) -> httpx.Response:
            urls.append(str(request.url))
            return httpx.Response(200, json=[_SAMPLE_OMM])

        f = SpacetrackFetcher(
            credential=_CRED,
            cache=cache,
            client=_client(_handler(on_query)),
            rate_limiter=_no_pacing(),
        )
        result = f.fetch(25544, start="2024-01-01", end="2024-06-30")

        assert result.source == "spacetrack"
        assert result.stale is False
        assert len(result.elsets) == 1
        assert result.elsets[0].norad_id == 25544
        query_url = urls[0]
        assert "class/gp_history" in query_url
        assert "NORAD_CAT_ID/25544" in query_url
        assert "EPOCH/" in query_url and "2024-01-01" in query_url and "2024-06-30" in query_url
        assert "--" in query_url  # the inclusive-range operator for a two-sided window
        assert "orderby/EPOCH" in query_url

    def test_open_ended_window_omits_epoch_predicate(self, cache: Cache) -> None:
        urls: list[str] = []

        def on_query(request: httpx.Request) -> httpx.Response:
            urls.append(str(request.url))
            return httpx.Response(200, json=[_SAMPLE_OMM])

        f = SpacetrackFetcher(
            credential=_CRED,
            cache=cache,
            client=_client(_handler(on_query)),
            rate_limiter=_no_pacing(),
        )
        f.fetch(25544)
        assert "EPOCH/" not in urls[0]

    def test_login_happens_once_across_two_fetches(self, cache: Cache) -> None:
        counts = {"login": 0, "query": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == _LOGIN_PATH:
                counts["login"] += 1
                return httpx.Response(200, headers={"Set-Cookie": "chocolatechip=abc; Path=/"})
            counts["query"] += 1
            return httpx.Response(200, json=[_SAMPLE_OMM])

        f = SpacetrackFetcher(
            credential=_CRED, cache=cache, client=_client(handler), rate_limiter=_no_pacing()
        )
        f.fetch(25544, start="2024-01-01")
        f.fetch(
            25544, start="2024-02-01"
        )  # different window -> different cache key -> second query
        assert counts["login"] == 1  # cookie reused
        assert counts["query"] == 2


class TestCredentials:
    def test_missing_credential_raises_before_network(self, cache: Cache) -> None:
        def on_query(request: httpx.Request) -> httpx.Response:
            raise AssertionError("network must not be touched without credentials")

        f = SpacetrackFetcher(
            cache=cache, client=_client(_handler(on_query)), rate_limiter=_no_pacing()
        )
        with pytest.raises(MissingCredentialError) as excinfo:
            f.fetch(25544)
        assert excinfo.value.source == "spacetrack"

    def test_env_credentials_are_used(self, cache: Cache, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPACETRACK_USERNAME", "bob@example.com")
        monkeypatch.setenv("SPACETRACK_PASSWORD", "pw")
        sent: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == _LOGIN_PATH:
                sent["body"] = request.content.decode()
                return httpx.Response(200, headers={"Set-Cookie": "chocolatechip=abc; Path=/"})
            return httpx.Response(200, json=[_SAMPLE_OMM])

        f = SpacetrackFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing())
        f.fetch(25544)
        assert "bob%40example.com" in sent["body"] or "bob@example.com" in sent["body"]

    def test_cache_hit_needs_no_credentials(self, cache: Cache) -> None:
        # Seed the cache under the key a no-window fetch computes; then a credential-less fetch
        # must serve it without reaching the network.
        cache.put("spacetrack", "catnr:25544|from:*|to:*", [_SAMPLE_OMM])

        def on_query(request: httpx.Request) -> httpx.Response:
            raise AssertionError("cache hit must not touch the network")

        f = SpacetrackFetcher(
            cache=cache, client=_client(_handler(on_query)), rate_limiter=_no_pacing()
        )
        result = f.fetch(25544)
        assert len(result.elsets) == 1
        assert result.stale is False


class TestSessionExpiry:
    def test_401_triggers_one_relogin_and_retry(self, cache: Cache) -> None:
        counts = {"login": 0, "query": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == _LOGIN_PATH:
                counts["login"] += 1
                return httpx.Response(200, headers={"Set-Cookie": "chocolatechip=abc; Path=/"})
            counts["query"] += 1
            if counts["query"] == 1:
                return httpx.Response(401)  # session silently expired
            return httpx.Response(200, json=[_SAMPLE_OMM])

        f = SpacetrackFetcher(
            credential=_CRED, cache=cache, client=_client(handler), rate_limiter=_no_pacing()
        )
        result = f.fetch(25544)
        assert counts["login"] == 2  # logged in again after the 401
        assert counts["query"] == 2  # retried once
        assert len(result.elsets) == 1

    def test_bad_credential_no_cookie_raises_data_source_error(self, cache: Cache) -> None:
        def on_query(request: httpx.Request) -> httpx.Response:
            raise AssertionError("query must not run when login never establishes a session")

        f = SpacetrackFetcher(
            credential=_CRED,
            cache=cache,
            client=_client(_handler(on_query, login_ok=False)),
            rate_limiter=_no_pacing(),
        )
        with pytest.raises(DataSourceError, match="login failed"):
            f.fetch(25544)


class TestOutageHandling:
    def test_outage_with_cache_serves_stale(self, cache: Cache) -> None:
        calls = {"query": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == _LOGIN_PATH:
                return httpx.Response(200, headers={"Set-Cookie": "chocolatechip=abc; Path=/"})
            calls["query"] += 1
            if calls["query"] == 1:
                return httpx.Response(200, json=[_SAMPLE_OMM])
            raise httpx.ConnectError("simulated outage")

        f = SpacetrackFetcher(
            credential=_CRED,
            cache=cache,
            client=_client(handler),
            rate_limiter=_no_pacing(),
            ttl_s=0.0,
        )
        first = f.fetch(25544)
        assert first.stale is False
        stale = f.fetch(25544)
        assert stale.stale is True
        assert stale.fetched_at == first.fetched_at

    def test_outage_with_no_cache_raises(self, cache: Cache) -> None:
        def on_query(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        f = SpacetrackFetcher(
            credential=_CRED,
            cache=cache,
            client=_client(_handler(on_query)),
            rate_limiter=_no_pacing(),
        )
        with pytest.raises(DataSourceError) as excinfo:
            f.fetch(25544)
        assert excinfo.value.source == "spacetrack"
