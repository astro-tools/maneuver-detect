"""Tests for ``maneuver_detect.data.celestrak`` — the no-auth current-GP fetcher.

The adapter is driven against ``httpx.MockTransport`` for the cache / outage / conditional-GET
paths. No test touches the network or the real XDG cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from maneuver_detect.data.cache import Cache
from maneuver_detect.data.celestrak import CelestrakFetcher
from maneuver_detect.data.ratelimit import RateLimiter
from maneuver_detect.errors import DataSourceError

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


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(directory=tmp_path)


def _no_pacing() -> RateLimiter:
    """A rate limiter that never sleeps, so tests don't burn the default 1s floor."""
    return RateLimiter(0.0)


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_construction_does_no_network() -> None:
    # Lazy client: nothing is built until the first fetch, so import / construction is offline.
    fetcher = CelestrakFetcher()
    assert fetcher._client is None


class TestHappyPathAndCache:
    def test_fetch_returns_parsed_elsets(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "CATNR=25544" in str(request.url)
            assert "FORMAT=json" in str(request.url)
            return httpx.Response(200, json=[_SAMPLE_OMM])

        with CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing()) as f:
            result = f.fetch(25544)

        assert result.source == "celestrak"
        assert result.stale is False
        assert result.norad_id == 25544
        assert len(result.elsets) == 1
        assert result.elsets[0].norad_id == 25544
        assert result.elsets[0].mean_motion == pytest.approx(15.5)
        assert result.fetched_at.tzinfo is not None

    def test_second_fetch_hits_cache_no_second_request(self, cache: Cache) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=[_SAMPLE_OMM])

        f = CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing())
        f.fetch(25544)
        again = f.fetch(25544)
        assert calls["n"] == 1
        assert again.stale is False

    def test_empty_payload_yields_no_elsets(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[])

        f = CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing())
        result = f.fetch(99999)
        assert result.elsets == ()


class TestDateWindow:
    def test_elset_outside_window_is_filtered_out(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM])  # epoch 2024-01-01

        f = CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing())
        # A closed window entirely before the current elset's epoch yields nothing.
        result = f.fetch(25544, start="2023-01-01", end="2023-12-31")
        assert result.elsets == ()

    def test_elset_inside_window_is_kept(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM])

        f = CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing())
        result = f.fetch(25544, start="2024-01-01", end="2024-01-02")
        assert len(result.elsets) == 1


class TestConditionalGet:
    def test_if_modified_since_sent_when_stale_entry_exists(self, cache: Cache) -> None:
        seen_headers: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(request.headers)
            return httpx.Response(200, json=[_SAMPLE_OMM])

        # ttl_s=0 makes the cached entry immediately expired -> the second fetch revalidates.
        f = CelestrakFetcher(
            cache=cache, client=_client(handler), rate_limiter=_no_pacing(), ttl_s=0.0
        )
        f.fetch(25544)  # seeds the cache
        f.fetch(25544)  # revalidates
        assert "if-modified-since" not in seen_headers[0]
        assert "if-modified-since" in seen_headers[1]

    def test_304_serves_cached_value_as_fresh(self, cache: Cache) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=[_SAMPLE_OMM])
            # No body — a 304 means "use what you cached".
            return httpx.Response(304)

        f = CelestrakFetcher(
            cache=cache, client=_client(handler), rate_limiter=_no_pacing(), ttl_s=0.0
        )
        f.fetch(25544)
        revalidated = f.fetch(25544)
        assert calls["n"] == 2
        assert revalidated.stale is False  # confirmed-current, not stale
        assert len(revalidated.elsets) == 1
        assert revalidated.elsets[0].mean_motion == pytest.approx(15.5)


class TestOutageHandling:
    def test_outage_with_cache_serves_stale(self, cache: Cache) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=[_SAMPLE_OMM])
            raise httpx.ConnectError("simulated outage")

        f = CelestrakFetcher(
            cache=cache, client=_client(handler), rate_limiter=_no_pacing(), ttl_s=0.0
        )
        first = f.fetch(25544)
        assert first.stale is False
        stale = f.fetch(25544)
        assert stale.stale is True
        assert len(stale.elsets) == 1
        assert stale.fetched_at == first.fetched_at  # original fetch time, not now

    def test_outage_with_no_cache_raises_data_source_error(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        f = CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing())
        with pytest.raises(DataSourceError) as excinfo:
            f.fetch(25544)
        assert excinfo.value.source == "celestrak"

    def test_http_500_with_no_cache_raises(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        f = CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing())
        with pytest.raises(DataSourceError):
            f.fetch(25544)

    def test_outage_with_unparseable_stale_cache_raises(self, cache: Cache) -> None:
        # A stale cache entry that no longer rebuilds (schema tightened since it was written) must
        # not mask the outage as success — fall through to the typed unreachable error.
        cache.put("celestrak", "catnr:25544", [{"garbage": True}])

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        f = CelestrakFetcher(
            cache=cache, client=_client(handler), rate_limiter=_no_pacing(), ttl_s=0.0
        )
        with pytest.raises(DataSourceError):
            f.fetch(25544)


class TestMalformedPayload:
    def test_non_list_payload_raises_and_is_not_cached(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "nope"})

        f = CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing())
        with pytest.raises(DataSourceError):
            f.fetch(25544)
        assert cache.get_stale("celestrak", "catnr:25544") is None

    def test_malformed_record_raises(self, cache: Cache) -> None:
        bad = {k: v for k, v in _SAMPLE_OMM.items() if k != "MEAN_MOTION"}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[bad])

        f = CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_no_pacing())
        with pytest.raises(DataSourceError, match="malformed"):
            f.fetch(25544)


def test_rate_limiter_is_invoked_on_network_fetch(cache: Cache) -> None:
    acquired = {"n": 0}

    class _CountingLimiter(RateLimiter):
        def acquire(self) -> None:
            acquired["n"] += 1

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_SAMPLE_OMM])

    f = CelestrakFetcher(cache=cache, client=_client(handler), rate_limiter=_CountingLimiter(0.0))
    f.fetch(25544)  # network -> one acquire
    f.fetch(25544)  # cache hit -> no acquire
    assert acquired["n"] == 1
