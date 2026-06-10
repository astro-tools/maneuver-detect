"""Tests for ``maneuver_detect.data.cache`` — the XDG on-disk cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maneuver_detect.data.cache import Cache


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(directory=tmp_path)


def test_put_then_get_roundtrips_value(cache: Cache) -> None:
    cache.put("celestrak", "catnr:25544", [{"NORAD_CAT_ID": 25544}])
    hit = cache.get("celestrak", "catnr:25544", ttl_s=3600)
    assert hit is not None
    assert hit.value == [{"NORAD_CAT_ID": 25544}]
    assert hit.fetched_at.tzinfo is not None


def test_get_misses_for_unknown_key(cache: Cache) -> None:
    assert cache.get("celestrak", "catnr:404", ttl_s=3600) is None


def test_get_misses_when_older_than_ttl_but_get_stale_hits(cache: Cache) -> None:
    cache.put("spacetrack", "k", {"v": 1})
    # A zero TTL makes any non-zero age a miss, while get_stale ignores age entirely.
    assert cache.get("spacetrack", "k", ttl_s=0.0) is None
    stale = cache.get_stale("spacetrack", "k")
    assert stale is not None and stale.value == {"v": 1}


def test_writes_are_one_file_per_source_key(cache: Cache, tmp_path: Path) -> None:
    cache.put("celestrak", "catnr:1", [1])
    cache.put("spacetrack", "catnr:1", [2])
    assert (tmp_path / "celestrak").is_dir()
    assert (tmp_path / "spacetrack").is_dir()
    # Distinct sources never collide even on an identical key.
    assert cache.get_stale("celestrak", "catnr:1") is not None
    assert cache.get_stale("spacetrack", "catnr:1") is not None


def test_corrupt_entry_is_treated_as_miss(cache: Cache) -> None:
    cache.put("celestrak", "k", [1])
    path = cache._path("celestrak", "k")
    path.write_text("{ not json", encoding="utf-8")
    assert cache.get_stale("celestrak", "k") is None


def test_put_is_atomic_no_tempfile_left_behind(cache: Cache, tmp_path: Path) -> None:
    cache.put("celestrak", "k", [1])
    leftovers = [p for p in (tmp_path / "celestrak").iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_stored_payload_carries_key_and_timestamp(cache: Cache) -> None:
    cache.put("celestrak", "catnr:25544", [1])
    raw = json.loads(cache._path("celestrak", "catnr:25544").read_text(encoding="utf-8"))
    assert raw["key"] == "catnr:25544"
    assert "fetched_at" in raw and raw["value"] == [1]


class TestDisabledCache:
    def test_explicit_empty_env_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MANEUVER_DETECT_CACHE_DIR", "")
        disabled = Cache()
        assert disabled.enabled is False
        assert disabled.directory is None
        disabled.put("celestrak", "k", [1])  # no-op, must not raise
        assert disabled.get("celestrak", "k", ttl_s=3600) is None
        assert disabled.get_stale("celestrak", "k") is None

    def test_env_dir_is_honoured(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MANEUVER_DETECT_CACHE_DIR", str(tmp_path / "xdg"))
        c = Cache()
        assert c.directory == tmp_path / "xdg"
        c.put("celestrak", "k", [1])
        assert c.get_stale("celestrak", "k") is not None

    def test_constructor_arg_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MANEUVER_DETECT_CACHE_DIR", "")
        c = Cache(directory=tmp_path)
        assert c.enabled is True
        assert c.directory == tmp_path
