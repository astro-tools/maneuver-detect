"""Tests for ``maneuver_detect.datasets.manifest`` — series hashing and the manifest."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from maneuver_detect.data.elset import Elset
from maneuver_detect.datasets.manifest import Manifest, SeriesDigest, series_sha256

_UTC = timezone.utc
_T0 = datetime(2024, 1, 1, 12, tzinfo=_UTC)


def _elset(epoch: datetime, mean_motion: float = 15.0) -> Elset:
    return Elset(
        norad_id=100,
        epoch=epoch,
        mean_motion=mean_motion,
        eccentricity=0.001,
        inclination=51.6,
        raan=200.0,
        arg_perigee=90.0,
        mean_anomaly=10.0,
        bstar=1.0e-4,
        mean_motion_dot=0.0,
        mean_motion_ddot=0.0,
        element_set_no=1,
        rev_at_epoch=1,
        classification="U",
        object_id="2024-001A",
    )


def _series() -> list[Elset]:
    return [_elset(_T0 + timedelta(days=d)) for d in range(5)]


def test_series_sha256_is_deterministic() -> None:
    series = _series()
    assert series_sha256(series) == series_sha256(series)
    # A fresh construction with identical values hashes identically (no object identity dependence).
    assert series_sha256(series) == series_sha256(_series())


def test_series_sha256_is_order_independent() -> None:
    series = _series()
    assert series_sha256(series) == series_sha256(list(reversed(series)))


def test_series_sha256_changes_with_elements() -> None:
    base = _series()
    bumped = [*base[:-1], _elset(base[-1].epoch, mean_motion=15.1)]
    assert series_sha256(base) != series_sha256(bumped)


def test_manifest_json_round_trip() -> None:
    manifest = Manifest(
        dataset_version="0.1.0",
        digests=(SeriesDigest(200, 3, "b" * 64), SeriesDigest(100, 5, "a" * 64)),
    )
    restored = Manifest.from_json(manifest.to_json())
    assert restored == manifest
    # Serialisation is NORAD-sorted regardless of input order.
    assert manifest.to_json().index('"norad_id": 100') < manifest.to_json().index('"norad_id": 200')


def test_verify_matching_is_empty() -> None:
    manifest = Manifest("0.1.0", (SeriesDigest(100, 5, "a" * 64),))
    assert manifest.verify(manifest) == []


def test_verify_detects_hash_and_count_mismatch() -> None:
    pinned = Manifest("0.1.0", (SeriesDigest(100, 5, "a" * 64),))
    changed = Manifest("0.1.0", (SeriesDigest(100, 5, "c" * 64),))
    mismatches = pinned.verify(changed)
    assert len(mismatches) == 1
    assert "hash mismatch" in mismatches[0]


def test_verify_detects_missing_and_version() -> None:
    pinned = Manifest("0.1.0", (SeriesDigest(100, 5, "a" * 64), SeriesDigest(200, 3, "b" * 64)))
    partial = Manifest("0.2.0", (SeriesDigest(100, 5, "a" * 64),))
    mismatches = pinned.verify(partial)
    assert any("dataset_version" in m for m in mismatches)
    assert any("NORAD 200" in m and "missing" in m for m in mismatches)
