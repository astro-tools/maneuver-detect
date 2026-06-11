"""Tests for ``maneuver_detect.datasets.reconstruct`` and ``…build`` — the reconstruction engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from maneuver_detect.data.base import Fetcher, FetchResult
from maneuver_detect.data.elset import Elset
from maneuver_detect.datasets.build import build_dataset, labels_from_json
from maneuver_detect.datasets.manifest import Manifest, SeriesDigest
from maneuver_detect.datasets.recipe import Recipe, RecipeEntry
from maneuver_detect.datasets.reconstruct import reconstruct, verify
from maneuver_detect.labels.record import (
    SOURCE_DORIS_IDS,
    SOURCE_GPS_NANU,
    ManeuverLabel,
    OrbitClass,
)
from maneuver_detect.schema import ManeuverType

_UTC = timezone.utc
_T0 = datetime(2024, 1, 1, 12, tzinfo=_UTC)
_FETCHED_AT = datetime(2024, 6, 1, tzinfo=_UTC)

_RECIPE = Recipe(
    dataset_version="test-0",
    entries=(
        RecipeEntry(100, OrbitClass.LEO, "Test-LEO", "fake", SOURCE_DORIS_IDS, "tst"),
        RecipeEntry(200, OrbitClass.MEO, "Test-MEO", "fake", SOURCE_GPS_NANU, "SVN99"),
    ),
)


def _elset(norad_id: int, epoch: datetime, mean_motion: float) -> Elset:
    return Elset(
        norad_id=norad_id,
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


def _series(norad_id: int, base_mm: float) -> list[Elset]:
    # Six daily elsets with an injected mean-motion step at day 3 (a stand-in for a maneuver).
    return [
        _elset(norad_id, _T0 + timedelta(days=d), base_mm + (0.02 if d >= 3 else 0.0))
        for d in range(6)
    ]


def _by_norad() -> dict[int, list[Elset]]:
    return {100: _series(100, 15.0), 200: _series(200, 2.0)}


def _label(norad_id: int, *, leo: bool) -> ManeuverLabel:
    epoch = _T0 + timedelta(days=2, hours=18)  # in the gap between the day-2 and day-3 elsets
    return ManeuverLabel(
        norad_id=norad_id,
        epoch=epoch,
        window_start=epoch,
        window_end=epoch,
        source=SOURCE_DORIS_IDS if leo else SOURCE_GPS_NANU,
        source_ref="ref",
        orbit_class=OrbitClass.LEO if leo else OrbitClass.MEO,
        maneuver_type=ManeuverType.IN_TRACK if leo else None,
        delta_v=1.5 if leo else None,
    )


def _labels() -> dict[int, list[ManeuverLabel]]:
    return {100: [_label(100, leo=True)], 200: [_label(200, leo=False)]}


class _FakeFetcher(Fetcher):
    """A fetcher that returns pinned in-memory elsets — no network."""

    source = "fake"

    def __init__(self, by_norad: dict[int, list[Elset]]) -> None:
        self._by_norad = by_norad

    def fetch(
        self,
        norad_id: int,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> FetchResult:
        return FetchResult(
            norad_id=norad_id,
            elsets=tuple(self._by_norad.get(norad_id, ())),
            fetched_at=_FETCHED_AT,
            source=self.source,
        )


def test_manifest_is_byte_deterministic() -> None:
    first = reconstruct(_RECIPE, _FakeFetcher(_by_norad()), _labels())
    # A fresh fetcher with freshly-built (but identical) elsets must hash the same.
    second = reconstruct(_RECIPE, _FakeFetcher(_by_norad()), _labels())
    assert first.manifest().to_json() == second.manifest().to_json()


def test_reconstruct_attaches_labels_and_series() -> None:
    dataset = reconstruct(_RECIPE, _FakeFetcher(_by_norad()), _labels())
    leo = dataset.by_norad()[100]
    assert len(leo.series) == 6
    assert len(leo.intervals) == 1
    assert leo.intervals[0].delta_v == 1.5
    assert leo.digest.n_elsets == 6


def test_coverage_reflects_class_scope() -> None:
    coverage = reconstruct(_RECIPE, _FakeFetcher(_by_norad()), _labels()).coverage()
    assert coverage.per_class[OrbitClass.LEO].n_with_delta_v == 1  # the Δv-labelled LEO event
    assert coverage.per_class[OrbitClass.MEO].n_events == 1
    assert coverage.per_class[OrbitClass.MEO].n_with_delta_v == 0  # MEO is epoch-only


def test_verify_passes_and_flags_tampering() -> None:
    manifest = reconstruct(_RECIPE, _FakeFetcher(_by_norad()), _labels()).manifest()
    assert verify(_RECIPE, _FakeFetcher(_by_norad()), manifest, _labels()) == []

    tampered = Manifest(
        manifest.dataset_version,
        (SeriesDigest(100, 6, "0" * 64), *(d for d in manifest.digests if d.norad_id != 100)),
    )
    assert verify(_RECIPE, _FakeFetcher(_by_norad()), tampered, _labels()) != []


def test_build_writes_and_round_trips(tmp_path: Path) -> None:
    report = build_dataset(_RECIPE, _FakeFetcher(_by_norad()), _labels(), tmp_path)
    assert report.n_objects == 2
    for name in ("recipe", "labels", "manifest"):
        assert report.paths[name].exists()

    assert Recipe.from_json(report.paths["recipe"].read_text(encoding="utf-8")) == _RECIPE
    pinned = reconstruct(_RECIPE, _FakeFetcher(_by_norad()), _labels()).manifest()
    assert Manifest.from_json(report.paths["manifest"].read_text(encoding="utf-8")) == pinned
    labels = labels_from_json(report.paths["labels"].read_text(encoding="utf-8"))
    assert len(labels) == 2
