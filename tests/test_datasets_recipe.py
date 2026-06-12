"""Tests for ``maneuver_detect.datasets.recipe`` and ``…catalogue`` — the pinned recipes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from maneuver_detect.datasets.catalogue import (
    GALILEO_CONSTELLATION,
    GEO_OBJECTS,
    GPS_CONSTELLATION,
    galileo_gsat_to_norad,
    gps_svn_to_norad,
    v01_recipe,
    v02_recipe,
)
from maneuver_detect.datasets.recipe import Recipe
from maneuver_detect.labels.record import (
    SOURCE_DORIS_IDS,
    SOURCE_GALILEO_NAGU,
    SOURCE_GPS_NANU,
    SOURCE_SELF_GEO,
    OrbitClass,
)


def test_per_class_counts_match_scope() -> None:
    counts = v01_recipe().per_class_counts()
    assert counts[OrbitClass.LEO] == 15  # the DORIS sats with man.txt files (altimetry + SPOTs)
    assert counts[OrbitClass.MEO] == len(GPS_CONSTELLATION) == 32  # the GPS constellation
    assert counts[OrbitClass.GEO] == 0  # no public GEO label source (deferred)


def test_entries_sorted_by_norad() -> None:
    norads = list(v01_recipe().norad_ids())
    assert norads == sorted(norads)
    assert len(set(norads)) == len(norads)  # no duplicates


def test_label_source_per_class() -> None:
    for entry in v01_recipe().entries:
        if entry.orbit_class is OrbitClass.LEO:
            assert entry.label_source == SOURCE_DORIS_IDS
        else:
            assert entry.orbit_class is OrbitClass.MEO
            assert entry.label_source == SOURCE_GPS_NANU
            assert entry.label_ref.startswith("SVN")


def test_known_entries_present() -> None:
    by_norad = {e.norad_id: e for e in v01_recipe().entries}
    assert by_norad[33105].object_name == "Jason-2"  # a DORIS LEO altimetry sat
    assert by_norad[33105].label_ref == "ja2"
    assert by_norad[36585].orbit_class is OrbitClass.MEO  # GPS SVN62


def test_gps_crosswalk() -> None:
    crosswalk = gps_svn_to_norad()
    assert len(crosswalk) == 32
    assert crosswalk["SVN62"] == 36585  # the V2-confirmed anchor
    assert crosswalk["SVN74"] == 43873  # GPS III SV01


def test_recipe_json_round_trip() -> None:
    recipe = v01_recipe()
    assert Recipe.from_json(recipe.to_json()) == recipe


# --- v0.2 growth: Galileo MEO + self-labelled GEO ---


def test_versions_are_pinned() -> None:
    assert v01_recipe().dataset_version == "0.1.0"
    assert v02_recipe().dataset_version == "0.2.0"


def test_v02_is_a_superset_of_v01() -> None:
    v01, v02 = set(v01_recipe().norad_ids()), set(v02_recipe().norad_ids())
    assert v01 <= v02
    # the new objects are exactly the Galileo constellation + the GEO set.
    assert len(v02 - v01) == len(GALILEO_CONSTELLATION) + len(GEO_OBJECTS)


def test_v02_per_class_counts_grow() -> None:
    v01, v02 = v01_recipe().per_class_counts(), v02_recipe().per_class_counts()
    assert v02[OrbitClass.LEO] == v01[OrbitClass.LEO]  # LEO unchanged
    assert v02[OrbitClass.MEO] == v01[OrbitClass.MEO] + len(GALILEO_CONSTELLATION)  # + Galileo
    assert v02[OrbitClass.GEO] == len(GEO_OBJECTS) >= 1  # GEO now non-empty


def test_v02_label_sources_per_class() -> None:
    for entry in v02_recipe().entries:
        if entry.orbit_class is OrbitClass.GEO:
            assert entry.label_source == SOURCE_SELF_GEO
            assert entry.label_ref == ""  # self-derived, no external ref
        elif entry.label_source == SOURCE_GALILEO_NAGU:
            assert entry.orbit_class is OrbitClass.MEO
            assert entry.label_ref.startswith("GSAT")
        else:
            assert entry.label_source in (SOURCE_DORIS_IDS, SOURCE_GPS_NANU)


def test_galileo_crosswalk() -> None:
    crosswalk = galileo_gsat_to_norad()
    assert len(crosswalk) == len(GALILEO_CONSTELLATION)
    assert crosswalk["GSAT0102"] == 37847  # the V2-follow-up anchor (GALILEO-FM2)


def test_v02_recipe_json_round_trip() -> None:
    recipe = v02_recipe()
    assert Recipe.from_json(recipe.to_json()) == recipe


_DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


@pytest.mark.parametrize("version, builder", [("0.1.0", v01_recipe), ("0.2.0", v02_recipe)])
def test_committed_recipe_matches_code(version: str, builder: Callable[[], Recipe]) -> None:
    """The committed ``recipe.json`` is exactly the in-code recipe's serialisation (CI-safe)."""
    committed = (_DATASET_DIR / f"v{version.rsplit('.', 1)[0]}" / "recipe.json").read_text(
        encoding="utf-8"
    )
    assert builder().to_json() == committed
