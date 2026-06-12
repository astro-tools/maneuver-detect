"""Tests for ``maneuver_detect.datasets.recipe`` and ``…catalogue`` — the pinned recipe."""

from __future__ import annotations

from pathlib import Path

from maneuver_detect.datasets.catalogue import (
    GALILEO_CONSTELLATION,
    GEO_OBJECTS,
    GPS_CONSTELLATION,
    galileo_gsat_to_norad,
    gps_svn_to_norad,
    recipe,
)
from maneuver_detect.datasets.recipe import Recipe
from maneuver_detect.labels.record import (
    SOURCE_DORIS_IDS,
    SOURCE_GALILEO_NAGU,
    SOURCE_GPS_NANU,
    SOURCE_SELF_GEO,
    OrbitClass,
)

_DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


def test_version_is_pinned() -> None:
    assert recipe().dataset_version == "0.2.0"


def test_per_class_counts_match_scope() -> None:
    counts = recipe().per_class_counts()
    assert counts[OrbitClass.LEO] == 15  # the DORIS sats with man.txt files (altimetry + SPOTs)
    # MEO is the GPS constellation plus the Galileo constellation.
    assert counts[OrbitClass.MEO] == len(GPS_CONSTELLATION) + len(GALILEO_CONSTELLATION) == 65
    assert counts[OrbitClass.GEO] == len(GEO_OBJECTS) == 6


def test_entries_sorted_by_norad() -> None:
    norads = list(recipe().norad_ids())
    assert norads == sorted(norads)
    assert len(set(norads)) == len(norads)  # no duplicates


def test_label_source_per_class() -> None:
    for entry in recipe().entries:
        if entry.orbit_class is OrbitClass.LEO:
            assert entry.label_source == SOURCE_DORIS_IDS
        elif entry.orbit_class is OrbitClass.GEO:
            assert entry.label_source == SOURCE_SELF_GEO
            assert entry.label_ref == ""  # self-derived, no external ref
        else:
            assert entry.orbit_class is OrbitClass.MEO
            if entry.label_source == SOURCE_GALILEO_NAGU:
                assert entry.label_ref.startswith("GSAT")
            else:
                assert entry.label_source == SOURCE_GPS_NANU
                assert entry.label_ref.startswith("SVN")


def test_known_entries_present() -> None:
    by_norad = {e.norad_id: e for e in recipe().entries}
    assert by_norad[33105].object_name == "Jason-2"  # a DORIS LEO altimetry sat
    assert by_norad[36585].orbit_class is OrbitClass.MEO  # GPS SVN62
    assert by_norad[37847].label_source == SOURCE_GALILEO_NAGU  # Galileo GSAT0102
    assert by_norad[41866].orbit_class is OrbitClass.GEO  # GOES-16


def test_gps_crosswalk() -> None:
    crosswalk = gps_svn_to_norad()
    assert len(crosswalk) == 32
    assert crosswalk["SVN62"] == 36585  # the V2-confirmed anchor


def test_galileo_crosswalk() -> None:
    crosswalk = galileo_gsat_to_norad()
    assert len(crosswalk) == len(GALILEO_CONSTELLATION) == 33
    assert crosswalk["GSAT0102"] == 37847  # the V2-follow-up anchor (GALILEO-FM2)


def test_recipe_json_round_trip() -> None:
    built = recipe()
    assert Recipe.from_json(built.to_json()) == built


def test_committed_recipe_matches_code() -> None:
    """The committed ``recipe.json`` is exactly the in-code recipe's serialisation (CI-safe)."""
    committed = (_DATASET_DIR / "v0.2" / "recipe.json").read_text(encoding="utf-8")
    assert recipe().to_json() == committed
