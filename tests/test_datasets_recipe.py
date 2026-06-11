"""Tests for ``maneuver_detect.datasets.recipe`` and ``…catalogue`` — the pinned v0.1 recipe."""

from __future__ import annotations

from maneuver_detect.datasets.catalogue import (
    GPS_CONSTELLATION,
    gps_svn_to_norad,
    v01_recipe,
)
from maneuver_detect.datasets.recipe import Recipe
from maneuver_detect.labels.record import SOURCE_DORIS_IDS, SOURCE_GPS_NANU, OrbitClass


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
