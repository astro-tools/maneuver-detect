"""The v0.1 dataset catalogue — the objects the recipe reconstructs, as public reference facts.

Two classes, matching the v0.1 label scope (D3):

- **LEO** — the DORIS/IDS altimetry satellites that publish a ``man.txt`` maneuver file *and* have a
  confident NORAD id (the Δv-labelled set). DORIS-tracked satellites without a maneuver file (the
  SPOTs) carry no labels and are excluded.
- **MEO** — the operational GPS constellation (``SVN / PRN / NORAD``), sourced and cross-checked
  against the CelesTrak GPS catalogue. This table doubles as the ``SVN → NORAD`` crosswalk the NANU
  label parser needs (it accepts an injectable crosswalk), so the dataset layer supplies the full
  mapping the label layer ships only a seed of.

GEO is deferred — there is no public GEO maneuver-label file source, so a GEO object would carry an
unlabelled series (best-effort per D3; out of v0.1).

The catalogue is a **pinned snapshot**: a satellite's ``SVN → NORAD`` is fixed for its lifetime,
while the constellation membership and PRN-slot assignments drift over time, so a recipe version
captures the set at sourcing time.
"""

from __future__ import annotations

from dataclasses import dataclass

from maneuver_detect.datasets.recipe import Recipe, RecipeEntry
from maneuver_detect.labels.doris import DORIS_SAT_TO_NORAD
from maneuver_detect.labels.record import SOURCE_DORIS_IDS, SOURCE_GPS_NANU, OrbitClass

__all__ = [
    "DATASET_VERSION",
    "GPS_CONSTELLATION",
    "GpsSatellite",
    "gps_svn_to_norad",
    "v01_recipe",
]

#: The dataset version (versioned in lockstep with a later Hub release and the manifest of it — D8).
DATASET_VERSION = "0.1.0"


@dataclass(frozen=True)
class GpsSatellite:
    """One GPS satellite — its space-vehicle number, broadcast PRN slot, NORAD id, and block.

    Attributes:
        svn: Space Vehicle Number (fixed per physical satellite).
        prn: Broadcast pseudo-random-noise code / slot (reassigned over the constellation's life).
        norad_id: NORAD catalogue id.
        block: GPS block — ``"IIR"``, ``"IIR-M"``, ``"IIF"``, or ``"III"``.
    """

    svn: int
    prn: int
    norad_id: int
    block: str


# The GPS constellation from the CelesTrak gps-ops catalogue, cross-checked against CelesTrak's
# GPSData.txt almanac and GPSrChive. Every NORAD id is confirmed by two independent CelesTrak
# sources. A handful are flagged unusable for navigation by NAVCEN (SVN65/70/79/80) or are newly
# commissioning (SVN83) — they remain catalogued objects with element histories and maneuver
# notices, so they stay in the maneuver-detection catalogue.
GPS_CONSTELLATION: tuple[GpsSatellite, ...] = (
    GpsSatellite(44, 22, 26407, "IIR"),
    GpsSatellite(48, 7, 32711, "IIR-M"),
    GpsSatellite(50, 5, 35752, "IIR-M"),
    GpsSatellite(52, 31, 29486, "IIR-M"),
    GpsSatellite(53, 17, 28874, "IIR-M"),
    GpsSatellite(55, 15, 32260, "IIR-M"),
    GpsSatellite(56, 16, 27663, "IIR"),
    GpsSatellite(57, 29, 32384, "IIR-M"),
    GpsSatellite(58, 12, 29601, "IIR-M"),
    GpsSatellite(59, 19, 28190, "IIR"),
    GpsSatellite(61, 2, 28474, "IIR"),
    GpsSatellite(62, 25, 36585, "IIF"),
    GpsSatellite(64, 30, 39533, "IIF"),
    GpsSatellite(65, 24, 38833, "IIF"),
    GpsSatellite(66, 27, 39166, "IIF"),
    GpsSatellite(67, 6, 39741, "IIF"),
    GpsSatellite(68, 9, 40105, "IIF"),
    GpsSatellite(69, 3, 40294, "IIF"),
    GpsSatellite(70, 32, 41328, "IIF"),
    GpsSatellite(71, 26, 40534, "IIF"),
    GpsSatellite(72, 8, 40730, "IIF"),
    GpsSatellite(73, 10, 41019, "IIF"),
    GpsSatellite(74, 4, 43873, "III"),
    GpsSatellite(75, 18, 44506, "III"),
    GpsSatellite(76, 23, 45854, "III"),
    GpsSatellite(77, 14, 46826, "III"),
    GpsSatellite(78, 11, 48859, "III"),
    GpsSatellite(79, 28, 55268, "III"),
    GpsSatellite(80, 1, 62339, "III"),
    GpsSatellite(81, 21, 64202, "III"),
    GpsSatellite(82, 20, 67588, "III"),
    GpsSatellite(83, 13, 68791, "III"),
)

# DORIS/IDS altimetry satellites with a published man.txt maneuver file: (DORIS code, name, man.txt
# basename prefix). The NORAD id comes from the shared DORIS crosswalk.
_LEO_ALTIMETRY: tuple[tuple[str, str, str], ...] = (
    ("TOPEX", "TOPEX/Poseidon", "top"),
    ("JASO1", "Jason-1", "ja1"),
    ("JASO2", "Jason-2", "ja2"),
    ("JASO3", "Jason-3", "ja3"),
    ("ENVI1", "Envisat", "en1"),
    ("CRYO2", "CryoSat-2", "cs2"),
    ("SARAL", "SARAL", "sr1"),
    ("HY-2A", "HY-2A", "h2a"),
    ("SEN3A", "Sentinel-3A", "s3a"),
    ("SEN3B", "Sentinel-3B", "s3b"),
    ("SEN6A", "Sentinel-6A", "s6a"),
)


def gps_svn_to_norad() -> dict[str, int]:
    """The ``SVN → NORAD`` crosswalk for the constellation (the NANU parser's ``svn_to_norad``)."""
    return {f"SVN{sat.svn}": sat.norad_id for sat in GPS_CONSTELLATION}


def v01_recipe(dataset_version: str = DATASET_VERSION) -> Recipe:
    """Build the pinned v0.1 reconstruction recipe — the LEO altimetry set + the GPS constellation.

    Every object fetches its multi-year series from Space-Track; LEO objects carry DORIS/IDS labels,
    MEO objects carry GPS NANU labels. Entries are ordered by NORAD id for a stable serialisation.
    """
    entries: list[RecipeEntry] = [
        RecipeEntry(
            norad_id=DORIS_SAT_TO_NORAD[code],
            orbit_class=OrbitClass.LEO,
            object_name=name,
            catalogue_source="spacetrack",
            label_source=SOURCE_DORIS_IDS,
            label_ref=man_ref,
        )
        for code, name, man_ref in _LEO_ALTIMETRY
    ]
    entries += [
        RecipeEntry(
            norad_id=sat.norad_id,
            orbit_class=OrbitClass.MEO,
            object_name=f"GPS SVN{sat.svn} (PRN{sat.prn}, {sat.block})",
            catalogue_source="spacetrack",
            label_source=SOURCE_GPS_NANU,
            label_ref=f"SVN{sat.svn}",
        )
        for sat in GPS_CONSTELLATION
    ]
    return Recipe(
        dataset_version=dataset_version,
        entries=tuple(sorted(entries, key=lambda entry: entry.norad_id)),
    )
