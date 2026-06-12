"""The dataset catalogue — the objects the recipe reconstructs, as public reference facts.

Three classes (D3, extended by D13):

- **LEO** — the DORIS/IDS satellites that publish a ``man.txt`` maneuver file *and* have a confident
  NORAD id: the altimetry missions (the Δv-labelled core) and the SPOT imaging satellites. A few
  with published files but no crosswalk entry yet (HY-2C/2D, Sentinel-6B, SWOT) are left out.
- **MEO** — two operator constellations: the operational **GPS** constellation (``SVN / PRN /
  NORAD``, labels from the NANU FCSTDV notices) and the **Galileo** constellation (``GSAT / NORAD``,
  labels from the NAGU ``PLN_MANV`` notices). Each table doubles as the source-id → NORAD crosswalk
  its label parser needs, so the dataset layer supplies the full mapping the label layer seeds.
- **GEO** — actively station-kept geostationary satellites; with no public GEO operator maneuver
  feed, their labels are **self-derived** from the element series by longitude-drift inspection
  (best-effort, derived; see :mod:`maneuver_detect.labels.longitude_shift`).

The catalogue is a **pinned snapshot**: a satellite's source-id → NORAD is fixed for its lifetime,
while constellation membership and slot assignments drift over time, so a recipe version pins the
set at sourcing time.
"""

from __future__ import annotations

from dataclasses import dataclass

from maneuver_detect.datasets.recipe import Recipe, RecipeEntry
from maneuver_detect.labels.doris import DORIS_SAT_TO_NORAD
from maneuver_detect.labels.record import (
    SOURCE_DORIS_IDS,
    SOURCE_GALILEO_NAGU,
    SOURCE_GPS_NANU,
    SOURCE_SELF_GEO,
    OrbitClass,
)

__all__ = [
    "DATASET_VERSION",
    "GALILEO_CONSTELLATION",
    "GEO_OBJECTS",
    "GPS_CONSTELLATION",
    "GalileoSatellite",
    "GpsSatellite",
    "galileo_gsat_to_norad",
    "gps_svn_to_norad",
    "recipe",
]

#: The dataset version (in lockstep with a later Hub release and the manifest — D8).
DATASET_VERSION = "0.2.0"


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

# DORIS/IDS satellites with a published man.txt maneuver file: (DORIS code, name, man.txt basename
# prefix). The NORAD id comes from the shared DORIS crosswalk; the prefixes are the IDS filenames.
_LEO_DORIS_SATS: tuple[tuple[str, str, str], ...] = (
    ("TOPEX", "TOPEX/Poseidon", "top"),
    ("JASO1", "Jason-1", "ja1"),
    ("JASO2", "Jason-2", "ja2"),
    ("JASO3", "Jason-3", "ja3"),
    ("ENVI1", "Envisat", "en1"),
    ("CRYO2", "CryoSat-2", "cs2"),
    ("SARAL", "SARAL", "srl"),
    ("HY-2A", "HY-2A", "h2a"),
    ("SEN3A", "Sentinel-3A", "s3a"),
    ("SEN3B", "Sentinel-3B", "s3b"),
    ("SEN6A", "Sentinel-6A", "s6a"),
    ("SPOT2", "SPOT-2", "sp2"),
    ("SPOT3", "SPOT-3", "sp3"),
    ("SPOT4", "SPOT-4", "sp4"),
    ("SPOT5", "SPOT-5", "sp5"),
)


@dataclass(frozen=True)
class GalileoSatellite:
    """One Galileo satellite — its GSAT id, NORAD id, and common name.

    Attributes:
        gsat: The GSAT designation (fixed per physical satellite, e.g. ``"GSAT0101"``) — the key the
            NAGU ``SATELLITE AFFECTED`` field carries and the label crosswalk resolves.
        norad_id: NORAD catalogue id.
        name: Common name (e.g. ``"GALILEO 5"`` / ``"GALILEO-PFM"``).
    """

    gsat: str
    norad_id: int
    name: str


# The Galileo constellation from the CelesTrak galileo catalogue (GSAT → NORAD, every id confirmed
# against CelesTrak). The two early In-Orbit-Validation satellites (GSAT0101/0102) and the eccentric
# GSAT0201/0202 (launched into a wrong orbit but usable) stay catalogued — like the GPS table, an
# object kept for navigation status or not still carries an element history and NAGU notices.
GALILEO_CONSTELLATION: tuple[GalileoSatellite, ...] = (
    GalileoSatellite("GSAT0101", 37846, "GALILEO-PFM"),
    GalileoSatellite("GSAT0102", 37847, "GALILEO-FM2"),
    GalileoSatellite("GSAT0103", 38857, "GALILEO-FM3"),
    GalileoSatellite("GSAT0201", 40128, "GALILEO 5"),
    GalileoSatellite("GSAT0202", 40129, "GALILEO 6"),
    GalileoSatellite("GSAT0203", 40544, "GALILEO 7"),
    GalileoSatellite("GSAT0204", 40545, "GALILEO 8"),
    GalileoSatellite("GSAT0205", 40889, "GALILEO 9"),
    GalileoSatellite("GSAT0206", 40890, "GALILEO 10"),
    GalileoSatellite("GSAT0208", 41175, "GALILEO 11"),
    GalileoSatellite("GSAT0209", 41174, "GALILEO 12"),
    GalileoSatellite("GSAT0210", 41550, "GALILEO 13"),
    GalileoSatellite("GSAT0211", 41549, "GALILEO 14"),
    GalileoSatellite("GSAT0207", 41859, "GALILEO 15"),
    GalileoSatellite("GSAT0212", 41860, "GALILEO 16"),
    GalileoSatellite("GSAT0213", 41861, "GALILEO 17"),
    GalileoSatellite("GSAT0214", 41862, "GALILEO 18"),
    GalileoSatellite("GSAT0215", 43055, "GALILEO 19"),
    GalileoSatellite("GSAT0216", 43056, "GALILEO 20"),
    GalileoSatellite("GSAT0217", 43057, "GALILEO 21"),
    GalileoSatellite("GSAT0218", 43058, "GALILEO 22"),
    GalileoSatellite("GSAT0219", 43566, "GALILEO 23"),
    GalileoSatellite("GSAT0220", 43567, "GALILEO 24"),
    GalileoSatellite("GSAT0221", 43564, "GALILEO 25"),
    GalileoSatellite("GSAT0222", 43565, "GALILEO 26"),
    GalileoSatellite("GSAT0223", 49809, "GALILEO 27"),
    GalileoSatellite("GSAT0224", 49810, "GALILEO 28"),
    GalileoSatellite("GSAT0225", 59598, "GALILEO 29"),
    GalileoSatellite("GSAT0226", 61183, "GALILEO 31"),
    GalileoSatellite("GSAT0227", 59600, "GALILEO 30"),
    GalileoSatellite("GSAT0232", 61182, "GALILEO 32"),
    GalileoSatellite("GSAT0233", 67160, "GALILEO 33"),
    GalileoSatellite("GSAT0234", 67162, "GALILEO 34"),
)

# Actively station-kept geostationary satellites for the self-labelled GEO class: NORAD id + name,
# confirmed GEO and operational against the CelesTrak SATCAT. A mix of tightly-controlled sats (the
# GOES/Himawari weather satellites, both E-W and N-S station-keeping) and inclined-orbit ones (the
# older Meteosat satellites, E-W only) — all with long, dense element histories.
GEO_OBJECTS: tuple[tuple[int, str], ...] = (
    (38552, "Meteosat-10"),
    (40267, "Himawari-8"),
    (40732, "Meteosat-11"),
    (41866, "GOES-16"),
    (43226, "GOES-17"),
    (51850, "GOES-18"),
)


def gps_svn_to_norad() -> dict[str, int]:
    """The ``SVN → NORAD`` crosswalk for the constellation (the NANU parser's ``svn_to_norad``)."""
    return {f"SVN{sat.svn}": sat.norad_id for sat in GPS_CONSTELLATION}


def galileo_gsat_to_norad() -> dict[str, int]:
    """The ``GSAT → NORAD`` crosswalk (the NAGU parser's ``gsat_to_norad``)."""
    return {sat.gsat: sat.norad_id for sat in GALILEO_CONSTELLATION}


def _base_entries() -> list[RecipeEntry]:
    """The LEO altimetry (DORIS/IDS) set + the GPS constellation."""
    entries: list[RecipeEntry] = [
        RecipeEntry(
            norad_id=DORIS_SAT_TO_NORAD[code],
            orbit_class=OrbitClass.LEO,
            object_name=name,
            catalogue_source="spacetrack",
            label_source=SOURCE_DORIS_IDS,
            label_ref=man_ref,
        )
        for code, name, man_ref in _LEO_DORIS_SATS
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
    return entries


def _galileo_entries() -> list[RecipeEntry]:
    """The Galileo MEO entries — NAGU PLN_MANV labels (``label_ref`` = GSAT id)."""
    return [
        RecipeEntry(
            norad_id=sat.norad_id,
            orbit_class=OrbitClass.MEO,
            object_name=f"Galileo {sat.gsat} ({sat.name})",
            catalogue_source="spacetrack",
            label_source=SOURCE_GALILEO_NAGU,
            label_ref=sat.gsat,
        )
        for sat in GALILEO_CONSTELLATION
    ]


def _geo_entries() -> list[RecipeEntry]:
    """The GEO entries — labels self-derived from the series (``label_ref=""``)."""
    return [
        RecipeEntry(
            norad_id=norad_id,
            orbit_class=OrbitClass.GEO,
            object_name=name,
            catalogue_source="spacetrack",
            label_source=SOURCE_SELF_GEO,
            label_ref="",
        )
        for norad_id, name in GEO_OBJECTS
    ]


def recipe(dataset_version: str = DATASET_VERSION) -> Recipe:
    """The pinned reconstruction recipe — every catalogue object and its fetch/label parameters.

    Three classes: the LEO altimetry (DORIS/IDS) set, the MEO constellations (GPS NANU + Galileo
    NAGU labels), and the actively station-kept GEO satellites (labels self-derived from the series
    by longitude-drift inspection). Every object fetches its multi-year series from Space-Track;
    entries are ordered by NORAD id for a stable serialisation.
    """
    entries = _base_entries() + _galileo_entries() + _geo_entries()
    return Recipe(
        dataset_version=dataset_version,
        entries=tuple(sorted(entries, key=lambda entry: entry.norad_id)),
    )
