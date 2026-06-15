"""The dataset catalogue — the objects the recipe reconstructs, as public reference facts.

Four populated classes + a reserved one (D3, extended by D13 and the v0.3 source survey):

- **LEO** — the DORIS/IDS satellites that publish a ``man.txt`` maneuver file *and* have a confident
  NORAD id: the altimetry missions (the Δv-labelled core) and the SPOT imaging satellites. A few
  with published files but no crosswalk entry yet (HY-2C/2D, Sentinel-6B, SWOT) are left out.
- **MEO** — two operator constellations: the operational **GPS** constellation (``SVN / PRN /
  NORAD``, labels from the NANU FCSTDV notices) and the **Galileo** constellation (``GSAT / NORAD``,
  labels from the NAGU ``PLN_MANV`` notices). Each table doubles as the source-id → NORAD crosswalk
  its label parser needs, so the dataset layer supplies the full mapping the label layer seeds.
- **GEO** — geostationary satellites. The **GOES** weather satellites carry operator-announced
  labels from the NOAA OSPO navigation summary (US-Government public domain); the Meteosat/Himawari
  satellites have no public operator feed, so their labels are **self-derived** from the element
  series by longitude-drift inspection (best-effort; see
  :mod:`maneuver_detect.labels.longitude_shift`). The two equatorial **QZSS** satellites (QZS-3/6)
  are GEO with operator-Δv OHI labels.
- **IGSO** — the inclined/eccentric-geosynchronous **QZSS** satellites (QZS-2/4/1R), labelled from
  the Cabinet Office of Japan's Operational History Information (OHI) files, the only surveyed
  operator feed that ships an executed Δv (:mod:`maneuver_detect.labels.qzss_ohi`).
- **HEO** — high-eccentricity apogee/perigee-control regime: a **reserved class with no objects** in
  v0.3. No machine-ingestible maneuver source exists (operator records are prose/PDF or
  ephemeris-only), and self-labelling from the noisy deep-space TLEs is perturbation-dominated, so
  HEO is deferred (D15). The enum member and the :mod:`maneuver_detect.labels.heo_self` deriver are
  retained for a future source.

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
    SOURCE_NOAA_GOES,
    SOURCE_QZSS_OHI,
    SOURCE_SELF_GEO,
    OrbitClass,
)

__all__ = [
    "DATASET_VERSION",
    "GALILEO_CONSTELLATION",
    "GOES_OBJECTS",
    "GPS_CONSTELLATION",
    "QZSS_CONSTELLATION",
    "SELF_GEO_OBJECTS",
    "GalileoSatellite",
    "GpsSatellite",
    "QzssSatellite",
    "galileo_gsat_to_norad",
    "goes_name_to_norad",
    "gps_svn_to_norad",
    "recipe",
]

#: The dataset version (in lockstep with a later Hub release and the manifest — D8).
DATASET_VERSION = "0.3.0"


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


@dataclass(frozen=True)
class QzssSatellite:
    """One Quasi-Zenith (QZSS) satellite — its label, NORAD id, orbit class, and OHI file reference.

    Attributes:
        label: The common designation (e.g. ``"QZS-2"``), used for provenance.
        norad_id: NORAD catalogue id.
        orbit_class: ``IGSO`` for the inclined/eccentric-geosynchronous birds (QZS-2/4/1R) or
            ``GEO`` for the equatorial ones (QZS-3/6) — fixed per satellite, not from elements.
        ohi_ref: The OHI file stem the label fetch uses (``ohi-{ohi_ref}.txt``, e.g. ``"qzs2"``).
    """

    label: str
    norad_id: int
    orbit_class: OrbitClass
    ohi_ref: str


# The QZSS constellation from the CelesTrak SATCAT, classed by orbit geometry: QZS-2/4/1R
# are inclined/eccentric geosynchronous (e~0.075, i~37-44 deg) -> IGSO; QZS-3/6 equatorial -> GEO.
# The decommissioned QZS-1 (drifted off-station) is left out. Each carries an OHI maneuver log.
QZSS_CONSTELLATION: tuple[QzssSatellite, ...] = (
    QzssSatellite("QZS-2", 42738, OrbitClass.IGSO, "qzs2"),
    QzssSatellite("QZS-4", 42965, OrbitClass.IGSO, "qzs4"),
    QzssSatellite("QZS-1R", 49336, OrbitClass.IGSO, "qzs1r"),
    QzssSatellite("QZS-3", 42917, OrbitClass.GEO, "qzs3"),
    QzssSatellite("QZS-6", 62876, OrbitClass.GEO, "qzs6"),
)

# The GOES weather satellites with operator-announced labels from the NOAA OSPO navigation summary
# (US-Government public domain): NORAD id + the navsum spacecraft name (the crosswalk key). All
# confirmed GEO against the CelesTrak SATCAT, with long, dense element histories.
GOES_OBJECTS: tuple[tuple[int, str], ...] = (
    (41866, "GOES-16"),
    (43226, "GOES-17"),
    (51850, "GOES-18"),
    (60133, "GOES-19"),
)

# Station-kept geostationary satellites with no public operator maneuver feed, for the self-labelled
# GEO track: NORAD id + name, confirmed GEO against the CelesTrak SATCAT. The Himawari weather
# satellite (E-W + N-S station-keeping) and the inclined-orbit older Meteosat satellites (E-W only).
SELF_GEO_OBJECTS: tuple[tuple[int, str], ...] = (
    (38552, "Meteosat-10"),
    (40267, "Himawari-8"),
    (40732, "Meteosat-11"),
)

# HEO (high-eccentricity) is a reserved class with no objects in v0.3: there is no ingestible HEO
# maneuver source (operator feeds are prose/PDF or ephemeris-only — re-deriving maneuvers from
# ephemeris is circular), and self-labelling from the noisy deep-space TLEs of HEO objects measured
# as perturbation-dominated, not maneuvers. So no HEO objects are catalogued; the class, the floor
# entry, and the ``labels.heo_self`` deriver are retained for a future source. See D15 + the spike.


def gps_svn_to_norad() -> dict[str, int]:
    """The ``SVN → NORAD`` crosswalk for the constellation (the NANU parser's ``svn_to_norad``)."""
    return {f"SVN{sat.svn}": sat.norad_id for sat in GPS_CONSTELLATION}


def galileo_gsat_to_norad() -> dict[str, int]:
    """The ``GSAT → NORAD`` crosswalk (the NAGU parser's ``gsat_to_norad``)."""
    return {sat.gsat: sat.norad_id for sat in GALILEO_CONSTELLATION}


def goes_name_to_norad() -> dict[str, int]:
    """The navsum ``GOES-N → NORAD`` crosswalk (the NOAA GOES parser's ``goes_name_to_norad``)."""
    return {name: norad_id for norad_id, name in GOES_OBJECTS}


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


def _qzss_entries() -> list[RecipeEntry]:
    """The QZSS entries — IGSO + GEO, operator-Δv OHI labels (``label_ref`` = the OHI file stem)."""
    return [
        RecipeEntry(
            norad_id=sat.norad_id,
            orbit_class=sat.orbit_class,
            object_name=f"QZSS {sat.label}",
            catalogue_source="spacetrack",
            label_source=SOURCE_QZSS_OHI,
            label_ref=sat.ohi_ref,
        )
        for sat in QZSS_CONSTELLATION
    ]


def _goes_entries() -> list[RecipeEntry]:
    """The GOES GEO entries — NOAA navsum labels (``label_ref`` = the navsum spacecraft name)."""
    return [
        RecipeEntry(
            norad_id=norad_id,
            orbit_class=OrbitClass.GEO,
            object_name=name,
            catalogue_source="spacetrack",
            label_source=SOURCE_NOAA_GOES,
            label_ref=name,
        )
        for norad_id, name in GOES_OBJECTS
    ]


def _self_geo_entries() -> list[RecipeEntry]:
    """The self-labelled GEO entries — labels derived from the series (``label_ref=""``)."""
    return [
        RecipeEntry(
            norad_id=norad_id,
            orbit_class=OrbitClass.GEO,
            object_name=name,
            catalogue_source="spacetrack",
            label_source=SOURCE_SELF_GEO,
            label_ref="",
        )
        for norad_id, name in SELF_GEO_OBJECTS
    ]


def recipe(dataset_version: str = DATASET_VERSION) -> Recipe:
    """The pinned reconstruction recipe — every catalogue object and its fetch/label parameters.

    Four populated classes: the LEO altimetry (DORIS/IDS) set; the MEO constellations (GPS NANU +
    Galileo NAGU labels); the GEO satellites (GOES from the NOAA navsum, QZS-3/6 from QZSS OHI,
    Meteosat/Himawari self-derived by longitude-drift); and the IGSO QZSS satellites (operator-Δv
    OHI labels). HEO is a reserved class with no objects (no ingestible maneuver source — see D15).
    Every object fetches its multi-year series from Space-Track; entries are ordered by NORAD id for
    a stable serialisation.
    """
    entries = (
        _base_entries()
        + _galileo_entries()
        + _qzss_entries()
        + _goes_entries()
        + _self_geo_entries()
    )
    return Recipe(
        dataset_version=dataset_version,
        entries=tuple(sorted(entries, key=lambda entry: entry.norad_id)),
    )
