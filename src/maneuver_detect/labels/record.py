"""The common maneuver-label record — the seam between heterogeneous operator logs and the labeller.

Every label source speaks a different on-disk format: the DORIS/IDS ``man.txt`` files
(:mod:`~maneuver_detect.labels.doris`) carry fixed-column burn blocks with per-axis ΔV; the GPS
NANUs (:mod:`~maneuver_detect.labels.gps_nanu`) are free-text maneuver-window notices with no ΔV.
Each per-source parser normalises one announced maneuver to a single :class:`ManeuverLabel`, which
the labeller (:mod:`~maneuver_detect.labels.labeller`) then maps onto the inter-elset gap that
brackets it. This module is the single source of truth for that record and its
:func:`to_frame` view.

One :class:`ManeuverLabel` is **one maneuver event** (a multi-burn DORIS event collapses to one
record carrying the event ΔV magnitude and dominant-axis type), so the count of labels matches the
operator's notion of a maneuver rather than a per-burn or per-second granularity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from maneuver_detect.schema import ManeuverType

__all__ = [
    "COLUMNS",
    "SOURCE_DORIS_IDS",
    "SOURCE_GALILEO_NAGU",
    "SOURCE_GPS_NANU",
    "SOURCE_NOAA_GOES",
    "SOURCE_QZSS_OHI",
    "SOURCE_SELF_GEO",
    "SOURCE_SELF_HEO",
    "ManeuverLabel",
    "OrbitClass",
    "to_frame",
]

#: Source tag for the DORIS/IDS ``man.txt`` maneuver files. The same files are the maneuver-history
#: source the ILRS service points at, so this single tag covers both services' history.
SOURCE_DORIS_IDS = "DORIS-IDS"
#: Source tag for the GPS NANU (Notice Advisory to Navstar Users) FCSTDV maneuver notices.
SOURCE_GPS_NANU = "GPS-NANU"
#: Source tag for the Galileo NAGU (Notice Advisory to Galileo Users) PLN_MANV maneuver notices —
#: the second MEO operator feed (European GNSS Service Centre), epoch-only like the GPS NANUs.
SOURCE_GALILEO_NAGU = "GALILEO-NAGU"
#: Source tag for the QZSS Operational History Information (OHI) maneuver logs — the Cabinet Office
#: of Japan's per-satellite executed-maneuver history, carrying actual Δv vector components
#: (the IGSO + GEO operator-truth source). See ``labels.qzss_ohi``.
SOURCE_QZSS_OHI = "QZSS-OHI"
#: Source tag for NOAA GOES operator maneuver epochs from the OSPO navigation-summary / weekly-plan
#: files — US-Government public-domain GEO operator announcements (epoch-only). See ``noaa_goes``.
SOURCE_NOAA_GOES = "NOAA-GOES"
#: Source tag for self-labelled GEO station-keeping epochs derived from the element series itself
#: (longitude-drift inspection) — a **derived, best-effort** source, not an operator announcement,
#: for GEO objects with no public operator maneuver feed. See ``labels.longitude_shift``.
SOURCE_SELF_GEO = "SELF-GEO"
#: Source tag for self-labelled HEO apogee/perigee-control epochs derived from the element series
#: itself (energy/eccentricity-step inspection) — a **derived, best-effort** source, like
#: :data:`SOURCE_SELF_GEO`, for the HEO class which has no public operator maneuver feed. See
#: ``labels.heo_self``.
SOURCE_SELF_HEO = "SELF-HEO"


class OrbitClass(str, Enum):
    """The orbit class of a labelled object — LEO, MEO, GEO, IGSO, or HEO.

    The canonical iteration order (``LEO`` → ``MEO`` → ``GEO`` → ``IGSO`` → ``HEO``) fixes the shape
    of every per-class report (coverage, split counts, scorer), so it is part of the contract.
    ``IGSO`` is the inclined/eccentric-geosynchronous regime (QZSS, operator-Δv); ``HEO`` is the
    high-eccentricity apogee/perigee-control regime (self-labelled, best-effort).
    """

    LEO = "LEO"
    MEO = "MEO"
    GEO = "GEO"
    IGSO = "IGSO"
    HEO = "HEO"


@dataclass(frozen=True)
class ManeuverLabel:
    """One operator-announced maneuver, normalised across sources.

    Attributes:
        norad_id: NORAD catalogue id of the object, or ``None`` when the source identifier has no
            entry in the source's catalogue crosswalk yet (the announcement is still ingested).
        epoch: A representative epoch of the maneuver (timezone-aware UTC) — the primary-burn epoch
            where the source gives one, else the announced-window midpoint. This is the epoch the
            labeller maps onto an inter-elset gap.
        window_start: Start of the announced maneuver window (timezone-aware UTC).
        window_end: End of the announced maneuver window (timezone-aware UTC).
        source: The label source (:data:`SOURCE_DORIS_IDS` / :data:`SOURCE_GPS_NANU`).
        source_ref: A human-readable reference to the specific announcement (e.g. the NANU number),
            for provenance.
        orbit_class: The object's orbit class (:class:`OrbitClass`).
        maneuver_type: The maneuver type from the dominant element change (:class:`ManeuverType`),
            or ``None`` for an epoch-only source that announces no direction.
        delta_v: Estimated ``|Δv|`` in m/s, or ``None`` for an epoch-only source that announces no
            magnitude.
    """

    norad_id: int | None
    epoch: datetime
    window_start: datetime
    window_end: datetime
    source: str
    source_ref: str
    orbit_class: OrbitClass
    maneuver_type: ManeuverType | None = None
    delta_v: float | None = None

    def __post_init__(self) -> None:
        for field_name, ts in (
            ("epoch", self.epoch),
            ("window_start", self.window_start),
            ("window_end", self.window_end),
        ):
            if ts.tzinfo is None:
                raise ValueError(f"{field_name} must be timezone-aware (UTC)")
        if self.window_start > self.window_end:
            raise ValueError(
                f"window_start {self.window_start.isoformat()} is after "
                f"window_end {self.window_end.isoformat()}"
            )
        if self.delta_v is not None and self.delta_v < 0.0:
            raise ValueError(f"delta_v must be non-negative, got {self.delta_v!r}")


#: The columns of the :func:`to_frame` view, in order. ``norad_id`` is nullable (``Int64``);
#: ``delta_v`` is ``NaN`` for an epoch-only source; ``maneuver_type`` is ``<NA>`` likewise.
COLUMNS: tuple[str, ...] = (
    "norad_id",
    "epoch",
    "window_start",
    "window_end",
    "maneuver_type",
    "delta_v",
    "orbit_class",
    "source",
    "source_ref",
)

_DATETIME_DTYPE = "datetime64[ns, UTC]"


def to_frame(labels: Sequence[ManeuverLabel]) -> pd.DataFrame:
    """Serialise ``labels`` to a DataFrame with the canonical :data:`COLUMNS` order and dtypes.

    A nullable ``Int64`` ``norad_id`` carries the un-crosswalked ``None``; ``delta_v`` and
    ``maneuver_type`` carry the epoch-only ``NaN`` / ``<NA>``. An empty sequence yields an empty
    frame that still carries the full schema.
    """
    data = {
        "norad_id": pd.array([m.norad_id for m in labels], dtype="Int64"),
        "epoch": pd.Series([m.epoch for m in labels], dtype=_DATETIME_DTYPE),
        "window_start": pd.Series([m.window_start for m in labels], dtype=_DATETIME_DTYPE),
        "window_end": pd.Series([m.window_end for m in labels], dtype=_DATETIME_DTYPE),
        "maneuver_type": pd.array(
            [None if m.maneuver_type is None else m.maneuver_type.value for m in labels],
            dtype="string",
        ),
        "delta_v": pd.Series(
            [float("nan") if m.delta_v is None else m.delta_v for m in labels], dtype="float64"
        ),
        "orbit_class": pd.Series([m.orbit_class.value for m in labels], dtype="string"),
        "source": pd.Series([m.source for m in labels], dtype="string"),
        "source_ref": pd.Series([m.source_ref for m in labels], dtype="string"),
    }
    return pd.DataFrame(data, columns=list(COLUMNS))
