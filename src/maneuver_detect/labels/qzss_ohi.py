"""QZSS Operational History Information (OHI) parser — the IGSO + GEO operator-Δv source.

The Cabinet Office of Japan publishes, per Quasi-Zenith satellite, an *Operational History
Information* file recording the satellite's executed orbit-maintenance maneuvers with their window
and a Δv vector — the only surveyed operator feed that gives a real, executed Δv (not just an outage
window). :func:`parse_qzss_ohi` turns one OHI file's maneuver section into normalised
:class:`~maneuver_detect.labels.record.ManeuverLabel` records; fetching is a recipe concern (the
parser never touches the network).

Format (flat text with ``#+SECTION`` / ``#-SECTION`` delimited blocks; the maneuver block is)::

    #+SATELLITE/MANEUVER
    #DATE TIME START(UTC),END(UTC),DURATION,DVX(m/s),DVY(m/s),DVZ(m/s)
    2017-11-15 11:03:31,2017-11-15 11:05:50,00:02:19,-2.325,0.004,0.032
    ...
    #-SATELLITE/MANEUVER

The GEO satellites (QZS-3/6) instead use a 7-column layout with an explicit ``NS/EW`` maneuver-type
marker before the Δv vector::

    #DATE TIME START(UTC),END(UTC),DURATION,NS/EW,DVX(m/s),DVY(m/s),DVZ(m/s)
    2017-10-12 08:33:48,2017-10-12 08:34:07,00:00:19,NS,-0.001,0.290,0.001

Two modelling choices, both documented in the v0.3 source spike:

- **Type from the operator's ``NS/EW`` marker, where the file gives one.** The GEO files mark each
  burn ``NS`` (north-south = inclination control → cross-track) or ``EW`` (east-west = longitude
  control → in-track) — the operator's own classification, so it is used directly. The IGSO files
  omit the marker and the raw ``DVX/DVY/DVZ`` axes carry no documented reference frame, so those
  labels are **magnitude-only** (``maneuver_type = None``) rather than fabricating a frame. Either
  way the frame-invariant ``|Δv| = ‖(DVX, DVY, DVZ)‖`` magnitude is kept.
- **Burns are collapsed into events.** A station-keeping campaign is a cluster of burns hours apart
  (e.g. a two-impulse ± pair, or a combined inclination + longitude campaign), then the next is
  weeks-to-months later. Consecutive burns within :data:`EVENT_GAP` of one another are collapsed to
  one maneuver event (the D4 "one label = one operator maneuver event" granularity), whose Δv is the
  **sum of the burns' magnitudes** (total campaign expenditure — robust to the ± cancellation a
  vector sum would suffer on a two-impulse correction) and whose type is the **dominant** (largest
  ``|Δv|``) burn's ``NS/EW`` marker.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from maneuver_detect.labels.record import SOURCE_QZSS_OHI, ManeuverLabel, OrbitClass
from maneuver_detect.schema import ManeuverType

__all__ = ["EVENT_GAP", "parse_qzss_ohi"]

_logger = logging.getLogger(__name__)

#: Section markers bracketing the executed-maneuver block in an OHI file.
_SECTION_START = "#+SATELLITE/MANEUVER"
_SECTION_END = "#-SATELLITE/MANEUVER"

#: The operator ``NS/EW`` burn marker → maneuver type: north-south = inclination control
#: (cross-track), east-west = longitude/drift control (in-track).
_MARKER_TYPE = {"NS": ManeuverType.CROSS_TRACK, "EW": ManeuverType.IN_TRACK}

#: Maximum gap between one burn's end and the next burn's start for them to belong to the same
#: maneuver event. Within-campaign burns are hours apart; successive campaigns are weeks-to-months
#: apart, so this cleanly separates events while collapsing a clustered campaign into one.
EVENT_GAP = timedelta(days=2)

_OHI_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class _Burn:
    """One parsed OHI burn row: window, |Δv| magnitude, and the operator NS/EW marker (if any)."""

    start: datetime
    end: datetime
    magnitude: float
    marker: str | None


def _parse_utc(value: str) -> datetime | None:
    """Parse an OHI ``YYYY-MM-DD HH:MM:SS`` timestamp as UTC, or ``None`` for the null sentinel."""
    text = value.strip()
    if not text or text.startswith("0000-00-00"):
        return None
    return datetime.strptime(text, _OHI_TIME_FORMAT).replace(tzinfo=timezone.utc)


def _maneuver_rows(text: str) -> list[_Burn]:
    """Extract one :class:`_Burn` per row from the file's maneuver section, in file order.

    Handles both layouts: the 6-column ``...,DURATION,DVX,DVY,DVZ`` (IGSO) and the 7-column
    ``...,DURATION,NS/EW,DVX,DVY,DVZ`` (GEO) — the NS/EW marker is captured when present.
    """
    rows: list[_Burn] = []
    in_section = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == _SECTION_START:
            in_section = True
            continue
        if line == _SECTION_END:
            break
        if not in_section or not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            raise ValueError(f"QZSS OHI maneuver row has too few fields: {line!r}")
        if len(fields) >= 7:  # 7-column GEO layout: an NS/EW marker precedes the Δv vector
            marker: str | None = fields[3].upper()
            dv = fields[4:7]
        else:  # 6-column IGSO layout: no marker
            marker = None
            dv = fields[3:6]
        start = _parse_utc(fields[0])
        end = _parse_utc(fields[1])
        if start is None or end is None:
            continue
        dvx, dvy, dvz = (float(dv[0]), float(dv[1]), float(dv[2]))
        rows.append(_Burn(start, end, math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz), marker))
    return rows


def parse_qzss_ohi(
    text: str,
    *,
    norad_id: int | None,
    orbit_class: OrbitClass,
    qzs_label: str,
    event_gap: timedelta = EVENT_GAP,
) -> list[ManeuverLabel]:
    """Parse one QZSS OHI file's maneuver section into normalised maneuver labels.

    ``norad_id`` and ``orbit_class`` are the satellite's pinned catalogue values (an OHI file is
    per-satellite, so the class — IGSO for QZS-1R/2/4, GEO for QZS-3/6 — is fixed by the caller, not
    derived here); ``qzs_label`` (e.g. ``"QZS-2"``) tags provenance. Consecutive burns within
    ``event_gap`` are collapsed into one event whose Δv is the sum of the burn magnitudes and whose
    type is the dominant burn's ``NS/EW`` marker (``None`` where the file omits it — the IGSO
    layout). Returns labels in chronological order.
    """
    rows = _maneuver_rows(text)
    if not rows:
        return []
    rows.sort(key=lambda burn: burn.start)

    labels: list[ManeuverLabel] = []
    first = rows[0]
    event_start, event_end, last_end = first.start, first.end, first.end
    event_dv, event_peak, event_marker = first.magnitude, first.magnitude, first.marker
    for burn in rows[1:]:
        if burn.start - last_end <= event_gap:  # same campaign: extend the event
            event_end = max(event_end, burn.end)
            event_dv += burn.magnitude
            if burn.magnitude > event_peak:  # the dominant burn sets the event type
                event_peak, event_marker = burn.magnitude, burn.marker
            last_end = max(last_end, burn.end)
        else:  # a new campaign: flush the previous event
            labels.append(
                _event_label(
                    norad_id, orbit_class, qzs_label, event_start, event_end, event_dv, event_marker
                )
            )
            event_start, event_end, event_dv = burn.start, burn.end, burn.magnitude
            event_marker, event_peak, last_end = burn.marker, burn.magnitude, burn.end
    labels.append(
        _event_label(
            norad_id, orbit_class, qzs_label, event_start, event_end, event_dv, event_marker
        )
    )
    _logger.debug("parsed %d QZSS OHI maneuver events for %s", len(labels), qzs_label)
    return labels


def _event_label(
    norad_id: int | None,
    orbit_class: OrbitClass,
    qzs_label: str,
    start: datetime,
    end: datetime,
    delta_v: float,
    marker: str | None,
) -> ManeuverLabel:
    """Build one collapsed-event label (epoch = the event's first burn start; type from NS/EW)."""
    return ManeuverLabel(
        norad_id=norad_id,
        epoch=start,
        window_start=start,
        window_end=end,
        source=SOURCE_QZSS_OHI,
        source_ref=f"QZSS OHI {qzs_label} {start.date()}",
        orbit_class=orbit_class,
        maneuver_type=_MARKER_TYPE.get(marker) if marker is not None else None,
        delta_v=delta_v,
    )
