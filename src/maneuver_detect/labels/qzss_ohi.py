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

Two modelling choices, both documented in the v0.3 source spike:

- **Magnitude-only, no maneuver type.** The ``DVX/DVY/DVZ`` axes carry no documented reference frame
  in the file, so a rigorous in-track / cross-track / radial classification is not derivable. The Δv
  *magnitude* ``‖(DVX, DVY, DVZ)‖`` is frame-invariant and is kept; the type is left ``None`` (a
  Δv-labelled but direction-untyped source), rather than fabricating a frame.
- **Burns are collapsed into events.** A station-keeping campaign is a cluster of burns hours apart
  (e.g. a two-impulse ± pair, or a multi-burn inclination campaign), then the next campaign is
  weeks-to-months later. Consecutive burns within :data:`EVENT_GAP` of one another are collapsed to
  one maneuver event (the D4 "one label = one operator maneuver event" granularity), whose Δv is the
  **sum of the burns' magnitudes** (total campaign expenditure — robust to the ± cancellation a
  vector sum would suffer on a two-impulse correction).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from maneuver_detect.labels.record import SOURCE_QZSS_OHI, ManeuverLabel, OrbitClass

__all__ = ["EVENT_GAP", "parse_qzss_ohi"]

_logger = logging.getLogger(__name__)

#: Section markers bracketing the executed-maneuver block in an OHI file.
_SECTION_START = "#+SATELLITE/MANEUVER"
_SECTION_END = "#-SATELLITE/MANEUVER"

#: Maximum gap between one burn's end and the next burn's start for them to belong to the same
#: maneuver event. Within-campaign burns are hours apart; successive campaigns are weeks-to-months
#: apart, so this cleanly separates events while collapsing a clustered campaign into one.
EVENT_GAP = timedelta(days=2)

_OHI_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_utc(value: str) -> datetime | None:
    """Parse an OHI ``YYYY-MM-DD HH:MM:SS`` timestamp as UTC, or ``None`` for the null sentinel."""
    text = value.strip()
    if not text or text.startswith("0000-00-00"):
        return None
    return datetime.strptime(text, _OHI_TIME_FORMAT).replace(tzinfo=timezone.utc)


def _maneuver_rows(text: str) -> list[tuple[datetime, datetime, float]]:
    """Extract ``(start, end, |Δv|)`` per burn from the file's maneuver section, in file order."""
    rows: list[tuple[datetime, datetime, float]] = []
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
        start = _parse_utc(fields[0])
        end = _parse_utc(fields[1])
        if start is None or end is None:
            continue
        dvx, dvy, dvz = (float(fields[3]), float(fields[4]), float(fields[5]))
        rows.append((start, end, math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)))
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
    ``event_gap`` are collapsed into one event whose Δv is the sum of the burn magnitudes; the type
    is ``None`` (the file's Δv frame is undocumented). Returns labels in chronological order.
    """
    rows = _maneuver_rows(text)
    if not rows:
        return []
    rows.sort(key=lambda row: row[0])

    labels: list[ManeuverLabel] = []
    event_start, event_end, event_dv = rows[0]
    last_end = event_end
    for start, end, magnitude in rows[1:]:
        if start - last_end <= event_gap:  # same campaign: extend the event
            event_end = max(event_end, end)
            event_dv += magnitude
            last_end = max(last_end, end)
        else:  # a new campaign: flush the previous event
            labels.append(
                _event_label(norad_id, orbit_class, qzs_label, event_start, event_end, event_dv)
            )
            event_start, event_end, event_dv = start, end, magnitude
            last_end = end
    labels.append(_event_label(norad_id, orbit_class, qzs_label, event_start, event_end, event_dv))
    _logger.debug("parsed %d QZSS OHI maneuver events for %s", len(labels), qzs_label)
    return labels


def _event_label(
    norad_id: int | None,
    orbit_class: OrbitClass,
    qzs_label: str,
    start: datetime,
    end: datetime,
    delta_v: float,
) -> ManeuverLabel:
    """Build one collapsed-event label (epoch = the event's first burn start; type ``None``)."""
    return ManeuverLabel(
        norad_id=norad_id,
        epoch=start,
        window_start=start,
        window_end=end,
        source=SOURCE_QZSS_OHI,
        source_ref=f"QZSS OHI {qzs_label} {start.date()}",
        orbit_class=orbit_class,
        maneuver_type=None,
        delta_v=delta_v,
    )
