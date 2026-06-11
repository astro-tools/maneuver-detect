"""GPS NANU (FCSTDV) maneuver-notice parser — the epoch-only MEO source.

The US Coast Guard NAVCEN publishes NANUs (Notice Advisory to Navstar Users) as public-domain
US-Government text, archived to 1997. The ``FCSTDV`` ("Forecast Delta-V") type announces a scheduled
GPS station-keeping maneuver as a start/stop window. :func:`parse_nanus` normalises every FCSTDV
notice in a file to a :class:`~maneuver_detect.labels.record.ManeuverLabel`; non-FCSTDV notices are
ignored.

NANUs are **epoch-only** — they carry no ΔV magnitude and no maneuver direction, so ``delta_v`` and
``maneuver_type`` are ``None`` and the orbit class is MEO. A notice carries the satellite's SVN/PRN,
not a NORAD id, so a SVN→NORAD crosswalk is applied; an SVN absent from the crosswalk ingests with
``norad_id=None`` (the maneuver epoch is still labelled). The full, current crosswalk is the
CelesTrak GPS catalogue; the built-in :data:`GPS_SVN_TO_NORAD` is a small seed a caller overrides.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from maneuver_detect.labels.record import SOURCE_GPS_NANU, ManeuverLabel, OrbitClass

__all__ = ["GPS_SVN_TO_NORAD", "parse_nanus"]

_logger = logging.getLogger(__name__)

#: Seed SVN→NORAD crosswalk. The authoritative, current mapping is the CelesTrak GPS catalogue;
#: callers override via ``parse_nanus(..., svn_to_norad=...)``. SVN62 = GPS IIF-1 (USA-213) = 36585.
GPS_SVN_TO_NORAD: dict[str, int] = {
    "SVN62": 36585,
}


def _field(block: str, key: str) -> str | None:
    # NANU fields may be indented and carry a list-number prefix, e.g. "1.     NANU TYPE: FCSTDV".
    match = re.search(rf"(?m)^\s*(?:\d+\.\s+)?{re.escape(key)}:\s*(.+)$", block)
    return match.group(1).strip() if match else None


def _require(block: str, key: str) -> str:
    value = _field(block, key)
    if value is None:
        raise ValueError(f"FCSTDV NANU is missing the required field {key!r}")
    return value


#: Month abbreviations as they appear in a NANU DTG (``DDHHMMZ MON YYYY``).
_DTG_MONTHS = {
    name: number
    for number, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}

#: A NANU is a forward-looking forecast, so its maneuver window falls on or shortly after the
#: notice's DTG. If a day-of-year placed in the DTG's own calendar year would land this many days
#: *before* the DTG, the maneuver belongs to the following year instead — a late-December notice
#: announcing an early-January burn, or a window whose STOP wraps past 31 December. Forecasts never
#: look months ahead, so the threshold only ever catches the year rollover.
_FORECAST_LOOKBACK_DAYS = 180


def _parse_dtg(dtg: str) -> datetime:
    """Parse a NANU DTG (``DDHHMMZ MON YYYY``) into a timezone-aware UTC datetime."""
    match = re.match(r"(?i)^(\d{2})(\d{2})(\d{2})Z\s+([A-Z]{3})\s+(\d{4})$", dtg.strip())
    if match is None:
        raise ValueError(f"FCSTDV NANU DTG is not 'DDHHMMZ MON YYYY': {dtg!r}")
    day, hour, minute, month_name, year = match.groups()
    month = _DTG_MONTHS.get(month_name.upper())
    if month is None:
        raise ValueError(f"FCSTDV NANU DTG names an unknown month: {dtg!r}")
    return datetime(int(year), month, int(day), int(hour), int(minute), tzinfo=timezone.utc)


def _window_dt(year: int, jday: int, zulu: str) -> datetime:
    """Day-of-year + ``HHMM`` Zulu → timezone-aware UTC datetime (NANU times are UTC)."""
    if len(zulu) != 4 or not zulu.isdigit():
        raise ValueError(f"NANU time-of-day is not HHMM Zulu: {zulu!r}")
    base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=jday - 1)
    return base.replace(hour=int(zulu[:2]), minute=int(zulu[2:]))


def _maneuver_window(dtg: datetime, jday: int, zulu: str) -> datetime:
    """Resolve a START/STOP day-of-year against the DTG, rolling to the next calendar year when the
    JDAY would otherwise sit far in the DTG's past (the year-boundary case). Applied independently
    to START and STOP, this dates a window that wraps past New Year's correctly without any
    cross-field comparison: only the small post-rollover JDAY trips the lookback.
    """
    when = _window_dt(dtg.year, jday, zulu)
    if dtg - when > timedelta(days=_FORECAST_LOOKBACK_DAYS):
        when = _window_dt(dtg.year + 1, jday, zulu)
    return when


def _parse_block(block: str, svn_to_norad: Mapping[str, int]) -> ManeuverLabel | None:
    """Parse one NANU block; return ``None`` for any notice that is not an FCSTDV maneuver."""
    if _field(block, "NANU TYPE") != "FCSTDV":
        return None

    dtg = _parse_dtg(_require(block, "NANU DTG"))
    svn = _require(block, "SVN")
    prn = _field(block, "PRN")
    window_start = _maneuver_window(
        dtg, int(_require(block, "START JDAY")), _require(block, "START TIME ZULU")
    )
    window_end = _maneuver_window(
        dtg, int(_require(block, "STOP JDAY")), _require(block, "STOP TIME ZULU")
    )
    number = _field(block, "NANU NUMBER") or "?"

    return ManeuverLabel(
        norad_id=svn_to_norad.get(f"SVN{svn}"),
        epoch=window_start + (window_end - window_start) / 2,
        window_start=window_start,
        window_end=window_end,
        source=SOURCE_GPS_NANU,
        source_ref=f"NANU {number} (FCSTDV, SVN{svn}/PRN{prn or '?'})",
        orbit_class=OrbitClass.MEO,
    )


def parse_nanus(text: str, *, svn_to_norad: Mapping[str, int] | None = None) -> list[ManeuverLabel]:
    """Parse the text of a NANU file into normalised maneuver labels (FCSTDV notices only).

    The file is split into individual notices on the ``NANU TYPE:`` boundary; each FCSTDV notice
    becomes one epoch-only MEO label, others are skipped. ``svn_to_norad`` overrides the built-in
    seed crosswalk (:data:`GPS_SVN_TO_NORAD`); an unmapped SVN yields ``norad_id=None``. A malformed
    FCSTDV notice raises :class:`ValueError`.
    """
    crosswalk = GPS_SVN_TO_NORAD if svn_to_norad is None else svn_to_norad
    labels: list[ManeuverLabel] = []
    for block in re.split(r"(?m)(?=^\s*(?:\d+\.\s+)?NANU TYPE:)", text):
        if not block.strip():
            continue
        label = _parse_block(block, crosswalk)
        if label is not None:
            labels.append(label)
    _logger.debug("parsed %d FCSTDV NANU labels", len(labels))
    return labels
