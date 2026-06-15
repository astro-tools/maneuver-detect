"""NOAA GOES navigation-summary parser — the US-Government public-domain GEO operator source.

NOAA's Office of Satellite and Product Operations publishes a *navigation summary* file
(``navsum.txt``) for the operational GOES fleet — one block per spacecraft carrying its current
osculating elements and a Comments footer stating the day of its **last maneuver**::

    Spacecraft :                                  GOES-16
    ...
    Comments:
    GOES-16 data propagated from orbit epoch 26/160 05:00:00.000 UTC.
    Fuel and oxidizer remaining are estimates after the last maneuver on 26/159.

This is an operator-announced GEO maneuver epoch — the non-circular ground truth the self-labelled
longitude-shift method (:mod:`~maneuver_detect.labels.longitude_shift`) lacks — and, being a NOAA
product, it is US-Government public domain (ship-labels). :func:`parse_navsum` turns one navsum
snapshot into normalised :class:`~maneuver_detect.labels.record.ManeuverLabel` records.

``navsum.txt`` is a **live-state** file: each snapshot reports only the *latest* maneuver per
spacecraft, at day-of-year granularity. A maneuver *history* is built by collecting the distinct
last-maneuver epochs across many snapshots over time (the reconstruction leg crawls the Internet
Archive for them); this parser handles one snapshot. Labels are epoch-only (no Δv, no type) and
GEO-classed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from maneuver_detect.labels.record import SOURCE_NOAA_GOES, ManeuverLabel, OrbitClass

__all__ = ["parse_navsum"]

_logger = logging.getLogger(__name__)

#: Block delimiter between spacecraft entries in ``navsum.txt``.
_BLOCK_DELIMITER = re.compile(r"^=+$", re.MULTILINE)
#: The spacecraft name line (e.g. ``Spacecraft :  GOES-16``).
_SPACECRAFT = re.compile(r"Spacecraft\s*:\s*(GOES-\d+)", re.IGNORECASE)
#: The Comments-footer last-maneuver epoch, ``yy/ddd`` (year-of-century / day-of-year).
_LAST_MANEUVER = re.compile(r"last maneuver on\s+(\d{2})/(\d{1,3})", re.IGNORECASE)


def _yy_ddd_to_utc_day(yy: int, ddd: int) -> datetime:
    """Convert a ``yy/ddd`` operator epoch to the UTC start-of-day datetime (2000-relative year)."""
    return datetime(2000 + yy, 1, 1, tzinfo=timezone.utc) + timedelta(days=ddd - 1)


def parse_navsum(text: str, *, goes_name_to_norad: Mapping[str, int]) -> list[ManeuverLabel]:
    """Parse one ``navsum.txt`` snapshot into the fleet's latest-maneuver labels.

    Each spacecraft block names the satellite and (in its Comments footer) its last-maneuver day at
    ``yy/ddd`` granularity. For every block whose ``GOES-N`` name resolves through
    ``goes_name_to_norad`` and that states a last-maneuver epoch, one epoch-only GEO
    :class:`~maneuver_detect.labels.record.ManeuverLabel` is emitted: the epoch is noon of the named
    UTC day and the window is that whole UTC day (the operator gives no finer time). A block without
    a name match or without a last-maneuver footer is skipped. Returns labels in block order; a
    snapshot reports at most one (the latest) maneuver per spacecraft.
    """
    labels: list[ManeuverLabel] = []
    # navsum.txt is served with CRLF endings; normalise so the block delimiter (a line of ``=``)
    # and the line-anchored field regexes match regardless of the source's line endings.
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    for block in _BLOCK_DELIMITER.split(normalised):
        name_match = _SPACECRAFT.search(block)
        epoch_match = _LAST_MANEUVER.search(block)
        if name_match is None or epoch_match is None:
            continue
        name = name_match.group(1).upper()
        norad_id = goes_name_to_norad.get(name)
        day_start = _yy_ddd_to_utc_day(int(epoch_match.group(1)), int(epoch_match.group(2)))
        labels.append(
            ManeuverLabel(
                norad_id=norad_id,
                epoch=day_start + timedelta(hours=12),
                window_start=day_start,
                window_end=day_start + timedelta(days=1),
                source=SOURCE_NOAA_GOES,
                source_ref=f"{name} navsum last-maneuver {day_start.date()}",
                orbit_class=OrbitClass.GEO,
            )
        )
    _logger.debug("parsed %d NOAA GOES last-maneuver labels from a navsum snapshot", len(labels))
    return labels
