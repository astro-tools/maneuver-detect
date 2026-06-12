"""Galileo NAGU (PLN_MANV) maneuver-notice parser — the second epoch-only MEO source.

The European GNSS Service Centre (GSC) publishes NAGUs (Notice Advisory to Galileo Users) as flat
``KEY: value`` text notices. The ``PLN_MANV`` type ("planned activity affecting the attitude and/or
orbit") announces a scheduled Galileo station-keeping maneuver as a start/end window in UTC.
:func:`parse_nagus` normalises every ``PLN_MANV`` notice in a file to a
:class:`~maneuver_detect.labels.record.ManeuverLabel`; other NAGU types are ignored.

NAGUs are **epoch-only** — they carry no ΔV magnitude and no maneuver direction, so ``delta_v`` and
``maneuver_type`` are ``None`` and the orbit class is MEO (like the GPS NANUs). A notice carries the
satellite's GSAT id and Space-Vehicle ID, not a NORAD id, so a GSAT→NORAD crosswalk is applied; a
GSAT absent from the crosswalk ingests with ``norad_id=None`` (the epoch is still kept). The full
crosswalk is the CelesTrak Galileo catalogue; the built-in :data:`GALILEO_GSAT_TO_NORAD` is a small
seed a caller overrides.

The announced window is the satellite-*unavailability* span (conservative, often several days); the
maneuver itself is at its onset, so the representative ``epoch`` is the window **start**, not the
midpoint the GPS NANU parser uses for its short maneuver windows.

NAGU content is reproduced from the GSC under its reuse terms (© EU); attribution is carried in the
dataset, not in this code.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import datetime, timezone

from maneuver_detect.labels.record import SOURCE_GALILEO_NAGU, ManeuverLabel, OrbitClass

__all__ = ["GALILEO_GSAT_TO_NORAD", "parse_nagus"]

_logger = logging.getLogger(__name__)

#: The NAGU type that announces a maneuver ("planned activity affecting the attitude and/or orbit").
_MANEUVER_TYPE = "PLN_MANV"

#: Seed GSAT→NORAD crosswalk. The authoritative mapping is the CelesTrak Galileo catalogue; callers
#: override via ``parse_nagus(..., gsat_to_norad=...)``. The two In-Orbit-Validation satellites are
#: confident: GSAT0101 (GALILEO-PFM) = 37846; GSAT0102 (GALILEO-FM2) = 37847.
GALILEO_GSAT_TO_NORAD: dict[str, int] = {
    "GSAT0101": 37846,
    "GSAT0102": 37847,
}


def _field(block: str, key: str) -> str | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*(.+)$", block)
    return match.group(1).strip() if match else None


def _require(block: str, key: str) -> str:
    value = _field(block, key)
    if value is None:
        raise ValueError(f"PLN_MANV NAGU is missing the required field {key!r}")
    return value


def _parse_event_dt(stamp: str) -> datetime:
    """Parse a NAGU event time (``YYYY-MM-DD HH:MM``, already UTC) to a tz-aware datetime."""
    try:
        return datetime.strptime(stamp.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"NAGU event time is not 'YYYY-MM-DD HH:MM': {stamp!r}") from exc


def _parse_block(block: str, gsat_to_norad: Mapping[str, int]) -> ManeuverLabel | None:
    """Parse one NAGU block; return ``None`` for any notice that is not a ``PLN_MANV`` maneuver."""
    if _field(block, "NAGU TYPE") != _MANEUVER_TYPE:
        return None

    gsat = _require(block, "SATELLITE AFFECTED")
    svid = _field(block, "SPACE VEHICLE ID")
    window_start = _parse_event_dt(_require(block, "START DATE EVENT (UTC)"))
    window_end = _parse_event_dt(_require(block, "END DATE EVENT (UTC)"))
    number = _field(block, "NAGU NUMBER") or "?"

    return ManeuverLabel(
        norad_id=gsat_to_norad.get(gsat),
        epoch=window_start,  # the maneuver is at the onset of the unavailability window
        window_start=window_start,
        window_end=window_end,
        source=SOURCE_GALILEO_NAGU,
        source_ref=f"NAGU {number} (PLN_MANV, {gsat}/SVID{svid or '?'})",
        orbit_class=OrbitClass.MEO,
    )


def parse_nagus(
    text: str, *, gsat_to_norad: Mapping[str, int] | None = None
) -> list[ManeuverLabel]:
    """Parse the text of a NAGU file into normalised maneuver labels (``PLN_MANV`` notices only).

    The text is split into individual notices on the ``NAGU TYPE:`` boundary; each ``PLN_MANV``
    notice becomes one epoch-only MEO label, other types are skipped. ``gsat_to_norad`` overrides
    the built-in seed crosswalk (:data:`GALILEO_GSAT_TO_NORAD`); an unmapped GSAT yields
    ``norad_id=None``. A malformed ``PLN_MANV`` notice raises :class:`ValueError`.
    """
    crosswalk = GALILEO_GSAT_TO_NORAD if gsat_to_norad is None else gsat_to_norad
    labels: list[ManeuverLabel] = []
    for block in re.split(r"(?m)(?=^\s*NAGU TYPE:)", text):
        if not block.strip():
            continue
        label = _parse_block(block, crosswalk)
        if label is not None:
            labels.append(label)
    _logger.debug("parsed %d PLN_MANV NAGU labels", len(labels))
    return labels
