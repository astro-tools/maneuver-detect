"""DORIS/IDS ``man.txt`` maneuver-file parser — the Δv-labelled LEO source.

The International DORIS Service (and, mirrored, NASA CDDIS) distributes one fixed-format maneuver
file per altimetry satellite — ``<code>man.txt`` (e.g. ``ja2man.txt``) governed by the published
``man.readme`` — recording every orbit-maintenance maneuver with its window and per-axis ΔV. These
are the richest public labels: LEO, with Δv. :func:`parse_doris` turns the file text into
normalised :class:`~maneuver_detect.labels.record.ManeuverLabel` records; fetching is a recipe
concern (the parser never touches the network).

**This parser also covers the ILRS maneuver-history source.** The ILRS service does not publish a
separate maneuver-file format; its maneuver *history* links out to these same IDS ``man.txt`` files,
and its maneuver *predictions* live inside the CPF ephemeris (no ΔV record) or a free-text
notification e-mail (no fixed schema). So the DORIS/IDS parser is the single ingest path for both
services' quantitative labels.

Format (one maneuver event per line, whitespace-delimited; times are **TAI**, day-of-year based):

- ``<code> <by> <bdoy> <bh> <bm> <ey> <edoy> <eh> <em>`` — the begin / end maneuver window. TOPEX
  lines stop here (window only, no burn detail).
- then, for the other satellites, an optional SPOT-only type token (``MCC`` routine / ``MCO``
  inclination), a **parameter-type code** that fixes the ΔV axis ordering (``005`` SPOT ``T,R,L``;
  ``006`` ENVISAT/CryoSat ``radial,along-track,cross-track``; ``007`` JASON ``Q,S,W``), a burn count
  ``N``, and ``N`` repeating 15-token burn blocks (median epoch + duration + three ΔV components in
  m/s + accelerations).

The event ΔV is the magnitude of the burns' vector sum; the maneuver type is the dominant summed
axis mapped onto :class:`~maneuver_detect.schema.ManeuverType` (along-track ≡ in-track).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from maneuver_detect.labels.record import SOURCE_DORIS_IDS, ManeuverLabel, OrbitClass
from maneuver_detect.schema import ManeuverType

__all__ = ["DORIS_SAT_TO_NORAD", "parse_doris"]

_logger = logging.getLogger(__name__)

#: DORIS 5-character satellite code → NORAD catalogue id, for the v0.1 altimetry set (all LEO).
#: These are stable public-catalogue ids; codes absent here (e.g. newer HY-2C/2D, SWOT) ingest with
#: ``norad_id=None`` until added. The authoritative correspondence is the IDS / CDDIS spacecraft-id
#: table.
DORIS_SAT_TO_NORAD: dict[str, int] = {
    "TOPEX": 22076,
    "JASO1": 26997,
    "JASO2": 33105,
    "JASO3": 41240,
    "ENVI1": 27386,
    "CRYO2": 36508,
    "SARAL": 39086,
    "HY-2A": 37781,
    "SEN3A": 41335,
    "SEN3B": 43437,
    "SEN6A": 46984,
    "SPOT2": 20436,
    "SPOT3": 22823,
    "SPOT4": 25260,
    "SPOT5": 27421,
}

# Parameter-type code → the ManeuverType of each ΔV component, in the file's axis order. Per the
# IDS man.readme, the axis labels cross-reference as Q = L = radial, W = T = cross-track, S = R =
# along-track, so 006 (radial, along-track, cross-track) and 007 (Q, S, W) both order radial /
# along-track / cross-track, while 005's T,R,L — Tangage (pitch), Roulis (roll), Lacet (yaw), NOT
# "tangential" — orders cross-track / along-track / radial. Along-track is the schema's in-track.
_AXIS_ORDER: dict[str, tuple[ManeuverType, ManeuverType, ManeuverType]] = {
    "005": (ManeuverType.CROSS_TRACK, ManeuverType.IN_TRACK, ManeuverType.RADIAL),
    "006": (ManeuverType.RADIAL, ManeuverType.IN_TRACK, ManeuverType.CROSS_TRACK),
    "007": (ManeuverType.RADIAL, ManeuverType.IN_TRACK, ManeuverType.CROSS_TRACK),
}

_BURN_TOKENS = 15  # 5 date sub-fields + duration + 3 ΔV + 3 acc + 3 delta-acc


def _doy_to_naive(year: int, doy: int, hour: int, minute: int, second: float = 0.0) -> datetime:
    """Build a naive datetime (the TAI clock reading) from a year + day-of-year + time of day."""
    whole = int(second)
    micros = round((second - whole) * 1_000_000)
    return datetime(year, 1, 1) + timedelta(
        days=doy - 1, hours=hour, minutes=minute, seconds=whole, microseconds=micros
    )


def _tai_to_utc(naive_tai: datetime) -> datetime:
    """Convert a naive TAI clock reading to a timezone-aware UTC datetime via astropy.

    The TAI-UTC offset (~37 s) is immaterial at the labeller's ±2-day matching tolerance, but the
    common record stores honest UTC, so the conversion is applied rather than reinterpreting the
    TAI clock as UTC. astropy is a core internal dependency; the import is deferred so importing
    this module stays light.
    """
    from astropy.time import Time

    converted = Time(naive_tai, scale="tai").utc.to_datetime()
    return datetime(
        converted.year,
        converted.month,
        converted.day,
        converted.hour,
        converted.minute,
        converted.second,
        converted.microsecond,
        tzinfo=timezone.utc,
    )


def _midpoint(start: datetime, end: datetime) -> datetime:
    return start + (end - start) / 2


def _summarise_burns(
    param_code: str, burns: list[tuple[datetime, tuple[float, float, float]]]
) -> tuple[datetime, float, ManeuverType | None]:
    """Collapse an event's burns to (representative epoch, |Δv| m/s, dominant-axis type)."""
    axes = _AXIS_ORDER.get(param_code)
    summed = [
        sum(burn[1][i] for burn in burns) for i in range(3)
    ]  # component-wise vector sum (same frame within an event)
    magnitude = math.hypot(*summed)
    maneuver_type: ManeuverType | None = None
    if axes is not None and magnitude > 0.0:
        dominant = max(range(3), key=lambda i: abs(summed[i]))
        maneuver_type = axes[dominant]
    epoch = burns[0][0]  # the primary (first) burn's median epoch
    return epoch, magnitude, maneuver_type


def _window_only_label(
    norad_id: int | None, code: str, tokens: list[str], start: datetime, end: datetime
) -> ManeuverLabel:
    """An epoch-only label from the window alone — for TOPEX lines and burn-less (N=0) maneuvers."""
    return ManeuverLabel(
        norad_id=norad_id,
        epoch=_midpoint(start, end),
        window_start=start,
        window_end=end,
        source=SOURCE_DORIS_IDS,
        source_ref=f"{code} {tokens[1]}/{tokens[2]}",
        orbit_class=OrbitClass.LEO,
    )


def _parse_line(line: str) -> ManeuverLabel | None:
    """Parse one ``man.txt`` event line into a :class:`ManeuverLabel`, or ``None`` to skip it."""
    tokens = line.split()
    if not tokens or tokens[0].startswith("#"):
        return None
    if len(tokens) < 9:
        raise ValueError(f"DORIS maneuver line has too few fields for a window: {line!r}")

    code = tokens[0].upper()
    norad_id = DORIS_SAT_TO_NORAD.get(code)
    window_start = _tai_to_utc(
        _doy_to_naive(int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4]))
    )
    window_end = _tai_to_utc(
        _doy_to_naive(int(tokens[5]), int(tokens[6]), int(tokens[7]), int(tokens[8]))
    )

    # TOPEX (and any window-only line): no burn detail, so the label is epoch-only.
    if len(tokens) == 9:
        return _window_only_label(norad_id, code, tokens, window_start, window_end)

    cursor = 9
    # SPOT lines carry a non-numeric maneuver-type token (MCC/MCO) before the parameter code.
    if not tokens[cursor].isdigit():
        cursor += 1
    param_code = tokens[cursor]
    n_burns = int(tokens[cursor + 1])
    cursor += 2

    burns: list[tuple[datetime, tuple[float, float, float]]] = []
    for _ in range(n_burns):
        block = tokens[cursor : cursor + _BURN_TOKENS]
        if len(block) < _BURN_TOKENS:
            raise ValueError(f"DORIS maneuver line has a truncated burn block: {line!r}")
        epoch = _tai_to_utc(
            _doy_to_naive(
                int(block[0]), int(block[1]), int(block[2]), int(block[3]), float(block[4])
            )
        )
        delta_v = (float(block[6]), float(block[7]), float(block[8]))
        burns.append((epoch, delta_v))
        cursor += _BURN_TOKENS

    # A maneuver announced with no burn detail (N=0) is an epoch-only event, like a TOPEX line.
    if not burns:
        return _window_only_label(norad_id, code, tokens, window_start, window_end)

    epoch, magnitude, maneuver_type = _summarise_burns(param_code, burns)
    return ManeuverLabel(
        norad_id=norad_id,
        epoch=epoch,
        window_start=window_start,
        window_end=window_end,
        source=SOURCE_DORIS_IDS,
        source_ref=f"{code} {tokens[1]}/{tokens[2]}",
        orbit_class=OrbitClass.LEO,
        maneuver_type=maneuver_type,
        delta_v=magnitude,
    )


def parse_doris(text: str) -> list[ManeuverLabel]:
    """Parse the text of a DORIS/IDS ``man.txt`` file into normalised maneuver labels.

    Each non-blank, non-comment line is one maneuver event. Blank and ``#``-comment lines are
    skipped; a line that starts a record but is structurally truncated raises :class:`ValueError`
    so format drift surfaces rather than passing silently. Times are converted from the file's TAI
    to UTC; the per-burn ΔV components are collapsed to the event ``|Δv|`` and dominant-axis type.
    """
    labels: list[ManeuverLabel] = []
    for line in text.splitlines():
        label = _parse_line(line)
        if label is not None:
            labels.append(label)
    _logger.debug("parsed %d DORIS maneuver labels", len(labels))
    return labels
