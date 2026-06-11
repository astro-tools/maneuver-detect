"""The parsed elset record — the seam between catalogue I/O and everything downstream.

An :class:`Elset` is one element set: the SGP4 mean elements at one epoch for one object. The
fetchers (:mod:`~maneuver_detect.data.celestrak`, :mod:`~maneuver_detect.data.spacetrack`) return
sequences of these; the cleaning and assembly layer turns a sequence into a per-object
mean-element time series.

Both CelesTrak and Space-Track serve **OMM** (Orbit Mean-Element Message) JSON, so a single
parser — :func:`from_omm` — covers both sources. ``from_omm`` is deliberately the only place that
knows the OMM field names: it is the function the canonical TLE / OMM reader from ``orbit-formats``
is expected to replace once that sibling is adopted, leaving the rest of the data layer untouched.

For a local **TLE file** (the classic two-line format, optionally with a leading name line), the
companion :func:`from_tle` parses one line pair into an :class:`Elset` and :func:`read_tle_file`
walks a whole file into a sequence — the offline counterpart to the OMM fetchers, used by the CLI's
``detect`` on a TLE path. The element values come from :class:`sgp4.api.Satrec`'s line parser (which
handles the TLE's assumed-decimal and exponent fields); the structural validation is ours, because
``twoline2rv`` accepts almost anything.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sgp4.api import Satrec

__all__ = ["Elset", "from_omm", "from_tle", "read_tle_file"]


@dataclass(frozen=True)
class Elset:
    """One SGP4 mean-element set at one epoch for one object.

    The six Keplerian mean elements plus the drag / rate terms an SGP4 propagation needs, the
    catalogue identifier, and the epoch. Angles are in degrees, ``mean_motion`` in revolutions
    per day, ``epoch`` is timezone-aware UTC. ``element_set_no`` and ``rev_at_epoch`` carry the
    catalogue bookkeeping the cleaning layer uses to spot duplicate-epoch and stale elsets.

    Attributes:
        norad_id: NORAD catalogue id of the object.
        epoch: Element-set epoch (timezone-aware UTC).
        mean_motion: Mean motion, revolutions per day.
        eccentricity: Eccentricity (dimensionless).
        inclination: Inclination, degrees.
        raan: Right ascension of the ascending node, degrees.
        arg_perigee: Argument of perigee, degrees.
        mean_anomaly: Mean anomaly, degrees.
        bstar: B\\* drag term (earth radii\\ :sup:`-1`).
        mean_motion_dot: First time-derivative of the mean motion (rev/day\\ :sup:`2`).
        mean_motion_ddot: Second time-derivative of the mean motion (rev/day\\ :sup:`3`).
        element_set_no: Element-set number assigned by the catalogue.
        rev_at_epoch: Revolution number at epoch.
        classification: Classification marker (``"U"`` unclassified by default).
        object_id: International designator, e.g. ``"1998-067A"`` (``""`` when not provided).
    """

    norad_id: int
    epoch: datetime
    mean_motion: float
    eccentricity: float
    inclination: float
    raan: float
    arg_perigee: float
    mean_anomaly: float
    bstar: float
    mean_motion_dot: float
    mean_motion_ddot: float
    element_set_no: int
    rev_at_epoch: int
    classification: str
    object_id: str

    def __post_init__(self) -> None:
        if self.epoch.tzinfo is None:
            raise ValueError("Elset.epoch must be timezone-aware (UTC)")


def _require(record: Mapping[str, Any], key: str) -> Any:
    try:
        return record[key]
    except KeyError:
        raise ValueError(f"OMM record is missing the required field {key!r}") from None


def _as_float(value: Any, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"OMM field {key!r} is not a number: {value!r}") from None


def _as_int(value: Any, key: str) -> int:
    try:
        # OMM integers occasionally arrive as numeric strings ("999"); int(float(...)) would
        # mask a genuinely fractional value, so go through int() on the stripped string.
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"OMM field {key!r} is not an integer: {value!r}") from None


def _parse_epoch(value: Any) -> datetime:
    """Parse an OMM ``EPOCH`` string to a timezone-aware UTC datetime.

    OMM epochs are UTC by convention and usually tz-naive. CelesTrak emits ISO-8601 with a ``T``
    separator (``2024-01-01T12:00:00.000000``); Space-Track uses a space (``2024-01-01 12:00:00``)
    and an occasional trailing ``Z``. Normalise both to what :meth:`datetime.fromisoformat`
    accepts on Python 3.10+, then stamp UTC when the parsed value is naive.
    """
    if not isinstance(value, str):
        raise ValueError(f"OMM field 'EPOCH' is not a string: {value!r}")
    text = value.strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"OMM field 'EPOCH' is not an ISO-8601 datetime: {value!r}") from None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def from_omm(record: Mapping[str, Any]) -> Elset:
    """Parse one OMM record (as a CelesTrak / Space-Track JSON dict) into an :class:`Elset`.

    The orbital-mechanics-critical fields — the catalogue id, epoch, the six mean elements, and
    the three drag / rate terms — are required; without them no usable element set exists, so a
    missing or non-numeric value raises :class:`ValueError`. The bookkeeping fields
    (``OBJECT_ID``, ``CLASSIFICATION_TYPE``, ``ELEMENT_SET_NO``, ``REV_AT_EPOCH``) carry
    OMM-spec defaults, because CelesTrak occasionally omits them on decay-tracking entries.

    This is the single OMM-aware function in the data layer; replacing it with the canonical
    ``orbit-formats`` reader is a mechanical swap that leaves the fetchers and the cache alone.
    """
    return Elset(
        norad_id=_as_int(_require(record, "NORAD_CAT_ID"), "NORAD_CAT_ID"),
        epoch=_parse_epoch(_require(record, "EPOCH")),
        mean_motion=_as_float(_require(record, "MEAN_MOTION"), "MEAN_MOTION"),
        eccentricity=_as_float(_require(record, "ECCENTRICITY"), "ECCENTRICITY"),
        inclination=_as_float(_require(record, "INCLINATION"), "INCLINATION"),
        raan=_as_float(_require(record, "RA_OF_ASC_NODE"), "RA_OF_ASC_NODE"),
        arg_perigee=_as_float(_require(record, "ARG_OF_PERICENTER"), "ARG_OF_PERICENTER"),
        mean_anomaly=_as_float(_require(record, "MEAN_ANOMALY"), "MEAN_ANOMALY"),
        bstar=_as_float(_require(record, "BSTAR"), "BSTAR"),
        mean_motion_dot=_as_float(_require(record, "MEAN_MOTION_DOT"), "MEAN_MOTION_DOT"),
        mean_motion_ddot=_as_float(_require(record, "MEAN_MOTION_DDOT"), "MEAN_MOTION_DDOT"),
        element_set_no=_as_int(record.get("ELEMENT_SET_NO", 0), "ELEMENT_SET_NO"),
        rev_at_epoch=_as_int(record.get("REV_AT_EPOCH", 0), "REV_AT_EPOCH"),
        classification=str(record.get("CLASSIFICATION_TYPE", "U")),
        object_id=str(record.get("OBJECT_ID", "")),
    )


# The standard fixed-width TLE columns are 69 characters wide (data fields 1-68 + a checksum).
_TLE_LINE_LENGTH = 69
# sgp4 carries the angular elements in radians and the rates in rad/min**k; the catalogue / OMM
# convention this :class:`Elset` follows is degrees and revolutions per day. One factor of
# 1440 min/day per derivative order converts the rates.
_REV_PER_DAY_PER_RAD_PER_MIN = 1440.0 / (2.0 * math.pi)
_REV_PER_DAY2_PER_RAD_PER_MIN2 = _REV_PER_DAY_PER_RAD_PER_MIN * 1440.0
_REV_PER_DAY3_PER_RAD_PER_MIN3 = _REV_PER_DAY_PER_RAD_PER_MIN * 1440.0 * 1440.0


def _two_digit_year(yy: int) -> int:
    """Expand a TLE two-digit year to four digits (57-99 -> 19xx, 00-56 -> 20xx)."""
    return yy + (1900 if yy >= 57 else 2000)


def _tle_epoch(satrec: Satrec) -> datetime:
    """The UTC epoch of a parsed TLE, from its year + fractional day-of-year.

    ``epochdays`` is the day of the year with ``1.0`` meaning January 1 at 00:00, so the offset
    from the year start is ``epochdays - 1``. Going through the day-of-year fields (already
    resolved by the line parser) avoids a Julian-date inverse and keeps the conversion exact to
    the sub-second precision a TLE epoch carries.
    """
    year = _two_digit_year(satrec.epochyr)
    start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
    return start_of_year + timedelta(days=satrec.epochdays - 1.0)


def _tle_object_id(line1: str) -> str:
    """The international designator (e.g. ``"1998-067A"``) from line 1, or ``""`` if absent.

    Columns 10-17 hold a two-digit launch year, a launch number, and the piece. CelesTrak / the
    spike fixtures occasionally leave them blank on older catalogue entries, so a missing or
    non-numeric designator yields the empty default rather than an error.
    """
    launch_year = line1[9:11].strip()
    launch_piece = line1[11:17].strip()
    if not launch_year.isdigit() or not launch_piece:
        return ""
    return f"{_two_digit_year(int(launch_year))}-{launch_piece}"


def from_tle(line1: str, line2: str) -> Elset:
    """Parse one TLE line pair into an :class:`Elset`.

    The two lines are the classic fixed-width TLE format; a leading name line (3LE) is not passed
    here — :func:`read_tle_file` strips it. The mean elements are read by
    :meth:`sgp4.api.Satrec.twoline2rv` (which decodes the assumed-decimal eccentricity and the
    exponent-notation drag / rate fields) and converted to this record's degree / rev-per-day
    convention. Because ``twoline2rv`` is permissive, the structural guard is here: both lines must
    be the standard width and carry their line number, and their catalogue numbers must agree, or a
    :class:`ValueError` is raised — the same contract :func:`from_omm` offers for a bad OMM record.
    """
    first = line1.rstrip("\r\n")
    second = line2.rstrip("\r\n")
    for number, line in (("1", first), ("2", second)):
        if len(line) < _TLE_LINE_LENGTH or line[0] != number or line[1] != " ":
            raise ValueError(f"malformed TLE line {number}: {line!r}")
    if first[2:7] != second[2:7]:
        raise ValueError(
            f"TLE line 1 / line 2 catalogue-number mismatch: {first[2:7]!r} vs {second[2:7]!r}"
        )
    try:
        satrec = Satrec.twoline2rv(first, second)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"could not parse TLE line pair: {exc}") from exc

    mean_motion = satrec.no_kozai * _REV_PER_DAY_PER_RAD_PER_MIN
    if not math.isfinite(mean_motion) or mean_motion <= 0.0:
        raise ValueError(f"TLE has a non-physical mean motion: {mean_motion!r}")

    return Elset(
        norad_id=int(satrec.satnum),
        epoch=_tle_epoch(satrec),
        mean_motion=mean_motion,
        eccentricity=satrec.ecco,
        inclination=math.degrees(satrec.inclo),
        raan=math.degrees(satrec.nodeo),
        arg_perigee=math.degrees(satrec.argpo),
        mean_anomaly=math.degrees(satrec.mo),
        bstar=satrec.bstar,
        mean_motion_dot=satrec.ndot * _REV_PER_DAY2_PER_RAD_PER_MIN2,
        mean_motion_ddot=satrec.nddot * _REV_PER_DAY3_PER_RAD_PER_MIN3,
        element_set_no=int(satrec.elnum),
        rev_at_epoch=int(satrec.revnum),
        classification=str(satrec.classification or "U"),
        object_id=_tle_object_id(first),
    )


def read_tle_file(path: str | Path) -> list[Elset]:
    """Read a TLE file into a list of :class:`Elset` records, in file order.

    Accepts both the two-line (2LE) and named three-line (3LE) layouts, intermixed: a line opening
    ``"1 "`` starts a pair and must be followed by its ``"2 "`` partner; any other non-blank line is
    treated as a name / title line for the next pair and ignored (this record carries no common
    name). Blank lines are skipped. Each pair is validated by :func:`from_tle`, so a malformed or
    orphaned line raises :class:`ValueError`. An empty file yields an empty list.
    """
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    elsets: list[Elset] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("1 "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("2 "):
                raise ValueError(f"TLE line 1 is not followed by a line 2: {line!r}")
            elsets.append(from_tle(line, lines[index + 1]))
            index += 2
        elif line.startswith("2 "):
            raise ValueError(f"TLE line 2 without a preceding line 1: {line!r}")
        else:
            # A name / title line (3LE) — informational only; the next pair follows.
            index += 1
    return elsets
