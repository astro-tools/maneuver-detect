"""The parsed elset record — the seam between catalogue I/O and everything downstream.

An :class:`Elset` is one element set: the SGP4 mean elements at one epoch for one object. The
fetchers (:mod:`~maneuver_detect.data.celestrak`, :mod:`~maneuver_detect.data.spacetrack`) return
sequences of these; the cleaning and assembly layer turns a sequence into a per-object
mean-element time series.

Both CelesTrak and Space-Track serve **OMM** (Orbit Mean-Element Message) JSON, so a single
parser — :func:`from_omm` — covers both sources. ``from_omm`` is deliberately the only place that
knows the OMM field names: it is the function the canonical TLE / OMM reader from ``orbit-formats``
is expected to replace once that sibling is adopted, leaving the rest of the data layer untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

__all__ = ["Elset", "from_omm"]


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
