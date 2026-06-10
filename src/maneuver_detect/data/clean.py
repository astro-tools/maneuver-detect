"""Elset cleaning — drop obvious catalog noise, then dedup to one elset per epoch.

``clean_elsets`` turns the raw :class:`~maneuver_detect.data.elset.Elset` sequence a fetcher
returns into a validated, epoch-ordered series with at most one elset per epoch — the input
:mod:`~maneuver_detect.data.history` assembles into the canonical mean-element DataFrame.

Two operations, in order:

1. **Validity** — reject only *obvious* catalog noise: an elset whose elements are non-physical
   (non-finite, eccentricity outside ``[0, 1)``, non-positive mean motion, inclination outside
   ``[0, 180]``) or which SGP4 cannot initialise / propagate at its own epoch. Telling a *bad
   elset* from a *maneuver* is the detector's job, not this layer's; this only removes elsets that
   are unusable on their face.
2. **Dedup duplicate-epoch elsets** — the catalog (and a multi-source merge) can carry more than
   one elset at a single epoch. The rule distinguishes two cases:

   - **Exact duplicates** (same epoch *and* identical elements — e.g. the same elset redistributed
     across sources) collapse to one, keeping the highest ``element_set_no``. This is robust even
     when ``element_set_no`` is a placeholder (CelesTrak frequently emits ``999``), because the
     elements decide identity, not the bookkeeping field.
   - A genuine **same-epoch re-fit** (same epoch, *differing* elements — the catalog re-issued a
     revised fit) keeps the highest ``element_set_no`` (the later revision), with a deterministic
     element-value fallback when ``element_set_no`` does not discriminate. Re-fit collisions are
     logged — a same-epoch elset that is not a mere redistribution is a small quality signal.

The dedup is fully deterministic (no dependence on input order), so a reconstructed series is
byte-stable for the benchmark.

Gaps are handled *passively*: a re-acquisition gap is a real feature of the series (and is where a
maneuver label lives), so cleaning never splits, fills, or drops an elset for sitting across one —
:mod:`~maneuver_detect.data.history` surfaces the inter-elset spacing instead.

Cleaning operates on the :class:`Elset` abstraction, so a future ``orbit-formats``-backed reader
slots in without touching this layer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import datetime, timezone

from sgp4.api import WGS72, Satrec, jday

from maneuver_detect.data.elset import Elset

__all__ = ["clean_elsets", "is_valid_elset"]

_logger = logging.getLogger(__name__)

# rev/day -> rad/min divisor (the sgp4 `xpdotp`), for the sgp4init element conversion.
_XPDOTP = 1440.0 / (2.0 * math.pi)
# The sgp4 epoch origin: days are counted from 1949 December 31 00:00 UTC.
_EPOCH_1949 = datetime(1949, 12, 31, tzinfo=timezone.utc)


# The orbital-element signature that defines "the same elset" for dedup — the six mean elements
# plus the drag / rate terms, excluding bookkeeping (element_set_no, rev_at_epoch, classification,
# object_id). Two elsets with identical signatures at one epoch are the same physical elset.
def _element_signature(elset: Elset) -> tuple[float, ...]:
    return (
        elset.mean_motion,
        elset.eccentricity,
        elset.inclination,
        elset.raan,
        elset.arg_perigee,
        elset.mean_anomaly,
        elset.bstar,
        elset.mean_motion_dot,
        elset.mean_motion_ddot,
    )


def _is_physical(elset: Elset) -> bool:
    """Cheap analytic bounds check: finite elements within their physical ranges."""
    if not all(math.isfinite(v) for v in _element_signature(elset)):
        return False
    if not 0.0 <= elset.eccentricity < 1.0:
        return False
    if elset.mean_motion <= 0.0:
        return False
    return 0.0 <= elset.inclination <= 180.0


def _sgp4_is_sane(elset: Elset) -> bool:
    """Whether SGP4 can initialise the elset and propagate it at its own epoch without error.

    Catches the "SGP4-insane" cases the analytic bounds miss — chiefly a decayed orbit whose
    perigee is inside the Earth — by building a ``Satrec`` from the elements and propagating at
    ``t = 0``. ``ndot`` / ``nddot`` are converted for completeness but do not drive the init or
    decay error codes, which the eccentricity / mean-motion / perigee geometry decide.
    """
    epoch_days = (elset.epoch - _EPOCH_1949).total_seconds() / 86400.0
    satrec = Satrec()
    try:
        satrec.sgp4init(
            WGS72,
            "i",
            elset.norad_id,
            epoch_days,
            elset.bstar,
            elset.mean_motion_dot / (_XPDOTP * 1440.0),
            elset.mean_motion_ddot / (_XPDOTP * 1440.0 * 1440.0),
            elset.eccentricity,
            math.radians(elset.arg_perigee),
            math.radians(elset.inclination),
            math.radians(elset.mean_anomaly),
            elset.mean_motion / _XPDOTP,
            math.radians(elset.raan),
        )
    except (ValueError, RuntimeError):
        return False
    if satrec.error != 0:
        return False
    epoch = elset.epoch
    jd, fr = jday(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second + epoch.microsecond / 1e6,
    )
    error, position, _velocity = satrec.sgp4(jd, fr)
    if error != 0:
        return False
    return all(math.isfinite(component) for component in position)


def is_valid_elset(elset: Elset) -> bool:
    """Whether ``elset`` survives the obvious-catalog-noise filter.

    ``True`` unless the elset is non-physical on its face or SGP4 cannot initialise / propagate it
    at its own epoch. This is deliberately permissive — it removes unusable elsets, not ones that
    merely look anomalous (that is the detector's call).
    """
    return _is_physical(elset) and _sgp4_is_sane(elset)


def clean_elsets(elsets: Sequence[Elset]) -> list[Elset]:
    """Validate and dedup ``elsets`` (one object's history) into an epoch-ordered series.

    Drops obvious catalog noise (:func:`is_valid_elset`), then keeps a single elset per epoch under
    the dedup rule documented in the module: exact duplicates collapse; a genuine same-epoch re-fit
    keeps the latest revision. The result is sorted by epoch ascending with strictly increasing
    epochs, and is deterministic regardless of input order.
    """
    valid = [elset for elset in elsets if is_valid_elset(elset)]
    deduped = _dedup_by_epoch(valid)
    if len(deduped) != len(elsets):
        _logger.debug(
            "cleaned series: %d elsets in -> %d valid -> %d after dedup",
            len(elsets),
            len(valid),
            len(deduped),
        )
    return deduped


def _dedup_by_epoch(elsets: list[Elset]) -> list[Elset]:
    """Keep one elset per epoch under the refined rule; return them epoch-ascending."""
    by_epoch: dict[datetime, list[Elset]] = {}
    for elset in elsets:
        by_epoch.setdefault(elset.epoch, []).append(elset)

    kept: list[Elset] = []
    for epoch in sorted(by_epoch):
        group = by_epoch[epoch]
        if len(group) == 1:
            kept.append(group[0])
            continue
        kept.append(_resolve_epoch_group(epoch, group))
    return kept


def _resolve_epoch_group(epoch: datetime, group: list[Elset]) -> Elset:
    """Collapse same-epoch elsets to one: exact-duplicate first, then highest-revision re-fit."""
    # Bucket by element signature; within an identical-elements bucket keep the highest revision —
    # this is the exact-duplicate collapse, robust to a placeholder element_set_no.
    by_signature: dict[tuple[float, ...], list[Elset]] = {}
    for elset in group:
        by_signature.setdefault(_element_signature(elset), []).append(elset)
    distinct = [
        max(bucket, key=lambda elset: elset.element_set_no) for bucket in by_signature.values()
    ]
    if len(distinct) == 1:
        return distinct[0]

    # A genuine same-epoch re-fit: differing elements at one epoch. Keep the latest revision, with
    # a deterministic element-value fallback when element_set_no does not discriminate.
    chosen = max(distinct, key=lambda elset: (elset.element_set_no, _element_signature(elset)))
    _logger.debug(
        "same-epoch re-fit for NORAD %s at %s: %d distinct elsets, kept element_set_no=%s",
        chosen.norad_id,
        epoch.isoformat(),
        len(distinct),
        chosen.element_set_no,
    )
    return chosen
