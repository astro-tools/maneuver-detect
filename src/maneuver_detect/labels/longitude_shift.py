"""Self-labelled GEO station-keeping epochs from longitude-drift inspection (best-effort source).

No public operator maneuver feed covers the GEO class (the licence-clean survey ruled BeiDou NABU
out as non-redistributable and uncrawlable, the rest sparse), so the GEO labels are **derived from
the element series itself** rather than announced. :func:`derive_geo_labels` reads a GEO object's
mean-element series and emits one :class:`~maneuver_detect.labels.record.ManeuverLabel` per detected
station-keeping maneuver, tagged :data:`~maneuver_detect.labels.record.SOURCE_SELF_GEO`.

The method is the classical "longitude-shift inspection": a geostationary satellite's sub-longitude
drift rate is set by its mean-motion offset from synchronous (the drift rate ~= 360*(n - n_sync)
deg/day), and **east-west station-keeping holds the satellite in a deadband by periodically
reversing that drift** — so an E-W maneuver shows as a *drift-rate sign reversal* (a vertex of the
longitude sawtooth). North-south station-keeping fights the luni-solar inclination build-up, so an
N-S maneuver shows as a *downward step in inclination*. Both are detected as robust, MAD-scaled
jumps across an inter-elset gap.

**Caveat (circularity).** This is a *derived, best-effort* label source, not ground truth: the E-W
signal (a mean-motion / semi-major-axis change) overlaps the in-track channel the benchmarked
detector triggers on, so a detector scored against these labels is partly graded on a related
quantity. The estimator here is deliberately distinct — it requires a drift-rate **reversal** (the
operational deadband-bounce signature), not merely any element step — but GEO stays the best-effort
class (D3) and should be reported as a separate, flagged track rather than folded into the headline
recall. The derivation is a pure, deterministic function of the cleaned series, so the committed
labels reproduce byte-for-byte at reconstruction (D8).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from maneuver_detect.labels.record import SOURCE_SELF_GEO, ManeuverLabel, OrbitClass
from maneuver_detect.schema import ManeuverType

__all__ = ["GEO_DRIFT_FLOOR_DEG_PER_DAY", "GEO_INCL_FLOOR_DEG", "derive_geo_labels"]

_logger = logging.getLogger(__name__)

#: Sidereal day (s) and the synchronous mean motion (rev/day) it implies (the zero-drift value).
_SIDEREAL_DAY_S = 86164.0905
_N_SYNC_REV_PER_DAY = 86400.0 / _SIDEREAL_DAY_S

#: Absolute floor on a significant east-west drift-rate step (deg/day) — ~0.4 km of semi-major axis.
GEO_DRIFT_FLOOR_DEG_PER_DAY = 0.005
#: Absolute floor on a significant north-south inclination step (deg).
GEO_INCL_FLOOR_DEG = 0.005


def _robust_scale(values: np.ndarray) -> float:
    """MAD-based robust scale (~sigma for Gaussian noise), floored so it can never be zero."""
    if values.size == 0:
        return float("inf")
    mad = float(np.median(np.abs(values - np.median(values))))
    return max(1.4826 * mad, 1e-12)


def derive_geo_labels(
    series: pd.DataFrame,
    *,
    norad_id: int | None = None,
    drift_sigma: float = 6.0,
    incl_sigma: float = 6.0,
    drift_floor_deg_per_day: float = GEO_DRIFT_FLOOR_DEG_PER_DAY,
    incl_floor_deg: float = GEO_INCL_FLOOR_DEG,
) -> list[ManeuverLabel]:
    """Derive GEO station-keeping labels from a mean-element ``series`` by drift inspection.

    For each inter-elset gap the longitude drift rate ``360*(mean_motion - n_sync)`` and the
    inclination are differenced. A gap is an **in-track** (E-W) maneuver when the drift-rate
    step both exceeds ``max(drift_sigma*robust_scale, drift_floor_deg_per_day)`` **and reverses the
    drift sign** (the deadband-bounce signature), and a **cross-track** (N-S) maneuver when the
    inclination steps **down** by more than ``max(incl_sigma*robust_scale, incl_floor_deg)``. A gap
    that trips both is one label of the type with the larger significance. The magnitude and the
    precise epoch are unknown (epoch-only, derived), so ``delta_v`` is ``None`` and the epoch is the
    gap midpoint; the label window is the bracketing elset pair. Returns labels sorted by epoch; a
    series shorter than three elsets yields none.

    ``norad_id`` tags the labels (defaults to the series' own ``norad_id`` column when present).
    """
    if len(series) < 3:
        return []
    ordered = series.sort_values("epoch").reset_index(drop=True)
    if norad_id is None and "norad_id" in ordered.columns and len(ordered):
        norad_id = int(ordered["norad_id"].iloc[0])

    epoch = ordered["epoch"].to_numpy()
    drift = 360.0 * (ordered["mean_motion"].to_numpy(dtype=float) - _N_SYNC_REV_PER_DAY)
    incl = ordered["inclination"].to_numpy(dtype=float)

    d_drift = np.diff(drift)  # per-gap drift-rate step (deg/day)
    d_incl = np.diff(incl)  # per-gap inclination step (deg)
    drift_thr = max(drift_sigma * _robust_scale(d_drift), drift_floor_deg_per_day)
    incl_thr = max(incl_sigma * _robust_scale(d_incl), incl_floor_deg)

    labels: list[ManeuverLabel] = []
    for gap in range(len(d_drift)):
        # East-west SK: a significant drift-rate step that reverses (or nulls) the drift direction.
        ew_step = abs(d_drift[gap])
        reversed_drift = drift[gap] * drift[gap + 1] <= 0.0
        ew_hit = ew_step > drift_thr and reversed_drift
        # North-south SK: a significant *downward* inclination step (a burn vs. the secular rise).
        ns_step = -d_incl[gap]
        ns_hit = ns_step > incl_thr

        if not ew_hit and not ns_hit:
            continue
        # One label per gap; the dominant type is the more significant of the two channels.
        ew_significance = ew_step / drift_thr if ew_hit else 0.0
        ns_significance = ns_step / incl_thr if ns_hit else 0.0
        maneuver_type = (
            ManeuverType.IN_TRACK
            if ew_significance >= ns_significance
            else ManeuverType.CROSS_TRACK
        )
        window_start = pd.Timestamp(epoch[gap]).to_pydatetime()
        window_end = pd.Timestamp(epoch[gap + 1]).to_pydatetime()
        labels.append(
            ManeuverLabel(
                norad_id=norad_id,
                epoch=window_start + (window_end - window_start) / 2,
                window_start=window_start,
                window_end=window_end,
                source=SOURCE_SELF_GEO,
                source_ref=f"longitude-shift gap {window_start.date()}..{window_end.date()}",
                orbit_class=OrbitClass.GEO,
                maneuver_type=maneuver_type,
            )
        )
    _logger.debug("derived %d self-labelled GEO maneuvers for NORAD %s", len(labels), norad_id)
    return labels
