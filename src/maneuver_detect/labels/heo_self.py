"""Self-labelled HEO apogee/perigee-control epochs from element-step inspection (best-effort).

No public operator maneuver feed covers the high-eccentricity (HEO) regime — the v0.3 source survey
found science-HEO maneuvers documented only in prose and the comms/early-warning HEO operators
silent — so, exactly as for GEO (:mod:`~maneuver_detect.labels.longitude_shift`), the HEO labels are
**derived from the element series itself** rather than announced. :func:`derive_heo_labels` reads an
HEO object's mean-element series and emits one
:class:`~maneuver_detect.labels.record.ManeuverLabel` per detected apogee/perigee-control maneuver,
tagged :data:`~maneuver_detect.labels.record.SOURCE_SELF_HEO`.

The method mirrors the GEO longitude-shift estimator on the channels that govern a highly elliptical
orbit: apogee/perigee maintenance changes the orbital **energy** (a step in semi-major axis ``a``,
the in-track channel) and reshapes the ellipse (a step in **eccentricity** ``e``, the radial
channel). Both are detected as robust, MAD-scaled jumps across an inter-elset gap, each gated by an
absolute floor so element noise alone never trips a label.

**Caveat (circularity).** Like the GEO source this is a *derived, best-effort* label, not ground
truth: the in-track (``a``) channel overlaps the quantity the benchmarked detector triggers on, so a
detector scored against these labels is partly graded on a related quantity. HEO therefore stays a
best-effort, separately-reported class (D3/D4), not folded into the headline recall. The derivation
is a pure, deterministic function of the cleaned series, so the committed labels reproduce
byte-for-byte at reconstruction (D8).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from maneuver_detect.labels.record import SOURCE_SELF_HEO, ManeuverLabel, OrbitClass
from maneuver_detect.schema import ManeuverType

__all__ = ["HEO_ECC_FLOOR", "HEO_SMA_FLOOR_KM", "derive_heo_labels"]

_logger = logging.getLogger(__name__)

#: Absolute floor on a significant semi-major-axis (energy) step, km — an apogee/perigee-maintenance
#: burn moves ``a`` by at least this much.
HEO_SMA_FLOOR_KM = 1.0
#: Absolute floor on a significant eccentricity (shape) step (dimensionless).
HEO_ECC_FLOOR = 1.0e-4


def _robust_scale(values: np.ndarray) -> float:
    """MAD-based robust scale (~sigma for Gaussian noise), floored so it can never be zero."""
    if values.size == 0:
        return float("inf")
    mad = float(np.median(np.abs(values - np.median(values))))
    return max(1.4826 * mad, 1e-12)


def derive_heo_labels(
    series: pd.DataFrame,
    *,
    norad_id: int | None = None,
    sma_sigma: float = 6.0,
    ecc_sigma: float = 6.0,
    sma_floor_km: float = HEO_SMA_FLOOR_KM,
    ecc_floor: float = HEO_ECC_FLOOR,
) -> list[ManeuverLabel]:
    """Derive HEO apogee/perigee-control labels from a ``series`` by element-step inspection.

    For each inter-elset gap the semi-major axis and eccentricity are differenced. A gap is an
    **in-track** maneuver when the ``a`` step exceeds ``max(sma_sigma*robust_scale, sma_floor_km)``,
    and **radial** when the ``e`` step exceeds ``max(ecc_sigma*robust_scale, ecc_floor)``.
    A gap that trips both is one label of the type with the larger significance. The magnitude and
    the precise epoch are unknown (epoch-only, derived), so ``delta_v`` is ``None`` and the epoch is
    the gap midpoint; the label window is the bracketing elset pair. Returns labels sorted by epoch;
    a series shorter than three elsets yields none.

    ``norad_id`` tags the labels (defaults to the series' own ``norad_id`` column when present).
    """
    if len(series) < 3:
        return []
    ordered = series.sort_values("epoch").reset_index(drop=True)
    if norad_id is None and "norad_id" in ordered.columns and len(ordered):
        norad_id = int(ordered["norad_id"].iloc[0])

    epoch = ordered["epoch"].to_numpy()
    sma = ordered["semi_major_axis"].to_numpy(dtype=float)
    ecc = ordered["eccentricity"].to_numpy(dtype=float)

    d_sma = np.abs(np.diff(sma))  # per-gap semi-major-axis step (km)
    d_ecc = np.abs(np.diff(ecc))  # per-gap eccentricity step
    sma_thr = max(sma_sigma * _robust_scale(d_sma), sma_floor_km)
    ecc_thr = max(ecc_sigma * _robust_scale(d_ecc), ecc_floor)

    labels: list[ManeuverLabel] = []
    for gap in range(len(d_sma)):
        sma_hit = d_sma[gap] > sma_thr
        ecc_hit = d_ecc[gap] > ecc_thr
        if not sma_hit and not ecc_hit:
            continue
        # One label per gap; the dominant type is the more significant of the two channels.
        sma_significance = d_sma[gap] / sma_thr if sma_hit else 0.0
        ecc_significance = d_ecc[gap] / ecc_thr if ecc_hit else 0.0
        maneuver_type = (
            ManeuverType.IN_TRACK if sma_significance >= ecc_significance else ManeuverType.RADIAL
        )
        window_start = pd.Timestamp(epoch[gap]).to_pydatetime()
        window_end = pd.Timestamp(epoch[gap + 1]).to_pydatetime()
        labels.append(
            ManeuverLabel(
                norad_id=norad_id,
                epoch=window_start + (window_end - window_start) / 2,
                window_start=window_start,
                window_end=window_end,
                source=SOURCE_SELF_HEO,
                source_ref=f"apogee/perigee-control gap {window_start.date()}..{window_end.date()}",
                orbit_class=OrbitClass.HEO,
                maneuver_type=maneuver_type,
            )
        )
    _logger.debug("derived %d self-labelled HEO maneuvers for NORAD %s", len(labels), norad_id)
    return labels
