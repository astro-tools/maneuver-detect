"""Per-token channel construction — the V5 / D11 irregular-sampling encoding, label-free.

:func:`build_channels` turns one object's cleaned mean-element series (the
:data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS` frame) into the per-token channel
matrix the sequence models consume, exactly as the V5 spike froze it (decision D11). One **token
per elset**, so one inter-elset gap is the transition between two adjacent tokens and the per-gap
maneuver target attaches to that transition (D4). The channels, in matrix-column order, are:

* **levels** — the element value at the token's epoch for each of the :data:`BASE_CHANNELS`
  (``a`` km, ``e``, ``sin i``, ``cos i``, the eccentricity vector ``h = e·cos ω`` / ``k = e·sin ω``,
  and the unwrapped node ``Ω``). Absolute context;
* **residuals** — each level minus its **secular trend**, a two-sided local-linear fit computed
  **once per series** (D11.3; the J2 nodal regression and apsidal precession are deg/day, far larger
  than any burn, so they must be removed or they read as a bogus step — the V4 failure mode). The
  anomaly;
* **deltas** — the **signed** level shift of each residual across the gap to the previous token
  (``residual[i] - residual[i-1]``; zero at the first token). The maneuver signal is in the
  *magnitude* of this step, but it is fed signed so the sign carries the burn direction the D5
  Δv-type classification needs (the non-linear model recovers the magnitude itself);
* **timing** — ``time2vec(Δt)``: a bounded linear term ``Δt_clip / scale`` plus sine/cosine pairs at
  :data:`TIME2VEC_PERIODS_DAYS`. ``Δt`` stays in the input because the step *rate* and the secular
  detrend need it; its modest, structural correlation with the label is handled at the protocol
  level, not here (D11.2);
* **mask** — a real-elset validity bit (always 1: the deltas encoding imputes no rows, unlike
  resample-to-grid) and a ``Δt``-saturation flag (the gap exceeded the clip cap).

The function takes **only** the element series — never a label, target, or split — so no feature can
be derived from the thing it predicts (the leak-free boundary the issue DoD asserts). Normalisation
(per-class, train-split statistics) is a separate pass in
:mod:`~maneuver_detect.features.normalize`; this layer emits the raw, un-normalised channels and the
column count is fixed, so the same series always yields a byte-identical matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.physics import orbit_class_of

__all__ = [
    "BASE_CHANNELS",
    "CHANNEL_NAMES",
    "CLIP_CAP_DAYS",
    "DETREND_HALFWIDTH",
    "N_CHANNELS",
    "N_ELEMENT_CHANNELS",
    "TIME2VEC_PERIODS_DAYS",
    "TIME2VEC_SCALE_DAYS",
    "RawChannels",
    "build_channels",
]

_logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

#: The base element channels, in matrix order. Inclination is carried as ``sin i`` / ``cos i`` and
#: the argument of perigee as the eccentricity vector ``(h, k)`` so no channel wraps; the node ``Ω``
#: is carried unwrapped (D11.3). Each base channel contributes a level, a residual, and a delta
#: column, so the element block is ``3·len(BASE_CHANNELS)`` columns wide.
BASE_CHANNELS: tuple[str, ...] = ("a", "e", "sin_i", "cos_i", "h", "k", "raan")

#: Fixed periods (days) of the ``time2vec`` timing block's sine/cosine pairs — a small, bounded
#: harmonic set spanning the sub-weekly cadence the catalogue actually carries. Fixed (not learned)
#: so the feature layer is deterministic and model-independent.
TIME2VEC_PERIODS_DAYS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

#: The ``Δt`` saturation cap (days): the timing block's linear term is ``min(Δt, cap) / scale`` and
#: the saturation flag fires above it. Bounds the linear term and stops a learned model
#: over-weighting outlier post-maneuver re-acquisition gaps (D11.2) — explicitly *not* a leak fix
#: (rank-AUC is invariant to the clip, so the structural timing leak is untouched).
CLIP_CAP_DAYS = 2.5

#: The divisor that maps the clipped ``Δt`` into ``[0, 1]`` (equal to the clip cap).
TIME2VEC_SCALE_DAYS = CLIP_CAP_DAYS

#: Half-width (samples per side) of the centred local-linear secular detrend; the full window is
#: ``2·halfwidth + 1`` samples. Matches the V5 proof's detrend span.
DETREND_HALFWIDTH = 10


def _timing_channel_names() -> list[str]:
    names = ["dt_clip"]
    for period in TIME2VEC_PERIODS_DAYS:
        names.append(f"dt_sin_p{period:g}")
        names.append(f"dt_cos_p{period:g}")
    return names


#: Every channel name, in matrix-column order: the level / residual / delta blocks over
#: :data:`BASE_CHANNELS`, then the ``time2vec`` timing block, then the two mask bits.
CHANNEL_NAMES: tuple[str, ...] = (
    tuple(f"level_{name}" for name in BASE_CHANNELS)
    + tuple(f"resid_{name}" for name in BASE_CHANNELS)
    + tuple(f"delta_{name}" for name in BASE_CHANNELS)
    + tuple(_timing_channel_names())
    + ("elset_valid", "dt_saturated")
)

#: The number of leading element columns (level + residual + delta blocks) the per-class normaliser
#: standardises; the timing and mask columns that follow are bounded by construction and pass
#: through unchanged.
N_ELEMENT_CHANNELS = 3 * len(BASE_CHANNELS)

#: The total channel count ``C`` of the emitted matrix.
N_CHANNELS = len(CHANNEL_NAMES)

# The element columns that must be finite for a token to be usable; a non-finite value poisons the
# secular detrend and the per-class statistics, so such rows are dropped before encoding.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "epoch",
    "norad_id",
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "raan",
    "arg_perigee",
)
_FINITE_COLUMNS: tuple[str, ...] = (
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "raan",
    "arg_perigee",
)

_DEG_TO_RAD = np.pi / 180.0


@dataclass(frozen=True, eq=False)
class RawChannels:
    """One object's un-normalised per-token channel matrix and its provenance.

    Attributes:
        norad_id: The object the channels were built for.
        orbit_class: The object's orbit class, from the median semi-major axis — selects the
            per-class normalisation statistics downstream.
        epochs: The token epochs (naive UTC ``datetime64[ns]``), one per row of :attr:`matrix`, in
            order — what aligns an externally-supplied per-gap target onto the tokens.
        matrix: The ``(n_tokens, N_CHANNELS)`` channel matrix in :data:`CHANNEL_NAMES` order, raw
            (un-normalised) ``float64``.
        channel_names: The column names of :attr:`matrix` (:data:`CHANNEL_NAMES`).
    """

    norad_id: int
    orbit_class: OrbitClass
    epochs: npt.NDArray[np.datetime64]
    matrix: FloatArray
    channel_names: tuple[str, ...] = CHANNEL_NAMES

    @property
    def n_tokens(self) -> int:
        """The number of tokens (rows) — one per elset."""
        return int(self.matrix.shape[0])

    def element_block(self) -> FloatArray:
        """The leading element columns the normaliser standardises (level/residual/delta blocks)."""
        block: FloatArray = self.matrix[:, :N_ELEMENT_CHANNELS]
        return block


def _line_at(times: FloatArray, values: FloatArray, at: float) -> float:
    """Least-squares line through ``(times, values)`` evaluated at ``at`` (mean if degenerate)."""
    n = times.shape[0]
    sum_t = float(times.sum())
    sum_v = float(values.sum())
    sum_tt = float((times * times).sum())
    sum_tv = float((times * values).sum())
    denom = n * sum_tt - sum_t * sum_t
    if denom == 0.0:  # all samples at one epoch (or a single sample) — no trend to fit
        return float(sum_v / n)
    slope = (n * sum_tv - sum_t * sum_v) / denom
    intercept = (sum_v - slope * sum_t) / n
    return float(slope * at + intercept)


def _local_linear_trend(t_days: FloatArray, values: FloatArray, halfwidth: int) -> FloatArray:
    """The two-sided local-linear secular trend of ``values`` over ``t_days`` (D11.3).

    A straight line is fit to the ``±halfwidth`` samples around each token and evaluated at that
    token's epoch, so the slowly-varying secular drift (J2 nodal regression, the semi-major-axis
    drag decay) is tracked and the residual ``values - trend`` keeps only the anomalous step.
    Computed once over the whole series — not per candidate gap — so it is ``O(n · halfwidth)``,
    not the ``O(n²)`` the per-gap recomputation the V5 note warns against would cost.
    """
    n = values.shape[0]
    trend = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - halfwidth)
        hi = min(n, i + halfwidth + 1)
        trend[i] = _line_at(t_days[lo:hi], values[lo:hi], float(t_days[i]))
    return trend


def _time2vec(dt_filled: FloatArray) -> FloatArray:
    """The ``time2vec(Δt)`` timing block: a clipped linear term plus periodic sine/cosine pairs.

    The linear term is ``min(Δt, cap) / scale`` (bounded in ``[0, 1]``); each period contributes a
    ``sin`` / ``cos`` of ``2π Δt / P`` (already bounded, so the *unclipped* ``Δt`` drives them).
    """
    n = dt_filled.shape[0]
    dt_clip = np.minimum(dt_filled, CLIP_CAP_DAYS)
    columns: list[FloatArray] = [(dt_clip / TIME2VEC_SCALE_DAYS).reshape(n, 1)]
    for period in TIME2VEC_PERIODS_DAYS:
        angle = 2.0 * np.pi * dt_filled / period
        columns.append(np.sin(angle).reshape(n, 1))
        columns.append(np.cos(angle).reshape(n, 1))
    return np.hstack(columns)


def _drop_non_finite(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows carrying a non-finite (NaN/inf) value in any element column, logging the count."""
    elements = frame[list(_FINITE_COLUMNS)].to_numpy(dtype=float)
    finite = np.isfinite(elements).all(axis=1)
    if bool(finite.all()):
        return frame
    dropped = int((~finite).sum())
    _logger.warning("dropping %d elset(s) with non-finite mean elements before encoding", dropped)
    kept: pd.DataFrame = frame.loc[finite]
    return kept


def build_channels(history: pd.DataFrame) -> RawChannels:
    """Encode one object's mean-element ``history`` into the V5 / D11 per-token channel matrix.

    ``history`` is a single-object mean-element series carrying :data:`_REQUIRED_COLUMNS` (the
    :data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS` frame, or any frame with those
    columns). It is sorted by epoch defensively and rows with a non-finite element are dropped. The
    result is the raw (un-normalised) :class:`RawChannels`; pass it through a
    :class:`~maneuver_detect.features.normalize.ClassNormaliser` to standardise per class.

    Takes no label or target — the encoding is a pure function of the element series, so it cannot
    leak the maneuver labels it is used to detect.

    Raises:
        ValueError: if a required column is missing, the series is empty (or all rows are
            non-finite), or it carries more than one ``norad_id`` (use
            :func:`~maneuver_detect.features.encode_history` for a multi-object frame).
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"history is missing required columns: {missing}")
    if history.empty:
        raise ValueError("cannot build channels from an empty history")

    norad_ids = history["norad_id"].unique()
    if len(norad_ids) > 1:
        raise ValueError(
            f"build_channels expects a single-object history, got {len(norad_ids)} norad_ids; "
            "use encode_history for a multi-object frame"
        )

    frame = _drop_non_finite(history.sort_values("epoch"))
    if frame.empty:
        raise ValueError("history has no finite elsets to encode")

    epoch_series = pd.to_datetime(frame["epoch"], utc=True)
    epochs: npt.NDArray[np.datetime64] = epoch_series.dt.tz_localize(None).to_numpy()
    origin = epoch_series.iloc[0]
    t_days = ((epoch_series - origin).dt.total_seconds() / 86400.0).to_numpy(dtype=np.float64)

    a_km = frame["semi_major_axis"].to_numpy(dtype=np.float64)
    ecc = frame["eccentricity"].to_numpy(dtype=np.float64)
    inc_rad = frame["inclination"].to_numpy(dtype=np.float64) * _DEG_TO_RAD
    argp_rad = frame["arg_perigee"].to_numpy(dtype=np.float64) * _DEG_TO_RAD
    raan_unwrapped = np.unwrap(frame["raan"].to_numpy(dtype=np.float64), period=360.0)

    base: dict[str, FloatArray] = {
        "a": a_km,
        "e": ecc,
        "sin_i": np.sin(inc_rad),
        "cos_i": np.cos(inc_rad),
        "h": ecc * np.cos(argp_rad),
        "k": ecc * np.sin(argp_rad),
        "raan": raan_unwrapped,
    }

    n = t_days.shape[0]
    levels = np.empty((n, len(BASE_CHANNELS)), dtype=np.float64)
    residuals = np.empty((n, len(BASE_CHANNELS)), dtype=np.float64)
    deltas = np.zeros((n, len(BASE_CHANNELS)), dtype=np.float64)
    for col, name in enumerate(BASE_CHANNELS):
        level = base[name]
        residual = level - _local_linear_trend(t_days, level, DETREND_HALFWIDTH)
        levels[:, col] = level
        residuals[:, col] = residual
        if n > 1:
            deltas[1:, col] = residual[1:] - residual[:-1]

    dt = np.empty(n, dtype=np.float64)
    dt[0] = np.nan
    if n > 1:
        dt[1:] = t_days[1:] - t_days[:-1]
    dt_filled = np.nan_to_num(dt, nan=0.0)
    timing = _time2vec(dt_filled)
    elset_valid = np.ones((n, 1), dtype=np.float64)
    dt_saturated = (dt_filled > CLIP_CAP_DAYS).astype(np.float64).reshape(n, 1)

    matrix = np.hstack([levels, residuals, deltas, timing, elset_valid, dt_saturated])

    return RawChannels(
        norad_id=int(frame["norad_id"].iloc[0]),
        orbit_class=orbit_class_of(float(np.median(a_km))),
        epochs=epochs,
        matrix=matrix,
    )
