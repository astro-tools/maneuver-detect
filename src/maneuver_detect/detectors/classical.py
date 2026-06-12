"""The classical reference detector — the baseline every learned model must beat.

A rule-based maneuver detector built directly on the physics: it reads a per-object mean-element
series and emits the canonical maneuver schema (:mod:`maneuver_detect.schema`), with a Δv estimate
and a maneuver type from the :mod:`maneuver_detect.physics` Gauss inversion. The pipeline is:

1. **Regularise and smooth.** The series is first collapsed to one representative elset per UTC day
   (real catalogues fit several per day in bursts, and each extra inter-elset gap is another chance
   to fire). Each element is then smoothed with a time-aware Holt level-plus-trend model (the trend
   is per-day, the one-step forecast is ``level + trend·Δt``); the trend absorbs the *secular* drift
   — the J2 nodal regression, the slow drag decay of the semi-major axis — so steady drift leaves no
   standing residual. (Full Holt-Winters' seasonal term is omitted: the irregular cadence makes a
   fixed seasonal index ill-posed, and the bounded periodic SRP / luni-solar variability is handled
   by the robust noise scale below.)

2. **Score the residual jump (rule-based threshold).** Across each inter-elset gap the *detrended*
   step in each element is measured with the two-sided local-linear fit of
   :func:`maneuver_detect.physics.local_step`, and scored against the robust ``1.4826·MAD`` spread
   of that step statistic over all gaps — a self-calibrated noise scale immune to the maneuver
   jumps and to the fit's extrapolation leverage. A gap is a candidate when the step on a *detection
   channel* exceeds ``threshold`` scales. The detection channels are the two well-observed elements,
   the **semi-major axis** (in-track) and **inclination** (cross-track) — keeping detection
   multi-element (not mean-motion alone, D4) while leaving out the node and eccentricity, which are
   an order of magnitude noisier and, measured against the operator labels, fire almost only false
   alarms. Candidates are reduced to the strongest gap of each footprint (non-maximum suppression),
   then transients — a single bad elset or same-epoch re-fit, whose step reverses on the adjacent
   gap — are dropped.

3. **Invert (the physics).** The detrended steps in *all four* elements ``(a, e, i, Ω)`` and the
   smoothed pre-gap reference orbit go to :func:`maneuver_detect.physics.invert`, which returns the
   RSW Δv decomposition, the total ``|Δv|``, and the dominant-component maneuver **type** (D5). So
   the node and eccentricity still inform the Δv and type even though they do not trigger.

4. **Gate and emit.** The Δv is reported only above the **per-type** detectability floor — the node
   and inclination are good only to arc-seconds while the semi-major axis is good to metres, so a
   cross-track burn must be far larger than an in-track one to be seen, and a single floor would
   mis-classify one or the other (D4/D5; a radial-dominated maneuver is additionally
   low-confidence). The confidence is a monotone function of the residual-jump significance, so the
   benchmark can threshold it to set an operating point. Each surviving gap becomes one row of the
   canonical schema, with the bounding elset epochs as provenance.

The detector registers under the name ``"classical"`` and is the default that
:func:`maneuver_detect.detect` dispatches to.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from maneuver_detect.detectors.base import Detector
from maneuver_detect.physics import (
    ElementStep,
    Inversion,
    Orbit,
    detectability_floor_ms,
    invert,
    local_step,
    orbit_class_of,
)
from maneuver_detect.schema import COLUMNS, Maneuver, ManeuverType, empty_frame, to_frame

__all__ = ["ClassicalDetector"]

_logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

# The mean-element columns the detector consumes (a subset of the history.MEAN_ELEMENT_COLUMNS
# contract). Referenced by name so the detector does not pull the data layer (and SGP4) into the
# detectors import chain.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "epoch",
    "norad_id",
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "raan",
    "arg_perigee",
)

# A floor on the inter-elset spacing (days) used in the smoother's per-day trend, so a duplicate or
# near-duplicate epoch that survived cleaning cannot blow up the trend update.
_DT_FLOOR_DAYS = 1.0 / 1440.0  # one minute

# A floor on the robust noise scale, in each element's own units, so a degenerate (noise-free)
# channel cannot divide-by-zero; a real step then reads as overwhelmingly significant and noise as
# zero, both correct.
_SIGMA_FLOOR = 1e-12

_DEG_TO_RAD = math.pi / 180.0

# The finite-valued mean-element columns. A single non-finite value (a corrupt elset that slipped
# through cleaning) poisons the robust noise scale and silently suppresses *every* detection for the
# object, so rows carrying one are dropped before scoring (see :func:`_drop_non_finite`).
_ELEMENT_COLUMNS: tuple[str, ...] = (
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "raan",
    "arg_perigee",
)


def _drop_non_finite(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a non-finite (NaN/inf) value in any mean-element column.

    A non-finite element propagates through the median/MAD noise scale and turns every per-gap
    significance into NaN, so ``NaN < threshold`` is always false and the object returns zero
    detections with no error — a silent false-negative across the whole satellite. Dropping the
    offending rows lets detection proceed on the good elsets (a corrupt elset is unusable anyway)
    and logs how many were removed rather than failing silently.
    """
    elements = frame[list(_ELEMENT_COLUMNS)].to_numpy(dtype=float)
    finite = np.isfinite(elements).all(axis=1)
    if bool(finite.all()):
        return frame
    dropped = int((~finite).sum())
    _logger.warning("dropping %d elset(s) with non-finite mean elements before detection", dropped)
    kept: pd.DataFrame = frame.loc[finite]
    return kept


class ClassicalDetector(Detector):
    """Rule-based reference detector: Holt smoothing, residual-jump detection, and Gauss inversion.

    Consumes a per-object mean-element series (the
    :data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS` frame) and returns the canonical
    maneuver DataFrame. A frame carrying more than one ``norad_id`` is processed object by object,
    so the detector is correct on a single-object series and on a concatenated multi-object one.

    The tunables are constructor arguments with literature-reasonable defaults; the detectability
    floor that gates the Δv estimate is calibrated per object and maneuver type from the element
    noise (:meth:`floor_for`). The default no-argument construction is what the registry
    instantiates.
    """

    name = "classical"

    def __init__(
        self,
        *,
        window: int = 4,
        threshold: float = 6.0,
        smoothing_level: float = 0.5,
        smoothing_trend: float = 0.1,
        radial_confidence_factor: float = 0.6,
        regularize_daily: bool = True,
        persistence_revert_fraction: float = 0.5,
    ) -> None:
        """Configure the detector.

        Args:
            window: Samples per side for the two-sided local-linear step fit and the smoothing
                warm-up; a gap needs at least this many elsets on each side to be scored.
            threshold: Residual-jump threshold in robust noise scales — a gap is a candidate when a
                detrended element step exceeds ``threshold`` standard deviations of the element's
                post-smoothing residual.
            smoothing_level: Holt level smoothing factor (alpha) in ``[0, 1]``.
            smoothing_trend: Holt trend smoothing factor (beta) in ``[0, 1]``.
            radial_confidence_factor: Multiplier applied to the confidence of a radial-dominated
                detection (D5: radial maneuvers are weakly observable and reported low-confidence).
            regularize_daily: Collapse the series to one representative elset per UTC day before
                detection. Real catalogues fit several elsets per day in bursts; left raw, each
                extra gap is another chance to fire, so the dense cadence inflates the false-alarm
                rate. The D4 matching tolerance (the bracketing gap plus or minus one, about two
                days) absorbs the small epoch shift the binning introduces.
            persistence_revert_fraction: A candidate is rejected as a transient (a single bad elset
                or a same-epoch re-fit, not a maneuver) when its dominant-element step reverses on
                an adjacent gap with at least this fraction of its magnitude — a real maneuver is a
                sustained step, not a spike that returns.
        """
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        for factor_name, factor in (
            ("smoothing_level", smoothing_level),
            ("smoothing_trend", smoothing_trend),
            ("radial_confidence_factor", radial_confidence_factor),
            ("persistence_revert_fraction", persistence_revert_fraction),
        ):
            if not 0.0 <= factor <= 1.0:
                raise ValueError(f"{factor_name} must be in [0, 1], got {factor}")
        self.window = window
        self.threshold = threshold
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.radial_confidence_factor = radial_confidence_factor
        self.regularize_daily = regularize_daily
        self.persistence_revert_fraction = persistence_revert_fraction

    def detect(self, history: pd.DataFrame) -> pd.DataFrame:
        """Detect maneuvers in ``history`` and return the canonical maneuver DataFrame.

        ``history`` is a mean-element series (it must carry :data:`_REQUIRED_COLUMNS`). An empty or
        too-short series yields an empty canonical frame. A frame with multiple objects is grouped
        by ``norad_id`` and each object detected independently; the rows are returned sorted by
        ``(norad_id, epoch)``.
        """
        missing = [column for column in _REQUIRED_COLUMNS if column not in history.columns]
        if missing:
            raise ValueError(f"history is missing required columns: {missing}")
        if history.empty:
            return empty_frame()

        maneuvers: list[Maneuver] = []
        for _, group in history.groupby("norad_id", sort=True):
            ordered = group.sort_values("epoch")
            maneuvers.extend(self._detect_object(ordered))

        frame = to_frame(maneuvers)
        ordered_frame = frame.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
        result: pd.DataFrame = ordered_frame[list(COLUMNS)]
        return result

    def floor_for(self, history: pd.DataFrame) -> dict[ManeuverType, float]:
        """The per-type Δv detectability floor (m/s) for a single-object ``history``.

        The Δv below which a maneuver of each type cannot be told from this object's TLE noise — the
        data-derived, TLE-quality-dependent floor D4 calls for, bounded below by the nominal
        per-class floor. The benchmark uses the floor for a label's type to decide whether it is in
        the above-floor population scored for recall, and the detector uses the floor for a
        detection's type to gate the reported Δv (D5). It is computed on the same regularised series
        the detector sees, so the two agree. Falls back to the nominal class floor when the series
        is too short to calibrate.
        """
        model = self._prepare(history)
        if model is None:
            finite = _drop_non_finite(history)
            if finite.empty:
                raise ValueError("cannot compute a floor for an empty or all-non-finite history")
            median_a = float(np.median(finite["semi_major_axis"].to_numpy(dtype=float)))
            nominal = detectability_floor_ms(orbit_class_of(median_a))
            return dict.fromkeys(ManeuverType, nominal)
        return self._object_floors(model)

    def _detect_object(self, frame: pd.DataFrame) -> list[Maneuver]:
        """Detect maneuvers in one object's epoch-sorted mean-element series."""
        model = self._prepare(frame)
        if model is None:
            return []
        floors = self._object_floors(model)

        # Trigger detection on the two well-observed channels only: the semi-major axis (in-track)
        # and the inclination (cross-track). The eccentricity and node are an order of magnitude
        # noisier — measured against the operator labels they fire almost only false alarms — so
        # they do not trigger a detection, though they still inform the Δv and type through the
        # full-element inversion below. This keeps detection multi-element (in-track and
        # cross-track, not mean-motion alone, D4) without paying the node/eccentricity false alarms.
        channels = (
            (model.step_a, model.scale_a),
            (model.step_inc, model.scale_inc),
        )

        candidates: list[_Candidate] = []
        for offset, gap in enumerate(model.gaps):
            dominant_step, significance = _dominant_channel(channels, offset)
            if significance < self.threshold:
                continue
            candidates.append(
                _Candidate(
                    gap=gap,
                    significance=significance,
                    offset=offset,
                    dominant=dominant_step,
                    step=ElementStep(
                        delta_a_km=model.step_a[offset],
                        delta_eccentricity=model.step_e[offset],
                        delta_inclination_rad=model.step_inc[offset] * _DEG_TO_RAD,
                        delta_raan_rad=model.step_raan[offset] * _DEG_TO_RAD,
                    ),
                )
            )

        maneuvers: list[Maneuver] = []
        for candidate in _suppress_neighbours(candidates, self.window):
            # Reject a transient (single bad elset / same-epoch re-fit) only after suppression has
            # kept the strongest gap of the footprint: a real maneuver is a sustained step, while a
            # spike's strongest gap has its opposite-sign partner immediately beside it.
            if _is_transient(
                candidate.dominant, candidate.offset, self.persistence_revert_fraction
            ):
                continue
            gap = candidate.gap
            orbit = _reference_orbit(
                model.level_a, model.level_e, model.level_inc, model.level_argp, gap
            )
            inversion = invert(candidate.step, orbit)
            maneuvers.append(
                Maneuver(
                    epoch=_gap_midpoint(model.epochs[gap - 1], model.epochs[gap]),
                    confidence=self._confidence(candidate.significance, inversion),
                    type=inversion.maneuver_type,
                    delta_v_estimate=inversion.delta_v_estimate(floors[inversion.maneuver_type]),
                    norad_id=model.norad_id,
                    elset_epoch_before=model.epochs[gap - 1],
                    elset_epoch_after=model.epochs[gap],
                )
            )
        return maneuvers

    def _prepare(self, frame: pd.DataFrame) -> _SeriesModel | None:
        """Regularise, smooth, and score every gap of one object's series — the shared core.

        Returns ``None`` when the (regularised) series is too short to fit a step on each side of a
        gap. Both :meth:`_detect_object` and :meth:`floor_for` build on this so the floor is
        calibrated against exactly the series the detector runs on.
        """
        frame = _drop_non_finite(frame)
        if self.regularize_daily:
            frame = _regularize_daily(frame)
        n = len(frame)
        if n < 2 * self.window + 1:
            return None

        epochs = list(frame["epoch"])
        t_days = _epochs_to_days(epochs).tolist()
        a_km = frame["semi_major_axis"].to_numpy(dtype=float)
        ecc = frame["eccentricity"].to_numpy(dtype=float)
        inc_deg = _unwrap_deg(frame["inclination"].to_numpy(dtype=float))
        raan_deg = _unwrap_deg(frame["raan"].to_numpy(dtype=float))
        argp_deg = _unwrap_deg(frame["arg_perigee"].to_numpy(dtype=float))

        gaps = range(self.window, n - self.window)
        # The detrended step in each element across every scorable gap (the residual jump). The
        # self-calibrated scale is the robust spread of that step statistic over all gaps, so the
        # threshold is in true standard deviations of the statistic — immune to the maneuver jumps
        # (MAD ignores the outliers) and to the extrapolation leverage of the two-sided fit.
        step_a = _step_series(t_days, a_km, gaps, self.window)
        step_e = _step_series(t_days, ecc, gaps, self.window)
        step_inc = _step_series(t_days, inc_deg, gaps, self.window)
        step_raan = _step_series(t_days, raan_deg, gaps, self.window)

        return _SeriesModel(
            epochs=epochs,
            norad_id=int(frame["norad_id"].iloc[0]),
            a_km=a_km,
            ecc=ecc,
            inc_deg=inc_deg,
            argp_deg=argp_deg,
            level_a=_holt_levels(t_days, a_km, self.smoothing_level, self.smoothing_trend),
            level_e=_holt_levels(t_days, ecc, self.smoothing_level, self.smoothing_trend),
            level_inc=_holt_levels(t_days, inc_deg, self.smoothing_level, self.smoothing_trend),
            level_argp=_holt_levels(t_days, argp_deg, self.smoothing_level, self.smoothing_trend),
            gaps=gaps,
            step_a=step_a,
            step_e=step_e,
            step_inc=step_inc,
            step_raan=step_raan,
            scale_a=_robust_scale(step_a),
            scale_e=_robust_scale(step_e),
            scale_inc=_robust_scale(step_inc),
            scale_raan=_robust_scale(step_raan),
        )

    def _object_floors(self, model: _SeriesModel) -> dict[ManeuverType, float]:
        """The per-type Δv floor (m/s): the smallest detectable maneuver of each type.

        A maneuver is seen only through the element it moves, and those elements differ enormously
        in TLE precision — the semi-major axis is good to metres while the node and inclination are
        good to arc-seconds — so an in-track burn of a few mm/s and a cross-track burn of a few
        hundred mm/s can be equally (un)detectable. A single floor would therefore mis-classify one
        type or the other, so the floor is computed per type: the Δv that, applied as that type's
        burn, produces a ``threshold``-scale step in its element. Each is bounded below by the
        nominal per-class floor (the SGP4 representation limit).
        """
        orbit = _median_orbit(model)
        snr = self.threshold
        nominal = detectability_floor_ms(orbit_class_of(orbit.semi_major_axis_km))

        in_track = invert(ElementStep(snr * model.scale_a, 0.0, 0.0, 0.0), orbit).delta_v_ms
        # Cross-track shows in both inclination and node; the most sensitive of the two sets the
        # floor (a burn at the right argument of latitude lands its signal in whichever is cleaner).
        cross_inc = invert(
            ElementStep(0.0, 0.0, snr * model.scale_inc * _DEG_TO_RAD, 0.0), orbit
        ).delta_v_ms
        cross_raan = invert(
            ElementStep(0.0, 0.0, 0.0, snr * model.scale_raan * _DEG_TO_RAD), orbit
        ).delta_v_ms
        radial = invert(ElementStep(0.0, snr * model.scale_e, 0.0, 0.0), orbit).delta_v_ms
        return {
            ManeuverType.IN_TRACK: max(in_track, nominal),
            ManeuverType.CROSS_TRACK: max(min(cross_inc, cross_raan), nominal),
            ManeuverType.RADIAL: max(radial, nominal),
        }

    def _confidence(self, significance: float, inversion: Inversion) -> float:
        """Map residual-jump significance to a calibrated confidence in ``[0, 1]``.

        A saturating ``s / (s + threshold)`` rises from ``0.5`` at the detection threshold toward
        ``1`` for an unmistakable jump, monotone in the evidence so the benchmark's confidence
        sweep is meaningful. A radial-dominated maneuver is down-weighted (D5).
        """
        base = significance / (significance + self.threshold)
        if inversion.radial_dominant:
            base *= self.radial_confidence_factor
        return float(min(1.0, max(0.0, base)))


@dataclass(frozen=True)
class _SeriesModel:
    """One object's regularised, smoothed, gap-scored series — the shared detection state.

    Built once by :meth:`ClassicalDetector._prepare` and consumed by both the detector and the
    floor calibration: the unwrapped element arrays, their Holt-smoothed levels, the per-gap
    detrended step series, and the self-calibrated noise scale of each.
    """

    epochs: list[pd.Timestamp]
    norad_id: int
    a_km: FloatArray
    ecc: FloatArray
    inc_deg: FloatArray
    argp_deg: FloatArray
    level_a: FloatArray
    level_e: FloatArray
    level_inc: FloatArray
    level_argp: FloatArray
    gaps: range
    step_a: list[float]
    step_e: list[float]
    step_inc: list[float]
    step_raan: list[float]
    scale_a: float
    scale_e: float
    scale_inc: float
    scale_raan: float


class _Candidate:
    """A scored candidate gap, before suppression, the transient check, and inversion."""

    __slots__ = ("dominant", "gap", "offset", "significance", "step")

    def __init__(
        self,
        *,
        gap: int,
        significance: float,
        offset: int,
        dominant: list[float],
        step: ElementStep,
    ) -> None:
        self.gap = gap
        self.significance = significance
        self.offset = offset  # index into the gap-aligned step series, for the transient check
        self.dominant = dominant  # the dominant channel's step series
        self.step = step


def _dominant_channel(
    channels: tuple[tuple[list[float], float], ...], offset: int
) -> tuple[list[float], float]:
    """The most significant channel at ``offset`` — its step series and its significance.

    Significance is ``|step| / scale``; the dominant channel decides whether the gap clears the
    detection threshold and supplies the persistence (revert) check.
    """
    best_steps = channels[0][0]
    best_significance = -1.0
    for steps, scale in channels:
        significance = abs(steps[offset]) / scale
        if significance > best_significance:
            best_significance = significance
            best_steps = steps
    return best_steps, best_significance


def _is_transient(steps: list[float], offset: int, fraction: float) -> bool:
    """Whether the step at ``offset`` reverses on an adjacent gap — a spike, not a maneuver.

    A single bad elset (or a same-epoch re-fit) makes the series jump and then jump straight back,
    so the strongest gap of its footprint has its opposite-sign partner on the very next gap. A
    real maneuver shifts the level once and keeps it, leaving the neighbouring gaps the same sign.
    Applied after suppression has reduced each footprint to its strongest gap, an adjacent-gap
    check is enough: the candidate is a transient when a neighbouring gap holds an opposite-sign
    step of at least ``fraction`` of its magnitude.
    """
    primary = steps[offset]
    if primary == 0.0:
        return False
    for neighbour in (offset - 1, offset + 1):
        if 0 <= neighbour < len(steps):
            adjacent = steps[neighbour]
            if adjacent * primary < 0.0 and abs(adjacent) >= fraction * abs(primary):
                return True
    return False


def _regularize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce a series to one representative elset per UTC day — the one nearest local noon.

    Dense catalogues fit several elsets per day in bursts; each extra inter-elset gap is another
    chance to fire, so the raw cadence inflates the false-alarm rate. Keeping the most central
    elset of each day regularises the cadence to the roughly-daily grid the detector assumes, while
    the D4 matching tolerance absorbs the sub-day epoch shift.
    """
    if len(frame) < 2:
        return frame
    indexed = frame.reset_index(drop=True)
    epoch = indexed["epoch"]
    day = epoch.dt.floor("D")
    distance_from_noon = ((epoch - day).dt.total_seconds() / 86400.0 - 0.5).abs()
    keep = distance_from_noon.groupby(day).idxmin()
    reduced: pd.DataFrame = indexed.loc[sorted(keep)].reset_index(drop=True)
    return reduced


def _median_orbit(model: _SeriesModel) -> Orbit:
    """The median orbit of the series — the reference the floor inversion linearises about."""
    eccentricity = min(max(float(np.median(model.ecc)), 0.0), 0.999_999)
    inclination = min(max(float(np.median(model.inc_deg)) * _DEG_TO_RAD, 0.0), math.pi)
    return Orbit(
        semi_major_axis_km=float(np.median(model.a_km)),
        eccentricity=eccentricity,
        inclination_rad=inclination,
        arg_perigee_rad=float(np.median(model.argp_deg)) * _DEG_TO_RAD,
    )


def _suppress_neighbours(candidates: list[_Candidate], radius: int) -> list[_Candidate]:
    """Non-maximum suppression: keep the strongest candidate in each ``radius``-gap neighbourhood.

    A single maneuver's step contaminates the two-sided fit at the gaps on either side of it as the
    step edge enters and leaves the fit windows — a footprint of about ``window`` gaps — producing
    weaker secondary candidates around the true gap. Accepting candidates strongest-first and
    suppressing any within ``radius`` gaps of one already kept collapses that footprint to the one
    real detection. The cost is that two genuine maneuvers closer than ``radius`` gaps merge into
    one; at the TLE cadence that is within the D4 matching tolerance.
    """
    accepted: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda c: -c.significance):
        if all(abs(candidate.gap - kept.gap) > radius for kept in accepted):
            accepted.append(candidate)
    return sorted(accepted, key=lambda c: c.gap)


def _step_series(t_days: list[float], values: FloatArray, gaps: range, window: int) -> list[float]:
    """The detrended two-sided step in ``values`` across each gap in ``gaps``.

    Each entry is :func:`maneuver_detect.physics.local_step` at that gap — a straight line fit to
    the ``window`` samples on each side of the gap, differenced at the gap midpoint, so the local
    secular trend is removed and only the anomalous step survives. The result is aligned
    positionally with ``gaps``.
    """
    series = values.tolist()
    return [local_step(t_days, series, gap, window=window) for gap in gaps]


def _robust_scale(steps: list[float]) -> float:
    """Robust scale (``1.4826·MAD``) of the step statistic, floored away from zero.

    The median-absolute-deviation scale of the per-gap step values is the noise of the step
    statistic itself, and is insensitive to the handful of gaps that hold a real maneuver. Falls
    back to the standard deviation, then to a tiny floor, when the MAD degenerates (a noise-free
    series), so a real step still reads as overwhelmingly significant and pure noise as zero.
    """
    if not steps:
        return _SIGMA_FLOOR
    array = np.asarray(steps, dtype=float)
    mad = float(np.median(np.abs(array - np.median(array))))
    if mad > 0.0:
        return 1.4826 * mad
    std = float(np.std(array))
    return max(std, _SIGMA_FLOOR)


def _holt_levels(t_days: list[float], values: FloatArray, alpha: float, beta: float) -> FloatArray:
    """Holt level path over an irregular cadence — level plus a per-day trend.

    The forecast for sample ``i`` from ``i-1`` is ``level + trend·Δt`` with ``Δt`` the elset
    spacing in days, so the trend tracks the element's secular rate (degrees per day for the node,
    km per day for the semi-major axis) regardless of the irregular sampling. Returns the filtered
    level at each sample.
    """
    n = values.shape[0]
    levels = np.empty(n, dtype=float)
    if n == 0:
        return levels
    level = float(values[0])
    levels[0] = level
    if n == 1:
        return levels
    trend = (float(values[1]) - level) / max(float(t_days[1] - t_days[0]), _DT_FLOOR_DAYS)
    for i in range(1, n):
        dt = max(float(t_days[i] - t_days[i - 1]), _DT_FLOOR_DAYS)
        forecast = level + trend * dt
        new_level = alpha * float(values[i]) + (1.0 - alpha) * forecast
        new_trend = beta * (new_level - level) / dt + (1.0 - beta) * trend
        levels[i] = new_level
        level, trend = new_level, new_trend
    return levels


def _reference_orbit(
    level_a: FloatArray,
    level_e: FloatArray,
    level_inc_deg: FloatArray,
    level_argp_deg: FloatArray,
    gap: int,
) -> Orbit:
    """The pre-gap reference orbit the inversion linearises about, from the smoothed levels.

    Each element is its Holt-smoothed level at the last pre-gap sample (``gap - 1``) — less noisy
    than a single elset. Eccentricity and inclination are clamped into their physical ranges so a
    smoothing excursion cannot raise a spurious :class:`~maneuver_detect.physics.Orbit` error.
    """
    pre = gap - 1
    eccentricity = min(max(float(level_e[pre]), 0.0), 0.999_999)
    inclination = min(max(float(level_inc_deg[pre]) * _DEG_TO_RAD, 0.0), math.pi)
    return Orbit(
        semi_major_axis_km=float(level_a[pre]),
        eccentricity=eccentricity,
        inclination_rad=inclination,
        arg_perigee_rad=float(level_argp_deg[pre]) * _DEG_TO_RAD,
    )


def _epochs_to_days(epochs: list[pd.Timestamp]) -> FloatArray:
    """Convert a list of timestamps to floating-point days since the first epoch."""
    origin = epochs[0]
    return np.array([(epoch - origin).total_seconds() / 86400.0 for epoch in epochs], dtype=float)


def _unwrap_deg(values: FloatArray) -> FloatArray:
    """Unwrap an angle series in degrees so a 360° wrap does not read as a jump."""
    unwrapped = np.unwrap(values, period=360.0)
    return np.asarray(unwrapped, dtype=float)


def _gap_midpoint(before: pd.Timestamp, after: pd.Timestamp) -> pd.Timestamp:
    """The midpoint epoch of the inter-elset gap a detection brackets."""
    return pd.Timestamp(before + (after - before) / 2)
