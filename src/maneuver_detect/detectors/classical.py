"""The classical reference detector — the baseline every learned model must beat.

A rule-based maneuver detector built directly on the physics: it reads a per-object mean-element
series and emits the canonical maneuver schema (:mod:`maneuver_detect.schema`), with a Δv estimate
and a maneuver type from the :mod:`maneuver_detect.physics` Gauss inversion. The pipeline, per
object and per detection element, is four steps:

1. **Smooth (Holt's linear method).** A level-plus-trend exponential smoother, generalised to the
   irregular TLE cadence (the trend is per-day and the one-step forecast is ``level + trend·Δt``).
   The trend term absorbs the *secular* drift of a mean element — the J2 nodal regression of the
   node, the slow drag decay of the semi-major axis — so steady natural drift leaves no standing
   residual. (The seasonal third component of full Holt-Winters is deliberately omitted: the
   irregular, gap-ridden TLE cadence makes a fixed seasonal index ill-posed, and the bounded
   periodic variability of SRP / luni-solar perturbations is handled by the robust noise scale
   below rather than modelled explicitly.)

2. **Score the residual jump (rule-based threshold).** The scatter of the post-smoothing residual,
   measured robustly as ``1.4826·MAD`` so a maneuver does not inflate its own threshold, sets a
   per-object, per-element noise scale. Across each inter-elset gap the *detrended* step in the
   element is measured with the two-sided local-linear fit of
   :func:`maneuver_detect.physics.local_step` — which removes the local secular trend on both sides
   of the gap — and a gap is a candidate when
   that step, on any element, exceeds ``threshold`` noise scales. This is what suppresses the
   natural-variability confounds (drag, SRP, luni-solar) and TLE noise: a smooth drift produces no
   step, and Gaussian scatter clears the threshold only with negligible probability.

3. **Invert (the physics).** The detrended steps in ``(a, e, i, Ω)`` and the smoothed pre-gap
   reference orbit go to :func:`maneuver_detect.physics.invert`, which returns the RSW Δv
   decomposition, the total ``|Δv|``, and the dominant-component maneuver **type** (D5). The
   detector is multi-element by construction (D4): the in-track channel is the semi-major axis, the
   cross-track channel the inclination and node, the radial channel the eccentricity — mean motion
   alone would catch only a fraction of real maneuvers.

4. **Gate and emit.** The Δv is reported only above the per-object detectability floor (D5: nothing
   below it, and a radial-dominated maneuver is low-confidence); the detection confidence is a
   monotone, calibrated function of the residual-jump significance, so the benchmark can threshold
   it to set an operating point. Each surviving gap becomes one row of the canonical schema, with
   the bounding elset epochs as provenance.

The detector registers under the name ``"classical"`` and is the default that
:func:`maneuver_detect.detect` dispatches to.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from maneuver_detect.detectors.base import Detector
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.physics import (
    ElementStep,
    Inversion,
    Orbit,
    detectability_floor_ms,
    invert,
    local_step,
)
from maneuver_detect.schema import COLUMNS, Maneuver, empty_frame, to_frame

__all__ = ["ClassicalDetector"]

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

# Semi-major-axis cut points (km) for the coarse orbit-class assignment that selects the nominal
# detectability floor: LEO below ~2000 km altitude, GEO near the geostationary radius, MEO between
# (the GPS constellation at ~26 560 km lands here).
_LEO_MAX_A_KM = 8378.0
_GEO_MIN_A_KM = 35000.0

# A floor on the inter-elset spacing (days) used in the smoother's per-day trend, so a duplicate or
# near-duplicate epoch that survived cleaning cannot blow up the trend update.
_DT_FLOOR_DAYS = 1.0 / 1440.0  # one minute

# A floor on the robust noise scale, in each element's own units, so a degenerate (noise-free)
# channel cannot divide-by-zero; a real step then reads as overwhelmingly significant and noise as
# zero, both correct.
_SIGMA_FLOOR = 1e-12

_DEG_TO_RAD = math.pi / 180.0


class ClassicalDetector(Detector):
    """Rule-based reference detector: Holt smoothing, residual-jump detection, and Gauss inversion.

    Consumes a per-object mean-element series (the
    :data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS` frame) and returns the canonical
    maneuver DataFrame. A frame carrying more than one ``norad_id`` is processed object by object,
    so the detector is correct on a single-object series and on a concatenated multi-object one.

    The tunables are constructor arguments with literature-reasonable defaults; the detectability
    floor that gates the Δv estimate is calibrated per object from its orbit class and residual
    noise. The default no-argument construction is what the registry instantiates.
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
        """
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        for factor_name, factor in (
            ("smoothing_level", smoothing_level),
            ("smoothing_trend", smoothing_trend),
        ):
            if not 0.0 <= factor <= 1.0:
                raise ValueError(f"{factor_name} must be in [0, 1], got {factor}")
        if not 0.0 <= radial_confidence_factor <= 1.0:
            raise ValueError(
                f"radial_confidence_factor must be in [0, 1], got {radial_confidence_factor}"
            )
        self.window = window
        self.threshold = threshold
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.radial_confidence_factor = radial_confidence_factor

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

    def _detect_object(self, frame: pd.DataFrame) -> list[Maneuver]:
        """Detect maneuvers in one object's epoch-sorted mean-element series."""
        n = len(frame)
        # A gap is scored only with `window` elsets on each side; below that no step can be fit.
        if n < 2 * self.window + 1:
            return []

        epochs = list(frame["epoch"])
        norad_id = int(frame["norad_id"].iloc[0])
        t_days = _epochs_to_days(epochs).tolist()

        a_km = frame["semi_major_axis"].to_numpy(dtype=float)
        ecc = frame["eccentricity"].to_numpy(dtype=float)
        inc_deg = _unwrap_deg(frame["inclination"].to_numpy(dtype=float))
        raan_deg = _unwrap_deg(frame["raan"].to_numpy(dtype=float))
        argp_deg = _unwrap_deg(frame["arg_perigee"].to_numpy(dtype=float))

        # Smooth each element with the time-aware Holt level/trend model; the smoothed level at the
        # last pre-gap sample is the reference orbit the inversion linearises about.
        level_a = _holt_levels(t_days, a_km, self.smoothing_level, self.smoothing_trend)
        level_e = _holt_levels(t_days, ecc, self.smoothing_level, self.smoothing_trend)
        level_inc = _holt_levels(t_days, inc_deg, self.smoothing_level, self.smoothing_trend)
        level_argp = _holt_levels(t_days, argp_deg, self.smoothing_level, self.smoothing_trend)

        gaps = range(self.window, n - self.window)
        # The detrended step in each element across every scorable gap (the residual jump).
        step_a = _step_series(t_days, a_km, gaps, self.window)
        step_e = _step_series(t_days, ecc, gaps, self.window)
        step_inc = _step_series(t_days, inc_deg, gaps, self.window)
        step_raan = _step_series(t_days, raan_deg, gaps, self.window)

        # Self-calibrated noise scale: the robust spread of the step statistic itself over all gaps,
        # so the threshold is in true standard deviations of that statistic — immune to both the
        # maneuver jumps (MAD ignores the outliers) and the extrapolation leverage of the two-sided
        # fit. This is what holds the false-alarm rate down on drift/SRP/luni-solar variability.
        scale_a = _robust_scale(step_a)
        scale_e = _robust_scale(step_e)
        scale_inc = _robust_scale(step_inc)
        scale_raan = _robust_scale(step_raan)

        floor_ms = self._floor_ms(a_km)

        candidates: list[_Candidate] = []
        for offset, gap in enumerate(gaps):
            significance = max(
                abs(step_a[offset]) / scale_a,
                abs(step_e[offset]) / scale_e,
                abs(step_inc[offset]) / scale_inc,
                abs(step_raan[offset]) / scale_raan,
            )
            if significance < self.threshold:
                continue
            candidates.append(
                _Candidate(
                    gap=gap,
                    significance=significance,
                    step=ElementStep(
                        delta_a_km=step_a[offset],
                        delta_eccentricity=step_e[offset],
                        delta_inclination_rad=step_inc[offset] * _DEG_TO_RAD,
                        delta_raan_rad=step_raan[offset] * _DEG_TO_RAD,
                    ),
                )
            )

        maneuvers: list[Maneuver] = []
        for candidate in _suppress_neighbours(candidates, self.window):
            gap = candidate.gap
            orbit = _reference_orbit(level_a, level_e, level_inc, level_argp, gap)
            inversion = invert(candidate.step, orbit)
            maneuvers.append(
                Maneuver(
                    epoch=_gap_midpoint(epochs[gap - 1], epochs[gap]),
                    confidence=self._confidence(candidate.significance, inversion),
                    type=inversion.maneuver_type,
                    delta_v_estimate=inversion.delta_v_estimate(floor_ms),
                    norad_id=norad_id,
                    elset_epoch_before=epochs[gap - 1],
                    elset_epoch_after=epochs[gap],
                )
            )
        return maneuvers

    def _floor_ms(self, a_km: FloatArray) -> float:
        """The per-object Δv detectability floor (m/s) from the orbit class (D4/D5)."""
        return detectability_floor_ms(_orbit_class_of(float(np.median(a_km))))

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


class _Candidate:
    """A scored candidate gap, before merging and inversion."""

    __slots__ = ("gap", "significance", "step")

    def __init__(self, *, gap: int, significance: float, step: ElementStep) -> None:
        self.gap = gap
        self.significance = significance
        self.step = step


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


def _orbit_class_of(a_km: float) -> OrbitClass:
    """Assign the coarse orbit class from a representative semi-major axis (km)."""
    if a_km < _LEO_MAX_A_KM:
        return OrbitClass.LEO
    if a_km >= _GEO_MIN_A_KM:
        return OrbitClass.GEO
    return OrbitClass.MEO


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
