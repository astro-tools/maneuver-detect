"""The Δv inversion — turning a detected mean-element jump into a maneuver type and a Δv estimate.

A maneuver detector sees a satellite's orbit only through its SGP4 *mean* elements: a step in the
semi-major axis, eccentricity, inclination, and node across the inter-elset gap that brackets a
burn. This module is the physics that reads a Δv back out of that step. It implements the impulsive
form of the **Gauss variational equations** — the exact first-order relation between an impulsive
Δv (decomposed into radial / in-track / cross-track, the RSW frame) and the resulting element
change — both forward (:func:`gauss_forward`) and inverse (:func:`invert`):

* **in-track** Δv shows up as a step in semi-major axis (vis-viva) and eccentricity;
* **cross-track** Δv shows up as a step in inclination and node, in closed form;
* **radial** Δv shows up as an eccentricity-vector change beyond what the in-track burn explains —
  weakly observable, so it is treated as low-confidence by default.

The maneuver **type** is the dominant component (:func:`classify_type`), and the magnitude of the
combined impulse is the **Δv estimate**. Two physical facts shape the implementation, both found in
the V4 spike and frozen as design decision D5:

* **Secular drift must be detrended first.** The natural J2 nodal regression of the node is several
  degrees per day in LEO — far larger than any station-keeping burn — so a raw element difference
  reads as a huge spurious cross-track Δv. :func:`local_step` removes it with a model-free,
  two-sided local-linear fit; :func:`j2_secular_rates` is the analytic drift it cancels.
* **There is a per-class detectability floor.** Below ~cm/s (LEO) / ~0.1 m/s (GEO) the element step
  is buried in TLE noise and neither the Δv nor the type is recoverable; above it the inversion is
  good to about ±25% (D5). :func:`is_above_floor` and :meth:`Inversion.delta_v_estimate` gate the
  estimate against that floor, so nothing is reported where it cannot be trusted.

The quantitative accuracy of the recovered Δv against *published* burn magnitudes is validated
downstream against the DORIS/IDS Δv ground truth; here the contract is method correctness — the
forward/inverse pair round-trips, the type rule is right above the floor, and the magnitudes match
the textbook impulsive-maneuver relations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import ManeuverType

__all__ = [
    "DETECTABILITY_FLOOR_MS",
    "EARTH_J2",
    "EARTH_MU_KM3_S2",
    "EARTH_RADIUS_KM",
    "ORBIT_CLASS_GEO_MIN_A_KM",
    "ORBIT_CLASS_LEO_MAX_A_KM",
    "ElementStep",
    "Inversion",
    "Orbit",
    "circular_speed_km_s",
    "classify_type",
    "detectability_floor_ms",
    "gauss_forward",
    "invert",
    "is_above_floor",
    "j2_secular_rates",
    "local_step",
    "mean_motion_rad_s",
    "orbit_class_of",
    "orbital_speed_km_s",
    "semi_major_axis_km",
]

#: Earth gravitational parameter, km³/s² — the WGS-72 value SGP4 uses, so the constant matches the
#: mean elements the inversion consumes (a different μ would bias the mean-motion → semi-major-axis
#: conversion). The Kozai-mean vs Brouwer-mean distinction in SGP4's own ``a`` cancels to first
#: order in the *difference* :data:`ElementStep.delta_a_km` across a burn.
EARTH_MU_KM3_S2 = 398600.8

#: Earth equatorial radius, km (WGS-72) — for the J2 secular-rate scaling.
EARTH_RADIUS_KM = 6378.135

#: Earth second zonal harmonic (WGS-72) — drives the secular drift that must be detrended.
EARTH_J2 = 1.082616e-3

# Below these thresholds an inversion input is treated as carrying no usable signal: a cross-track
# magnitude this small is noise (the burn true anomaly cannot be located), and an in-plane 2x2
# system with a determinant this small is singular (the burn location leaves a and e unconstrained).
_CROSS_TRACK_EPS_KM_S = 1e-12
_DET_EPS = 1e-15


@dataclass(frozen=True)
class Orbit:
    """The reference (pre-maneuver) mean orbit the inversion linearises about.

    Only the elements the Gauss relations need: the in-plane size/shape (``semi_major_axis_km``,
    ``eccentricity``), the inclination, and the argument of perigee that maps a burn's argument of
    latitude to its true anomaly. The node and anomaly do not enter a first-order impulsive
    inversion, so they are omitted.

    Attributes:
        semi_major_axis_km: Semi-major axis, km.
        eccentricity: Eccentricity (dimensionless, ``[0, 1)``).
        inclination_rad: Inclination, radians.
        arg_perigee_rad: Argument of perigee, radians (defaults to 0, the only value that matters
            for a circular orbit, where perigee is undefined).
    """

    semi_major_axis_km: float
    eccentricity: float
    inclination_rad: float
    arg_perigee_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.semi_major_axis_km <= 0.0:
            raise ValueError(
                f"semi_major_axis_km must be positive, got {self.semi_major_axis_km!r}"
            )
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError(f"eccentricity must be in [0, 1), got {self.eccentricity!r}")
        if not 0.0 <= self.inclination_rad <= math.pi:
            raise ValueError(f"inclination_rad must be in [0, pi], got {self.inclination_rad!r}")

    @property
    def mean_motion_rad_s(self) -> float:
        """Mean motion, radians per second."""
        return mean_motion_rad_s(self.semi_major_axis_km)

    @property
    def semi_latus_rectum_km(self) -> float:
        """Semi-latus rectum ``p = a(1 - e²)``, km."""
        return self.semi_major_axis_km * (1.0 - self.eccentricity * self.eccentricity)

    @property
    def specific_angular_momentum(self) -> float:
        """Specific angular momentum ``h = sqrt(μ p)``, km²/s."""
        return math.sqrt(EARTH_MU_KM3_S2 * self.semi_latus_rectum_km)

    def radius_at(self, true_anomaly_rad: float) -> float:
        """Orbital radius ``r = p / (1 + e cos nu)`` at true anomaly ``true_anomaly_rad``, km."""
        return self.semi_latus_rectum_km / (1.0 + self.eccentricity * math.cos(true_anomaly_rad))


@dataclass(frozen=True)
class ElementStep:
    """The detrended anomalous step in the mean elements across a maneuver.

    The four mean-element changes a TLE detector can read reliably across the inter-elset gap, with
    natural secular drift already removed (see :func:`local_step`). The argument of perigee is
    omitted on purpose: it is ill-determined for the near-circular orbits in scope and contributes
    no robust signal.

    Attributes:
        delta_a_km: Change in semi-major axis, km.
        delta_eccentricity: Change in eccentricity (dimensionless).
        delta_inclination_rad: Change in inclination, radians.
        delta_raan_rad: Change in right ascension of the ascending node, radians.
    """

    delta_a_km: float
    delta_eccentricity: float
    delta_inclination_rad: float
    delta_raan_rad: float


@dataclass(frozen=True)
class Inversion:
    """A recovered impulsive maneuver — the RSW Δv decomposition, the total, and the type.

    The cross-track and radial components are stored as magnitudes: their sign is not observable
    from a mean-element step without knowing where in the orbit the burn occurred. The in-track
    component keeps its sign — positive raises the orbit (a prograde burn), negative lowers it —
    because that *is* fixed by the sign of the semi-major-axis step.

    Attributes:
        delta_v_ms: Total impulse magnitude ``|Δv|``, m/s — the root-sum-square of the components.
        radial_ms: Radial component magnitude, m/s (low-confidence; weakly observable).
        in_track_ms: In-track (transverse) component, m/s, signed.
        cross_track_ms: Cross-track (normal) component magnitude, m/s.
        maneuver_type: The dominant-component type (:func:`classify_type`).
    """

    delta_v_ms: float
    radial_ms: float
    in_track_ms: float
    cross_track_ms: float
    maneuver_type: ManeuverType

    @property
    def radial_dominant(self) -> bool:
        """Whether the maneuver is radial-dominated — low-confidence by default (D5)."""
        return self.maneuver_type is ManeuverType.RADIAL

    def is_above_floor(self, floor_ms: float) -> bool:
        """Whether ``delta_v_ms`` clears the per-object detectability floor ``floor_ms`` (m/s)."""
        return self.delta_v_ms >= floor_ms

    def delta_v_estimate(self, floor_ms: float) -> float | None:
        """The reportable Δv (m/s), or ``None`` below ``floor_ms``.

        Maps straight onto the schema's optional ``delta_v_estimate`` column: D5 reports a Δv only
        above the floor, so a below-floor inversion yields ``None`` rather than a noise figure.
        """
        return self.delta_v_ms if self.delta_v_ms >= floor_ms else None


def semi_major_axis_km(mean_motion_rev_per_day: float) -> float:
    """Semi-major axis (km) from SGP4 mean motion (revolutions per day), via Kepler's third law."""
    if mean_motion_rev_per_day <= 0.0:
        raise ValueError(f"mean_motion must be positive, got {mean_motion_rev_per_day!r}")
    n_rad_s = mean_motion_rev_per_day * 2.0 * math.pi / 86400.0
    return float((EARTH_MU_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0))


def mean_motion_rad_s(semi_major_axis: float) -> float:
    """Mean motion ``n = sqrt(μ / a³)`` (rad/s) from the semi-major axis (km)."""
    if semi_major_axis <= 0.0:
        raise ValueError(f"semi_major_axis must be positive, got {semi_major_axis!r}")
    return math.sqrt(EARTH_MU_KM3_S2 / semi_major_axis**3)


def circular_speed_km_s(semi_major_axis: float) -> float:
    """Circular orbital speed ``sqrt(μ / a)`` (km/s) at semi-major axis ``a`` (km)."""
    if semi_major_axis <= 0.0:
        raise ValueError(f"semi_major_axis must be positive, got {semi_major_axis!r}")
    return math.sqrt(EARTH_MU_KM3_S2 / semi_major_axis)


def orbital_speed_km_s(orbit: Orbit, true_anomaly_rad: float) -> float:
    """Orbital speed (km/s) at ``true_anomaly_rad`` from vis-viva ``v² = μ(2/r - 1/a)``."""
    r = orbit.radius_at(true_anomaly_rad)
    return math.sqrt(EARTH_MU_KM3_S2 * (2.0 / r - 1.0 / orbit.semi_major_axis_km))


def gauss_forward(
    *,
    radial_ms: float,
    in_track_ms: float,
    cross_track_ms: float,
    orbit: Orbit,
    true_anomaly_rad: float,
) -> ElementStep:
    """The forward Gauss VOP — element step produced by an impulsive Δv applied at ``true_anomaly``.

    The exact first-order (impulsive) Gauss variational equations for ``(Δa, Δe, Δi, ΔΩ)`` given the
    RSW components of the impulse, evaluated at the burn true anomaly. This is the model the
    inversion inverts and the generator the round-trip tests drive; it makes no circular-orbit
    approximation.
    """
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    inc = orbit.inclination_rad
    nu = true_anomaly_rad
    u = orbit.arg_perigee_rad + nu
    n = orbit.mean_motion_rad_s
    s = math.sqrt(1.0 - e * e)
    p = orbit.semi_latus_rectum_km
    h = orbit.specific_angular_momentum
    r = orbit.radius_at(nu)
    cos_e = (e + math.cos(nu)) / (1.0 + e * math.cos(nu))

    dv_r = radial_ms * 1e-3  # km/s
    dv_s = in_track_ms * 1e-3
    dv_w = cross_track_ms * 1e-3

    delta_a = (2.0 / (n * s)) * (e * math.sin(nu) * dv_r + (p / r) * dv_s)
    delta_e = (s / (n * a)) * (math.sin(nu) * dv_r + (math.cos(nu) + cos_e) * dv_s)
    delta_i = (r * math.cos(u) / h) * dv_w
    delta_raan = (r * math.sin(u) / (h * math.sin(inc))) * dv_w
    return ElementStep(
        delta_a_km=delta_a,
        delta_eccentricity=delta_e,
        delta_inclination_rad=delta_i,
        delta_raan_rad=delta_raan,
    )


def invert(
    step: ElementStep,
    orbit: Orbit,
    *,
    true_anomaly_rad: float | None = None,
) -> Inversion:
    """Recover the impulsive Δv (RSW components, total, type) from a detrended ``step``.

    The cross-track component comes from ``(Δi, ΔΩ)`` in closed form, and the burn argument of
    latitude — hence the true anomaly — from their ratio. The in-plane components come from
    ``(Δa, Δe)``: when the burn true anomaly is known (passed in, or recovered from a cross-track
    signal) and the resulting 2x2 system is well-conditioned, it is solved exactly; otherwise the
    burn location is unobservable, and the inversion falls back to the V4-validated, location-free
    estimator — vis-viva for the in-track component from ``Δa``, and the residual
    eccentricity-vector kick for the (low-confidence) radial component.

    Pass ``true_anomaly_rad`` when the burn location is known (e.g. validating against the forward
    model); leave it ``None`` for the realistic TLE case, where it is inferred or marginalised.
    """
    inc = orbit.inclination_rad
    h = orbit.specific_angular_momentum
    sin_i = math.sin(inc)

    # --- cross-track: closed form from the inclination/node step ---
    cross_signal = math.hypot(step.delta_inclination_rad, sin_i * step.delta_raan_rad)
    nu_from_cross: float | None = None
    if cross_signal > _CROSS_TRACK_EPS_KM_S:
        u = math.atan2(sin_i * step.delta_raan_rad, step.delta_inclination_rad)
        nu_from_cross = u - orbit.arg_perigee_rad
        r_burn = orbit.radius_at(nu_from_cross)
        cross_kms = (h / r_burn) * cross_signal
    else:
        cross_kms = 0.0

    # --- in-plane: 2x2 Gauss solve when the burn location is known, else location-free estimate ---
    # Only an explicitly supplied true anomaly drives the exact 2x2 solve. A nu *recovered* from the
    # cross-track step is unreliable when that step is mostly noise (an in-track burn has none), and
    # feeding it to the 2x2 would amplify the noise — so it serves only as a speed-radius hint.
    radial_kms, in_track_kms = _invert_in_plane(
        step, orbit, nu_exact=true_anomaly_rad, nu_hint=nu_from_cross
    )

    radial_ms = abs(radial_kms) * 1e3
    in_track_ms = in_track_kms * 1e3
    cross_track_ms = abs(cross_kms) * 1e3
    delta_v_ms = math.sqrt(radial_ms**2 + in_track_ms**2 + cross_track_ms**2)
    return Inversion(
        delta_v_ms=delta_v_ms,
        radial_ms=radial_ms,
        in_track_ms=in_track_ms,
        cross_track_ms=cross_track_ms,
        maneuver_type=classify_type(
            radial_ms=radial_ms, in_track_ms=in_track_ms, cross_track_ms=cross_track_ms
        ),
    )


def _invert_in_plane(
    step: ElementStep, orbit: Orbit, *, nu_exact: float | None, nu_hint: float | None
) -> tuple[float, float]:
    """Solve ``(Δa, Δe)`` for the in-plane ``(radial, in-track)`` Δv (km/s).

    With a known, well-conditioned burn location (``nu_exact``) the 2x2 Gauss system is exact.
    Otherwise the location is unobservable from ``(Δa, Δe)`` alone: the in-track component follows
    from vis-viva on ``Δa`` and the radial component from the residual of the eccentricity-vector
    kick — the weakly-observable path V4 flagged and validated. ``nu_hint`` (the nu recovered from a
    cross-track step, if any) only sets the radius at which the speed is evaluated; it never enters
    the solve, so noise on the cross-track channel cannot corrupt the in-plane estimate.
    """
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    n = orbit.mean_motion_rad_s
    s = math.sqrt(1.0 - e * e)
    p = orbit.semi_latus_rectum_km

    if nu_exact is not None:
        r = orbit.radius_at(nu_exact)
        cos_e = (e + math.cos(nu_exact)) / (1.0 + e * math.cos(nu_exact))
        # [Δa, Δe] = M · [dv_radial, dv_in_track]
        m00 = (2.0 / (n * s)) * (e * math.sin(nu_exact))
        m01 = (2.0 / (n * s)) * (p / r)
        m10 = (s / (n * a)) * math.sin(nu_exact)
        m11 = (s / (n * a)) * (math.cos(nu_exact) + cos_e)
        det = m00 * m11 - m01 * m10
        if abs(det) > _DET_EPS:
            radial = (step.delta_a_km * m11 - m01 * step.delta_eccentricity) / det
            in_track = (m00 * step.delta_eccentricity - step.delta_a_km * m10) / det
            return radial, in_track

    # Location-free estimator: vis-viva in-track from Δa, residual radial from Δe.
    nu_for_speed = nu_exact if nu_exact is not None else nu_hint
    r = orbit.radius_at(nu_for_speed) if nu_for_speed is not None else a
    v = math.sqrt(EARTH_MU_KM3_S2 * (2.0 / r - 1.0 / a))
    in_track = (n * n * a / (2.0 * v)) * step.delta_a_km
    residual = (v * step.delta_eccentricity) ** 2 - (2.0 * in_track) ** 2
    radial = math.sqrt(residual) if residual > 0.0 else 0.0
    return radial, in_track


def classify_type(*, radial_ms: float, in_track_ms: float, cross_track_ms: float) -> ManeuverType:
    """Attribute the maneuver type to the dominant Δv component (D5).

    Ties resolve in-track → cross-track → radial, the order of decreasing observability, so a
    coin-flip never lands on the least-trustworthy class.
    """
    by_type = {
        ManeuverType.IN_TRACK: abs(in_track_ms),
        ManeuverType.CROSS_TRACK: abs(cross_track_ms),
        ManeuverType.RADIAL: abs(radial_ms),
    }
    return max(by_type, key=lambda kind: by_type[kind])


def j2_secular_rates(orbit: Orbit) -> tuple[float, float, float]:
    """The J2 secular rates ``(Ω̇, ω̇, Ṁ)`` of node, perigee, and mean anomaly (rad/s).

    The dominant natural drift of a mean orbit: the node regresses, the apsides rotate, and the
    mean anomaly drifts, all secularly under Earth oblateness. This is the trend
    :func:`local_step` removes before an inversion — quoted here so a caller can predict or
    cross-check it. The node rate vanishes at the poles, the perigee rate at the critical
    inclination (≈63.4°), and the mean-anomaly rate where ``3cos²i = 1``.
    """
    e = orbit.eccentricity
    inc = orbit.inclination_rad
    n = orbit.mean_motion_rad_s
    p = orbit.semi_latus_rectum_km
    factor = n * EARTH_J2 * (EARTH_RADIUS_KM / p) ** 2
    cos_i = math.cos(inc)
    raan_dot = -1.5 * factor * cos_i
    argp_dot = 0.75 * factor * (5.0 * cos_i * cos_i - 1.0)
    mean_anomaly_dot = 0.75 * factor * math.sqrt(1.0 - e * e) * (3.0 * cos_i * cos_i - 1.0)
    return raan_dot, argp_dot, mean_anomaly_dot


def local_step(
    times: Sequence[float], values: Sequence[float], gap_index: int, *, window: int = 4
) -> float:
    """The detrended step in ``values`` across the gap before ``gap_index``, removing secular drift.

    A two-sided local-linear fit: a straight line is fit to the ``window`` samples on each side of
    the gap and both are evaluated at the gap midpoint; their difference is the anomalous step with
    the local secular trend (J2 nodal regression and the rest) subtracted out. ``gap_index`` is the
    index of the first sample *after* the gap, so the gap spans ``[gap_index - 1, gap_index]``.
    Without this detrending a maneuver-free node drift of degrees per day reads as a large spurious
    cross-track Δv — the V4 failure mode.

    Raises:
        ValueError: if ``window < 2``, the series lengths differ, or there are fewer than ``window``
            samples on either side of the gap.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    if len(times) != len(values):
        raise ValueError(f"times and values differ in length: {len(times)} vs {len(values)}")
    if gap_index < window or gap_index > len(times) - window:
        raise ValueError(
            f"gap_index {gap_index} leaves fewer than window={window} samples on a side "
            f"of a series of length {len(times)}"
        )
    midpoint = 0.5 * (times[gap_index - 1] + times[gap_index])
    before = _line_at(
        times[gap_index - window : gap_index], values[gap_index - window : gap_index], midpoint
    )
    after = _line_at(
        times[gap_index : gap_index + window], values[gap_index : gap_index + window], midpoint
    )
    return after - before


def _line_at(times: Sequence[float], values: Sequence[float], at: float) -> float:
    """Least-squares line through ``(times, values)`` evaluated at ``at``."""
    count = len(times)
    sum_t = math.fsum(times)
    sum_v = math.fsum(values)
    sum_tt = math.fsum(t * t for t in times)
    sum_tv = math.fsum(t * v for t, v in zip(times, values, strict=True))
    denom = count * sum_tt - sum_t * sum_t
    if denom == 0.0:  # all samples at one epoch — no trend to fit; fall back to the mean
        return sum_v / count
    slope = (count * sum_tv - sum_t * sum_v) / denom
    intercept = (sum_v - slope * sum_t) / count
    return slope * at + intercept


#: Semi-major-axis cut points (km) for the coarse orbit-class assignment that selects the nominal
#: detectability floor and the per-class feature normalisation statistics: LEO below ~2000 km
#: altitude, GEO near the geostationary radius, MEO between (the GPS / Galileo constellations at
#: ~26 560 / ~29 600 km land here).
ORBIT_CLASS_LEO_MAX_A_KM = 8378.0
ORBIT_CLASS_GEO_MIN_A_KM = 35000.0


def orbit_class_of(semi_major_axis_km: float) -> OrbitClass:
    """The coarse runtime orbit class of a representative semi-major axis (km).

    A single seam both the detector (selecting the nominal Δv floor) and the feature layer
    (selecting per-class normalisation statistics) read, so the class boundaries are defined once.
    The cuts are :data:`ORBIT_CLASS_LEO_MAX_A_KM` and :data:`ORBIT_CLASS_GEO_MIN_A_KM`.

    This returns only ``LEO`` / ``MEO`` / ``GEO`` — semi-major axis alone cannot distinguish the
    eccentric classes (``IGSO`` is geosynchronous so it lands in ``GEO`` here; a high-``e`` ``HEO``
    object lands wherever its ``a`` falls). The benchmark's per-class scoring uses the **pinned**
    class from the dataset recipe, not this runtime classifier, so ``IGSO`` / ``HEO`` are still
    scored as themselves; this seam only picks the detector's working floor / normalisation, where
    treating them as the nearest coarse class is an accepted first-pass approximation.
    """
    if semi_major_axis_km < ORBIT_CLASS_LEO_MAX_A_KM:
        return OrbitClass.LEO
    if semi_major_axis_km >= ORBIT_CLASS_GEO_MIN_A_KM:
        return OrbitClass.GEO
    return OrbitClass.MEO


#: Nominal per-class detectability floor for the Δv inversion, m/s (D4/D5). LEO and GEO are the
#: spike-measured values (LEO ~cm/s; GEO ~0.05-0.15 m/s, taken at the mid); MEO is a provisional
#: analytical placeholder between them. IGSO mirrors GEO (geosynchronous, comparable TLE quality);
#: HEO is an analytical placeholder (D4 left the HEO floor open). The *per-object* floor is
#: TLE-quality-dependent and is calibrated by the detector and benchmark — pass it explicitly when
#: known; this is the default. ``IGSO`` / ``HEO`` are present so a lookup by a pinned class never
#: fails, even though :func:`orbit_class_of` (the runtime path) never returns them.
DETECTABILITY_FLOOR_MS: dict[OrbitClass, float] = {
    OrbitClass.LEO: 0.01,
    OrbitClass.MEO: 0.03,
    OrbitClass.GEO: 0.10,
    OrbitClass.IGSO: 0.10,
    OrbitClass.HEO: 0.05,
}


def detectability_floor_ms(orbit_class: OrbitClass) -> float:
    """The nominal per-class detectability floor (m/s); see :data:`DETECTABILITY_FLOOR_MS`."""
    return DETECTABILITY_FLOOR_MS[orbit_class]


def is_above_floor(delta_v_ms: float, orbit_class: OrbitClass) -> bool:
    """Whether a Δv (m/s) clears the nominal per-class detectability floor for ``orbit_class``."""
    return delta_v_ms >= DETECTABILITY_FLOOR_MS[orbit_class]
