"""Tests for ``maneuver_detect.physics`` — the vis-viva + Gauss Δv inversion and type rule.

Validated four ways, matching the V4 spike's design-of-experiment: the forward/inverse Gauss pair
round-trips an injected impulse exactly when the burn location is known; clean single-axis burns
classify to the right type and recover the textbook magnitudes (vis-viva for in-track, ``v·Δi`` for
a plane change); a noisy above-floor in-track burn recovers ``|Δv|`` within the ±25% D5 band while a
sub-floor burn is gated to no estimate; and the supporting physics — the J2 secular rates that must
be detrended and the model-free detrending itself — match known references (a sun-synchronous node
rate, the critical inclination) and recover an injected step through a secular trend.
"""

from __future__ import annotations

import math
import random
from statistics import median

import pytest

from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.physics import (
    DETECTABILITY_FLOOR_MS,
    EARTH_MU_KM3_S2,
    ElementStep,
    Orbit,
    circular_speed_km_s,
    classify_type,
    detectability_floor_ms,
    gauss_forward,
    invert,
    is_above_floor,
    j2_secular_rates,
    local_step,
    mean_motion_rad_s,
    orbital_speed_km_s,
    semi_major_axis_km,
)
from maneuver_detect.schema import ManeuverType

_DEG = math.pi / 180.0


# --------------------------------------------------------------------------- orbit kinematics
def test_mean_motion_and_semi_major_axis_round_trip() -> None:
    # An ISS-like orbit: ~92.7 min period ≈ 15.5 rev/day.
    a = semi_major_axis_km(15.5)
    assert 6700.0 < a < 6850.0
    n_rad_s = mean_motion_rad_s(a)
    rev_per_day = n_rad_s * 86400.0 / (2.0 * math.pi)
    assert rev_per_day == pytest.approx(15.5, rel=1e-12)


def test_vis_viva_matches_circular_speed_at_zero_eccentricity() -> None:
    orbit = Orbit(semi_major_axis_km=6778.0, eccentricity=0.0, inclination_rad=51.6 * _DEG)
    # A 400 km circular orbit is ~7.67 km/s.
    assert orbital_speed_km_s(orbit, 0.0) == pytest.approx(7.668, abs=1e-2)
    assert orbital_speed_km_s(orbit, 1.234) == pytest.approx(circular_speed_km_s(6778.0), rel=1e-12)


def test_eccentric_orbit_is_faster_at_perigee_than_apogee() -> None:
    orbit = Orbit(semi_major_axis_km=26600.0, eccentricity=0.74, inclination_rad=63.4 * _DEG)
    assert orbital_speed_km_s(orbit, 0.0) > orbital_speed_km_s(orbit, math.pi)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"semi_major_axis_km": -1.0, "eccentricity": 0.0, "inclination_rad": 0.0},
        {"semi_major_axis_km": 7000.0, "eccentricity": 1.0, "inclination_rad": 0.0},
        {"semi_major_axis_km": 7000.0, "eccentricity": -0.1, "inclination_rad": 0.0},
        {"semi_major_axis_km": 7000.0, "eccentricity": 0.0, "inclination_rad": 4.0},
    ],
)
def test_orbit_rejects_unphysical_elements(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Orbit(**kwargs)


# --------------------------------------------------------------------------- forward ↔ inverse
def test_gauss_forward_inverse_round_trip_recovers_all_three_components() -> None:
    # An eccentric, inclined orbit so every Gauss coefficient is exercised (not the circular limit).
    orbit = Orbit(
        semi_major_axis_km=7000.0,
        eccentricity=0.05,
        inclination_rad=50.0 * _DEG,
        arg_perigee_rad=30.0 * _DEG,
    )
    nu = 70.0 * _DEG
    step = gauss_forward(
        radial_ms=0.3, in_track_ms=1.0, cross_track_ms=0.6, orbit=orbit, true_anomaly_rad=nu
    )
    recovered = invert(step, orbit, true_anomaly_rad=nu)
    assert recovered.radial_ms == pytest.approx(0.3, rel=1e-6)
    assert recovered.in_track_ms == pytest.approx(1.0, rel=1e-6)
    assert recovered.cross_track_ms == pytest.approx(0.6, rel=1e-6)
    assert recovered.delta_v_ms == pytest.approx(math.sqrt(0.3**2 + 1.0**2 + 0.6**2), rel=1e-6)


def test_in_plane_and_out_of_plane_burns_are_decoupled() -> None:
    # A pure cross-track burn must leave a and e untouched (Δa, Δe depend only on R/S).
    orbit = Orbit(semi_major_axis_km=7200.0, eccentricity=0.02, inclination_rad=80.0 * _DEG)
    step = gauss_forward(
        radial_ms=0.0, in_track_ms=0.0, cross_track_ms=2.0, orbit=orbit, true_anomaly_rad=1.0
    )
    assert step.delta_a_km == pytest.approx(0.0, abs=1e-12)
    assert step.delta_eccentricity == pytest.approx(0.0, abs=1e-15)


def test_forward_sign_of_in_track_burn_raises_semi_major_axis() -> None:
    orbit = Orbit(semi_major_axis_km=7000.0, eccentricity=0.001, inclination_rad=66.0 * _DEG)
    raised = gauss_forward(
        radial_ms=0.0, in_track_ms=1.0, cross_track_ms=0.0, orbit=orbit, true_anomaly_rad=0.5
    )
    lowered = gauss_forward(
        radial_ms=0.0, in_track_ms=-1.0, cross_track_ms=0.0, orbit=orbit, true_anomaly_rad=0.5
    )
    assert raised.delta_a_km > 0.0 > lowered.delta_a_km


# --------------------------------------------------------------------------- textbook magnitudes
def test_in_track_burn_recovers_vis_viva_magnitude_and_type() -> None:
    # A tangential burn on a circular orbit: Δv ≈ (n/2)·Δa (the vis-viva tangential relation).
    a, delta_a = 7000.0, 2.0
    n = mean_motion_rad_s(a)
    step = ElementStep(
        delta_a_km=delta_a, delta_eccentricity=0.0, delta_inclination_rad=0.0, delta_raan_rad=0.0
    )
    orbit = Orbit(semi_major_axis_km=a, eccentricity=0.0, inclination_rad=51.6 * _DEG)
    result = invert(step, orbit)
    assert result.maneuver_type is ManeuverType.IN_TRACK
    assert result.in_track_ms == pytest.approx((n / 2.0) * delta_a * 1e3, rel=1e-9)
    assert result.cross_track_ms == pytest.approx(0.0, abs=1e-9)
    assert result.radial_ms == pytest.approx(0.0, abs=1e-9)


def test_cross_track_burn_recovers_plane_change_magnitude_and_type() -> None:
    # A pure inclination change on a circular orbit: Δv ≈ v·Δi for small Δi.
    a, delta_i = 7000.0, 0.01
    v_circ = circular_speed_km_s(a)
    step = ElementStep(
        delta_a_km=0.0,
        delta_eccentricity=0.0,
        delta_inclination_rad=delta_i,
        delta_raan_rad=0.0,
    )
    orbit = Orbit(semi_major_axis_km=a, eccentricity=0.0, inclination_rad=60.0 * _DEG)
    result = invert(step, orbit)
    assert result.maneuver_type is ManeuverType.CROSS_TRACK
    assert result.cross_track_ms == pytest.approx(v_circ * delta_i * 1e3, rel=1e-9)


def test_node_change_reads_as_cross_track() -> None:
    a = 7000.0
    orbit = Orbit(semi_major_axis_km=a, eccentricity=0.0, inclination_rad=60.0 * _DEG)
    step = ElementStep(
        delta_a_km=0.0,
        delta_eccentricity=0.0,
        delta_inclination_rad=0.0,
        delta_raan_rad=0.01,
    )
    result = invert(step, orbit)
    assert result.maneuver_type is ManeuverType.CROSS_TRACK
    assert result.cross_track_ms > 0.0


def test_radial_burn_at_quadrature_is_classified_radial() -> None:
    # A radial impulse at true anomaly 90 deg kicks the eccentricity vector with no transverse part.
    orbit = Orbit(semi_major_axis_km=8000.0, eccentricity=0.01, inclination_rad=45.0 * _DEG)
    step = gauss_forward(
        radial_ms=2.0,
        in_track_ms=0.0,
        cross_track_ms=0.0,
        orbit=orbit,
        true_anomaly_rad=math.pi / 2.0,
    )
    result = invert(step, orbit)
    assert result.maneuver_type is ManeuverType.RADIAL
    assert result.radial_dominant


# --------------------------------------------------------------------------- V4 floor behaviour
def test_above_floor_in_track_burn_recovers_within_d5_band_under_noise() -> None:
    # V4 Part A, in-track leg: a 1 m/s LEO burn is recovered to within ±25% with TLE-scale noise,
    # and classified correctly — the realistic unknown-burn-location (robust) path.
    rng = random.Random(0)
    orbit = Orbit(semi_major_axis_km=7000.0, eccentricity=0.001, inclination_rad=66.0 * _DEG)
    injected_ms = 1.0
    errors: list[float] = []
    correct = 0
    trials = 300
    for _ in range(trials):
        nu = rng.uniform(0.0, 2.0 * math.pi)
        clean = gauss_forward(
            radial_ms=0.0,
            in_track_ms=injected_ms,
            cross_track_ms=0.0,
            orbit=orbit,
            true_anomaly_rad=nu,
        )
        noisy = ElementStep(
            delta_a_km=clean.delta_a_km + rng.gauss(0.0, 5e-3),
            delta_eccentricity=clean.delta_eccentricity + rng.gauss(0.0, 1e-5),
            delta_inclination_rad=clean.delta_inclination_rad + rng.gauss(0.0, 1e-5),
            delta_raan_rad=clean.delta_raan_rad + rng.gauss(0.0, 1e-5),
        )
        result = invert(noisy, orbit)
        errors.append(abs(result.delta_v_ms - injected_ms) / injected_ms)
        correct += result.maneuver_type is ManeuverType.IN_TRACK
    assert median(errors) < 0.25
    assert correct / trials > 0.9


def test_sub_floor_burn_is_gated_to_no_estimate() -> None:
    # A 1 mm/s in-track burn sits below the 1 cm/s LEO floor: no Δv is reported (D5).
    orbit = Orbit(semi_major_axis_km=7000.0, eccentricity=0.0, inclination_rad=66.0 * _DEG)
    step = gauss_forward(
        radial_ms=0.0, in_track_ms=0.001, cross_track_ms=0.0, orbit=orbit, true_anomaly_rad=0.0
    )
    result = invert(step, orbit)
    floor = detectability_floor_ms(OrbitClass.LEO)
    assert not result.is_above_floor(floor)
    assert result.delta_v_estimate(floor) is None


def test_above_floor_estimate_is_reported() -> None:
    orbit = Orbit(semi_major_axis_km=7000.0, eccentricity=0.0, inclination_rad=66.0 * _DEG)
    step = gauss_forward(
        radial_ms=0.0, in_track_ms=0.5, cross_track_ms=0.0, orbit=orbit, true_anomaly_rad=0.0
    )
    result = invert(step, orbit)
    floor = detectability_floor_ms(OrbitClass.LEO)
    assert result.is_above_floor(floor)
    assert result.delta_v_estimate(floor) == pytest.approx(result.delta_v_ms)


# --------------------------------------------------------------------------- type classification
def test_classify_type_breaks_ties_toward_observability() -> None:
    # Equal magnitudes resolve in-track → cross-track → radial.
    assert (
        classify_type(radial_ms=1.0, in_track_ms=1.0, cross_track_ms=1.0) is ManeuverType.IN_TRACK
    )
    assert (
        classify_type(radial_ms=1.0, in_track_ms=0.5, cross_track_ms=1.0)
        is ManeuverType.CROSS_TRACK
    )
    assert classify_type(radial_ms=2.0, in_track_ms=0.5, cross_track_ms=1.0) is ManeuverType.RADIAL


# --------------------------------------------------------------------------- floor table
def test_floor_table_orders_leo_below_geo_and_covers_every_class() -> None:
    assert set(DETECTABILITY_FLOOR_MS) == set(OrbitClass)
    assert DETECTABILITY_FLOOR_MS[OrbitClass.LEO] < DETECTABILITY_FLOOR_MS[OrbitClass.GEO]
    assert is_above_floor(0.2, OrbitClass.GEO)
    assert not is_above_floor(0.02, OrbitClass.GEO)


# --------------------------------------------------------------------------- J2 secular drift
def test_j2_node_rate_matches_sun_synchronous_condition() -> None:
    # An ~800 km sun-synchronous orbit precesses its node eastward at +360°/year (0.9856°/day).
    orbit = Orbit(semi_major_axis_km=7178.0, eccentricity=0.0, inclination_rad=98.6 * _DEG)
    raan_dot, _, _ = j2_secular_rates(orbit)
    deg_per_day = raan_dot * 86400.0 / _DEG
    assert deg_per_day == pytest.approx(0.9856, abs=0.05)


def test_j2_perigee_rate_vanishes_at_critical_inclination() -> None:
    orbit = Orbit(semi_major_axis_km=26600.0, eccentricity=0.1, inclination_rad=63.4349 * _DEG)
    _, argp_dot, _ = j2_secular_rates(orbit)
    assert argp_dot == pytest.approx(0.0, abs=1e-12)


def test_j2_node_regresses_for_prograde_orbit() -> None:
    orbit = Orbit(semi_major_axis_km=7000.0, eccentricity=0.0, inclination_rad=51.6 * _DEG)
    raan_dot, _, _ = j2_secular_rates(orbit)
    assert raan_dot < 0.0


# --------------------------------------------------------------------------- detrending
def test_local_step_recovers_an_injected_step_through_a_secular_trend() -> None:
    # A linear drift of 0.5/day with a 3.0 step injected across the gap at index 5.
    times = [float(day) for day in range(10)]
    values = [10.0 + 0.5 * t + (3.0 if k >= 5 else 0.0) for k, t in enumerate(times)]
    assert local_step(times, values, 5, window=4) == pytest.approx(3.0, abs=1e-9)


def test_local_step_returns_zero_for_pure_drift_with_no_maneuver() -> None:
    times = [float(day) for day in range(10)]
    values = [100.0 - 1.25 * t for t in times]  # pure secular drift, no step
    assert local_step(times, values, 5, window=4) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("gap_index", "window"),
    [(5, 1), (2, 4), (8, 4)],
)
def test_local_step_guards_against_insufficient_support(gap_index: int, window: int) -> None:
    times = [float(day) for day in range(10)]
    values = [float(day) for day in range(10)]
    with pytest.raises(ValueError):
        local_step(times, values, gap_index, window=window)


def test_local_step_rejects_mismatched_series_lengths() -> None:
    with pytest.raises(ValueError):
        local_step([0.0, 1.0, 2.0], [0.0, 1.0], 1, window=2)


# --------------------------------------------------------------------------- constant sanity
def test_gravitational_parameter_is_the_sgp4_wgs72_value() -> None:
    assert EARTH_MU_KM3_S2 == 398600.8
