"""Tests for the per-token channel construction (the V5 / D11 encoding).

Correctness is checked on synthetic mean-element series with a known background and an injected
step, the same discipline the detector tests use: a pure secular drift must detrend to zero (so the
model sees maneuver steps, not J2 nodal regression — the V4 failure mode), an injected step must
surface on the delta channel of the gap that brackets it, and the timing / mask channels must encode
exactly as the frozen contract specifies.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from maneuver_detect.features import build_channels
from maneuver_detect.features.channels import (
    BASE_CHANNELS,
    CHANNEL_NAMES,
    CLIP_CAP_DAYS,
    DETREND_HALFWIDTH,
    N_CHANNELS,
    N_ELEMENT_CHANNELS,
)
from maneuver_detect.labels.record import OrbitClass

_DEG = math.pi / 180.0


def _frame(
    *,
    n: int = 120,
    norad_id: int = 25544,
    a0: float = 6778.0,
    drift_km_day: float = -0.002,
    raan_rate_deg_day: float = 5.0,
    inc0_deg: float = 66.0,
    step_at: int | None = None,
    step_a_km: float = 0.0,
    step_inc_deg: float = 0.0,
    gaps_days: np.ndarray | None = None,
) -> pd.DataFrame:
    """A synthetic single-object mean-element series with secular drift and an optional step."""
    if gaps_days is None:
        gaps_days = np.ones(n - 1)
    times = np.concatenate([[0.0], np.cumsum(gaps_days)])
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    epochs = [start + timedelta(days=float(t)) for t in times]
    a = a0 + drift_km_day * times
    inc = np.full(n, inc0_deg)
    if step_at is not None:
        a[step_at:] += step_a_km
        inc[step_at:] += step_inc_deg
    raan = (20.0 + raan_rate_deg_day * times) % 360.0
    return pd.DataFrame(
        {
            "epoch": pd.Series(epochs, dtype="datetime64[ns, UTC]"),
            "norad_id": np.full(n, norad_id, dtype=int),
            "semi_major_axis": a,
            "eccentricity": np.full(n, 0.001),
            "inclination": inc,
            "raan": raan,
            "arg_perigee": np.full(n, 90.0),
        }
    )


def test_shape_and_channel_names_are_the_frozen_contract() -> None:
    rc = build_channels(_frame(n=100))
    assert rc.matrix.shape == (100, N_CHANNELS)
    assert len(CHANNEL_NAMES) == N_CHANNELS
    assert rc.channel_names == CHANNEL_NAMES
    assert 3 * len(BASE_CHANNELS) == N_ELEMENT_CHANNELS
    # the node Ω is carried (D11.3 / the V5 proof) — a level, a residual, and a delta channel.
    assert {"level_raan", "resid_raan", "delta_raan"} <= set(CHANNEL_NAMES)
    assert rc.matrix.dtype == np.float64


def test_build_is_byte_deterministic() -> None:
    frame = _frame(n=130, step_at=70, step_a_km=0.4)
    one = build_channels(frame).matrix
    two = build_channels(frame).matrix
    assert one.tobytes() == two.tobytes()


def test_pure_secular_drift_detrends_to_zero() -> None:
    # A pure linear drift in a and a pure linear (wrapping) ramp in the node must leave no residual:
    # the secular J2 / drag trend is exactly what the local-linear detrend removes.
    rc = build_channels(_frame(n=140, drift_km_day=-0.01, raan_rate_deg_day=8.0))
    resid_a = rc.matrix[:, CHANNEL_NAMES.index("resid_a")]
    resid_raan = rc.matrix[:, CHANNEL_NAMES.index("resid_raan")]
    assert np.max(np.abs(resid_a)) < 1e-6
    assert np.max(np.abs(resid_raan)) < 1e-6


def test_injected_step_surfaces_on_the_delta_of_the_bracketing_gap() -> None:
    rc = build_channels(_frame(n=140, step_at=70, step_a_km=0.5))
    delta_a = rc.matrix[:, CHANNEL_NAMES.index("delta_a")]
    assert int(np.argmax(np.abs(delta_a))) == 70
    # the two-sided detrend absorbs a little of the step near the gap, but the signed delta recovers
    # the bulk of the injected +0.5 km jump (~95%), with the right sign.
    assert 0.4 < delta_a[70] < 0.5


def test_cross_track_step_surfaces_on_inclination_channels() -> None:
    rc = build_channels(_frame(n=140, step_at=70, step_inc_deg=0.02))
    delta_sin_i = rc.matrix[:, CHANNEL_NAMES.index("delta_sin_i")]
    assert int(np.argmax(np.abs(delta_sin_i))) == 70


def test_two_sided_detrend_smears_the_step_into_exactly_halfwidth_neighbours() -> None:
    # The centred (two-sided) local-linear detrend is symmetric by design (D11.3), so a real step
    # bleeds a small, opposite-signed share onto the DETREND_HALFWIDTH tokens either side. Pin both
    # the magnitude and the radius so the intra-object smear cannot silently grow: the bracketing
    # gap keeps 2h/(2h+1) of the step and each neighbour within the window carries -1/(2h+1) of it.
    step, at, n = 0.5, 80, 160
    rc = build_channels(_frame(n=n, step_at=at, step_a_km=step))
    delta_a = rc.matrix[:, CHANNEL_NAMES.index("delta_a")]
    window = 2 * DETREND_HALFWIDTH + 1

    assert delta_a[at] == pytest.approx(step * (2 * DETREND_HALFWIDTH) / window, abs=1e-6)
    smear = -step / window
    for offset in range(1, DETREND_HALFWIDTH + 1):
        assert delta_a[at - offset] == pytest.approx(smear, abs=1e-6)
        assert delta_a[at + offset] == pytest.approx(smear, abs=1e-6)
    # Just outside the ±halfwidth window the residual is clean again — the radius is exactly H.
    assert delta_a[at - DETREND_HALFWIDTH - 1] == pytest.approx(0.0, abs=1e-9)
    assert delta_a[at + DETREND_HALFWIDTH + 1] == pytest.approx(0.0, abs=1e-9)


def test_eccentricity_vector_tracks_arg_perigee_across_all_quadrants() -> None:
    # ω is carried as the eccentricity vector (h, k) = (e·cos ω, e·sin ω) so the channel never wraps
    # (D11.3). Sweep ω through every quadrant and past 360° and assert h, k match the definition and
    # stay continuous across the wrap — the failure a raw-angle ω channel would show.
    n = 120
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    epochs = [start + timedelta(days=i) for i in range(n)]
    ecc = 0.02
    argp = (10.0 + 7.0 * np.arange(n)) % 360.0  # ramps through all four quadrants and wraps
    frame = pd.DataFrame(
        {
            "epoch": pd.Series(epochs, dtype="datetime64[ns, UTC]"),
            "norad_id": np.full(n, 25544, dtype=int),
            "semi_major_axis": np.full(n, 6778.0),
            "eccentricity": np.full(n, ecc),
            "inclination": np.full(n, 66.0),
            "raan": np.full(n, 20.0),
            "arg_perigee": argp,
        }
    )
    rc = build_channels(frame)
    argp_rad = argp * _DEG
    assert np.allclose(rc.matrix[:, CHANNEL_NAMES.index("level_h")], ecc * np.cos(argp_rad))
    assert np.allclose(rc.matrix[:, CHANNEL_NAMES.index("level_k")], ecc * np.sin(argp_rad))
    # Continuous across 360°→0°: consecutive level steps stay bounded by e·Δω, never a full jump.
    assert np.max(np.abs(np.diff(rc.matrix[:, CHANNEL_NAMES.index("level_h")]))) < 0.005
    assert np.max(np.abs(np.diff(rc.matrix[:, CHANNEL_NAMES.index("level_k")]))) < 0.005


def test_duplicate_epoch_rows_encode_without_error() -> None:
    # A real catalogue can emit two elsets at the same epoch (a zero-Δt gap); the encoding must stay
    # finite and treat it as a zero-duration interval rather than crashing the local-linear fit.
    frame = _frame(n=40)
    frame.loc[20, "epoch"] = frame.loc[19, "epoch"]
    rc = build_channels(frame)
    assert rc.matrix.shape == (40, N_CHANNELS)
    assert np.isfinite(rc.matrix).all()
    dt_clip = rc.matrix[:, CHANNEL_NAMES.index("dt_clip")]
    assert np.count_nonzero(dt_clip[1:] == 0.0) == 1  # exactly one interior zero-Δt gap


def test_level_channels_are_the_raw_elements() -> None:
    frame = _frame(n=60, inc0_deg=55.0)
    rc = build_channels(frame)
    inc_rad = frame["inclination"].to_numpy() * _DEG
    argp_rad = frame["arg_perigee"].to_numpy() * _DEG
    ecc = frame["eccentricity"].to_numpy()
    assert np.allclose(rc.matrix[:, CHANNEL_NAMES.index("level_sin_i")], np.sin(inc_rad))
    assert np.allclose(rc.matrix[:, CHANNEL_NAMES.index("level_cos_i")], np.cos(inc_rad))
    assert np.allclose(rc.matrix[:, CHANNEL_NAMES.index("level_h")], ecc * np.cos(argp_rad))
    assert np.allclose(rc.matrix[:, CHANNEL_NAMES.index("level_k")], ecc * np.sin(argp_rad))


def test_first_token_deltas_are_zero_and_elset_valid_is_set() -> None:
    rc = build_channels(_frame(n=50))
    delta_cols = [i for i, name in enumerate(CHANNEL_NAMES) if name.startswith("delta_")]
    assert np.all(rc.matrix[0, delta_cols] == 0.0)
    assert np.all(rc.matrix[:, CHANNEL_NAMES.index("elset_valid")] == 1.0)


def test_dt_saturation_flag_fires_exactly_above_the_clip_cap() -> None:
    # one 5-day gap at token 30 (> 2.5-day cap), every other gap one day.
    n = 60
    gaps = np.ones(n - 1)
    gaps[29] = 5.0  # the gap entering token 30
    rc = build_channels(_frame(n=n, gaps_days=gaps))
    sat = rc.matrix[:, CHANNEL_NAMES.index("dt_saturated")]
    dt_clip = rc.matrix[:, CHANNEL_NAMES.index("dt_clip")]
    assert sat[30] == 1.0
    assert sat.sum() == 1.0  # only that gap saturates
    assert sat[0] == 0.0
    assert abs(dt_clip[30] - 1.0) < 1e-9  # min(5, 2.5)/2.5
    assert abs(dt_clip[1] - 1.0 / CLIP_CAP_DAYS) < 1e-9  # a one-day gap


def test_single_row_series_encodes_without_error() -> None:
    rc = build_channels(_frame(n=1))
    assert rc.matrix.shape == (1, N_CHANNELS)
    delta_cols = [i for i, name in enumerate(CHANNEL_NAMES) if name.startswith("delta_")]
    assert np.all(rc.matrix[0, delta_cols] == 0.0)


def test_orbit_class_is_derived_from_the_median_semi_major_axis() -> None:
    assert build_channels(_frame(a0=6778.0)).orbit_class is OrbitClass.LEO
    assert build_channels(_frame(a0=26560.0)).orbit_class is OrbitClass.MEO
    assert build_channels(_frame(a0=42164.0)).orbit_class is OrbitClass.GEO


def test_non_finite_rows_are_dropped() -> None:
    frame = _frame(n=40)
    frame.loc[10, "semi_major_axis"] = np.nan
    rc = build_channels(frame)
    assert rc.n_tokens == 39


def test_all_non_finite_history_raises() -> None:
    frame = _frame(n=20)
    frame["semi_major_axis"] = np.nan
    with pytest.raises(ValueError, match="no finite elsets"):
        build_channels(frame)


def test_multi_object_history_is_rejected() -> None:
    frame = pd.concat([_frame(n=30, norad_id=1), _frame(n=30, norad_id=2)], ignore_index=True)
    with pytest.raises(ValueError, match="single-object"):
        build_channels(frame)


def test_empty_and_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="empty history"):
        build_channels(_frame(n=10).iloc[0:0])
    with pytest.raises(ValueError, match="missing required columns"):
        build_channels(_frame(n=10).drop(columns=["raan"]))
