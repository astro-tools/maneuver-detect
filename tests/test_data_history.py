"""Tests for ``maneuver_detect.data.history`` — mean-element series assembly."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from maneuver_detect.data.elset import Elset
from maneuver_detect.data.history import MEAN_ELEMENT_COLUMNS, assemble, build_series

_T0 = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)


def _elset(epoch: datetime = _T0, **overrides: object) -> Elset:
    fields: dict[str, object] = {
        "norad_id": 25544,
        "epoch": epoch,
        "mean_motion": 15.4916,
        "eccentricity": 0.0005881,
        "inclination": 51.6361,
        "raan": 333.6061,
        "arg_perigee": 172.368,
        "mean_anomaly": 187.7399,
        "bstar": 1.045167e-4,
        "mean_motion_dot": 0.0001,
        "mean_motion_ddot": 0.0,
        "element_set_no": 100,
        "rev_at_epoch": 57070,
        "classification": "U",
        "object_id": "1998-067A",
    }
    fields.update(overrides)
    return Elset(**fields)  # type: ignore[arg-type]


def test_schema_columns_and_dtypes() -> None:
    df = assemble([_elset()])
    assert tuple(df.columns) == MEAN_ELEMENT_COLUMNS
    assert str(df["epoch"].dtype) == "datetime64[ns, UTC]"
    assert df["norad_id"].dtype == "int64"
    for col in (
        "mean_motion",
        "semi_major_axis",
        "eccentricity",
        "inclination",
        "raan",
        "arg_perigee",
        "mean_anomaly",
        "bstar",
        "dt_days",
    ):
        assert df[col].dtype == "float64"


def test_elements_carried_through_verbatim() -> None:
    row = assemble([_elset()]).iloc[0]
    assert row["mean_motion"] == pytest.approx(15.4916)
    assert row["eccentricity"] == pytest.approx(0.0005881)
    assert row["inclination"] == pytest.approx(51.6361)
    assert row["raan"] == pytest.approx(333.6061)
    assert row["arg_perigee"] == pytest.approx(172.368)
    assert row["mean_anomaly"] == pytest.approx(187.7399)
    assert int(row["norad_id"]) == 25544


def test_semi_major_axis_from_mean_motion() -> None:
    # ISS mean motion ~15.49 rev/day -> a ~6795 km.
    a = float(assemble([_elset()]).iloc[0]["semi_major_axis"])
    assert a == pytest.approx(6795.0, abs=5.0)


def test_dt_days_first_is_nan_then_gaps() -> None:
    elsets = [
        _elset(_T0),
        _elset(_T0 + timedelta(days=1)),
        _elset(_T0 + timedelta(days=3, hours=12)),
    ]
    dt = assemble(elsets)["dt_days"].tolist()
    assert math.isnan(dt[0])
    assert dt[1] == pytest.approx(1.0)
    assert dt[2] == pytest.approx(2.5)


def test_assemble_sorts_by_epoch() -> None:
    e0, e1, e2 = (_elset(_T0 + timedelta(days=d)) for d in (0, 1, 2))
    df = assemble([e2, e0, e1])
    assert df["epoch"].is_monotonic_increasing
    assert df["epoch"].iloc[0] == pd.Timestamp(e0.epoch)


def test_empty_input_is_empty_frame_with_schema() -> None:
    df = assemble([])
    assert len(df) == 0
    assert tuple(df.columns) == MEAN_ELEMENT_COLUMNS
    assert str(df["epoch"].dtype) == "datetime64[ns, UTC]"


def test_build_series_cleans_then_assembles() -> None:
    # A multi-elset history with an injected bad elset and an exact duplicate; build_series should
    # drop the bad one, collapse the duplicate, and assemble a clean monotonic series.
    good = [_elset(_T0 + timedelta(days=d)) for d in range(5)]
    duplicate = _elset(_T0 + timedelta(days=2), element_set_no=500)  # exact dup of good[2]
    bad = _elset(_T0 + timedelta(days=10), eccentricity=1.4)  # non-physical
    df = build_series([*good, duplicate, bad])
    assert len(df) == 5  # 5 good epochs; duplicate collapsed, bad dropped
    assert df["epoch"].is_monotonic_increasing
    assert df["epoch"].is_unique


def test_build_series_multi_year_synthetic() -> None:
    # The DoD's "clean multi-year series": ~2 years of weekly elsets assembles cleanly.
    elsets = [_elset(_T0 + timedelta(weeks=w), mean_anomaly=(w * 37) % 360) for w in range(104)]
    df = build_series(elsets)
    assert len(df) == 104
    assert df["epoch"].is_monotonic_increasing
    assert df["epoch"].is_unique
    assert tuple(df.columns) == MEAN_ELEMENT_COLUMNS
    # The series spans ~2 years.
    span_days = (df["epoch"].iloc[-1] - df["epoch"].iloc[0]).days
    assert span_days > 700
