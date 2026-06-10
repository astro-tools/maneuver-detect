"""Tests for the canonical maneuver schema — the frozen library contract."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from maneuver_detect.schema import (
    COLUMNS,
    Maneuver,
    ManeuverType,
    empty_frame,
    from_frame,
    to_frame,
    validate_frame,
)


def _sample_maneuvers() -> list[Maneuver]:
    return [
        Maneuver(
            epoch=pd.Timestamp("2024-03-01T12:00:00", tz="UTC"),
            confidence=0.9,
            type=ManeuverType.IN_TRACK,
            delta_v_estimate=0.42,
            norad_id=25544,
            elset_epoch_before=pd.Timestamp("2024-03-01T00:00:00", tz="UTC"),
            elset_epoch_after=pd.Timestamp("2024-03-02T00:00:00", tz="UTC"),
        ),
        Maneuver(
            epoch=pd.Timestamp("2024-05-10T06:30:00", tz="UTC"),
            confidence=0.5,
            type=ManeuverType.CROSS_TRACK,
            delta_v_estimate=None,  # not reported
            norad_id=40000,
            elset_epoch_before=pd.Timestamp("2024-05-09T18:00:00", tz="UTC"),
            elset_epoch_after=pd.Timestamp("2024-05-10T18:00:00", tz="UTC"),
        ),
        Maneuver(
            epoch=pd.Timestamp("2024-07-20T00:00:00", tz="UTC"),
            confidence=0.1,
            type=ManeuverType.RADIAL,
            delta_v_estimate=1.5,
            norad_id=40000,
            elset_epoch_before=pd.Timestamp("2024-07-19T12:00:00", tz="UTC"),
            elset_epoch_after=pd.Timestamp("2024-07-20T12:00:00", tz="UTC"),
        ),
    ]


def test_to_frame_has_canonical_columns_in_order() -> None:
    frame = to_frame(_sample_maneuvers())
    assert tuple(frame.columns) == COLUMNS


def test_to_frame_dtypes() -> None:
    frame = to_frame(_sample_maneuvers())
    assert str(frame["epoch"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["elset_epoch_before"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["elset_epoch_after"].dtype) == "datetime64[ns, UTC]"
    assert frame["confidence"].dtype == "float64"
    assert frame["delta_v_estimate"].dtype == "float64"
    assert frame["norad_id"].dtype == "int64"
    assert str(frame["type"].dtype) == "string"


def test_schema_roundtrips_without_drift() -> None:
    maneuvers = _sample_maneuvers()
    assert from_frame(to_frame(maneuvers)) == maneuvers


def test_delta_v_none_roundtrips_as_na() -> None:
    frame = to_frame(_sample_maneuvers())
    assert math.isnan(frame["delta_v_estimate"].iloc[1])
    assert from_frame(frame)[1].delta_v_estimate is None


def test_empty_frame_has_schema_and_roundtrips() -> None:
    frame = empty_frame()
    assert tuple(frame.columns) == COLUMNS
    assert len(frame) == 0
    assert from_frame(frame) == []
    assert tuple(to_frame([]).columns) == COLUMNS


def test_validate_frame_rejects_missing_columns() -> None:
    frame = to_frame(_sample_maneuvers()).drop(columns=["confidence"])
    with pytest.raises(ValueError, match="missing canonical columns"):
        validate_frame(frame)


def test_maneuver_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Maneuver(
            epoch=pd.Timestamp("2024-03-01T12:00:00", tz="UTC"),
            confidence=1.5,
            type=ManeuverType.IN_TRACK,
            delta_v_estimate=None,
            norad_id=1,
            elset_epoch_before=pd.Timestamp("2024-03-01T00:00:00", tz="UTC"),
            elset_epoch_after=pd.Timestamp("2024-03-02T00:00:00", tz="UTC"),
        )


def test_maneuver_rejects_negative_delta_v() -> None:
    with pytest.raises(ValueError, match="delta_v_estimate"):
        Maneuver(
            epoch=pd.Timestamp("2024-03-01T12:00:00", tz="UTC"),
            confidence=0.5,
            type=ManeuverType.IN_TRACK,
            delta_v_estimate=-0.1,
            norad_id=1,
            elset_epoch_before=pd.Timestamp("2024-03-01T00:00:00", tz="UTC"),
            elset_epoch_after=pd.Timestamp("2024-03-02T00:00:00", tz="UTC"),
        )


def test_maneuver_rejects_naive_epoch() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Maneuver(
            epoch=pd.Timestamp("2024-03-01T12:00:00"),  # naive
            confidence=0.5,
            type=ManeuverType.IN_TRACK,
            delta_v_estimate=None,
            norad_id=1,
            elset_epoch_before=pd.Timestamp("2024-03-01T00:00:00", tz="UTC"),
            elset_epoch_after=pd.Timestamp("2024-03-02T00:00:00", tz="UTC"),
        )
