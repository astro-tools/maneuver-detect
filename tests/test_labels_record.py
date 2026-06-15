"""Tests for ``maneuver_detect.labels.record`` — the common ManeuverLabel record."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from maneuver_detect.labels.record import (
    COLUMNS,
    SOURCE_DORIS_IDS,
    SOURCE_GALILEO_NAGU,
    SOURCE_GPS_NANU,
    SOURCE_NOAA_GOES,
    SOURCE_QZSS_OHI,
    SOURCE_SELF_GEO,
    SOURCE_SELF_HEO,
    ManeuverLabel,
    OrbitClass,
    to_frame,
)
from maneuver_detect.schema import ManeuverType

_UTC = timezone.utc
_START = datetime(2024, 1, 1, 12, 0, tzinfo=_UTC)
_END = datetime(2024, 1, 1, 12, 30, tzinfo=_UTC)


def _label(**overrides: object) -> ManeuverLabel:
    fields: dict[str, object] = {
        "norad_id": 26997,
        "epoch": _START,
        "window_start": _START,
        "window_end": _END,
        "source": SOURCE_DORIS_IDS,
        "source_ref": "JASO1 2024/010",
        "orbit_class": OrbitClass.LEO,
        "maneuver_type": ManeuverType.IN_TRACK,
        "delta_v": 1.5,
    }
    fields.update(overrides)
    return ManeuverLabel(**fields)  # type: ignore[arg-type]


def test_orbit_class_canonical_order() -> None:
    # The iteration order is the frozen per-class report shape (coverage / splits / scorer).
    assert [oc.value for oc in OrbitClass] == ["LEO", "MEO", "GEO", "IGSO", "HEO"]


def test_source_tags_are_distinct() -> None:
    sources = {
        SOURCE_DORIS_IDS,
        SOURCE_GPS_NANU,
        SOURCE_GALILEO_NAGU,
        SOURCE_QZSS_OHI,
        SOURCE_NOAA_GOES,
        SOURCE_SELF_GEO,
        SOURCE_SELF_HEO,
    }
    assert len(sources) == 7  # every source tag is unique


def test_igso_and_heo_labels_construct() -> None:
    assert (
        _label(orbit_class=OrbitClass.IGSO, source=SOURCE_QZSS_OHI).orbit_class is OrbitClass.IGSO
    )
    assert _label(orbit_class=OrbitClass.HEO, source=SOURCE_SELF_HEO).orbit_class is OrbitClass.HEO


def test_construct_and_defaults() -> None:
    label = ManeuverLabel(
        norad_id=None,
        epoch=_START,
        window_start=_START,
        window_end=_END,
        source=SOURCE_DORIS_IDS,
        source_ref="ref",
        orbit_class=OrbitClass.MEO,
    )
    assert label.maneuver_type is None
    assert label.delta_v is None
    assert label.norad_id is None


def test_naive_epoch_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _label(epoch=datetime(2024, 1, 1, 12, 0))


def test_window_order_validated() -> None:
    with pytest.raises(ValueError, match="after"):
        _label(window_start=_END, window_end=_START)


def test_negative_delta_v_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _label(delta_v=-0.1)


def test_to_frame_columns_and_dtypes() -> None:
    df = to_frame([_label()])
    assert tuple(df.columns) == COLUMNS
    assert df["norad_id"].dtype == "Int64"
    assert str(df["epoch"].dtype) == "datetime64[ns, UTC]"
    assert str(df["window_start"].dtype) == "datetime64[ns, UTC]"
    assert df["delta_v"].dtype == "float64"
    assert df["maneuver_type"].dtype == "string"
    assert df["orbit_class"].dtype == "string"


def test_to_frame_carries_nullable_fields() -> None:
    df = to_frame([_label(norad_id=None, maneuver_type=None, delta_v=None)])
    assert pd.isna(df["norad_id"].iloc[0])
    assert pd.isna(df["maneuver_type"].iloc[0])
    assert pd.isna(df["delta_v"].iloc[0])


def test_to_frame_empty_keeps_schema() -> None:
    df = to_frame([])
    assert len(df) == 0
    assert tuple(df.columns) == COLUMNS
    assert str(df["epoch"].dtype) == "datetime64[ns, UTC]"
