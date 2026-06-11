"""Tests for ``maneuver_detect.labels.labeller`` — epoch→gap mapping and coverage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from maneuver_detect.data.elset import Elset
from maneuver_detect.data.history import assemble
from maneuver_detect.labels.labeller import (
    INTERVAL_COLUMNS,
    intervals_to_frame,
    label_coverage,
    label_series,
)
from maneuver_detect.labels.record import (
    SOURCE_DORIS_IDS,
    SOURCE_GPS_NANU,
    ManeuverLabel,
    OrbitClass,
)
from maneuver_detect.schema import ManeuverType

_UTC = timezone.utc
_T0 = datetime(2024, 1, 1, 12, tzinfo=_UTC)


def _elset(epoch: datetime) -> Elset:
    return Elset(
        norad_id=25544,
        epoch=epoch,
        mean_motion=15.4916,
        eccentricity=0.0005881,
        inclination=51.6361,
        raan=333.6061,
        arg_perigee=172.368,
        mean_anomaly=187.7399,
        bstar=1.045167e-4,
        mean_motion_dot=0.0001,
        mean_motion_ddot=0.0,
        element_set_no=100,
        rev_at_epoch=57070,
        classification="U",
        object_id="1998-067A",
    )


def _series() -> pd.DataFrame:
    # Six daily elsets at 12:00 UTC → five inter-elset gaps.
    return assemble([_elset(_T0 + timedelta(days=d)) for d in range(6)])


def _label(epoch: datetime, **overrides: object) -> ManeuverLabel:
    fields: dict[str, object] = {
        "norad_id": 25544,
        "epoch": epoch,
        "window_start": epoch,
        "window_end": epoch,
        "source": SOURCE_DORIS_IDS,
        "source_ref": "ref",
        "orbit_class": OrbitClass.LEO,
        "maneuver_type": ManeuverType.IN_TRACK,
        "delta_v": 1.0,
    }
    fields.update(overrides)
    return ManeuverLabel(**fields)  # type: ignore[arg-type]


def test_maps_epoch_onto_bracketing_gap() -> None:
    # A maneuver mid-gap between the day-3 and day-4 elsets.
    label = _label(datetime(2024, 1, 3, 18, tzinfo=_UTC))
    result = label_series(_series(), [label])
    assert len(result.intervals) == 1
    assert result.unmatched == []
    interval = result.intervals[0]
    assert interval.elset_epoch_before == pd.Timestamp(2024, 1, 3, 12, tz="UTC")
    assert interval.elset_epoch_after == pd.Timestamp(2024, 1, 4, 12, tz="UTC")


def test_tolerance_spans_one_adjacent_gap_each_side() -> None:
    label = _label(datetime(2024, 1, 3, 18, tzinfo=_UTC))
    interval = label_series(_series(), [label]).intervals[0]
    # Bracketing gap is [day3, day4); ±1 adjacent gap → [day2, day5].
    assert interval.tol_start == pd.Timestamp(2024, 1, 2, 12, tz="UTC")
    assert interval.tol_end == pd.Timestamp(2024, 1, 5, 12, tz="UTC")


def test_maneuver_on_elset_epoch_matches_following_gap() -> None:
    label = _label(datetime(2024, 1, 2, 12, tzinfo=_UTC))  # exactly the day-2 elset epoch
    interval = label_series(_series(), [label]).intervals[0]
    assert interval.elset_epoch_before == pd.Timestamp(2024, 1, 2, 12, tz="UTC")
    assert interval.elset_epoch_after == pd.Timestamp(2024, 1, 3, 12, tz="UTC")


def test_maneuvers_outside_span_are_unmatched() -> None:
    before = _label(datetime(2023, 12, 31, tzinfo=_UTC))
    after = _label(datetime(2024, 1, 10, tzinfo=_UTC))
    at_last = _label(datetime(2024, 1, 6, 12, tzinfo=_UTC))  # the last elset epoch — no gap after
    result = label_series(_series(), [before, after, at_last])
    assert result.intervals == []
    assert len(result.unmatched) == 3


def test_intervals_to_frame_schema() -> None:
    label = _label(datetime(2024, 1, 3, 18, tzinfo=_UTC))
    df = intervals_to_frame(label_series(_series(), [label]).intervals)
    assert tuple(df.columns) == INTERVAL_COLUMNS
    assert df["norad_id"].dtype == "Int64"
    assert str(df["elset_epoch_before"].dtype) == "datetime64[ns, UTC]"
    assert str(df["tol_end"].dtype) == "datetime64[ns, UTC]"


def test_coverage_matches_class_scope() -> None:
    labels = [
        _label(_T0, source=SOURCE_DORIS_IDS, delta_v=2.5, norad_id=26997),  # LEO, Δv-labelled
        _label(  # LEO epoch-only (TOPEX-style)
            _T0, source=SOURCE_DORIS_IDS, delta_v=None, maneuver_type=None, norad_id=22076
        ),
        _label(  # MEO epoch-only (NANU)
            _T0,
            source=SOURCE_GPS_NANU,
            orbit_class=OrbitClass.MEO,
            delta_v=None,
            maneuver_type=None,
            norad_id=None,
        ),
    ]
    report = label_coverage(labels)
    assert report.total == 3

    leo = report.per_class[OrbitClass.LEO]
    assert leo.n_events == 2
    assert leo.n_with_delta_v == 1  # only the Δv-labelled LEO event
    assert leo.n_with_norad == 2
    assert leo.sources == (SOURCE_DORIS_IDS,)

    meo = report.per_class[OrbitClass.MEO]
    assert meo.n_events == 1
    assert meo.n_with_delta_v == 0  # MEO is epoch-only
    assert meo.sources == (SOURCE_GPS_NANU,)

    assert report.per_class[OrbitClass.GEO].n_events == 0  # no shippable GEO source in v0.1
    assert "LEO" in report.summary()
