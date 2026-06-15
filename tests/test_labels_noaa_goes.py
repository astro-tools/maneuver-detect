"""Tests for ``maneuver_detect.labels.noaa_goes`` — the NOAA GOES navsum maneuver parser.

The inline fixture reproduces real ``navsum.txt`` structure (``=``-delimited per-spacecraft blocks,
a Comments footer naming the last maneuver) and uses CRLF endings, as the live file is served.
"""

from __future__ import annotations

from datetime import datetime, timezone

from maneuver_detect.labels.noaa_goes import parse_navsum
from maneuver_detect.labels.record import SOURCE_NOAA_GOES, OrbitClass

_GOES_NAME_TO_NORAD = {"GOES-16": 41866, "GOES-17": 43226, "GOES-18": 51850}

# Two spacecraft blocks and one with no last-maneuver footer; CRLF endings (as served).
_NAVSUM = (
    "Below are the weekly GOES element sets:\r\n"
    "=============================================================\r\n"
    "Spacecraft :                                  GOES-16\r\n"
    "Epoch (yy/ddd hh:mm:ss.sss GMT) :             26/160 05:00:00.000\r\n"
    "Comments:\r\n"
    "Fuel and oxidizer remaining are estimates after the last maneuver on 26/159.\r\n"
    "=============================================================\r\n"
    "Spacecraft :                                  GOES-17\r\n"
    "Epoch (yy/ddd hh:mm:ss.sss GMT) :             26/160 12:00:00.000\r\n"
    "Comments:\r\n"
    "Fuel and oxidizer remaining are estimates after the last maneuver on 25/001.\r\n"
    "=============================================================\r\n"
    "Spacecraft :                                  GOES-18\r\n"
    "Epoch (yy/ddd hh:mm:ss.sss GMT) :             26/160 18:00:00.000\r\n"
    "Comments:\r\n"
    "GOES-18 data propagated from orbit epoch 26/160.\r\n"  # no last-maneuver footer -> skipped
)


def test_parses_one_label_per_spacecraft_with_a_maneuver_footer() -> None:
    labels = parse_navsum(_NAVSUM, goes_name_to_norad=_GOES_NAME_TO_NORAD)
    assert len(labels) == 2  # GOES-16 and GOES-17; GOES-18 has no last-maneuver footer
    assert {label.norad_id for label in labels} == {41866, 43226}


def test_label_fields_are_normalised() -> None:
    goes16 = next(
        label
        for label in parse_navsum(_NAVSUM, goes_name_to_norad=_GOES_NAME_TO_NORAD)
        if label.norad_id == 41866
    )
    # 26/159 -> 2026 day-of-year 159 -> 2026-06-08; epoch is noon, window is the whole UTC day.
    day = datetime(2026, 6, 8, tzinfo=timezone.utc)
    assert goes16.window_start == day
    assert goes16.epoch == datetime(2026, 6, 8, 12, tzinfo=timezone.utc)
    assert goes16.window_end == datetime(2026, 6, 9, tzinfo=timezone.utc)
    assert goes16.orbit_class is OrbitClass.GEO
    assert goes16.source == SOURCE_NOAA_GOES
    assert goes16.delta_v is None and goes16.maneuver_type is None  # epoch-only


def test_unmapped_spacecraft_yields_none_norad() -> None:
    labels = parse_navsum(_NAVSUM, goes_name_to_norad={})
    # The maneuver is still ingested when the name is not in the crosswalk; norad_id is left None.
    assert labels and all(label.norad_id is None for label in labels)


def test_no_blocks_yields_no_labels() -> None:
    assert parse_navsum("nothing here\n", goes_name_to_norad=_GOES_NAME_TO_NORAD) == []
