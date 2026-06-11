"""Tests for ``maneuver_detect.data.elset`` — OMM parsing, the orbit-formats seam."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Any

import pytest

from maneuver_detect.data.elset import Elset, from_omm, from_tle, read_tle_file

_SAMPLE_OMM: dict[str, Any] = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2024-01-01T12:00:00.000000",
    "MEAN_MOTION": 15.5,
    "ECCENTRICITY": 0.0001,
    "INCLINATION": 51.64,
    "RA_OF_ASC_NODE": 90.0,
    "ARG_OF_PERICENTER": 80.0,
    "MEAN_ANOMALY": 270.0,
    "NORAD_CAT_ID": 25544,
    "BSTAR": 0.00018,
    "MEAN_MOTION_DOT": 0.0001,
    "MEAN_MOTION_DDOT": 0.0,
    "CLASSIFICATION_TYPE": "U",
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 42,
}


def test_from_omm_parses_all_fields() -> None:
    elset = from_omm(_SAMPLE_OMM)
    assert elset.norad_id == 25544
    assert elset.epoch.year == 2024 and elset.epoch.hour == 12
    assert elset.epoch.tzinfo is not None
    assert elset.mean_motion == pytest.approx(15.5)
    assert elset.eccentricity == pytest.approx(0.0001)
    assert elset.inclination == pytest.approx(51.64)
    assert elset.raan == pytest.approx(90.0)
    assert elset.arg_perigee == pytest.approx(80.0)
    assert elset.mean_anomaly == pytest.approx(270.0)
    assert elset.bstar == pytest.approx(0.00018)
    assert elset.element_set_no == 999
    assert elset.rev_at_epoch == 42
    assert elset.classification == "U"
    assert elset.object_id == "1998-067A"


def test_epoch_is_normalised_to_utc() -> None:
    assert from_omm(_SAMPLE_OMM).epoch.utcoffset() == timezone.utc.utcoffset(None)


@pytest.mark.parametrize(
    "epoch_text",
    [
        "2024-01-01T12:00:00.000000",  # CelesTrak ISO, T separator
        "2024-01-01 12:00:00",  # Space-Track, space separator
        "2024-01-01T12:00:00Z",  # trailing Z
        "2024-01-01 12:00:00.500000Z",  # space + fractional + Z
    ],
)
def test_epoch_separator_and_zulu_variants(epoch_text: str) -> None:
    record = {**_SAMPLE_OMM, "EPOCH": epoch_text}
    elset = from_omm(record)
    assert elset.epoch.tzinfo is not None
    assert elset.epoch.year == 2024 and elset.epoch.month == 1 and elset.epoch.hour == 12


def test_metadata_fields_default_when_absent() -> None:
    thin = {
        k: v
        for k, v in _SAMPLE_OMM.items()
        if k
        not in {
            "OBJECT_ID",
            "CLASSIFICATION_TYPE",
            "ELEMENT_SET_NO",
            "REV_AT_EPOCH",
        }
    }
    elset = from_omm(thin)
    assert elset.object_id == ""
    assert elset.classification == "U"
    assert elset.element_set_no == 0
    assert elset.rev_at_epoch == 0


def test_numeric_strings_are_accepted() -> None:
    # Space-Track sometimes serialises numbers as strings; the parser coerces.
    record = {**_SAMPLE_OMM, "MEAN_MOTION": "15.5", "NORAD_CAT_ID": "25544", "ELEMENT_SET_NO": "7"}
    elset = from_omm(record)
    assert elset.mean_motion == pytest.approx(15.5)
    assert elset.norad_id == 25544
    assert elset.element_set_no == 7


@pytest.mark.parametrize(
    "missing",
    ["NORAD_CAT_ID", "EPOCH", "MEAN_MOTION", "ECCENTRICITY", "BSTAR", "MEAN_MOTION_DOT"],
)
def test_missing_required_field_raises(missing: str) -> None:
    record = {k: v for k, v in _SAMPLE_OMM.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        from_omm(record)


def test_non_numeric_required_field_raises() -> None:
    with pytest.raises(ValueError, match="MEAN_MOTION"):
        from_omm({**_SAMPLE_OMM, "MEAN_MOTION": "not-a-number"})


def test_unparseable_epoch_raises() -> None:
    with pytest.raises(ValueError, match="EPOCH"):
        from_omm({**_SAMPLE_OMM, "EPOCH": "last tuesday"})


def test_non_string_epoch_raises() -> None:
    with pytest.raises(ValueError, match="EPOCH"):
        from_omm({**_SAMPLE_OMM, "EPOCH": 20240101})


def test_non_integer_metadata_field_raises() -> None:
    with pytest.raises(ValueError, match="ELEMENT_SET_NO"):
        from_omm({**_SAMPLE_OMM, "ELEMENT_SET_NO": "not-an-int"})


def test_naive_epoch_construction_is_rejected() -> None:
    from datetime import datetime

    with pytest.raises(ValueError, match="timezone-aware"):
        Elset(
            norad_id=1,
            epoch=datetime(2024, 1, 1, 12),  # naive
            mean_motion=15.0,
            eccentricity=0.0,
            inclination=0.0,
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            bstar=0.0,
            mean_motion_dot=0.0,
            mean_motion_ddot=0.0,
            element_set_no=0,
            rev_at_epoch=0,
            classification="U",
            object_id="",
        )


# --- TLE-file parsing (the local-file counterpart to the OMM fetchers) ---------------------

_ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005"
_ISS_LINE2 = "2 25544  51.6400 208.0000 0006703 130.0000 325.0000 15.50000000123456"


def test_from_tle_parses_elements() -> None:
    elset = from_tle(_ISS_LINE1, _ISS_LINE2)
    assert elset.norad_id == 25544
    # Epoch day-of-year 1.5 -> Jan 1 12:00 UTC, timezone-aware.
    assert elset.epoch.year == 2024 and elset.epoch.month == 1 and elset.epoch.day == 1
    assert elset.epoch.hour == 12 and elset.epoch.tzinfo is not None
    assert elset.mean_motion == pytest.approx(15.5)
    assert elset.inclination == pytest.approx(51.64)
    assert elset.raan == pytest.approx(208.0)
    assert elset.arg_perigee == pytest.approx(130.0)
    assert elset.mean_anomaly == pytest.approx(325.0)
    assert elset.eccentricity == pytest.approx(0.0006703)
    assert elset.bstar == pytest.approx(0.00010270)
    assert elset.element_set_no == 900
    assert elset.rev_at_epoch == 12345
    assert elset.classification == "U"
    assert elset.object_id == "1998-067A"


def test_from_tle_two_digit_year_window() -> None:
    # A two-digit year >= 57 maps to the 1900s (the standard TLE convention).
    line1 = "1 00001U 57001A   57001.00000000  .00000000  00000-0  00000-0 0  9990"
    line2 = "2 00001  51.6400 208.0000 0006703 130.0000 325.0000 15.50000000123456"
    assert from_tle(line1, line2).epoch.year == 1957


@pytest.mark.parametrize(
    ("line1", "line2", "match"),
    [
        ("1 25544U ...", _ISS_LINE2, "malformed TLE line 1"),
        (_ISS_LINE1, "2 25544 too short", "malformed TLE line 2"),
        (_ISS_LINE2, _ISS_LINE1, "malformed TLE line 1"),  # swapped line order
        (
            _ISS_LINE1,
            "2 99999  51.6400 208.0000 0006703 130.0000 325.0000 15.50000000123456",
            "catalogue-number mismatch",
        ),
        (
            _ISS_LINE1,
            "2 25544  51.6400 208.0000 0006703 130.0000 325.0000 00.00000000123456",
            "non-physical mean motion",
        ),
    ],
)
def test_from_tle_malformed_raises(line1: str, line2: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        from_tle(line1, line2)


def test_read_tle_file_two_line(tmp_path: Path) -> None:
    path = tmp_path / "history.tle"
    path.write_text(f"{_ISS_LINE1}\n{_ISS_LINE2}\n")
    elsets = read_tle_file(path)
    assert len(elsets) == 1 and elsets[0].norad_id == 25544


def test_read_tle_file_three_line_skips_name(tmp_path: Path) -> None:
    # A leading name line (3LE) and blank lines are ignored; two pairs parse in file order.
    line1b = "1 25544U 98067A   24002.50000000  .00016717  00000-0  10270-3 0  9013"
    line2b = "2 25544  51.6400 207.0000 0006703 131.0000 326.0000 15.50000000123463"
    path = tmp_path / "history.tle"
    path.write_text(f"ISS (ZARYA)\n{_ISS_LINE1}\n{_ISS_LINE2}\n\nISS (ZARYA)\n{line1b}\n{line2b}\n")
    elsets = read_tle_file(path)
    assert [e.epoch.day for e in elsets] == [1, 2]


def test_read_tle_file_empty_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.tle"
    path.write_text("\n\n")
    assert read_tle_file(path) == []


def test_read_tle_file_line1_without_line2_raises(tmp_path: Path) -> None:
    path = tmp_path / "dangling.tle"
    path.write_text(f"{_ISS_LINE1}\n")
    with pytest.raises(ValueError, match="not followed by a line 2"):
        read_tle_file(path)


def test_read_tle_file_orphan_line2_raises(tmp_path: Path) -> None:
    path = tmp_path / "orphan.tle"
    path.write_text(f"{_ISS_LINE2}\n")
    with pytest.raises(ValueError, match="without a preceding line 1"):
        read_tle_file(path)
