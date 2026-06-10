"""Tests for ``maneuver_detect.data.elset`` — OMM parsing, the orbit-formats seam."""

from __future__ import annotations

from datetime import timezone
from typing import Any

import pytest

from maneuver_detect.data.elset import Elset, from_omm

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
