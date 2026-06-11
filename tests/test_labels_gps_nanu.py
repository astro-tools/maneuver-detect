"""Tests for ``maneuver_detect.labels.gps_nanu`` — the FCSTDV NANU parser.

NANUs are US-Government public domain; the fixtures are illustrative notices in the public NANU
field format (the same form the V2 ingest spike used).
"""

from __future__ import annotations

from datetime import timezone

import pytest

from maneuver_detect.labels.gps_nanu import parse_nanus
from maneuver_detect.labels.record import SOURCE_GPS_NANU, OrbitClass

_FCSTDV_SVN62 = """\
NANU TYPE: FCSTDV
NANU NUMBER: 2025087
NANU DTG: 211200Z MAR 2025
SVN: 62
PRN: 25
START JDAY: 086
START TIME ZULU: 1300
STOP JDAY: 086
STOP TIME ZULU: 1900
"""

_FCSTDV_SVN74 = """\
NANU TYPE: FCSTDV
NANU NUMBER: 2025104
NANU DTG: 080900Z APR 2025
SVN: 74
PRN: 04
START JDAY: 100
START TIME ZULU: 0600
STOP JDAY: 100
STOP TIME ZULU: 1200
"""

# A non-FCSTDV notice (a usable-summary notice) — must be ignored, it is not a maneuver.
_FCSTSUMM = """\
NANU TYPE: FCSTSUMM
NANU NUMBER: 2025088
NANU DTG: 281200Z MAR 2025
SVN: 62
PRN: 25
"""

_FIXTURE = _FCSTDV_SVN62 + _FCSTSUMM + _FCSTDV_SVN74


def test_parses_only_fcstdv_notices() -> None:
    labels = parse_nanus(_FIXTURE)
    assert len(labels) == 2  # the FCSTSUMM notice is skipped


def test_known_svn_resolves_unknown_is_none() -> None:
    labels = parse_nanus(_FIXTURE)
    by_svn = {label.source_ref: label for label in labels}
    svn62 = next(label for label in labels if "SVN62" in label.source_ref)
    svn74 = next(label for label in labels if "SVN74" in label.source_ref)
    assert svn62.norad_id == 36585  # seed crosswalk
    assert svn74.norad_id is None  # not in the seed crosswalk
    assert len(by_svn) == 2


def test_labels_are_meo_epoch_only() -> None:
    label = parse_nanus(_FCSTDV_SVN62)[0]
    assert label.orbit_class is OrbitClass.MEO
    assert label.delta_v is None
    assert label.maneuver_type is None
    assert label.source == SOURCE_GPS_NANU
    assert label.window_start.tzinfo is timezone.utc


def test_epoch_is_window_midpoint() -> None:
    label = parse_nanus(_FCSTDV_SVN62)[0]
    # 13:00 to 19:00 Zulu on the same day → midpoint 16:00 UTC.
    assert label.epoch.hour == 16
    assert label.epoch.minute == 0


def test_injected_crosswalk_overrides_seed() -> None:
    labels = parse_nanus(_FCSTDV_SVN74, svn_to_norad={"SVN74": 99999})
    assert labels[0].norad_id == 99999


# The real archived NANU format: fields are indented and the type carries a list-number prefix,
# under a NOTICE/SUBJ header.
_ARCHIVE_FCSTDV = """\
NOTICE ADVISORY TO NAVSTAR USERS (NANU) 2024002
SUBJ: SVN65 (PRN24) FORECAST OUTAGE JDAY 011/1130 - JDAY 011/2330
1.     NANU TYPE: FCSTDV
       NANU NUMBER: 2024002
       NANU DTG: 071844Z JAN 2024
       SVN: 65
       PRN: 24
       START JDAY: 011
       START TIME ZULU: 1130
       STOP JDAY: 011
       STOP TIME ZULU: 2330

2.  CONDITION: GPS SATELLITE SVN65 (PRN24) WILL BE UNUSABLE ON JDAY 011.
"""


def test_parses_indented_numbered_archive_format() -> None:
    labels = parse_nanus(_ARCHIVE_FCSTDV, svn_to_norad={"SVN65": 38833})
    assert len(labels) == 1
    assert labels[0].norad_id == 38833
    assert labels[0].orbit_class is OrbitClass.MEO
    assert labels[0].window_start.hour == 11
    assert labels[0].window_end.hour == 23


def test_malformed_fcstdv_raises() -> None:
    missing_stop = """\
NANU TYPE: FCSTDV
NANU NUMBER: 2025200
NANU DTG: 010900Z MAY 2025
SVN: 62
START JDAY: 120
START TIME ZULU: 0600
"""
    with pytest.raises(ValueError, match="STOP JDAY"):
        parse_nanus(missing_stop)


# A late-December notice announcing an early-January burn: the day-of-year belongs to the *next*
# calendar year, not the DTG's year. Applying the DTG year to the JDAY would mis-date it by a year
# and drop the label into the wrong (or no) elset gap.
_FCSTDV_YEAR_ROLLOVER = """\
NANU TYPE: FCSTDV
NANU NUMBER: 2024366
NANU DTG: 281200Z DEC 2024
SVN: 62
PRN: 25
START JDAY: 003
START TIME ZULU: 0600
STOP JDAY: 003
STOP TIME ZULU: 1200
"""


def test_dtg_year_rollover_dates_window_in_next_year() -> None:
    label = parse_nanus(_FCSTDV_YEAR_ROLLOVER)[0]
    # JDAY 003 announced on 28 Dec 2024 → 3 Jan 2025, not 3 Jan 2024.
    assert (label.window_start.year, label.window_start.month, label.window_start.day) == (
        2025,
        1,
        3,
    )
    assert label.window_end.year == 2025


# A window that straddles New Year's: START on the last day of a (leap) year, STOP in the next.
# 2024 is a leap year, so JDAY 366 is 31 Dec 2024 and JDAY 001 is 1 Jan 2025.
_FCSTDV_WINDOW_WRAPS_NEW_YEAR = """\
NANU TYPE: FCSTDV
NANU NUMBER: 2024365
NANU DTG: 301200Z DEC 2024
SVN: 62
PRN: 25
START JDAY: 366
START TIME ZULU: 2200
STOP JDAY: 001
STOP TIME ZULU: 0200
"""


def test_window_wrapping_new_year_resolves_stop_in_next_year() -> None:
    label = parse_nanus(_FCSTDV_WINDOW_WRAPS_NEW_YEAR)[0]
    assert (label.window_start.year, label.window_start.month, label.window_start.day) == (
        2024,
        12,
        31,
    )
    assert (label.window_end.year, label.window_end.month, label.window_end.day) == (2025, 1, 1)
    assert label.window_end > label.window_start


def test_malformed_dtg_raises() -> None:
    bad_dtg = """\
NANU TYPE: FCSTDV
NANU NUMBER: 2025201
NANU DTG: MARCH 2025
SVN: 62
START JDAY: 120
START TIME ZULU: 0600
STOP JDAY: 120
STOP TIME ZULU: 1200
"""
    with pytest.raises(ValueError, match="DTG"):
        parse_nanus(bad_dtg)
