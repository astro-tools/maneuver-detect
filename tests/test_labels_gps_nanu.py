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
