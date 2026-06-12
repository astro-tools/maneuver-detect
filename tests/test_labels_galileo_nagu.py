"""Tests for ``maneuver_detect.labels.galileo_nagu`` — the PLN_MANV NAGU parser.

NAGU content is reproduced from the GSC under its reuse terms (© EU); the fixtures are in the public
NAGU ``.txt`` field format (the form the V2 follow-up ingest spike used).
"""

from __future__ import annotations

from datetime import datetime, timezone

from maneuver_detect.labels.galileo_nagu import GALILEO_GSAT_TO_NORAD, parse_nagus
from maneuver_detect.labels.record import SOURCE_GALILEO_NAGU, OrbitClass

# NAGU 2025001 (GSAT0102) verbatim from the GSC, the real PLN_MANV .txt format.
_PLN_MANV_GSAT0102 = """\
NOTICE ADVISORY TO GALILEO USERS (NAGU) 2025001
DATE GENERATED (UTC): 2025-01-17 15:15

NAGU TYPE: PLN_MANV
NAGU NUMBER: 2025001
NAGU SUBJECT: PLANNED MANOEUVRE FROM 2025-01-22 UNTIL 2025-02-01
NAGU REFERENCED TO: N/A
START DATE EVENT (UTC): 2025-01-22 06:00
END DATE EVENT (UTC): 2025-02-01 23:05
SATELLITE AFFECTED: GSAT0102
SPACE VEHICLE ID: 12
SIGNAL(S) AFFECTED: ALL
"""

# A PLN_MANV for a GSAT not in the seed crosswalk — ingested with norad_id None.
_PLN_MANV_UNMAPPED = """\
NAGU TYPE: PLN_MANV
NAGU NUMBER: 2025006
START DATE EVENT (UTC): 2025-02-10 07:30
END DATE EVENT (UTC): 2025-02-14 18:45
SATELLITE AFFECTED: GSAT0299
SPACE VEHICLE ID: 99
"""

# A non-maneuver NAGU type — must be ignored.
_PLN_OUTAGE = """\
NAGU TYPE: PLN_OUTAGE
NAGU NUMBER: 2025007
START DATE EVENT (UTC): 2025-03-01 00:00
END DATE EVENT (UTC): 2025-03-02 00:00
SATELLITE AFFECTED: GSAT0101
SPACE VEHICLE ID: 11
"""


def test_parses_pln_manv() -> None:
    (label,) = parse_nagus(_PLN_MANV_GSAT0102)
    assert label.norad_id == 37847  # GSAT0102 via the seed crosswalk
    assert label.source == SOURCE_GALILEO_NAGU
    assert label.orbit_class is OrbitClass.MEO
    assert label.maneuver_type is None  # epoch-only, no direction
    assert label.delta_v is None
    assert label.window_start == datetime(2025, 1, 22, 6, 0, tzinfo=timezone.utc)
    assert label.window_end == datetime(2025, 2, 1, 23, 5, tzinfo=timezone.utc)
    assert label.epoch == label.window_start  # the maneuver is at the window onset
    assert "2025001" in label.source_ref


def test_ignores_non_maneuver_types() -> None:
    assert parse_nagus(_PLN_OUTAGE) == []


def test_unmapped_gsat_keeps_label_with_none_norad() -> None:
    (label,) = parse_nagus(_PLN_MANV_UNMAPPED)
    assert label.norad_id is None
    assert "GSAT0299" in label.source_ref


def test_concatenated_file_filters_to_maneuvers() -> None:
    labels = parse_nagus(_PLN_MANV_GSAT0102 + _PLN_OUTAGE + _PLN_MANV_UNMAPPED)
    assert [label.norad_id for label in labels] == [37847, None]


def test_crosswalk_override() -> None:
    (label,) = parse_nagus(_PLN_MANV_UNMAPPED, gsat_to_norad={"GSAT0299": 99999})
    assert label.norad_id == 99999


def test_seed_crosswalk_has_the_iov_satellites() -> None:
    assert GALILEO_GSAT_TO_NORAD["GSAT0101"] == 37846
    assert GALILEO_GSAT_TO_NORAD["GSAT0102"] == 37847
