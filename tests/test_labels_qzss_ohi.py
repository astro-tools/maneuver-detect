"""Tests for ``maneuver_detect.labels.qzss_ohi`` — the QZSS OHI executed-maneuver parser.

The inline fixtures reproduce real excerpts of both layouts: the 6-column IGSO ``ohi-qzs2.txt``
(magnitude-only) and the 7-column GEO ``ohi-qzs6.txt`` (with the operator ``NS/EW`` type marker).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from maneuver_detect.labels.qzss_ohi import parse_qzss_ohi
from maneuver_detect.labels.record import SOURCE_QZSS_OHI, OrbitClass
from maneuver_detect.schema import ManeuverType

# A real ohi-qzs2.txt excerpt: a SATELLITE/MASS block (ignored) then a SATELLITE/MANEUVER block with
# two same-day burns in Nov 2017 (one campaign) and two more ~52 days later (a second campaign).
_OHI_QZS2 = """\
#+SATELLITE/MASS
#DATE TIME START(UTC),END(UTC),MASS(kg)
2017-11-15 10:34:47,2018-01-06 06:55:39,2320
2026-04-26 20:54:48,0000-00-00 00:00:00,2199
#-SATELLITE/MASS

#+SATELLITE/MANEUVER
#DATE TIME START(UTC),END(UTC),DURATION,DVX(m/s),DVY(m/s),DVZ(m/s)
2017-11-15 11:03:31,2017-11-15 11:05:50,00:02:19,-2.325,0.004,0.032
2017-11-15 18:33:48,2017-11-15 18:35:32,00:01:44,1.756,0.017,0.024
2018-01-06 07:23:19,2018-01-06 07:24:18,00:00:59,-0.916,0.001,0.012
2018-01-06 14:53:27,2018-01-06 14:54:10,00:00:43,0.692,-0.006,0.008
#-SATELLITE/MANEUVER
"""

_NO_MANEUVERS = """\
#+SATELLITE/MASS
#DATE TIME START(UTC),END(UTC),MASS(kg)
2017-11-15 10:34:47,2018-01-06 06:55:39,2320
#-SATELLITE/MASS
"""

# A real ohi-qzs6.txt excerpt: the 7-column GEO layout with an NS/EW marker. The Aug-22 NS burn and
# the Aug-23 EW burn fall within a day (one campaign); the Sep-14 EW burn is a separate campaign.
_OHI_QZS6_GEO = """\
#+SATELLITE/MANEUVER
#DATE TIME START(UTC),END(UTC),DURATION,NS/EW,DVX(m/s),DVY(m/s),DVZ(m/s)
2025-08-22 13:44:42,2025-08-22 13:49:53,00:05:11,NS,0.006,-5.146,0.095
2025-08-23 12:59:47,2025-08-23 13:00:12,00:00:25,EW,-0.098,0.0,0.002
2025-09-14 15:59:51,2025-09-14 16:00:08,00:00:17,EW,-0.066,0.0,0.002
#-SATELLITE/MANEUVER
"""


def _mag(dvx: float, dvy: float, dvz: float) -> float:
    return math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)


def test_collapses_burns_into_two_events() -> None:
    labels = parse_qzss_ohi(
        _OHI_QZS2, norad_id=42738, orbit_class=OrbitClass.IGSO, qzs_label="QZS-2"
    )
    assert len(labels) == 2  # the two same-day burns collapse per campaign


def test_event_fields_are_normalised() -> None:
    first, second = parse_qzss_ohi(
        _OHI_QZS2, norad_id=42738, orbit_class=OrbitClass.IGSO, qzs_label="QZS-2"
    )
    # Event Δv is the sum of the burns' magnitudes; the type is left None (frame undocumented).
    assert first.delta_v == _mag(-2.325, 0.004, 0.032) + _mag(1.756, 0.017, 0.024)
    assert first.maneuver_type is None
    assert first.delta_v is not None and first.delta_v > 0.0
    # Epoch is the event's first burn start; the window spans first-start..last-end.
    assert first.epoch == datetime(2017, 11, 15, 11, 3, 31, tzinfo=timezone.utc)
    assert first.window_start == datetime(2017, 11, 15, 11, 3, 31, tzinfo=timezone.utc)
    assert first.window_end == datetime(2017, 11, 15, 18, 35, 32, tzinfo=timezone.utc)
    assert first.source == SOURCE_QZSS_OHI
    assert first.norad_id == 42738
    assert first.orbit_class is OrbitClass.IGSO
    assert second.epoch == datetime(2018, 1, 6, 7, 23, 19, tzinfo=timezone.utc)


def test_orbit_class_is_passed_through() -> None:
    (label,) = parse_qzss_ohi(
        "#+SATELLITE/MANEUVER\n2017-11-15 11:03:31,2017-11-15 11:05:50,00:02:19,-2.3,0.0,0.0\n"
        "#-SATELLITE/MANEUVER\n",
        norad_id=42917,
        orbit_class=OrbitClass.GEO,
        qzs_label="QZS-3",
    )
    assert label.orbit_class is OrbitClass.GEO  # QZS-3/6 are GEO, not IGSO


def test_geo_layout_types_from_ns_ew_marker() -> None:
    first, second = parse_qzss_ohi(
        _OHI_QZS6_GEO, norad_id=62876, orbit_class=OrbitClass.GEO, qzs_label="QZS-6"
    )
    # The mixed NS+EW campaign collapses to one event; the dominant (larger-Δv) NS burn sets the
    # type to cross-track. The standalone EW campaign is in-track.
    assert first.maneuver_type is ManeuverType.CROSS_TRACK
    assert first.delta_v == _mag(0.006, -5.146, 0.095) + _mag(-0.098, 0.0, 0.002)
    assert second.maneuver_type is ManeuverType.IN_TRACK


def test_no_maneuver_section_yields_no_labels() -> None:
    assert (
        parse_qzss_ohi(
            _NO_MANEUVERS, norad_id=42738, orbit_class=OrbitClass.IGSO, qzs_label="QZS-2"
        )
        == []
    )


def test_is_deterministic() -> None:
    def run() -> list[object]:
        return list(
            parse_qzss_ohi(
                _OHI_QZS2, norad_id=42738, orbit_class=OrbitClass.IGSO, qzs_label="QZS-2"
            )
        )

    assert run() == run()
