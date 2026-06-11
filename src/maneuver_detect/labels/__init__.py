"""Operator-announcement ingest and the epoch-to-elset-gap labeller.

One module per maneuver-label source, each normalising a heterogeneous operator log to the common
:class:`~maneuver_detect.labels.record.ManeuverLabel` record, plus the labeller that maps a maneuver
epoch onto the inter-elset gap that brackets it under the frozen matching tolerance:

- :mod:`~maneuver_detect.labels.doris` — the DORIS/IDS ``man.txt`` files (LEO, with Δv). The same
  files are the maneuver-history source the ILRS service points at, so there is no separate ILRS
  module.
- :mod:`~maneuver_detect.labels.gps_nanu` — the GPS NANU FCSTDV notices (MEO, epoch-only).
- :mod:`~maneuver_detect.labels.labeller` — the epoch-to-gap mapping and the coverage report.
"""

from __future__ import annotations

from maneuver_detect.labels.doris import DORIS_SAT_TO_NORAD, parse_doris
from maneuver_detect.labels.gps_nanu import GPS_SVN_TO_NORAD, parse_nanus
from maneuver_detect.labels.labeller import (
    ClassCoverage,
    CoverageReport,
    LabelledInterval,
    LabellingResult,
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

__all__ = [
    "DORIS_SAT_TO_NORAD",
    "GPS_SVN_TO_NORAD",
    "SOURCE_DORIS_IDS",
    "SOURCE_GPS_NANU",
    "ClassCoverage",
    "CoverageReport",
    "LabelledInterval",
    "LabellingResult",
    "ManeuverLabel",
    "OrbitClass",
    "intervals_to_frame",
    "label_coverage",
    "label_series",
    "parse_doris",
    "parse_nanus",
]
