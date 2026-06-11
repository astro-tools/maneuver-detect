"""Tests for ``maneuver_detect.datasets.build`` — the NANU-archive crawl (mocked HTTP, offline)."""

from __future__ import annotations

import httpx

from maneuver_detect.datasets.build import fetch_nanu_labels
from maneuver_detect.labels.record import OrbitClass

_INDEX_2024 = """
<a href="/GPS/NANU/2024/nanu.2024001.txt">nanu.2024001.txt</a>
<a href="/GPS/NANU/2024/nanu.2024002.txt">nanu.2024002.txt</a>
"""

_FCSTSUMM = """\
1.     NANU TYPE: FCSTSUMM
       NANU NUMBER: 2024001
       NANU DTG: 011200Z JAN 2024
       SVN: 65
       PRN: 24
"""

_FCSTDV = """\
NOTICE ADVISORY TO NAVSTAR USERS (NANU) 2024002
SUBJ: SVN65 (PRN24) FORECAST OUTAGE
1.     NANU TYPE: FCSTDV
       NANU NUMBER: 2024002
       NANU DTG: 071844Z JAN 2024
       SVN: 65
       PRN: 24
       START JDAY: 011
       START TIME ZULU: 1130
       STOP JDAY: 011
       STOP TIME ZULU: 2330
"""


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/GPS/NANU/2023/":  # a missing year index
        return httpx.Response(404)
    if path == "/GPS/NANU/2024/":
        return httpx.Response(200, text=_INDEX_2024)
    if path.endswith("nanu.2024001.txt"):
        return httpx.Response(200, text=_FCSTSUMM)
    if path.endswith("nanu.2024002.txt"):
        return httpx.Response(200, text=_FCSTDV)
    return httpx.Response(404)


def test_crawls_archive_and_keeps_only_fcstdv() -> None:
    with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
        labels = fetch_nanu_labels(
            client, start_year=2023, end_year=2024, svn_to_norad={"SVN65": 38833}
        )
    # The FCSTSUMM is dropped; only the FCSTDV becomes a label. The 2023 index (404) is skipped.
    assert len(labels) == 1
    assert labels[0].norad_id == 38833
    assert labels[0].orbit_class is OrbitClass.MEO
    assert labels[0].delta_v is None  # NANUs are epoch-only


def test_unmapped_svn_is_dropped() -> None:
    with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
        labels = fetch_nanu_labels(client, start_year=2024, end_year=2024, svn_to_norad={})
    assert labels == []  # SVN65 not in the crosswalk -> no NORAD -> dropped
