"""Tests for ``maneuver_detect.datasets.build`` — the NANU-archive crawl (mocked HTTP, offline)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from maneuver_detect.data.cache import Cache
from maneuver_detect.data.ratelimit import RateLimiter
from maneuver_detect.datasets.build import (
    fetch_labels,
    fetch_nanu_labels,
    fetch_noaa_goes_labels,
    fetch_qzss_ohi_labels,
)
from maneuver_detect.datasets.recipe import Recipe, RecipeEntry
from maneuver_detect.labels.record import (
    SOURCE_DORIS_IDS,
    SOURCE_GPS_NANU,
    SOURCE_NOAA_GOES,
    SOURCE_QZSS_OHI,
    OrbitClass,
)

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


def test_crawls_archive_and_keeps_only_fcstdv(tmp_path: Path) -> None:
    with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
        labels = fetch_nanu_labels(
            client,
            start_year=2023,
            end_year=2024,
            svn_to_norad={"SVN65": 38833},
            cache=Cache(tmp_path),
        )
    # The FCSTSUMM is dropped; only the FCSTDV becomes a label. The 2023 index (404) is skipped.
    assert len(labels) == 1
    assert labels[0].norad_id == 38833
    assert labels[0].orbit_class is OrbitClass.MEO
    assert labels[0].delta_v is None  # NANUs are epoch-only


def test_cache_prevents_repeat_downloads(tmp_path: Path) -> None:
    calls = {"n": 0}

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _handler(request)

    cache = Cache(tmp_path)
    kwargs = {"start_year": 2024, "end_year": 2024, "svn_to_norad": {"SVN65": 38833}}
    with httpx.Client(transport=httpx.MockTransport(counting_handler)) as client:
        first = fetch_nanu_labels(client, cache=cache, **kwargs)  # type: ignore[arg-type]
        after_first = calls["n"]
        second = fetch_nanu_labels(client, cache=cache, **kwargs)  # type: ignore[arg-type]
    assert after_first > 0  # the first crawl hit the network
    assert (
        calls["n"] == after_first
    )  # the second crawl re-downloaded nothing (all served from cache)
    assert second == first  # and produced the same labels


def test_unmapped_svn_is_dropped(tmp_path: Path) -> None:
    with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
        labels = fetch_nanu_labels(
            client, start_year=2024, end_year=2024, svn_to_norad={}, cache=Cache(tmp_path)
        )
    assert labels == []  # SVN65 not in the crosswalk -> no NORAD -> dropped


# --- fetch_labels: the DORIS + NANU ingest orchestration (offline) ---

# One JASON-1 (007: Q,S,W = radial, along, cross) along-track 2.5 m/s burn → one IN_TRACK label for
# NORAD 26997. 15 burn tokens: 5 date + duration + 3 ΔV + 3 acc + 3 delta-acc.
_JA1_MAN_TXT = (
    "JASO1 2024 285 00 00 2024 286 00 00 007 1 "
    "2024 285 00 30 00.000 5.0e+00 0.000000e+00 2.500000e+00 0.000000e+00 0 0 0 0 0 0\n"
)

# A NANU FCSTDV for SVN62, which the real GPS crosswalk resolves to NORAD 36585.
_NANU_INDEX_2024 = '<a href="nanu.2024050.txt">nanu.2024050.txt</a>\n'
_NANU_SVN62 = """\
NANU TYPE: FCSTDV
NANU NUMBER: 2024050
NANU DTG: 071200Z FEB 2024
SVN: 62
PRN: 25
START JDAY: 040
START TIME ZULU: 0600
STOP JDAY: 040
STOP TIME ZULU: 1200
"""


def _labels_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/ja1man.txt"):
        return httpx.Response(200, text=_JA1_MAN_TXT)
    if path.endswith("/missingman.txt"):  # a renamed/missing DORIS file
        return httpx.Response(404)
    if path == "/GPS/NANU/2024/":
        return httpx.Response(200, text=_NANU_INDEX_2024)
    if path.endswith("nanu.2024050.txt"):
        return httpx.Response(200, text=_NANU_SVN62)
    return httpx.Response(404)


def _recipe() -> Recipe:
    def doris(norad_id: int, ref: str) -> RecipeEntry:
        return RecipeEntry(
            norad_id=norad_id,
            orbit_class=OrbitClass.LEO,
            object_name=ref,
            catalogue_source="spacetrack",
            label_source=SOURCE_DORIS_IDS,
            label_ref=ref,
        )

    return Recipe(
        dataset_version="test",
        entries=(
            doris(26997, "ja1"),  # the JASON file
            doris(26997, "ja1"),  # a duplicate label_ref → fetched once (seen_refs dedup)
            doris(99999, "missing"),  # a 404 file → skipped with a warning, not aborting
            RecipeEntry(  # a non-DORIS entry → skipped in the DORIS loop (NANU is its own crawl)
                norad_id=36585,
                orbit_class=OrbitClass.MEO,
                object_name="SVN62",
                catalogue_source="spacetrack",
                label_source=SOURCE_GPS_NANU,
                label_ref="SVN62",
            ),
        ),
    )


def test_fetch_labels_merges_doris_and_nanu_and_dedups(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with httpx.Client(transport=httpx.MockTransport(_labels_handler)) as client:
        by_norad = fetch_labels(
            _recipe(),
            client,
            nanu_start_year=2024,
            nanu_end_year=2024,
            rate_limiter=RateLimiter(0.0),  # disabled limiter, but exercises the acquire() path
            cache=Cache(tmp_path),
        )
    # DORIS JASON (one event) and NANU SVN62 both land, keyed by NORAD.
    assert set(by_norad) == {26997, 36585}
    # The duplicate "ja1" entry was fetched once: exactly one DORIS label, not two.
    assert len(by_norad[26997]) == 1
    assert by_norad[26997][0].delta_v == pytest.approx(2.5)
    assert by_norad[36585][0].source == SOURCE_GPS_NANU
    # The missing DORIS file was skipped with a loud warning rather than aborting the build.
    assert "not found (404)" in capsys.readouterr().err
    assert 99999 not in by_norad


# --- fetch_qzss_ohi_labels: per-satellite OHI fetch (offline) ---

_OHI_QZS2 = """\
#+SATELLITE/MANEUVER
#DATE TIME START(UTC),END(UTC),DURATION,DVX(m/s),DVY(m/s),DVZ(m/s)
2018-05-20 22:04:06,2018-05-20 22:06:16,00:02:10,-1.99,0.0,0.026
#-SATELLITE/MANEUVER
"""


def _qzss_recipe() -> Recipe:
    def qzss(norad_id: int, ref: str, orbit_class: OrbitClass) -> RecipeEntry:
        return RecipeEntry(
            norad_id=norad_id,
            orbit_class=orbit_class,
            object_name=f"QZSS {ref}",
            catalogue_source="spacetrack",
            label_source=SOURCE_QZSS_OHI,
            label_ref=ref,
        )

    return Recipe(
        dataset_version="test",
        entries=(
            qzss(42738, "qzs2", OrbitClass.IGSO),
            qzss(62876, "qzs6", OrbitClass.GEO),  # a freshly-launched bird with no OHI yet (404)
        ),
    )


def _qzss_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/ohi-qzs2.txt"):
        return httpx.Response(200, text=_OHI_QZS2)
    return httpx.Response(404)  # ohi-qzs6.txt missing


def test_fetch_qzss_ohi_skips_missing_files(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with httpx.Client(transport=httpx.MockTransport(_qzss_handler)) as client:
        labels = fetch_qzss_ohi_labels(
            _qzss_recipe(), client, rate_limiter=RateLimiter(0.0), cache=Cache(tmp_path)
        )
    assert len(labels) == 1  # only QZS-2 has an OHI file
    assert labels[0].norad_id == 42738
    assert labels[0].orbit_class is OrbitClass.IGSO
    assert labels[0].delta_v is not None  # QZSS labels carry a Δv magnitude
    assert "ohi-qzs6.txt' not found (404)" in capsys.readouterr().err


# --- fetch_noaa_goes_labels: navsum history from the Internet Archive (offline) ---

_GOES_NAME_TO_NORAD = {"GOES-16": 41866}


def _navsum(yy_ddd: str) -> str:
    return (
        "=============================================================\r\n"
        "Spacecraft :                                  GOES-16\r\n"
        "Comments:\r\n"
        f"Fuel and oxidizer remaining are estimates after the last maneuver on {yy_ddd}.\r\n"
    )


def _noaa_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/cdx/search/cdx":
        # The CDX listing: a header row then two content-distinct snapshots.
        return httpx.Response(200, text='[["timestamp"],["20240115000000"],["20260610000000"]]')
    if "20240115000000" in path:
        return httpx.Response(200, text=_navsum("24/015"))
    if "20260610000000" in path:
        return httpx.Response(200, text=_navsum("26/159"))
    if path.endswith("/resources/cemscs/navsum.txt"):  # the live file, repeating the latest
        return httpx.Response(200, text=_navsum("26/159"))
    return httpx.Response(404)


def test_fetch_noaa_goes_dedups_across_snapshots(tmp_path: Path) -> None:
    with httpx.Client(transport=httpx.MockTransport(_noaa_handler)) as client:
        labels = fetch_noaa_goes_labels(
            client,
            goes_name_to_norad=_GOES_NAME_TO_NORAD,
            rate_limiter=RateLimiter(0.0),
            cache=Cache(tmp_path),
        )
    # Two distinct maneuver days (24/015, 26/159); the live file repeats 26/159 and is deduped.
    assert len(labels) == 2
    assert {label.norad_id for label in labels} == {41866}
    assert all(label.source == SOURCE_NOAA_GOES for label in labels)
    days = sorted(label.window_start.date().isoformat() for label in labels)
    assert days == ["2024-01-15", "2026-06-08"]
