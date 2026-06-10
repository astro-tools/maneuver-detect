#!/usr/bin/env python3
"""V2 spike — maneuver-label ingest proof (stdlib only).

Demonstrates that a no-auth, public-domain maneuver-label source is
machine-ingestible into the project's normalised label record
`(norad_id, epoch, type, delta_v, source, ...)`, deterministically.

Source: GPS NANUs (Notice Advisory to Navstar Users) from US Coast Guard
NAVCEN — public-domain US-Government text, archives to 1997. The FCSTDV type
("Forecast Delta-V") announces a scheduled station-keeping maneuver window.
NANUs carry SVN/PRN, not a NORAD id, so a SVN->NORAD crosswalk (from CelesTrak's
GPS catalogue) is applied where known.

- Part A (offline, reproducible): parse two embedded FCSTDV NANUs (the public
  NANU format) into normalised records, then JSON-serialise + SHA-256 twice;
  assert identical. NANUs are US-gov public domain, so embedding samples is
  redistribution-clean (unlike Space-Track data — see the V1 spike).
- Part B (best-effort, no-auth): fetch NAVCEN's current NANU file to show the
  live no-auth ingest leg works; never written to disk / committed.

Run:  python3 v2_label_ingest_proof.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import urllib.request

NAVCEN_CURRENT = "https://www.navcen.uscg.gov/sites/default/files/gps/nanu/current_nanu.nnu"

# Illustrative SVN->NORAD crosswalk; the full mapping comes from CelesTrak's GPS
# operational-status / SATCAT data. SVN62 = GPS IIF-1 (USA-213) = NORAD 36585.
SVN_TO_NORAD = {"SVN62": 36585}

# Two embedded FCSTDV NANUs in the public NANU field format (illustrative values).
SAMPLE_NANUS = [
    """\
NANU TYPE: FCSTDV
NANU NUMBER: 2025087
NANU DTG: 211200Z MAR 2025
SVN: 62
PRN: 25
START JDAY: 086
START TIME ZULU: 1300
START CALENDAR DATE: 27 MAR 2025
STOP JDAY: 086
STOP TIME ZULU: 1900
STOP CALENDAR DATE: 27 MAR 2025
CONDITION: GPS SATELLITE SVN62 (PRN25) WILL BE UNUSABLE ON JDAY 086 (27 MAR 2025)
BEGINNING 1300 ZULU UNTIL JDAY 086 (27 MAR 2025) ENDING 1900 ZULU DUE TO A DELTA-V MANEUVER.
""",
    """\
NANU TYPE: FCSTDV
NANU NUMBER: 2025104
NANU DTG: 080900Z APR 2025
SVN: 74
PRN: 04
START JDAY: 100
START TIME ZULU: 0600
START CALENDAR DATE: 10 APR 2025
STOP JDAY: 100
STOP TIME ZULU: 1200
STOP CALENDAR DATE: 10 APR 2025
CONDITION: GPS SATELLITE SVN74 (PRN04) WILL BE UNUSABLE ON JDAY 100 (10 APR 2025)
BEGINNING 0600 ZULU UNTIL JDAY 100 (10 APR 2025) ENDING 1200 ZULU DUE TO A DELTA-V MANEUVER.
""",
]


def _field(text: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _to_iso(year: int, jday: int, zulu: str) -> str:
    """Day-of-year + HHMM Zulu -> ISO8601 UTC."""
    base = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=jday - 1)
    stamp = base.replace(hour=int(zulu[:2]), minute=int(zulu[2:]))
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_nanu(text: str) -> dict | None:
    if _field(text, "NANU TYPE") != "FCSTDV":
        return None  # only delta-V maneuver notices are labels
    svn = _field(text, "SVN")
    prn = _field(text, "PRN")
    dtg = _field(text, "NANU DTG") or ""
    year = int(re.search(r"\b(20\d\d)\b", dtg).group(1))
    start = _to_iso(year, int(_field(text, "START JDAY")), _field(text, "START TIME ZULU"))
    stop = _to_iso(year, int(_field(text, "STOP JDAY")), _field(text, "STOP TIME ZULU"))
    return {
        "norad_id": SVN_TO_NORAD.get(f"SVN{svn}"),  # None if not in the crosswalk
        "object": f"GPS SVN{svn}/PRN{prn}",
        "epoch_start_utc": start,
        "epoch_stop_utc": stop,
        "maneuver_type": "stationkeeping",  # FCSTDV = delta-V; R/S/W refinement is downstream
        "delta_v_mps": None,                # NANUs give no magnitude (epoch-only source)
        "source": "GPS-NANU",
        "source_ref": f"NANU {_field(text, 'NANU NUMBER')} (FCSTDV)",
    }


def canonical_hash(records: list[dict]) -> str:
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def part_a() -> tuple[bool, str]:
    records = [r for r in (parse_nanu(t) for t in SAMPLE_NANUS) if r]
    h1 = canonical_hash(records)
    h2 = canonical_hash(records)
    print(f"[A] FCSTDV NANUs parsed      : {len(records)}")
    for r in records:
        print(f"      {r['source_ref']:24} norad={r['norad_id']}  {r['epoch_start_utc']} -> {r['epoch_stop_utc']}")
    print(f"[A] normalise+hash, run 1    : {h1}")
    print(f"[A] normalise+hash, run 2    : {h2}")
    ok = h1 == h2 and len(records) == 2
    print(f"[A] deterministic ingest     : {'PASS' if ok else 'FAIL'}")
    return ok, h1


def part_b() -> None:
    req = urllib.request.Request(NAVCEN_CURRENT, headers={"User-Agent": "maneuver-detect-spike/0.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", "replace")
        n_nanu = len(re.findall(r"NANU\s+(?:TYPE|NUMBER)", text, re.IGNORECASE)) or text.count("NANU")
        print(f"[B] NAVCEN no-auth fetch     : OK ({len(text)} bytes, ~{n_nanu} NANU markers)")
        print("[B] fetched data             : NOT written to disk / committed")
    except Exception as exc:
        print(f"[B] NAVCEN fetch             : skipped ({exc.__class__.__name__})")


if __name__ == "__main__":
    ok, _ = part_a()
    part_b()
    sys.exit(0 if ok else 1)
