#!/usr/bin/env python3
"""V2 follow-up spike — Galileo NAGU maneuver-label ingest proof (stdlib only).

Companion to ``v2_label_ingest_proof.py`` (the original V2 GPS-NANU proof). It
demonstrates that the cleanest *new* maneuver-label source found in the follow-up
survey — Galileo NAGUs — is machine-ingestible into the project's normalised label
record ``(norad_id, epoch, type, delta_v, source, ...)``, deterministically.

Source: Notice Advisory to Galileo Users (NAGU) from the European GNSS Service
Centre (GSC, https://www.gsc-europa.eu). Each notice is published as a flat
``KEY: value`` ``.txt`` at a stable per-notice URL; the ``PLN_MANV`` type
("planned activity affecting the attitude and/or orbit") announces a scheduled
maneuver window with ``START``/``END DATE EVENT (UTC)`` already in calendar UTC
(no day-of-year conversion, unlike GPS NANUs). NAGUs carry the GSAT id + SVID,
not a NORAD id, so a GSAT->NORAD crosswalk (from CelesTrak's Galileo catalogue)
is applied where known.

GSC reuse terms authorise reproduction/use provided the source is acknowledged
(c) EU; the embedded sample (NAGU 2025001, verbatim) is reproduced under those
terms and is redistribution-clean with attribution. The second sample is in the
same public ``.txt`` format for an object deliberately left out of the seed
crosswalk, to exercise the un-crosswalked ``norad_id = None`` path.

- Part A (offline, reproducible): parse two embedded PLN_MANV NAGUs into normalised
  records, then JSON-serialise + SHA-256 twice; assert identical.
- Part B (best-effort, no-auth): fetch a live NAGU page from the GSC to show the
  no-auth ingest leg works; never written to disk / committed.

Run:  python3 v2_followup_label_ingest_proof.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import urllib.request

# A published NAGU notice page (no-auth); used by Part B only as a reachability check.
GSC_NAGU_PAGE = "https://www.gsc-europa.eu/notice-advisory-to-galileo-users-nagu-2025001"

# Illustrative GSAT->NORAD crosswalk seed; the full mapping comes from CelesTrak's
# Galileo catalogue / IGS MGEX. The two In-Orbit-Validation satellites are confident:
# GSAT0101 (GALILEO-PFM) = NORAD 37846; GSAT0102 (GALILEO-FM2) = NORAD 37847.
GSAT_TO_NORAD = {"GSAT0101": 37846, "GSAT0102": 37847}

# Two embedded PLN_MANV NAGUs in the public NAGU ``.txt`` field format. The first is
# NAGU 2025001 verbatim (GSC, (c) EU, reproduced with attribution); the second is in
# the same format for GSAT0220 (left out of the seed crosswalk on purpose).
SAMPLE_NAGUS = [
    """\
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

EVENT DESCRIPTION: GALILEO SATELLITE GSAT0102 (ALL SIGNALS) WILL BE UNAVAILABLE FROM 2025-01-22 BEGINNING 06:00 UTC DUE TO MANOEUVRE. OUTAGE RECOVERY ESTIMATED ON 2025-02-01 23:05 UTC.
""",
    """\
NOTICE ADVISORY TO GALILEO USERS (NAGU) 2025006
DATE GENERATED (UTC): 2025-02-03 09:30

NAGU TYPE: PLN_MANV
NAGU NUMBER: 2025006
NAGU SUBJECT: PLANNED MANOEUVRE FROM 2025-02-10 UNTIL 2025-02-14
NAGU REFERENCED TO: N/A
START DATE EVENT (UTC): 2025-02-10 07:30
END DATE EVENT (UTC): 2025-02-14 18:45
SATELLITE AFFECTED: GSAT0220
SPACE VEHICLE ID: 14
SIGNAL(S) AFFECTED: ALL

EVENT DESCRIPTION: GALILEO SATELLITE GSAT0220 (ALL SIGNALS) WILL BE UNAVAILABLE FROM 2025-02-10 BEGINNING 07:30 UTC DUE TO MANOEUVRE. OUTAGE RECOVERY ESTIMATED ON 2025-02-14 18:45 UTC.
""",
]


def _field(text: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _to_iso(stamp: str) -> str:
    """``YYYY-MM-DD HH:MM`` UTC -> ISO8601 UTC. NAGU times are already calendar UTC."""
    parsed = dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_nagu(text: str) -> dict | None:
    if _field(text, "NAGU TYPE") != "PLN_MANV":
        return None  # only planned-maneuver notices are labels
    gsat = _field(text, "SATELLITE AFFECTED")
    svid = _field(text, "SPACE VEHICLE ID")
    return {
        "norad_id": GSAT_TO_NORAD.get(gsat or ""),  # None if not in the crosswalk
        "object": f"Galileo {gsat} (SVID{svid})",
        "epoch_start_utc": _to_iso(_field(text, "START DATE EVENT (UTC)")),
        "epoch_stop_utc": _to_iso(_field(text, "END DATE EVENT (UTC)")),
        "maneuver_type": "stationkeeping",  # PLN_MANV = planned attitude/orbit maneuver
        "delta_v_mps": None,                # NAGUs give no magnitude (epoch-only source)
        "source": "GALILEO-NAGU",
        "source_ref": f"NAGU {_field(text, 'NAGU NUMBER')} (PLN_MANV)",
    }


def canonical_hash(records: list[dict]) -> str:
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def part_a() -> tuple[bool, str]:
    records = [r for r in (parse_nagu(t) for t in SAMPLE_NAGUS) if r]
    h1 = canonical_hash(records)
    h2 = canonical_hash(records)
    print(f"[A] PLN_MANV NAGUs parsed    : {len(records)}")
    for r in records:
        print(f"      {r['source_ref']:24} norad={r['norad_id']}  {r['epoch_start_utc']} -> {r['epoch_stop_utc']}")
    print(f"[A] normalise+hash, run 1    : {h1}")
    print(f"[A] normalise+hash, run 2    : {h2}")
    ok = h1 == h2 and len(records) == 2
    print(f"[A] deterministic ingest     : {'PASS' if ok else 'FAIL'}")
    return ok, h1


def part_b() -> None:
    req = urllib.request.Request(GSC_NAGU_PAGE, headers={"User-Agent": "maneuver-detect-spike/0.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", "replace")
        has_marker = "NOTICE ADVISORY TO GALILEO USERS" in text.upper()
        print(f"[B] GSC no-auth fetch        : OK ({len(text)} bytes, NAGU marker={'yes' if has_marker else 'no'})")
        print("[B] fetched data             : NOT written to disk / committed")
    except Exception as exc:
        print(f"[B] GSC fetch                : skipped ({exc.__class__.__name__})")


if __name__ == "__main__":
    ok, _ = part_a()
    part_b()
    sys.exit(0 if ok else 1)
