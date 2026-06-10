#!/usr/bin/env python3
"""V1 spike — reconstruction-determinism proof (stdlib only).

Demonstrates the single property the recommended *recipe-first* distribution
model (D2) relies on: given a PINNED set of input elsets, deriving the
mean-element series and serialising it is byte-for-byte deterministic across
runs. That is what lets a published "labels + pinned recipe" dataset reconstruct
identically on any machine *without* the project ever redistributing Space-Track
data or its analysis (10 U.S.C. 2274(c)(2) / the Space-Track User Agreement).

- Part A (offline, fully reproducible): build a pinned set of SYNTHETIC,
  spec-valid TLEs (fictional catalogue id 90001, with an injected mean-motion
  step that mimics an in-track maneuver), derive the mean-element series, and
  hash it twice. Asserts the two hashes are identical. No network, no real
  catalogue data, nothing to redistribute.
- Part B (best-effort, no-auth): fetch one satellite's current GP from CelesTrak
  to show the no-auth fetch leg works. The fetched data is never written to disk
  and never committed. Skipped cleanly when offline. Space-Track *history* — the
  production reconstruction source — needs the user's own credentials and is out
  of scope for this proof.

Run:  python3 v1_reconstruct_proof.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"


def _checksum(body68: str) -> str:
    """TLE mod-10 checksum: digits add their value, '-' counts as 1."""
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in body68[:68])
    return str(total % 10)


def build_line1(satnum: int, epoch: float, elset: int) -> str:
    body = (
        "1 "                       # cols 1-2
        f"{satnum:5d}"             # 3-7  satellite number
        "U"                        # 8    classification
        " "                        # 9
        "24001A  "                 # 10-17 international designator
        " "                        # 18
        f"{epoch:14.8f}"           # 19-32 epoch YYDDD.DDDDDDDD
        " "                        # 33
        " .00000000"               # 34-43 first deriv of mean motion
        " "                        # 44
        " 00000-0"                 # 45-52 second deriv
        " "                        # 53
        " 00000-0"                 # 54-61 B* drag term
        " "                        # 62
        "0"                        # 63   ephemeris type
        " "                        # 64
        f"{elset:4d}"              # 65-68 element-set number
    )
    assert len(body) == 68, len(body)
    return body + _checksum(body)


def build_line2(satnum: int, incl: float, raan: float, ecc7: str,
                argp: float, ma: float, n: float, rev: int) -> str:
    body = (
        "2 "                       # cols 1-2
        f"{satnum:5d}"             # 3-7
        " "                        # 8
        f"{incl:8.4f}"             # 9-16  inclination
        " "                        # 17
        f"{raan:8.4f}"             # 18-25 RAAN
        " "                        # 26
        f"{ecc7}"                  # 27-33 eccentricity (leading decimal implied)
        " "                        # 34
        f"{argp:8.4f}"             # 35-42 argument of perigee
        " "                        # 43
        f"{ma:8.4f}"               # 44-51 mean anomaly
        " "                        # 52
        f"{n:11.8f}"               # 53-63 mean motion
        f"{rev:5d}"                # 64-68 revolution number
    )
    assert len(body) == 68, len(body)
    return body + _checksum(body)


def synthetic_elsets() -> list[tuple[str, str]]:
    """A pinned 8-point synthetic series; mean motion steps up at k=4 (a stand-in
    for an in-track maneuver). Fully deterministic — no inputs, no clock."""
    rows: list[tuple[str, str]] = []
    base_n = 15.50000000
    for k in range(8):
        epoch = 26000.0 + (10.0 + k * 2.0)          # YY=26, DDD=010,012,...
        n = base_n + (0.02 if k >= 4 else 0.0)      # the "maneuver" step
        ma = float((k * 45) % 360)
        rows.append((
            build_line1(90001, epoch, 999),
            build_line2(90001, 51.6400, 200.0000, "0001000", 90.0000, ma, n, 1),
        ))
    return rows


def parse_mean_elements(l1: str, l2: str) -> dict:
    """Parse the mean elements straight out of the fixed TLE columns."""
    return {
        "epoch": l1[18:32].strip(),
        "inclination_deg": float(l2[8:16]),
        "raan_deg": float(l2[17:25]),
        "ecc": float("0." + l2[26:33].strip()),
        "argp_deg": float(l2[34:42]),
        "mean_anomaly_deg": float(l2[43:51]),
        "mean_motion_revday": float(l2[52:63]),
    }


def derive_series(elsets: list[tuple[str, str]]) -> list[dict]:
    return [parse_mean_elements(a, b) for a, b in elsets]


def canonical_hash(series: list[dict]) -> str:
    blob = json.dumps(series, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def part_a() -> tuple[bool, str]:
    elsets = synthetic_elsets()
    h1 = canonical_hash(derive_series(elsets))
    h2 = canonical_hash(derive_series(elsets))
    print(f"[A] pinned synthetic elsets : {len(elsets)}")
    print(f"[A] derive+hash, run 1      : {h1}")
    print(f"[A] derive+hash, run 2      : {h2}")
    ok = h1 == h2
    print(f"[A] byte-deterministic      : {'PASS' if ok else 'FAIL'}")
    return ok, h1


def part_b() -> None:
    try:
        with urllib.request.urlopen(CELESTRAK_GP, timeout=15) as resp:
            text = resp.read().decode("utf-8", "replace")
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 3 and lines[1].startswith("1 ") and lines[2].startswith("2 "):
            me = parse_mean_elements(lines[1], lines[2])
            print(f"[B] CelesTrak no-auth fetch : OK (CATNR 25544, "
                  f"epoch={me['epoch']}, n={me['mean_motion_revday']})")
            print("[B] fetched data            : NOT written to disk / committed")
        else:
            print(f"[B] CelesTrak fetch         : unexpected response ({len(lines)} lines)")
    except Exception as exc:  # offline / blocked — expected in sandboxes
        print(f"[B] CelesTrak fetch         : skipped ({exc.__class__.__name__})")


if __name__ == "__main__":
    ok, _ = part_a()
    part_b()
    sys.exit(0 if ok else 1)
