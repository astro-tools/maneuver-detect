#!/usr/bin/env python3
"""V3 spike — detectability-floor + matching-tolerance analysis (stdlib + numpy).

Empirically characterises, per orbit class, (a) the TLE cadence, (b) how a
maneuver shows up as a residual jump in the mean-motion series vs. the
quiet-interval noise floor, and (c) the resulting smallest detectable Δv —
to fix the detection-matching tolerance and the detectability floor (decision
D4). Mean motion is read straight from the TLE; no sgp4 needed.

Data: the Shorten benchmark (15 satellites, TLE history + ground-truth maneuver
timestamps), used as a dev-only oracle (it is unlicensed — see the V2 spike — so
it is analysed locally, never redistributed). Get it with:

    git clone https://github.com/dpshorten/TLE_observation_benchmark_dataset

and point --data-dir at its `processed_files/` directory.

Run:  python3 v3_detectability_analysis.py --data-dir <path>/processed_files
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import os
import re

import numpy as np

MU = 398600.4418  # km^3/s^2, Earth GM
EPOCH0 = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)


def tle_epoch_to_days(field: str) -> float:
    """TLE line-1 epoch 'YYDDD.DDDDDDDD' -> days since 2000-01-01 UTC."""
    yy = int(field[:2])
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(field[2:])
    d = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=doy - 1.0)
    return (d - EPOCH0).total_seconds() / 86400.0


def parse_tle(path: str) -> tuple[np.ndarray, np.ndarray]:
    times, mm = [], []
    lines = [ln.rstrip("\n") for ln in open(path) if ln.strip()]
    for i in range(0, len(lines) - 1):
        l1, l2 = lines[i], lines[i + 1]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                times.append(tle_epoch_to_days(l1[18:32]))
                mm.append(float(l2[52:63]))
            except ValueError:
                continue
    t = np.array(times)
    n = np.array(mm)
    order = np.argsort(t)
    t, n = t[order], n[order]
    keep = np.concatenate(([True], np.diff(t) > 1e-9))  # drop duplicate-epoch elsets
    return t[keep], n[keep]


def parse_maneuvers(path: str) -> np.ndarray:
    out = []
    for line in open(path):
        m = re.match(r"\s*-\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", line)
        if m:
            d = dt.datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}").replace(tzinfo=dt.timezone.utc)
            out.append((d - EPOCH0).total_seconds() / 86400.0)
    return np.array(sorted(out))


def moving_median(x: np.ndarray, w: int) -> np.ndarray:
    half = w // 2
    pad = np.pad(x, half, mode="edge")
    return np.array([np.median(pad[i:i + w]) for i in range(len(x))])


def dv_from_dn(dn: float, n_revday: float, a_km: float) -> float:
    """|Δv| (m/s) for an in-track burn that produces mean-motion change dn (rev/day)."""
    da = (2.0 / 3.0) * a_km * (abs(dn) / n_revday)   # km
    n_rad = n_revday * 2.0 * math.pi / 86400.0       # rad/s
    return (n_rad * da / 2.0) * 1000.0               # m/s


def analyse(name: str, tle: str, man: str) -> dict | None:
    t, n = parse_tle(tle)
    if len(t) < 50:
        return None
    mans = parse_maneuvers(man)
    n_mean = float(np.median(n))
    a_km = (MU / (n_mean * 2 * math.pi / 86400.0) ** 2) ** (1.0 / 3.0)
    klass = "GEO" if n_mean < 2.0 else "LEO"  # Fengyun ~1.0 rev/day; altimetry ~12-14

    resid = n - moving_median(n, 21)            # detrend drag/secular drift
    gap_dt = np.diff(t)
    # robust per-gap signal: level shift of the residual across the gap (medians of
    # k elsets before vs after), which suppresses single-elset noise and captures the
    # in-track/E-W step a maneuver leaves in mean motion.
    k = 5
    jump = np.full(len(t) - 1, np.nan)
    for i in range(len(t) - 1):
        pre, post = resid[max(0, i - k + 1):i + 1], resid[i + 1:i + 1 + k]
        if len(pre) >= 2 and len(post) >= 2:
            jump[i] = abs(np.median(post) - np.median(pre))
    valid = ~np.isnan(jump)
    # a gap [t_i, t_{i+1}) is a "maneuver gap" if a labelled maneuver falls inside it
    man_gap = np.array([np.any((mans >= t[i]) & (mans < t[i + 1])) for i in range(len(t) - 1)])

    quiet = jump[valid & ~man_gap]
    mvr = jump[valid & man_gap]
    span_yr = (t[-1] - t[0]) / 365.25
    quiet_per_yr = len(quiet) / span_yr

    def operating_point(fa_per_yr: float) -> tuple[float, float]:
        """Threshold giving `fa_per_yr` quiet false-alarms/yr; return (Δv floor, recall)."""
        if quiet_per_yr <= fa_per_yr:
            thr = float(np.min(quiet))
        else:
            thr = float(np.percentile(quiet, 100.0 * (1.0 - fa_per_yr / quiet_per_yr)))
        rec = float(np.mean(mvr > thr)) if len(mvr) else float("nan")
        return dv_from_dn(thr, n_mean, a_km), rec

    floor_dv, recall = operating_point(1.0)   # primary operating point: 1 false-alarm/sat-year
    floor_dv3, recall3 = operating_point(3.0)

    return dict(
        name=name, klass=klass, n_elsets=len(t), span_yr=span_yr,
        cad_med=float(np.median(gap_dt)), cad_p90=float(np.percentile(gap_dt, 90)),
        n_man=len(mans), man_gaps=int(man_gap.sum()), a_km=a_km, n_mean=n_mean,
        floor_dv=floor_dv, recall=recall, floor_dv3=floor_dv3, recall3=recall3,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("SHORTEN_DIR", ""),
                    help="path to the Shorten benchmark processed_files/ directory")
    args = ap.parse_args()
    if not args.data_dir or not os.path.isdir(args.data_dir):
        raise SystemExit("provide --data-dir <Shorten processed_files/> (see module docstring)")

    rows = []
    for tle in sorted(glob.glob(os.path.join(args.data_dir, "*.tle"))):
        name = os.path.basename(tle)[:-4]
        man = os.path.join(args.data_dir, f"manoeuvres_{name}.yaml")
        if os.path.exists(man):
            r = analyse(name, tle, man)
            if r:
                rows.append(r)

    hdr = (f"{'satellite':14}{'cls':4}{'elsets':>7}{'yr':>6}{'cad_d':>7}{'man':>5}"
           f"{'floorΔv@1':>11}{'rec@1':>7}{'rec@3':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:14}{r['klass']:4}{r['n_elsets']:>7}{r['span_yr']:>6.1f}"
              f"{r['cad_med']:>7.2f}{r['n_man']:>5}{r['floor_dv']:>11.3f}{r['recall']:>7.2f}{r['recall3']:>7.2f}")

    print("\nPer-class aggregate (median across satellites):")
    for k in ("LEO", "GEO"):
        sub = [r for r in rows if r["klass"] == k]
        if not sub:
            continue
        cad = np.median([r["cad_med"] for r in sub])
        cad90 = np.median([r["cad_p90"] for r in sub])
        floor = np.median([r["floor_dv"] for r in sub])
        rec1 = np.median([r["recall"] for r in sub])
        rec3 = np.median([r["recall3"] for r in sub])
        print(f"  {k}: sats={len(sub)} cadence_med={cad:.2f}d (p90 {cad90:.2f}d) | "
              f"@1 FA/sat-yr: floorΔv≈{floor:.3f} m/s, recall≈{rec1:.2f} | @3 FA/sat-yr: recall≈{rec3:.2f}")
    print("\nOperating point = mean-motion level-shift threshold giving N quiet false-alarms per")
    print("satellite-year. floorΔv@1 = vis-viva in-track Δv at the 1-FA/sat-yr threshold;")
    print("rec@1 / rec@3 = fraction of labelled-maneuver gaps detected at 1 / 3 FA/sat-yr (mean motion only).")


if __name__ == "__main__":
    main()
