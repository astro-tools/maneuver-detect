#!/usr/bin/env python3
"""V4 spike — Δv-inversion validation (stdlib + numpy).

Validates the inversion that turns a detected mean-element jump into a maneuver
type (R/S/W) and a Δv estimate, using the linearised circular-orbit impulsive
(Gauss) relations:

    in-track    Δv_T = n_rad · Δa / 2
    cross-track Δv_W = v · sqrt(Δi² + (sin i · ΔΩ)²)
    radial      Δv_R = sqrt(max(0, (v·Δe)² − (2 Δv_T)²))   # ecc change beyond transverse
    |Δv| = sqrt(Δv_R² + Δv_T² + Δv_W²);  type = argmax component

Element jumps must be the *anomalous* step, with natural secular drift removed
(J2 nodal regression of Ω is several deg/day in LEO — far larger than any burn).
`local_jump` does a two-sided local-linear fit and reads the step across the gap.

- Part A — synthetic round-trip: inject known Δv (pure in-track / cross-track /
  radial) at LEO/GEO regimes, run forward → add element noise (σ from the Shorten
  quiet gaps, same detrended metric) → inverse; report recovery error + type
  accuracy vs magnitude. Establishes per-component accuracy and failure modes.
- Part B — empirical magnitude cross-check: run the inverse on real labelled-
  maneuver element jumps for one LEO (Jason-2) and one GEO (Fengyun-2F) satellite;
  check the Δv magnitudes/types are physically sensible vs literature ranges.
  (Shorten has no Δv, so this is a magnitude sanity check, not accuracy.)

Data: Shorten benchmark (dev-only oracle, V2) — `--data-dir <…>/processed_files`.
Forward model = linearised circular-orbit approximation; #14 implements full Gauss.
The inversion operates on SGP4 *mean* element changes (the step a TLE detector sees).

Run:  python3 v4_dv_inversion.py --data-dir <Shorten processed_files>
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re

import numpy as np

MU = 398600.4418  # km^3/s^2
EPOCH0 = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
DEG = math.pi / 180.0


# ---------- orbital helpers ----------
def a_from_n(n_revday):
    n_rad = np.asarray(n_revday) * 2 * math.pi / 86400.0
    return (MU / n_rad ** 2) ** (1.0 / 3.0)


def v_circ(a):
    return math.sqrt(MU / a)


def n_rad_of(a):
    return math.sqrt(MU / a ** 3)


def invert(da, di, dRA, de, a, i):
    """element jumps (da km, di rad, dRA rad, de) -> (dvT, dvW, dvR, |dv|, type) in m/s."""
    v, nrad = v_circ(a), n_rad_of(a)
    dvT = nrad * da / 2.0
    dvW = v * math.sqrt(di ** 2 + (math.sin(i) * dRA) ** 2)
    dvR = math.sqrt(max(0.0, (v * de) ** 2 - (2.0 * dvT) ** 2))
    comp = {"in-track": abs(dvT), "cross-track": abs(dvW), "radial": abs(dvR)}
    dv = math.sqrt(dvR ** 2 + dvT ** 2 + dvW ** 2)
    return dvT * 1e3, dvW * 1e3, dvR * 1e3, dv * 1e3, max(comp, key=comp.get)


def forward(dvR, dvT, dvW, a, i, u):
    """inject Δv (km/s) -> element jumps (km, rad, rad, -)."""
    v, nrad = v_circ(a), n_rad_of(a)
    return (2.0 * dvT / nrad,
            dvW * math.cos(u) / v,
            dvW * math.sin(u) / (v * math.sin(i)),
            math.sqrt((2.0 * dvT) ** 2 + dvR ** 2) / v)


# ---------- parsing ----------
def tle_epoch_days(field):
    yy = int(field[:2]); year = 2000 + yy if yy < 57 else 1900 + yy
    d = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=float(field[2:]) - 1.0)
    return (d - EPOCH0).total_seconds() / 86400.0


def parse_tle_full(path):
    t, n, inc, ra, ec = [], [], [], [], []
    lines = [ln.rstrip("\n") for ln in open(path) if ln.strip()]
    for j in range(len(lines) - 1):
        l1, l2 = lines[j], lines[j + 1]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                t.append(tle_epoch_days(l1[18:32])); n.append(float(l2[52:63]))
                inc.append(float(l2[8:16])); ra.append(float(l2[17:25]))
                ec.append(float("0." + l2[26:33].strip()))
            except ValueError:
                continue
    t = np.array(t); order = np.argsort(t)
    t, n, inc, ra, ec = (np.array(x)[order] for x in (t, n, inc, ra, ec))
    keep = np.concatenate(([True], np.diff(t) > 1e-9))
    return t[keep], n[keep], inc[keep], ra[keep], ec[keep]


def parse_man(path):
    out = []
    for line in open(path):
        m = re.match(r"\s*-\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", line)
        if m:
            d = dt.datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}").replace(tzinfo=dt.timezone.utc)
            out.append((d - EPOCH0).total_seconds() / 86400.0)
    return np.array(sorted(out))


def elements(path):
    """-> dict of time + per-element arrays with Ω unwrapped, ready for local_jump."""
    t, n, inc, ra, ec = parse_tle_full(path)
    return dict(t=t, a=a_from_n(n), i=inc * DEG, RA=np.unwrap(ra * DEG), e=ec,
                a_mean=float(np.median(a_from_n(n))), i_mean=float(np.median(inc) * DEG),
                n0=float(n[0]))


def local_jump(t, x, idx, k=4):
    """Two-sided local-linear step in x across the gap [idx-1, idx] (removes secular drift)."""
    if idx < k or idx > len(t) - k:
        return float("nan")
    tg = 0.5 * (t[idx - 1] + t[idx])
    sp, ip = np.polyfit(t[idx - k:idx], x[idx - k:idx], 1)
    sa, ia = np.polyfit(t[idx:idx + k], x[idx:idx + k], 1)
    return (sa * tg + ia) - (sp * tg + ip)


# ---------- empirical noise (detrended quiet-gap scatter) ----------
def quiet_sigma(path):
    el = elements(path)
    t, mans = el["t"], parse_man(path.replace(os.path.basename(path), f"manoeuvres_{os.path.basename(path)[:-4]}.yaml"))
    man_gap = np.array([np.any((mans >= t[g]) & (mans < t[g + 1])) for g in range(len(t) - 1)])
    rstd = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    out = {"a_km": el["a_mean"], "i_rad": el["i_mean"], "n0": el["n0"]}
    for key, col in (("a", "a"), ("i", "i"), ("RA", "RA"), ("e", "e")):
        j = np.array([local_jump(t, el[col], g + 1) for g in range(len(t) - 1) if not man_gap[g]])
        j = j[~np.isnan(j)]
        out[key] = float(rstd(j))
    return out


# ---------- Part A ----------
def part_a(sig_leo, sig_geo):
    rng = np.random.default_rng(0)
    print("== Part A: synthetic round-trip (median |Δv| recovery error vs injected Δv) ==")
    comps = {"in-track": (0, 1, 0), "cross-track": (0, 0, 1), "radial": (1, 0, 0)}
    for rk, s in (("LEO", sig_leo), ("GEO", sig_geo)):
        a, i = s["a_km"], s["i_rad"]
        print(f"\n  {rk} (a={a:.0f} km | noise σ: Δa={s['a']*1e3:.2f} m, Δi={s['i']/DEG*1e3:.2f} mdeg, "
              f"ΔΩ={s['RA']/DEG*1e3:.2f} mdeg, Δe={s['e']:.1e})")
        print(f"    {'inject':>8}{'in-track':>10}{'cross-trk':>11}{'radial':>9}{'type-acc':>10}")
        for mag_ms in (0.002, 0.01, 0.05, 0.2, 1.0, 5.0):  # m/s, spanning the floor
            mag = mag_ms / 1e3  # m/s -> km/s
            errs = {c: [] for c in comps}; correct = 0; trials = 300
            for _ in range(trials):
                c = list(comps)[rng.integers(3)]
                dvR, dvT, dvW = (mag * x for x in comps[c])
                u = rng.uniform(0, 2 * math.pi)
                da, di, dRA, de = forward(dvR, dvT, dvW, a, i, u)
                da += rng.normal(0, s["a"]); di += rng.normal(0, s["i"])
                dRA += rng.normal(0, s["RA"]); de += rng.normal(0, s["e"])
                _, _, _, dv_h, typ = invert(da, di, dRA, de, a, i)
                errs[c].append(abs(dv_h - mag_ms) / mag_ms)
                correct += (typ == c)
            cells = {c: f"{np.median(errs[c]) * 100:.0f}%" for c in comps}
            print(f"    {mag_ms:>6g}m/s{cells['in-track']:>10}{cells['cross-track']:>11}"
                  f"{cells['radial']:>9}{correct / trials * 100:>9.0f}%")
    print("\n  type-acc = correct dominant-component classification across the 3 injected types.")


# ---------- Part B ----------
def part_b(data_dir):
    print("\n== Part B: empirical Δv magnitudes on real labelled maneuvers (detrended) ==")
    for name, lit in (("Jason-2", "LEO altimetry maintenance ~mm/s–cm/s, in-track"),
                      ("Fengyun-2F", "GEO E-W ~mm/s–cm/s in-track; N-S ~1–2 m/s cross-track")):
        tle = os.path.join(data_dir, f"{name}.tle")
        if not os.path.exists(tle):
            continue
        el = elements(tle); t = el["t"]
        mans = parse_man(os.path.join(data_dir, f"manoeuvres_{name}.yaml"))
        dvs, types = [], {"in-track": 0, "cross-track": 0, "radial": 0}
        for m in mans:
            idx = int(np.searchsorted(t, m))
            da = local_jump(t, el["a"], idx); di = local_jump(t, el["i"], idx)
            dRA = local_jump(t, el["RA"], idx); de = local_jump(t, el["e"], idx)
            if any(math.isnan(x) for x in (da, di, dRA, de)):
                continue
            _, _, _, dv, typ = invert(da, di, dRA, de, el["a_mean"], el["i_mean"])
            dvs.append(dv); types[typ] += 1
        dvs = np.array(dvs); klass = "GEO" if el["n0"] < 2 else "LEO"
        print(f"\n  {name} ({klass}, {len(dvs)} maneuvers): expect {lit}")
        print(f"    Δv estimate  median={np.median(dvs):.4f}  p10={np.percentile(dvs,10):.4f}  "
              f"p90={np.percentile(dvs,90):.3f}  max={dvs.max():.2f}  (m/s)")
        print(f"    dominant-type counts: {types}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("SHORTEN_DIR", ""))
    args = ap.parse_args()
    if not args.data_dir or not os.path.isdir(args.data_dir):
        raise SystemExit("provide --data-dir <Shorten processed_files/>")
    sig_leo = quiet_sigma(os.path.join(args.data_dir, "Jason-2.tle"))
    sig_geo = quiet_sigma(os.path.join(args.data_dir, "Fengyun-2F.tle"))
    part_a(sig_leo, sig_geo)
    part_b(args.data_dir)
    print("\nForward = linearised circular-orbit impulsive (Gauss) approx; inversion on SGP4 mean")
    print("elements. Element jumps are detrended (J2 nodal drift removed). Full Gauss eqns -> #14.")


if __name__ == "__main__":
    main()
