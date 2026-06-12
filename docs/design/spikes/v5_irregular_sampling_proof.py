#!/usr/bin/env python3
"""V5 spike — irregular-sampling sequence-model input encoding (stdlib + numpy).

Prototypes and compares the three candidate encodings of the irregular TLE
cadence the v0.2 sequence models (BiLSTM, transformer) will consume, and frozen
as decision D11:

  1. resample-to-regular-grid  — linear interpolation onto a uniform daily grid
                                 plus a sampling mask;
  2. time-encoded deltas       — (Δt, Δelement) per inter-elset gap, no interp;
  3. continuous-time / time2vec — a bounded (clipped, periodic) encoding of Δt.

Each is judged on the two properties the feature layer needs (issue DoD):

  (a) LEAK — does the *timing channel alone* separate maneuver gaps from quiet
      gaps? Post-maneuver re-acquisition gaps run long, so a raw Δt (or a
      resample sampling-mask run-length) lets a model flag a maneuver by gap
      length without ever looking at the elements. Measured as the rank-AUC of
      a timing-only classifier; 0.5 = chance = no leak.

  (b) SIGNAL — with timing held constant, does the *element-delta content*
      recover the maneuvers, and specifically the above-floor ones (V3/D4)? A
      cross-fit LDA score over the detrended element-delta vector, reported as
      AUC and as recall@1-FA/sat-yr over the above-floor population, vs. the
      mean-motion-only lower bound V3 established.

Data: the Shorten benchmark (15 satellites — 10 LEO altimetry, 5 GEO Fengyun —
TLE history + ground-truth maneuver timestamps), a dev-only oracle (unlicensed,
see the V2 spike) analysed locally and never redistributed. Mean elements are
parsed straight from the TLE lines; no sgp4 needed. Get it with:

    git clone https://github.com/dpshorten/TLE_observation_benchmark_dataset

and point --data-dir at its `processed_files/` directory.

Run:  python3 v5_irregular_sampling_proof.py --data-dir <path>/processed_files
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

# V3/D4 per-class in-track detectability floor (m/s) used to define the
# "above-floor" (physically detectable) maneuver population for the signal test.
FLOOR_DV = {"LEO": 0.03, "GEO": 0.11}
CLIP_CAP_DAYS = 2.5  # the time2vec saturation cap used for the leak mitigation


# --------------------------------------------------------------------------- #
# Parsing — mean elements straight from the TLE (no sgp4)                      #
# --------------------------------------------------------------------------- #
def tle_epoch_to_days(field: str) -> float:
    """TLE line-1 epoch 'YYDDD.DDDDDDDD' -> days since 2000-01-01 UTC."""
    yy = int(field[:2])
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(field[2:])
    d = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=doy - 1.0)
    return (d - EPOCH0).total_seconds() / 86400.0


def parse_tle(path: str) -> dict[str, np.ndarray]:
    """Return epoch-sorted, duplicate-pruned mean-element channels."""
    t, inc, raan, ecc, argp, manom, mm = [], [], [], [], [], [], []
    lines = [ln.rstrip("\n") for ln in open(path) if ln.strip()]
    for i in range(len(lines) - 1):
        l1, l2 = lines[i], lines[i + 1]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        try:
            t.append(tle_epoch_to_days(l1[18:32]))
            inc.append(float(l2[8:16]))
            raan.append(float(l2[17:25]))
            ecc.append(float("0." + l2[26:33].strip()))
            argp.append(float(l2[34:42]))
            manom.append(float(l2[43:51]))
            mm.append(float(l2[52:63]))
        except ValueError:
            continue
    t = np.array(t)
    order = np.argsort(t)
    keep = np.concatenate(([True], np.diff(t[order]) > 1e-9))  # drop dup epochs
    idx = order[keep]
    n = np.array(mm)[idx]
    n_rad = n * 2.0 * math.pi / 86400.0
    a = (MU / n_rad**2) ** (1.0 / 3.0)  # km, Kozai-mean
    return dict(
        t=t[idx], n=n, a=a,
        e=np.array(ecc)[idx], i=np.array(inc)[idx],
        raan=np.array(raan)[idx], argp=np.array(argp)[idx], m=np.array(manom)[idx],
    )


def parse_maneuvers(path: str) -> np.ndarray:
    out = []
    for line in open(path):
        m = re.match(r"\s*-\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", line)
        if m:
            d = dt.datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}").replace(tzinfo=dt.timezone.utc)
            out.append((d - EPOCH0).total_seconds() / 86400.0)
    return np.array(sorted(out))


# --------------------------------------------------------------------------- #
# Detrending and per-gap features                                             #
# --------------------------------------------------------------------------- #
def moving_median(x: np.ndarray, w: int) -> np.ndarray:
    half = w // 2
    pad = np.pad(x, half, mode="edge")
    return np.array([np.median(pad[i:i + w]) for i in range(len(x))])


def detrend(x: np.ndarray, t: np.ndarray, angle: bool, w: int = 21) -> np.ndarray:
    """Residual after removing secular drift. Angles are unwrapped first (J2 nodal
    regression / apsidal precession are deg/day, far larger than any burn — V4)."""
    if angle:
        x = np.degrees(np.unwrap(np.radians(x)))
    return x - moving_median(x, w)


def level_shift(resid: np.ndarray, i: int, k: int = 5) -> float:
    """Signed level shift of a residual across gap [t_i, t_{i+1}): median of the k
    elsets after minus the k before (suppresses single-elset noise)."""
    pre, post = resid[max(0, i - k + 1):i + 1], resid[i + 1:i + 1 + k]
    if len(pre) < 2 or len(post) < 2:
        return np.nan
    return float(np.median(post) - np.median(pre))


def dv_from_dn(dn: float, n_revday: float, a_km: float) -> float:
    """Vis-viva in-track |Δv| (m/s) for a mean-motion step dn (rev/day)."""
    da = (2.0 / 3.0) * a_km * (abs(dn) / n_revday)
    n_rad = n_revday * 2.0 * math.pi / 86400.0
    return (n_rad * da / 2.0) * 1000.0


def daily_regularize(t: np.ndarray) -> np.ndarray:
    """Indices of one elset per UTC day (the first) — the #37 cadence regularization,
    which removes the sub-daily-burst structure from the gap series."""
    day = np.floor(t).astype(np.int64)
    keep = np.concatenate(([True], np.diff(day) > 0))
    return np.where(keep)[0]


# --------------------------------------------------------------------------- #
# Metrics — rank-AUC and a deterministic cross-fit LDA                         #
# --------------------------------------------------------------------------- #
def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney rank-AUC with tie-averaged ranks."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    s = score[order]
    r = np.arange(1, len(score) + 1, dtype=float)
    # average ranks over ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = r
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def separability(score: np.ndarray, y: np.ndarray) -> float:
    """Direction-agnostic AUC in [0.5, 1]: how well the feature(s) split the classes."""
    a = auc(score, y)
    return float(max(a, 1.0 - a)) if not math.isnan(a) else float("nan")


def lda_crossfit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """2-fold (even/odd) cross-fit regularised-LDA projection — out-of-fold scores,
    so the reported separability isn't inflated by in-sample fitting. Deterministic."""
    X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12)
    X = np.nan_to_num(X)
    out = np.full(len(y), np.nan)
    folds = (np.arange(len(y)) % 2)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        Xp, Xn = X[tr & (y == 1)], X[tr & (y == 0)]
        if len(Xp) < 2 or len(Xn) < 2:
            continue
        mu = Xp.mean(0) - Xn.mean(0)
        sw = np.cov(Xp.T) * (len(Xp) - 1) + np.cov(Xn.T) * (len(Xn) - 1)
        sw = np.atleast_2d(sw) / (len(Xp) + len(Xn) - 2)
        w = np.linalg.solve(sw + lam * np.eye(sw.shape[0]), mu)
        out[te] = X[te] @ w
    return out


# --------------------------------------------------------------------------- #
# Per-satellite encoding build                                                #
# --------------------------------------------------------------------------- #
def build(series: dict[str, np.ndarray], mans: np.ndarray, klass: str) -> dict | None:
    t = series["t"]
    if len(t) < 80 or len(mans) < 3:
        return None
    n_mean = float(np.median(series["n"]))
    a_km = float(np.median(series["a"]))

    # detrended residual per channel (angles unwrapped) — the element content
    resid = {
        "a": detrend(series["a"], t, angle=False),
        "e": detrend(series["e"], t, angle=False),
        "i": detrend(series["i"], t, angle=True),
        "raan": detrend(series["raan"], t, angle=True),
    }
    # eccentricity vector (h, k) sidesteps the argp/M wrap for the radial channel
    h = series["e"] * np.cos(np.radians(series["argp"]))
    kk = series["e"] * np.sin(np.radians(series["argp"]))
    resid["h"] = detrend(h, t, angle=False)
    resid["k"] = detrend(kk, t, angle=False)

    ng = len(t) - 1
    dt_gap = np.diff(t)
    man_gap = np.array([np.any((mans >= t[i]) & (mans < t[i + 1])) for i in range(ng)])

    # element-delta vector per gap (encoding 2/3 share this — timing is what differs).
    # The maneuver signal is in the *size* of the discontinuity, not its sign (a burn
    # can raise or lower a / i / Ω), so the feature is the magnitude of each level shift.
    chans = ["a", "e", "i", "raan", "h", "k"]
    elem_delta = np.abs(np.column_stack([[level_shift(resid[c], i) for i in range(ng)] for c in chans]))

    # mean-motion |Δv| per gap → above-floor mask (V3 floor)
    n_resid = detrend(series["n"], t, angle=False)
    dn = np.array([level_shift(n_resid, i) for i in range(ng)])
    dv_mm = np.array([dv_from_dn(d, n_mean, a_km) if not math.isnan(d) else np.nan for d in dn])

    # encoding 1 — resample to a uniform daily grid, linear interp + sampling mask
    grid = np.arange(math.floor(t[0]), math.ceil(t[-1]) + 1.0, 1.0)
    interp = {c: np.interp(grid, t, series[c]) for c in ("a", "e", "i", "raan")}
    interp_resid = {c: interp[c] - moving_median(interp[c], 21) for c in interp}
    # nearest grid index for each gap boundary; interpolated level shift across the gap
    gi = np.clip(np.searchsorted(grid, t), 0, len(grid) - 1)
    rs_delta = np.full((ng, len(interp)), np.nan)
    for j, c in enumerate(("a", "e", "i", "raan")):
        ir = interp_resid[c]
        for i in range(ng):
            lo, hi = gi[i], gi[i + 1]
            pre, post = ir[max(0, lo - 5):lo + 1], ir[hi:hi + 6]
            if len(pre) >= 2 and len(post) >= 2:
                rs_delta[i, j] = abs(np.median(post) - np.median(pre))
    # sampling-mask run-length in the gap = number of empty grid days it spans
    mask_runlen = np.array([max(0.0, gi[i + 1] - gi[i] - 1) for i in range(ng)], float)

    # timing representations for the leak test
    dt_raw = dt_gap.copy()
    dt_clip = np.minimum(dt_gap, CLIP_CAP_DAYS)  # time2vec saturation cap
    # daily-regularized timing series (one elset per UTC day — the #37 regularization):
    # its own gap series + maneuver-gap labels, to test whether collapsing the
    # sub-daily-burst structure shrinks the timing leak.
    reg = daily_regularize(t)
    treg = t[reg]
    dt_reg = np.diff(treg)
    y_reg = np.array([np.any((mans >= treg[j]) & (mans < treg[j + 1])) for j in range(len(treg) - 1)]).astype(int)

    valid = ~np.isnan(elem_delta).any(1) & ~np.isnan(rs_delta).any(1) & ~np.isnan(dv_mm)
    span_yr = (t[-1] - t[0]) / 365.25
    return dict(
        klass=klass, span_yr=span_yr, ng=int(valid.sum()), n_man=int(man_gap.sum()),
        y=man_gap[valid].astype(int),
        elem_delta=elem_delta[valid], rs_delta=rs_delta[valid],
        dt_raw=dt_raw[valid], dt_clip=dt_clip[valid], mask_runlen=mask_runlen[valid],
        dt_reg=dt_reg, y_reg=y_reg, dv_mm=dv_mm[valid],
        above_floor=(dv_mm[valid] > FLOOR_DV[klass]) & (man_gap[valid] == 1),
        quiet_per_yr=float((man_gap[valid] == 0).sum()) / span_yr,
    )


def recall_at_fa(score: np.ndarray, y: np.ndarray, above: np.ndarray,
                 quiet_per_yr: float, fa: float = 1.0) -> float:
    """Recall over above-floor maneuvers at the score threshold giving `fa` quiet
    false-alarms per satellite-year (the V3/D4 operating point)."""
    quiet = score[y == 0]
    if len(quiet) == 0 or above.sum() == 0:
        return float("nan")
    if quiet_per_yr <= fa:
        thr = float(np.min(quiet))
    else:
        thr = float(np.percentile(quiet, 100.0 * (1.0 - fa / quiet_per_yr)))
    return float(np.mean(score[above] > thr))


# --------------------------------------------------------------------------- #
# Aggregate over a class and report                                           #
# --------------------------------------------------------------------------- #
def pooled(rows: list[dict], key: str) -> np.ndarray:
    return np.concatenate([r[key] for r in rows])


def report_class(klass: str, rows: list[dict]) -> None:
    if not rows:
        return
    y = pooled(rows, "y")
    elem = np.vstack([r["elem_delta"] for r in rows])
    rs = np.vstack([r["rs_delta"] for r in rows])
    above = pooled(rows, "above_floor")
    qpy = float(np.mean([r["quiet_per_yr"] for r in rows]))

    # SIGNAL — does the element content recover maneuvers (timing neutralised)?
    # in-track = the |Δa| channel alone (where the in-track / E-W signal lives, and
    # the channel interpolation most directly smears); multi-elem = cross-fit LDA over
    # the full element-delta vector (adds GEO cross-track; on LEO the arc-second-noisy
    # node/ecc channels dilute it — the #37 finding, an argument about the *readout*,
    # not the encoding).
    rs_intrack = separability(rs[:, 0], y)
    dl_intrack = separability(elem[:, 0], y)
    rs_multi = separability(lda_crossfit(rs, y), y)
    dl_multi = separability(lda_crossfit(elem, y), y)

    # LEAK — each encoding's native timing channel, alone
    leak_resample = separability(pooled(rows, "mask_runlen"), y)
    leak_deltas = separability(pooled(rows, "dt_raw"), y)
    leak_time2vec = separability(pooled(rows, "dt_clip"), y)
    leak_reg = separability(pooled(rows, "dt_reg"), pooled(rows, "y_reg"))

    # recall @ 1 FA/sat-yr over the above-floor population, in-track channel per encoding
    rec_rs = recall_at_fa(np.concatenate([r["rs_delta"][:, 0] for r in rows]), y, above, qpy)
    rec_dl = recall_at_fa(np.concatenate([r["elem_delta"][:, 0] for r in rows]), y, above, qpy)

    nman = int(y.sum())
    print(f"\n=== {klass}  (sats={len(rows)}, gaps={len(y)}, maneuver-gaps={nman}, "
          f"above-floor={int(above.sum())}) ===")
    print(f"{'encoding':24}{'in-track':>10}{'multi-elem':>12}{'leak':>8}{'rec@1FA':>9}")
    print("-" * 63)
    print(f"{'resample-to-grid':24}{rs_intrack:>10.3f}{rs_multi:>12.3f}{leak_resample:>8.3f}{rec_rs:>9.2f}")
    print(f"{'time-encoded deltas':24}{dl_intrack:>10.3f}{dl_multi:>12.3f}{leak_deltas:>8.3f}{rec_dl:>9.2f}")
    print(f"{'continuous (time2vec)':24}{dl_intrack:>10.3f}{dl_multi:>12.3f}{leak_time2vec:>8.3f}{rec_dl:>9.2f}")
    print(f"  timing leak: per-elset Δt {leak_deltas:.3f} | daily-regularized Δt {leak_reg:.3f} "
          f"| resample mask {leak_resample:.3f}")
    print("  note: rank-AUC is invariant to monotonic Δt transforms (clip/log/time2vec),")
    print("        so the timing leak is structural — it is not encodable away by clipping.")


def classify(n_mean: float) -> str:
    return "GEO" if n_mean < 2.0 else "LEO"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("SHORTEN_DIR", ""),
                    help="path to the Shorten benchmark processed_files/ directory")
    args = ap.parse_args()
    if not args.data_dir or not os.path.isdir(args.data_dir):
        raise SystemExit("provide --data-dir <Shorten processed_files/> (see module docstring)")

    by_class: dict[str, list[dict]] = {"LEO": [], "GEO": []}
    detail = []
    for tle in sorted(glob.glob(os.path.join(args.data_dir, "*.tle"))):
        name = os.path.basename(tle)[:-4]
        man = os.path.join(args.data_dir, f"manoeuvres_{name}.yaml")
        if not os.path.exists(man):
            continue
        series = parse_tle(tle)
        klass = classify(float(np.median(series["n"])))
        row = build(series, parse_maneuvers(man), klass)
        if row:
            by_class[klass].append(row)
            detail.append((name, klass, row))

    print("Per-satellite coverage")
    print(f"{'satellite':14}{'cls':4}{'yr':>6}{'gaps':>7}{'man-gaps':>10}{'above-floor':>13}")
    print("-" * 54)
    for name, klass, r in detail:
        print(f"{name:14}{klass:4}{r['span_yr']:>6.1f}{r['ng']:>7}"
              f"{r['n_man']:>10}{int(r['above_floor'].sum()):>13}")

    for klass in ("LEO", "GEO"):
        report_class(klass, by_class[klass])

    print("\nin-track / multi-elem = cross-fit signal separability (AUC) from the element")
    print("       content with timing neutralised: the |Δa| channel alone vs. the full")
    print("       element-delta vector. Higher = the encoding preserves the signal.")
    print("leak = direction-agnostic rank-AUC of the encoding's timing channel ALONE vs.")
    print("       the maneuver-gap label (0.5 = chance = no leak).")
    print("rec@1FA = recall over the above-floor population at 1 quiet FA/sat-year (in-track).")


if __name__ == "__main__":
    main()
