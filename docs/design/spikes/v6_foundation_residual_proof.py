#!/usr/bin/env python3
"""V6 spike — foundation-model forecast-residual maneuver detector (stdlib + numpy + package).

Prototypes the detector recipe **D14** rests on: turn a time-series *forecaster* into a maneuver
detector by thresholding its forecast residual **per orbit class**, and feed the result straight
into the shipped benchmark scorer (D4 matching, D7 metrics). The foundation model is therefore a
drop-in replacement for the classical detector's hand-built quiet-dynamics model — nothing
downstream (the canonical schema D6, the D4 matcher, the D5 Δv/type inversion, the D7 scorer)
changes.

The forecaster here is a deterministic **robust drift-continuation stand-in** (forecast = last value
+ a per-object robust secular drift; per-object MAD scale), not Chronos/TimesFM: the spike proves the
*wiring and the per-class-threshold contract*, the property that does not depend on which forecaster
fills the slot. A foundation model only raises forecast quality — a stronger zero-shot
quiet-dynamics prior with calibrated predictive intervals — on the very same contract; its measured
zero-shot / fine-tuned recall lands on the v0.3 baseline's model card, exactly as V7 pinned the
single-GPU compute budget the runs later measure against. The mechanism the proof asserts:

  1. **The residual spikes at a maneuver and is quiet otherwise.** A burst steps an element (Δa for
     in-track, Δi for cross-track); a forecaster trained on quiet dynamics cannot anticipate the
     step, so the standardized one-step residual jumps far above its quiet-gap level.
  2. **A per-class threshold separates above-floor maneuvers from quiet gaps and from a below-floor
     maneuver** (V3/D4: detectability is per class; the floor is a per-class residual cutoff).
  3. **Thresholded residuals become canonical `Maneuver` records** (D6) — dominant channel → type
     (D5) — that the *real* `score()` matches and scores at the 1-FA/sat-year operating point (D4/D7).

stdlib + numpy + the installed package only; no network, no GPU, no torch. Deterministic across
runs (seeded noise, printed figures rounded), with fictional catalogue ids 9000x so the artifact
ships no redistributed TLEs (the V1/D2 no-redistribution practice).

Run:  python docs/design/spikes/v6_foundation_residual_proof.py   (or `uv run`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maneuver_detect.benchmark.matching import ScoredLabel
from maneuver_detect.benchmark.metrics import ObjectExposure
from maneuver_detect.benchmark.scoring import ScoreReport, score
from maneuver_detect.labels.labeller import LabelledInterval
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import Maneuver, ManeuverType

DAY = pd.Timedelta(days=1)
BASE = pd.Timestamp("2026-03-01T00:00:00", tz="UTC")

# Per-orbit-class residual-z detection threshold — the D4 per-class detectability floor expressed in
# standardized-residual units. This is the operating threshold the v0.3 baseline calibrates per
# class; here it is fixed so the proof shows the separation it buys.
CLASS_THRESHOLD: dict[OrbitClass, float] = {OrbitClass.LEO: 4.5, OrbitClass.GEO: 4.5}

NMS_DAYS = 3  # suppress secondary residual spikes within this many gaps of a stronger one
SEED = 60


# --------------------------------------------------------------------------------------------------
# Synthetic mean-element series: a quiet two-channel (a, i) signal with injected maneuver steps.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Injected:
    """A maneuver injected as a permanent element step appearing on inter-elset gap ``gap``."""

    gap: int  # the maneuver sits in the gap between elset ``gap`` and elset ``gap + 1``
    channel: str  # "a" (in-track) or "i" (cross-track)
    delta: float  # step magnitude in the channel's unit (km for a, deg for i)
    mtype: ManeuverType
    above_floor: bool
    delta_v: float | None


@dataclass(frozen=True)
class SyntheticObject:
    """A fictional object's quiet dynamics + the maneuvers stepped into its element series."""

    norad_id: int
    orbit_class: OrbitClass
    a0: float
    a_drift: float  # secular da/dt (km/day): atmospheric decay for LEO, ~0 for GEO
    a_noise: float  # per-elset a noise std (km)
    i0: float
    i_noise: float  # per-elset i noise std (deg)
    n_days: int
    maneuvers: tuple[Injected, ...]

    def series(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """The realised (a, i) series over ``n_days + 1`` daily elsets, with the steps applied."""
        days = np.arange(self.n_days + 1)
        a = self.a0 + self.a_drift * days + rng.normal(0.0, self.a_noise, days.size)
        i = self.i0 + rng.normal(0.0, self.i_noise, days.size)
        for man in self.maneuvers:
            # The step appears between elset ``gap`` and ``gap + 1`` and persists thereafter.
            mask = days > man.gap
            if man.channel == "a":
                a[mask] += man.delta
            else:
                i[mask] += man.delta
        return a, i


# --------------------------------------------------------------------------------------------------
# The forecast-residual detector — forecaster-agnostic; a local-linear stand-in fills the model slot.
# --------------------------------------------------------------------------------------------------


def _standardized_residuals(values: np.ndarray) -> np.ndarray:
    """Per-gap forecast residuals, standardized by a per-object robust predictive scale.

    The quiet-dynamics forecast is ``last value + a secular drift``, so the residual on a gap is its
    first-difference minus the drift. The drift is the **median** of the object's inter-elset
    first-differences and the predictive scale their robust **MAD (·1.4826)** — both estimated over
    the *whole* series, which is the D4 per-object floor calibration: a handful of maneuver steps are
    outliers the median/MAD shrug off, so the scale is stable (a small rolling window's MAD is itself
    noisy and randomly inflates an ordinary gap's z). A maneuver then lands as **one** large
    standardized residual on the gap it sits on; the shifted-but-quiet gaps after return to near
    zero. This is the shipped detector's detrend-then-look-at-the-per-gap-delta approach; a foundation
    model replaces the constant-drift prior with a learned conditional forecast on the same contract.
    Returns one z per gap: entry ``k`` is the residual for the gap between elset ``k`` and ``k + 1``.
    """
    diffs = np.diff(values)  # diffs[k] is the change across gap k (between elset k and k + 1)
    drift = float(np.median(diffs))
    scale = max(1.4826 * float(np.median(np.abs(diffs - drift))), 1e-9)
    return (diffs - drift) / scale


@dataclass(frozen=True)
class GapResidual:
    """The detector's per-gap evidence: the standardized residual and the dominant channel."""

    gap: int  # between elset ``gap`` and ``gap + 1``
    z_abs: float  # max |standardized residual| across channels
    mtype: ManeuverType  # dominant channel → D5 type


def gap_residuals(obj: SyntheticObject, rng: np.random.Generator) -> list[GapResidual]:
    """Run the forecaster on both channels and reduce to one (z, type) per gap."""
    a, i = obj.series(rng)
    z_a = _standardized_residuals(a)  # one z per gap, indexed by gap
    z_i = _standardized_residuals(i)
    out: list[GapResidual] = []
    for gap in range(z_a.size):
        za, zi = abs(float(z_a[gap])), abs(float(z_i[gap]))
        mtype = ManeuverType.IN_TRACK if za >= zi else ManeuverType.CROSS_TRACK
        out.append(GapResidual(gap=gap, z_abs=max(za, zi), mtype=mtype))
    return out


def detect(obj: SyntheticObject, rng: np.random.Generator) -> list[Maneuver]:
    """Threshold the per-gap residuals at the object's class floor, NMS, emit canonical records."""
    threshold = CLASS_THRESHOLD[obj.orbit_class]
    flagged = [g for g in gap_residuals(obj, rng) if g.z_abs >= threshold]
    # Non-maximum suppression: keep the strongest gap in each NMS_DAYS neighbourhood (a re-fit one
    # gap after a step can produce a weaker secondary spike — the classical detector suppresses it).
    kept: list[GapResidual] = []
    for g in sorted(flagged, key=lambda g: g.z_abs, reverse=True):
        if all(abs(g.gap - k.gap) > NMS_DAYS for k in kept):
            kept.append(g)
    detections: list[Maneuver] = []
    for g in sorted(kept, key=lambda g: g.gap):
        gap_start = BASE + int(g.gap) * DAY
        detections.append(
            Maneuver(
                epoch=gap_start + pd.Timedelta(hours=12),
                confidence=round(1.0 - math.exp(-g.z_abs / threshold), 6),
                type=g.mtype,
                delta_v_estimate=None,  # |Δv| / type via the unchanged D5 Gauss inversion (physics.py)
                norad_id=obj.norad_id,
                elset_epoch_before=gap_start,
                elset_epoch_after=gap_start + DAY,
            )
        )
    return detections


# --------------------------------------------------------------------------------------------------
# The held-out benchmark — labels + exposure the real scorer consumes (V7's construction).
# --------------------------------------------------------------------------------------------------


def scored_labels(obj: SyntheticObject) -> list[ScoredLabel]:
    """One held-out label per injected maneuver, on its bracketing gap with the D4 ±1 tolerance."""
    labels: list[ScoredLabel] = []
    for man in obj.maneuvers:
        gap_start = BASE + man.gap * DAY
        gap_end = gap_start + DAY
        labels.append(
            ScoredLabel(
                LabelledInterval(
                    norad_id=obj.norad_id,
                    epoch=gap_start + pd.Timedelta(hours=12),
                    elset_epoch_before=gap_start,
                    elset_epoch_after=gap_end,
                    tol_start=gap_start - DAY,  # ±1 adjacent gap (D4)
                    tol_end=gap_end + DAY,
                    maneuver_type=man.mtype,
                    delta_v=man.delta_v,
                    source="SYNTH",
                    source_ref=f"{obj.norad_id}:{man.gap}",
                    orbit_class=obj.orbit_class,
                ),
                above_floor=man.above_floor,
            )
        )
    return labels


def build_fleet() -> tuple[SyntheticObject, ...]:
    """A LEO and a GEO object, each with above- and below-floor maneuvers on disjoint gaps."""
    leo = SyntheticObject(
        norad_id=90001,
        orbit_class=OrbitClass.LEO,
        a0=7000.0,
        a_drift=-0.0025,
        a_noise=0.020,
        i0=98.6,
        i_noise=0.0015,
        n_days=44,
        maneuvers=(
            Injected(8, "a", 0.30, ManeuverType.IN_TRACK, True, 0.16),
            Injected(20, "i", 0.018, ManeuverType.CROSS_TRACK, True, 0.40),
            Injected(33, "a", 0.26, ManeuverType.IN_TRACK, True, 0.14),
            Injected(40, "a", 0.030, ManeuverType.IN_TRACK, False, 0.016),  # below floor
        ),
    )
    geo = SyntheticObject(
        norad_id=90002,
        orbit_class=OrbitClass.GEO,
        a0=42164.0,
        a_drift=0.0,
        a_noise=0.030,
        i0=0.05,
        i_noise=0.0012,
        n_days=44,
        maneuvers=(
            Injected(11, "i", 0.014, ManeuverType.CROSS_TRACK, True, 0.13),
            Injected(28, "a", 0.34, ManeuverType.IN_TRACK, True, 0.11),
        ),
    )
    return (leo, geo)


# --------------------------------------------------------------------------------------------------
# Evidence the mechanism works: the separation margin, then the real scorer's verdict.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Separation:
    """Per-object residual separation: above-floor maneuvers vs quiet gaps vs the below-floor one."""

    norad_id: int
    orbit_class: str
    threshold: float
    min_above_floor_z: float
    max_quiet_z: float
    below_floor_z: float | None


def separation(obj: SyntheticObject, rng: np.random.Generator) -> Separation:
    """Compare the standardized residual at maneuver gaps against quiet gaps for one object."""
    by_gap = {g.gap: g.z_abs for g in gap_residuals(obj, rng)}
    man_gaps = {m.gap: m for m in obj.maneuvers}
    above = [by_gap[m.gap] for m in obj.maneuvers if m.above_floor and m.gap in by_gap]
    below = [by_gap[m.gap] for m in obj.maneuvers if not m.above_floor and m.gap in by_gap]
    quiet = [z for gap, z in by_gap.items() if gap not in man_gaps]
    return Separation(
        norad_id=obj.norad_id,
        orbit_class=obj.orbit_class.value,
        threshold=CLASS_THRESHOLD[obj.orbit_class],
        min_above_floor_z=round(min(above), 2),
        max_quiet_z=round(max(quiet), 2),
        below_floor_z=round(below[0], 2) if below else None,
    )


def public_report(report: ScoreReport) -> dict[str, object]:
    """The headline the scorer returns: per-class above-floor recall at the operating point."""
    return {
        "operating_point_fa_per_sat_year": report.operating_point,
        "headline_recall_above_floor": {
            oc.value: report.headline()[oc] for oc in report.per_class
        },
    }


def run() -> tuple[list[Separation], dict[str, object], int, int]:
    """Detect on the fleet, measure the separation, and score against the held-out labels."""
    seps: list[Separation] = []
    detections: list[Maneuver] = []
    labels: list[ScoredLabel] = []
    exposure: list[ObjectExposure] = []
    for obj in build_fleet():
        # A fresh RNG seeded per object, so the separation analysis and the detector both see the
        # identical series and the whole pipeline is reproducible (D8).
        seps.append(separation(obj, np.random.default_rng(SEED + obj.norad_id)))
        detections += detect(obj, np.random.default_rng(SEED + obj.norad_id))
        labels += scored_labels(obj)
        exposure.append(ObjectExposure(obj.norad_id, obj.orbit_class, obj.n_days / 365.25))
    report = score(detections, labels, exposure)
    n_above_floor = sum(1 for label in labels if label.above_floor)
    return seps, public_report(report), len(detections), n_above_floor


def main() -> None:
    seps, result, n_detections, n_above_floor = run()

    # --- Assertions: the three claims D14 rests on ------------------------------------------------
    for sep in seps:
        assert sep.min_above_floor_z >= sep.threshold, f"above-floor maneuver missed: {sep}"
        assert sep.max_quiet_z < sep.threshold, f"quiet gap would false-alarm: {sep}"
        if sep.below_floor_z is not None:
            assert sep.below_floor_z < sep.threshold, f"below-floor maneuver flagged: {sep}"
    # Every above-floor maneuver recovered, nothing spurious emitted, at 1 FA/sat-year.
    assert n_detections == n_above_floor, f"emitted {n_detections}, expected {n_above_floor}"
    recalls = result["headline_recall_above_floor"]
    # LEO and GEO carry labels here; the scorer also lists MEO (no labels → null), which we skip.
    scored = [r for r in recalls.values() if r is not None]  # type: ignore[union-attr]
    assert scored and all(r == 1.0 for r in scored), f"above-floor recall below 1.0: {result}"
    # Determinism (D8): the whole pipeline re-runs byte-identically.
    again = run()
    assert json.dumps(again[1], sort_keys=True) == json.dumps(result, sort_keys=True)

    print("V6 — foundation-model forecast-residual detector (real scorer, stand-in forecaster)")
    print("=" * 82)
    print("\n[1] Residual separation per object (standardized one-step forecast residual):")
    print(f"    {'object':>8}  {'class':>5}  {'thr':>4}  {'min(man z)':>10}  "
          f"{'max(quiet z)':>12}  {'below-floor z':>13}")
    for sep in seps:
        below = "n/a" if sep.below_floor_z is None else f"{sep.below_floor_z:.2f}"
        print(f"    {sep.norad_id:>8}  {sep.orbit_class:>5}  {sep.threshold:>4.1f}  "
              f"{sep.min_above_floor_z:>10.2f}  {sep.max_quiet_z:>12.2f}  {below:>13}")
    print("\n    => above-floor maneuvers clear the per-class threshold; quiet gaps and the")
    print("       below-floor maneuver stay under it — the threshold separates them cleanly.")

    print("\n[2] Mechanism checks (all assert-backed, passed):")
    print("    - residual spikes at maneuvers, quiet elsewhere (margin above)")
    print("    - per-class threshold rejects quiet gaps and the below-floor maneuver")
    print(f"    - thresholded residuals → {n_detections} canonical records "
          f"(= {n_above_floor} above-floor labels, no false alarms)")
    print("    - scoring is byte-deterministic across runs (D8)")

    print("\n[3] Real scorer verdict on the emitted detections:")
    print(json.dumps(result, indent=2, default=str))
    print("\n    => the forecast-residual detector plugs into the shipped D4 matcher / D7 scorer")
    print("       unchanged and recovers the above-floor population at 1 FA/sat-year.")


if __name__ == "__main__":
    main()
