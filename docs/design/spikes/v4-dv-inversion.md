# V4 spike — Δv-inversion accuracy

**Status:** findings + recommendation, feeding **D5** (Δv-inversion method, type-classification rule,
per-class tolerance, failure modes) and gating the `physics.py` implementation (#14). Validated by a
synthetic forward/inverse round-trip + an empirical magnitude cross-check on real maneuvers.

## Question

Does the vis-viva + Gauss-variational inversion reproduce a maneuver's Δv and type from a detected
mean-element jump, within what tolerance, per class; what is the type-classification rule; and what are
the failure modes?

## Method

[`v4_dv_inversion.py`](v4_dv_inversion.py) (stdlib + numpy) implements the linearised circular-orbit
impulsive (Gauss) relations:

```
in-track    Δv_T = n_rad · Δa / 2
cross-track Δv_W = v · sqrt(Δi² + (sin i · ΔΩ)²)
radial      Δv_R = sqrt(max(0, (v·Δe)² − (2 Δv_T)²))    # eccentricity change beyond transverse
|Δv| = sqrt(Δv_R² + Δv_T² + Δv_W²);   type = argmax component
```

- **Part A — synthetic round-trip:** inject a known Δv (pure in-track / cross-track / radial) at LEO and
  GEO regimes, run forward → add element-measurement noise (σ measured from the Shorten quiet gaps) →
  inverse, and measure |Δv| recovery error and type-classification accuracy vs magnitude.
- **Part B — empirical magnitude cross-check:** run the inverse on real labelled-maneuver element jumps
  for one LEO (Jason-2) and one GEO (Fengyun-2F) satellite (Shorten data, dev oracle), and check the Δv
  magnitudes and types are physically sensible vs literature.

Element jumps are the **anomalous step with natural secular drift removed** (a two-sided local-linear
fit) — J2 nodal regression of Ω is several deg/day in LEO, far larger than any burn, and *must* be
detrended or it reads as a bogus ~hundreds-of-m/s cross-track Δv (a failure mode found and fixed here).

## Findings

### 1. The inversion is correct above the detectability floor (Part A)

| inject \|Δv\| | LEO recovery err / type-acc | GEO recovery err / type-acc |
|---|---|---|
| 2 mm/s | ~800% / 34% (chance) | ~1500% / 37% |
| 10 mm/s | ~100% / 47% | ~260% / 42% |
| **50 mm/s** | **~15% / 98%** | ~32% / 80% |
| 0.2 m/s | ~3% / 100% | ~6% / 100% |
| ≥ 1 m/s | ~0% / 100% | ~0% / 100% |

Above the per-class floor the inversion recovers \|Δv\| to **a few percent and classifies the type with
100% accuracy**; below it, both collapse to noise. The **reliable-inversion floor is ~5 cm/s (LEO) and
~0.1–0.2 m/s (GEO)** — the *same* floor V3 found for detection. So one physical floor governs both
detection *and* Δv/type estimation.

### 2. Type-classification rule

- **in-track** ↔ a step in semi-major axis (mean motion), Δa;
- **cross-track** ↔ a step in inclination and/or node, Δi & ΔΩ;
- **radial** ↔ an eccentricity-vector change beyond what the transverse burn explains, Δe.

Reliable **above** the per-class floor; near/below it, classification is noise-dominated.

### 3. Real maneuvers invert to physically sensible Δv (Part B)

- **Jason-2 (LEO, 88 maneuvers):** median **0.046 m/s** (p10 0.010, p90 1.76, max 7.0) — cm/s-scale,
  consistent with altimetry orbit-maintenance (~mm/s–cm/s). Type counts are mixed (in-track 12 /
  cross-track 59 / radial 17) because most of these maneuvers sit *near the floor*, where the
  ~cm/s cross-track noise rivals the in-track signal — type is unreliable there (consistent with Part A).
- **Fengyun-2F (GEO, 67 maneuvers):** median **0.36 m/s** (p10 0.04, p90 1.9, max ~40) with an in-track
  (21) + cross-track (40) mix — **exactly the GEO station-keeping signature**: N-S (cross-track, ~1–2 m/s)
  + E-W (in-track, cm/s), plus a ~40 m/s relocation outlier. Matches the literature (GEO N-S ≈ 50 m/s/yr
  ≈ ~2 m/s/maneuver; E-W ≈ mm/s–cm/s).

### 4. Failure modes

- **Sub-floor maneuvers** (< ~cm/s LEO, < ~0.1 m/s GEO): not reliably invertible — Δv error explodes and
  type → chance. (Same population V3 flagged as undetectable.)
- **Radial is the weakest component** — recovered by subtracting the transverse contribution from the
  eccentricity change, which is ill-conditioned when the transverse burn dominates; often best left
  unreported.
- **Natural secular drift must be removed** before inversion (J2 nodal regression especially), or it
  produces a large spurious cross-track Δv.
- **Cross-track noise floor** (~cm/s in LEO) limits type classification of near-floor maneuvers even when
  the magnitude is roughly right.

## Recommendation (→ D5)

- **D5.1 — method.** Vis-viva in-track (from Δa) + Gauss cross-track (from Δi, ΔΩ) + radial from the
  residual eccentricity change; `|Δv|` and `type = argmax component`. `#14` implements the **full Gauss
  equations** (not just the linearised circular form) and **detrends secular drift** before inverting.
- **D5.2 — type rule.** As in Finding 2; emit a type only **above the per-object floor**.
- **D5.3 — Δv tolerance.** Report \|Δv\| with a **~±25% band for above-floor maneuvers** (tightening to
  <5% well above the floor); flag estimates within ~2× the floor as low-confidence; **do not report a
  Δv/type below the floor**. Treat **radial** as low-confidence by default.
- **D5.4 — failure modes** documented (Finding 4) — the detector (#15) and the Δv column inherit them.
- **D5.5 — accuracy vs published Δv** is validated here by *method correctness* (synthetic) + *magnitude
  realism* (matches literature ranges on real maneuvers). The **quantitative error-vs-published-Δv table
  is deferred to the data-layer work (#10)**, where the DORIS/IDS burn Δv (V2) provides per-maneuver
  ground truth; the protocol is: ingest DORIS Δv, run the inverse on the matching TLE jumps, tabulate
  per-class error above the floor.

## Reproducibility

`v4_dv_inversion.py` is stdlib + numpy, deterministic across runs (fixed RNG seed; data parsed from the
TLE). The Shorten data is fetched separately and not committed (unlicensed dev oracle, V2):
`git clone https://github.com/dpshorten/TLE_observation_benchmark_dataset`, then `--data-dir
<…>/processed_files`.

## Caveats / open items

- Forward model is the **linearised circular-orbit** approximation; the eccentric/zonal terms are
  small for these near-circular orbits but `#14` uses the full Gauss VOP.
- The inversion runs on SGP4 **mean** element changes — it estimates the Δv consistent with the
  mean-element step a TLE detector sees (not an osculating-state Δv).
- Empirical leg is a **magnitude/type sanity check** (Shorten has no Δv); absolute accuracy → #10 via
  DORIS Δv.
- Ratify D5 at the design freeze.

## References

- D. Vallado, *Fundamentals of Astrodynamics and Applications* — Gauss variational equations of motion
  (the forward/inverse relations used here).
- Lemmens & Krag (2014), TLE maneuver detection — <https://arc.aiaa.org/doi/10.2514/1.61300>.
- Orbital station-keeping Δv budgets (GEO N-S ≈ 50 m/s/yr; E-W ≈ mm/s–cm/s) —
  <https://en.wikipedia.org/wiki/Orbital_station-keeping>.
- Shorten et al. benchmark (real maneuver element jumps) —
  <https://github.com/dpshorten/TLE_observation_benchmark_dataset>.
- DORIS/IDS maneuver (Δv) files for the quantitative #10 validation — NASA CDDIS / IDS (see the V2 spike).
