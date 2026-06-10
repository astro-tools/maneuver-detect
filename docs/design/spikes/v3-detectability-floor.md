# V3 spike — detectability floor + maneuver-to-elset-gap labelling granularity

**Status:** findings + recommendation, feeding **D4** (labelling granularity, detection-matching
tolerance, detectability floor, FPR unit/operating point) and informing the detector (#15) and benchmark
(#12/#13). Empirical analysis of public data + first principles + literature.

## Question

How does a maneuver appear in the TLE mean-element series given irregular cadence; what is the smallest
Δv reliably detectable per class; what detection-matching tolerance should the benchmark freeze; and what
is the false-alarm-rate unit and operating point?

## Method

Empirical analysis of the **Shorten benchmark** (15 satellites — 10 LEO altimetry, 5 GEO Fengyun — TLE
history + ground-truth maneuver timestamps; a dev-only oracle per V2, analysed locally, not
redistributed) via [`v3_detectability_analysis.py`](v3_detectability_analysis.py) (stdlib + numpy):
detrend mean motion (moving median), measure the **residual level-shift** across each inter-elset gap,
split gaps into labelled-maneuver vs. quiet, and at a chosen false-alarm rate read off the threshold,
the recall, and the **vis-viva Δv** the threshold corresponds to. Cross-checked against first principles
and the TLE-maneuver-detection literature (Lemmens & Krag 2014; Kelecy 2007).

## Findings

### 1. Observability and matching granularity

A maneuver is observable only as a **discontinuity between two consecutive elsets**, so the detection
target is "**this inter-elset interval contains a maneuver**", and the matching tolerance is a window in
elset-gaps/days, not seconds. Empirically the catalogue cadence is **≈ 1 day median** for both classes
(LEO 1.01 d, p90 1.33 d; GEO 1.03 d, p90 1.62 d).

### 2. Detectability floor (per class)

At an operating point of **1 quiet false-alarm per satellite-year**, the in-track Δv at the
mean-motion detection threshold is:

| Class | cadence (median / p90) | floor Δv @ 1 FA/sat-yr (median) | spread |
|---|---|---|---|
| **LEO** (altimetry) | 1.01 d / 1.33 d | ≈ **0.03 m/s** | well-behaved sats reach **cm/s** (Jason-1 0.03, Sentinel-3A 0.015, TOPEX 0.003); noisy-TLE outliers higher (Jason-3 ~4.9, Jason-2 ~0.8) |
| **GEO** (Fengyun) | 1.03 d / 1.62 d | ≈ **0.11 m/s** | 0.02–0.13 m/s |

This matches the literature: Lemmens & Krag (2014) report reliable LEO TLE detection "down to
delta-velocity magnitudes at the centimetre-per-second level or less", with the same σ-threshold ↔
false-alarm trade-off. The GEO floor is higher in Δv (larger semi-major axis), but GEO E-W
station-keeping is m/s-class, so it sits well above the floor. **The floor is per-satellite and
TLE-quality-dependent** — a fixed Δv number is not meaningful; the benchmark must treat the floor as a
calibrated, per-object/per-class quantity.

### 3. Mean-motion alone has low recall — the detector must be multi-element

At 1 (3) false-alarms/sat-year, the fraction of *labelled* maneuvers detected from **mean motion alone**
is only ≈ **5% (12%)** for LEO and ≈ **8% (17%)** for GEO. This is a real, important result, not a metric
artifact:

- Most operational station-keeping maneuvers are **small** — at or below the TLE mean-motion noise floor
  individually (CryoSat-2 alone has 168 maneuvers in 12 years; altimetry sats fly tight orbit tubes with
  frequent tiny burns).
- Mean motion only captures the **in-track / energy-changing** component; **cross-track** maneuvers
  (inclination — GEO north-south, LEO inclination control) barely move `n`.

Two consequences the rest of v0.1 must absorb:

1. **The classical detector (#15) must use the full mean-element vector** — inclination for cross-track,
   the eccentricity vector for E-W — not mean motion alone. The numbers above are a *lower bound* on
   achievable recall (single observable); a multi-element (and learned) detector will do materially
   better, especially on cross-track maneuvers.
2. **The benchmark (#12/#13) must score precision/recall at a fixed false-alarm rate** and **stratify by
   maneuver detectability** — sub-floor maneuvers are physically undetectable from TLEs by *any* method,
   so raw recall over all labels is the wrong headline; recall over the **above-floor (detectable)**
   population, at a stated FPR, is the meaningful metric.

### 4. False-alarm-rate unit and operating point

Because maneuvers are rare and the catalogue is large, **the FPR unit is false-alarms per
satellite-year** (not per-elset). The primary **operating point is 1 FA/sat-year**, with the benchmark
also reporting a small sweep (e.g. 0.3 / 1 / 3 FA/sat-year) so methods are compared on a precision-recall
curve rather than a single point.

## Recommendation (→ D4)

- **D4.1 — labelling granularity.** A label is the **inter-elset interval** that brackets a maneuver
  epoch (the gap `[t_i, t_{i+1})` containing it); detection is per-gap, not per-second.
- **D4.2 — detection-matching tolerance.** A detection matches a label if it lands in the labelled gap
  **or within ±1 adjacent gap** — i.e. **≈ ±2 days** (covers the ~1-day median cadence and its p90 tail,
  and the ambiguity of which gap first reflects the burn). Frozen into the benchmark.
- **D4.3 — detectability floor.** In-track Δv floor ≈ **cm/s for LEO**, **~0.05–0.15 m/s for GEO**,
  calibrated **per object/class** (TLE-quality-dependent), not a single global constant. The benchmark
  scores/stratifies by the above-floor population.
- **D4.4 — FPR unit + operating point.** **False-alarms per satellite-year**, primary operating point
  **1 FA/sat-year**, reported as a curve over a small FPR sweep.
- **D4.5 — design input (for #15/#13).** Detection must be **multi-element**; mean-motion-only is a lower
  bound. Set project expectations accordingly: full-population recall is bounded by the floor.

## Reproducibility

`v3_detectability_analysis.py` is stdlib + numpy, deterministic across runs given the pinned dataset
(mean motion parsed straight from the TLE; no sgp4). The Shorten data is fetched separately
(`git clone https://github.com/dpshorten/TLE_observation_benchmark_dataset`, point `--data-dir` at
`processed_files/`) and is **not** committed — it is an unlicensed dev oracle (V2).

## Caveats / open items

- Empirical base is **LEO + GEO** (the Shorten classes); **MEO** floor is analytical (GPS NANUs carry no
  Δv — V2) and **HEO** is deferred (V2). Refine MEO/GEO with operator Δv at the data-layer stage (#10).
- The Δv floor is the **in-track vis-viva equivalent**; cross-track Δv is not captured by mean motion —
  resolved by the multi-element detector and by **V4** (the Δv-inversion validation against published Δv).
- Shorten labels are **epoch-only**, so this fixes the floor as a mean-motion-jump bound; absolute Δv
  calibration comes from the DORIS/altimetry Δv set (V2) via V4.
- Per-satellite TLE quality varies widely (Jason-3 noise inflates its floor) — the per-object calibration
  in D4.3 handles this.
- Ratify D4 at the design freeze.

## References

- Shorten et al., benchmark dataset — <https://github.com/dpshorten/TLE_observation_benchmark_dataset>
  (arXiv:2212.08662).
- Lemmens & Krag, "Two-Line-Elements-Based Maneuver Detection Methods for Satellites in Low Earth Orbit",
  *J. Guidance, Control, and Dynamics* (2014) — <https://arc.aiaa.org/doi/10.2514/1.61300>.
- Kelecy et al., "Satellite Maneuver Detection Using Two-line Element (TLE) Data", AMOS 2007 —
  <https://amostech.com/TechnicalPapers/2007/Modeling_Analysis_Simulation/Kelecy.pdf>.
