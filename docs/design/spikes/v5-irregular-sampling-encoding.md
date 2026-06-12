# V5 spike — irregular-sampling sequence-model input encoding

**Status:** findings + recommendation, feeding **D11** (the irregular-sampling input contract) and the
architectural gate for the v0.2 learned baselines (the feature layer + the BiLSTM and transformer). The
chosen encoding is frozen here so those three can be built without re-deciding. Empirical comparison of
candidate encodings on real labelled element series + first principles.

## Question

TLE cadence is irregular (≈1-day median, gappy, with longer post-maneuver re-acquisition gaps), and a
maneuver is observable only as a discontinuity between consecutive elsets. How should that irregular
series be presented to a sequence model so that the encoding **(a) does not leak the label** — the elset
gap that defines the target must not be a trivially separable input feature — and **(b) does not destroy
the maneuver signal** — a sanity model must still recover known maneuvers above the V3 detectability
floor? Compare the three candidate encodings and freeze the winner as the tensor contract the feature
layer implements.

## Method

[`v5_irregular_sampling_proof.py`](v5_irregular_sampling_proof.py) (stdlib + numpy; mean elements parsed
straight from the TLE lines, no sgp4) builds and scores the three candidates on the **Shorten benchmark**
(15 satellites — 10 LEO altimetry, 5 GEO Fengyun — TLE history + ground-truth maneuver timestamps; a
dev-only oracle per V2, analysed locally, never redistributed):

1. **resample-to-regular-grid** — linear interpolation of each element channel onto a uniform daily grid
   plus a sampling mask;
2. **time-encoded deltas** — `(Δt, Δelement)` per inter-elset gap, no interpolation;
3. **continuous-time / time2vec** — a bounded (clipped + periodic) encoding of `Δt`, standing in for a
   time-aware recurrence (time-LSTM / Neural-ODE) without training one in numpy.

The prediction unit is the **inter-elset gap** (D4). Element deltas are the level shift of the
secular-detrended residual across the gap (median of *k* elsets after minus *k* before), with angle
channels unwrapped first — J2 nodal regression and apsidal precession are deg/day, far larger than any
burn, and must be removed or they read as a bogus cross-track step (the V4 failure mode).

Two measurements per encoding, one per DoD criterion:

- **Leak** — the direction-agnostic rank-AUC of the encoding's **timing channel alone** (raw `Δt` for
  deltas; the sampling-mask run-length for resample; clipped `Δt` for time2vec) against the
  maneuver-gap label. `0.5` = chance = no leak; a high value means a model can flag maneuvers by gap
  length without ever looking at the elements.
- **Signal** — cross-fit (2-fold, deterministic) LDA separability from the **element content with timing
  neutralised**: the `|Δa|` in-track channel alone (where the LEO in-track / GEO E-W signal lives and
  where interpolation smears most directly) and the full element-delta vector; plus **recall over the
  above-floor population at 1 quiet false-alarm/sat-year** (the V3/D4 operating point), with the
  mean-motion-only result as the V3 lower bound.

## Findings

Per-class aggregates (pooled across satellites; AUC, higher = more separable):

| encoding | in-track (AUC) | multi-elem (AUC) | leak (AUC) | recall @ 1 FA/sat-yr |
|---|---|---|---|---|
| **LEO** (10 sats, 29 677 gaps, 658 maneuver-gaps, 50 above-floor) | | | | |
| resample-to-grid | 0.816 | 0.624 | 0.533 | 0.74 |
| **time-encoded deltas** | 0.823 | 0.630 | 0.619 | **0.86** |
| continuous (time2vec) | 0.823 | 0.630 | 0.619 | 0.86 |
| **GEO** (5 sats, 8 900 gaps, 197 maneuver-gaps, 16 above-floor) | | | | |
| resample-to-grid | 0.896 | 0.871 | 0.631 | 0.19 |
| **time-encoded deltas** | 0.892 | **0.910** | 0.677 | **0.44** |
| continuous (time2vec) | 0.892 | 0.910 | 0.677 | 0.44 |

### 1. Resampling-to-grid destroys signal where it matters

Interpolating across a gap **fabricates element values on exactly the interval that carries the maneuver**
and smears the discontinuity below the floor. Bulk-distribution AUC barely moves (the damage is
concentrated near the decision threshold), but the metric that counts — **above-floor recall at 1
FA/sat-year — drops sharply**: LEO 0.86 → 0.74 and GEO 0.44 → 0.19 (less than half). On GEO the
multi-element AUC also falls 0.910 → 0.871 because the cross-track step (N-S station-keeping) is the
worst hit by interpolation. The sampling mask gives resample the *lowest* timing leak (≈0.53 LEO / 0.63
GEO), but that is no consolation for halving the recoverable recall — and the mask is itself a gap-length
feature.

### 2. Time-encoded deltas preserve the signal

The deltas encoding keeps the discontinuity intact: in-track AUC ≈ 0.82 (LEO) / 0.89 (GEO) — matching the
V3 mean-motion result — and the **best above-floor recall** (0.86 LEO / 0.44 GEO). The full element-delta
vector adds the GEO cross-track channel (multi-element AUC 0.910 > the 0.892 in-track). On LEO the full
vector (0.630) is *weaker* than the in-track channel alone (0.823): the node/eccentricity channels are
arc-second-noisy for LEO and **dilute** the strong in-track signal under an equal-weight linear readout —
the same effect #37 found when node/ecc were poor trigger channels. That is an argument about the
*readout* (per-channel normalisation + the per-type floor), **not** the encoding: the deltas encoding
exposes every channel faithfully; how the model weights them is a downstream choice the per-type floor
already informs.

**The maneuver signal lives in the *magnitude* of the element step, not its sign** — a burn can raise or
lower `a` / `i` / Ω, so quiet gaps and maneuver gaps both centre a *signed* delta near zero and a linear
separator finds nothing (the in-track AUC above collapses to chance on signed deltas; it is `|Δa|` that
scores 0.82). The proof's probe therefore scores `|Δelement|`. The frozen contract nonetheless feeds the
model **signed** deltas: a non-linear BiLSTM/transformer recovers the magnitude itself, and the sign
carries the burn *direction* the Δv-type classification (D5) needs. Magnitude-vs-sign is thus a readout
choice — the encoding carries both.

### 3. The timing leak is real, modest, and structural — not "trivially separable"

Post-maneuver re-acquisition gaps run long, so `Δt` alone carries a **measurable but weak** correlation
with the label: AUC ≈ **0.62 (LEO) / 0.68 (GEO)** — well above chance, but far below the element signal
(0.82–0.89). So the encoding does **not** make the gap a *trivially separable* feature (that would be
AUC ≳ 0.95); a model cannot win on gap length alone. Two consequences:

- **`Δt` is necessary and stays in the contract.** It is needed to interpret a step's *rate* (a 1-day vs
  a 5-day Δa mean different things) and to detrend secular drift over the gap. Dropping it would hurt the
  signal more than it helps the leak.
- **The leak cannot be encoded away.** Rank-AUC is invariant to any monotonic transform of a scalar, so
  clipping, log, or a time2vec saturation leave the timing leak unchanged (the proof's `Δt`-leak is
  identical raw vs clipped), and daily-cadence regularization (the #37 trick) does not move it either
  (0.619 → 0.620): the re-acquisition gap survives regularization. The leak is therefore handled at the
  **protocol** level (see D11), not by transforming the input.

### 4. Continuous-time / time2vec buys nothing over deltas for the v0.2 baselines

The time2vec row is identical to deltas on both axes — same signal (it *is* the deltas content) and same
leak (rank invariance). A genuine time-aware recurrence (time-LSTM / Neural-ODE) would add runtime and
implementation cost for no leak or signal advantage at the BiLSTM/transformer **baseline** tier, whose
whole point is a simple, reproducible reference. The transferable part of the continuous-time idea — a
**bounded, smooth `Δt` representation** the attention/recurrence can consume — is adopted *as the timing
block of the deltas encoding*; the recurrence itself is deferred (revisit at v0.3 foundation models if
warranted).

## Recommendation (→ D11)

- **D11.1 — encoding.** **Time-encoded deltas, no interpolation.** Reject resample-to-grid (halves
  above-floor recall by smearing the discontinuity it interpolates across). Reject a continuous-time
  recurrence for the v0.2 baselines (no measured advantage over deltas; cost not justified) — adopt its
  bounded-`Δt` (time2vec) representation as the deltas encoding's timing block.
- **D11.2 — leak handling (protocol, not encoding).** `Δt` stays in the input (needed for step-rate +
  detrending). Because the timing leak is structural (≈0.62–0.68 AUC, not removable by monotonic
  transform), the benchmark **must report a timing-only baseline** — the AUC/recall a `Δt`-only model
  reaches — as the "cheating floor" any submitted model must beat, and the **headline metric stays recall
  over the above-floor population** (D4/D7), where the element signal dominates the timing leak. `Δt` is
  **clipped** (time2vec saturation, ~2–3 days) for training stability and to stop a learned model
  over-weighting outlier re-acquisition gaps — explicitly *not* as a leak fix.
- **D11.3 — normalisation + detrending.** Per-class (LEO/MEO/GEO) **robust** standardisation
  (median / IQR) per element channel, statistics fit on the **train split only** (leak-free). Secular
  drift (J2 nodal regression / apsidal precession) is removed by a two-sided local-linear fit **before**
  the delta is computed; angle channels are carried as the eccentricity vector `(h, k)` and an unwrapped
  Ω so there is no wrap discontinuity. The arc-second-noisy node/ecc channels are exposed but governed by
  the per-type floor (#37), not fed raw at equal weight.
- **D11.4 — tensor contract.** Frozen below; the feature layer implements it verbatim.

## Tensor contract (the frozen input the feature layer emits)

One **token per elset** (so one inter-elset gap = the transition between two adjacent tokens; the
per-gap maneuver target attaches to that transition, D4). For object class *c* ∈ {LEO, MEO, GEO}:

**Channels** (all `float32`, per-class robust-normalised, train-split statistics):

1. *Element levels* at the token's epoch — `a` (km), `e`, `sin i`, `cos i`, `h = e·cos ω`,
   `k = e·sin ω`, and the unwrapped node `Ω` (the angle channels carried as `(h, k)` + unwrapped `Ω`
   per D11.3, so none wraps), plus the secular-detrended residual of each (the level the local-linear
   fit predicts is subtracted). Levels give absolute context; residuals give the anomaly.
2. *Element deltas across the gap to the previous token* — the **signed** level shift of each detrended
   channel above. The maneuver signal is in the *magnitude* of the step (Finding 2), but signed is fed so
   the sign carries burn direction for the D5 Δv-type classification; the model recovers magnitude itself.
3. *Timing block* — `time2vec(Δt)`: `[Δt_clip / scale, {sin, cos}(2π Δt / P_j) for j=1..m]` with `Δt`
   clipped at the saturation cap (D11.2). Bounded and smooth.
4. *Mask / validity* — a real-elset bit (always 1 here — the deltas encoding imputes no rows, unlike
   resample), and a `Δt`-saturation flag (gap exceeded the clip cap).

**Windowing.** Train on sliding windows of **W = 64 consecutive elsets** (≈2 months at 1-day cadence),
stride < W so every gap appears with bidirectional context; the per-token (per-gap) target is "this gap
contains a maneuver". Sequences never cross a satellite boundary (split integrity, D7). At inference the
model runs over the full per-object series; predictions remain per-gap and are scored by the D4 matching
rule.

**Shape.** `(batch, W, C)` for the channel count *C* above, plus a `(batch, W)` target and a `(batch, W)`
validity mask; class *c* selects the normalisation statistics, not a separate tensor.

**Implementation note.** Compute the secular detrend (and the per-channel normalisation statistics)
**once per object series**, then read deltas off the precomputed residual — not per candidate gap.
Recomputing the local-linear/rolling detrend inside a per-gap loop is `O(n²)` in the series length and
needlessly slow on the multi-thousand-elset histories (a trap hit and fixed while writing the proof).

## Reproducibility

`v5_irregular_sampling_proof.py` is stdlib + numpy, deterministic across runs (cross-fit folds are
even/odd index, no RNG; elements parsed from the TLE). The Shorten data is fetched separately and not
committed (unlicensed dev oracle, V2): `git clone
https://github.com/dpshorten/TLE_observation_benchmark_dataset`, then `--data-dir <…>/processed_files`.

## Caveats / open items

- The **above-floor population is small** (50 LEO / 16 GEO maneuver-gaps), so the recall figures are
  directional, not precise — but the resample-vs-deltas gap is consistent across both classes and matches
  the V3 expectation (GEO recall < LEO).
- The signal readout here is a **linear** LDA; the learned baselines are non-linear and will exploit the
  multi-element / cross-track content better than the 0.63 LEO multi-element figure (a floor on the
  achievable readout, set deliberately simple so the *encoding* comparison is what varies).
- The leak/signal evidence is **LEO + GEO** (the Shorten classes); MEO has no Δv-labelled oracle (epoch
  only, V2) — the contract applies to it by class normalisation, validated when MEO labels are scored.
- The empirical base is mean-element steps from one catalogue; the credentialed Space-Track DORIS replay
  (the v0.1 real-eval path) is the cross-check when the feature layer lands.
- **Ratified.** D11 is ratified by the v0.2 feature layer (`maneuver_detect/features/`), which emits
  this contract verbatim: the seven base channels (`a`, `e`, `sin i`, `cos i`, `h`, `k`, unwrapped
  `Ω`) with their secular-detrended residuals and signed gap deltas, the `time2vec(Δt)` timing block,
  and the validity / `Δt`-saturation mask, windowed at `W = 64`. The detrend is the once-per-series
  two-sided local-linear fit (not the per-gap recompute the implementation note warns against).

## References

- Charter §3 prerequisite validation **V5**; the `features/` architecture sketch (§3).
- [`v3-detectability-floor.md`](v3-detectability-floor.md) — the detectability floor and matching
  tolerance the signal test scores against (D4).
- [`v4-dv-inversion.md`](v4-dv-inversion.md) — the secular-drift detrending requirement (the cross-track
  failure mode) reused here.
- [`../benchmark-protocol.md`](../benchmark-protocol.md) — the above-floor metric and per-gap matching
  the leak/signal tests are framed against (D7).
- Kazemi et al., "Time2Vec: Learning a Vector Representation of Time", 2019 —
  <https://arxiv.org/abs/1907.05321> (the bounded `Δt` representation).
- Shorten et al., benchmark dataset — <https://github.com/dpshorten/TLE_observation_benchmark_dataset>
  (arXiv:2212.08662).
