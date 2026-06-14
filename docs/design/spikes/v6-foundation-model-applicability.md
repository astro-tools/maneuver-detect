# V6 spike — foundation-model applicability (Chronos / TimesFM forecast-residual)

**Status:** findings + recommendation, feeding **D14** (foundation-model baseline + the `[foundation]`
extra contract) and gating the v0.3 foundation-model baseline detector (`detectors/foundation.py`).
The question this settles, before the v0.3 milestone commits to it: **can a time-series foundation
model be turned into a maneuver detector at all**, on a permissive enough licence, within the
single-GPU budget V7 already pinned — and **which model** (Chronos or TimesFM) leads the baseline.
The mechanism is settled by a dry-run of the residual detector against the *real* shipped scorer; the
licence by reading the checkpoint cards; the budget by reusing the V7 `6·N·T` framing.

## Question

The v0.3 deliverable is a foundation-model baseline that should beat the classical detector. Four
things must be true before building it:

1. **The recipe works.** A pretrained forecaster can be made into a detector by **forecast-residual
   thresholding** — forecast the element series, flag the intervals where the realised series departs
   from the forecast beyond a per-class threshold — and the result drops into the existing benchmark
   (D4 matching, D5 type/Δv, D6 schema, D7 scorer) unchanged.
2. **The weights are usable.** The checkpoints are licensed permissively enough to ship a
   redistributable fine-tune (both families claim Apache-2.0 — verify the specific checkpoints).
3. **It fits the budget.** Zero-shot and a light fine-tune are feasible on the element series within
   the single-GPU budget V7 framed.
4. **The model is chosen.** Chronos vs TimesFM (or both) for the v0.3 baseline.

---

## Part A — the forecast-residual detector recipe

### A.1 The model fills exactly one slot

The classical detector already *is* a residual detector: it removes the secular drift (J2 nodal
regression, decay) from each element channel and flags the per-gap deltas that exceed a per-object,
per-type **detectability floor** (D4). A foundation model replaces **one** component of that — the
hand-built quiet-dynamics model — with a **learned conditional forecast**. Everything else is reused
verbatim:

- **Forecast.** Run the model over each object's mean-element series (the D11 irregular-sampling
  encoding: time-encoded deltas, no interpolation), one-step-ahead (or short-horizon) and rolling, to
  get a predicted next value **and a predictive interval** per channel.
- **Residual → score.** The standardized residual is `(realised − forecast) / predictive-scale`. A
  maneuver steps an element (Δa for in-track, Δi/ΔΩ for cross-track) the quiet-trained forecaster
  cannot anticipate, so its standardized residual spikes; quiet gaps sit near zero.
- **Per-class threshold.** Threshold the standardized residual **per orbit class** — the D4 floor
  expressed in residual units — then non-maximum-suppress per object (a step's neighbouring gaps can
  re-spike). This is the v0.3 baseline's calibrated operating point.
- **Emit canonical records.** Each surviving gap becomes a canonical `Maneuver` (D6): dominant
  channel → **type** via the unchanged **D5** Gauss inversion (which also gives `delta_v_estimate`);
  `confidence` from the residual quantile; provenance from the bracketing elset epochs.

So the foundation model touches nothing downstream: it is a stronger quiet-dynamics prior feeding the
same matcher, inversion, schema, and scorer the classical detector and the v0.2 learned baselines all
use. Its **calibrated predictive interval** is the part the classical floor approximates by hand — and
is exactly what the v0.3 uncertainty-calibration deliverable needs to make the `confidence` column
mean something (reliability diagrams / conformal intervals over the residual quantiles).

### A.2 Why a pretrained forecaster is the right prior

A maneuver is, by construction, the part of the series the *quiet dynamics do not explain*. The better
the quiet-dynamics forecast, the cleaner the residual separates maneuvers from noise. A foundation
model brings a broad, pretrained prior over time-series shape (trend, seasonality, noise scale) that a
per-object linear detrend cannot — and it does so **zero-shot**, with no labels, which matters because
the labelled set is small and class-imbalanced (the V2 / D13 finding). The residual recipe also keeps
the detector **honest against the D11 timing-only leak**: it reads element *content* (the forecast
residual), not gap length, so it sits above the published "cheating floor."

### A.3 The mechanism, proven against the real scorer

[`v6_foundation_residual_proof.py`](v6_foundation_residual_proof.py) (stdlib + numpy + the installed
package; no network, no GPU, no torch; deterministic; fictional catalogue ids `9000x` per the V1/D2
no-redistribution practice) runs the recipe end-to-end with a **deterministic robust drift-continuation
stand-in** in the model slot — forecast = last value + a per-object robust secular drift, standardized
by a per-object MAD scale (the D4 per-object-floor calibration). It is **not** Chronos/TimesFM: it
proves the *wiring and the per-class-threshold contract*, the property that holds whatever forecaster
fills the slot. A
foundation model only raises forecast quality on the same contract; its measured zero-shot / fine-tuned
recall lands on the v0.3 baseline's model card, exactly as V7 pinned the budget the runs later measure
against. Verbatim output:

```
V6 — foundation-model forecast-residual detector (real scorer, stand-in forecaster)
==================================================================================

[1] Residual separation per object (standardized one-step forecast residual):
      object  class   thr  min(man z)  max(quiet z)  below-floor z
       90001    LEO   4.5        8.17          2.35           1.16
       90002    GEO   4.5        8.62          2.78            n/a

    => above-floor maneuvers clear the per-class threshold; quiet gaps and the
       below-floor maneuver stay under it — the threshold separates them cleanly.

[2] Mechanism checks (all assert-backed, passed):
    - residual spikes at maneuvers, quiet elsewhere (margin above)
    - per-class threshold rejects quiet gaps and the below-floor maneuver
    - thresholded residuals → 5 canonical records (= 5 above-floor labels, no false alarms)
    - scoring is byte-deterministic across runs (D8)

[3] Real scorer verdict on the emitted detections:
{
  "operating_point_fa_per_sat_year": 1.0,
  "headline_recall_above_floor": {
    "LEO": 1.0,
    "MEO": null,
    "GEO": 1.0
  }
}

    => the forecast-residual detector plugs into the shipped D4 matcher / D7 scorer
       unchanged and recovers the above-floor population at 1 FA/sat-year.
```

What the run shows: the standardized residual at an above-floor maneuver (z ≈ 8) sits a wide margin
above both the quiet gaps (z ≲ 2.8) and the below-floor maneuver (z ≈ 1.2), so a single per-class
threshold separates them; the thresholded residuals become canonical records that the **real**
`benchmark.scoring.score` matches under the D4 tolerance and scores at 1 FA/sat-year; and the whole
pipeline re-runs byte-identically (D8). The recipe is therefore a property of the shipped scorer, not
a mock — the foundation model swaps in at the one slot the stand-in occupies.

---

## Part B — weight licence

Both candidate families publish **Apache-2.0** checkpoints, which is permissive enough to ship a
redistributable fine-tune (a fine-tune inherits the base licence — already stated in **D9**) and to
take as a runtime dependency (the base install excludes it regardless — D10).

- **Chronos.** The `chronos-forecasting` library is Apache-2.0; the weights `amazon/chronos-t5-{tiny,
  mini,small,base,large}` (8M–710M) and the faster `amazon/chronos-bolt-{tiny,mini,small,base}` are
  published Apache-2.0 on the Hub.
- **TimesFM.** The `timesfm` library is Apache-2.0; `google/timesfm-1.0-200m` and
  `google/timesfm-2.0-500m` are published Apache-2.0.

**Caveat (issue DoD — "verify the specific checkpoints"):** a licence is a property of the *exact
checkpoint revision* pulled, not the family, and model cards can change between revisions. The v0.3
build **pins the checkpoint id + revision and confirms its card is Apache-2.0 at ingest** before any
fine-tune is published — the same "confirm at ingest" discipline D13 applied to the BeiDou NABU type
string. No GLONASS-style 150-character restriction is in play here; both licences are standard
permissive OSS terms.

---

## Part C — zero-shot + light fine-tune feasibility (single-GPU budget)

Reusing the V7 budget framing (`≈ 6 · N · T` training FLOP; realistic small-model single-GPU
throughput), both modes land inside the budget V7 already pinned for the v0.2 baselines — with room to
spare, because the load-bearing mode needs **no training at all**.

- **Zero-shot = inference only.** Residual detection from a *pretrained* forecaster trains nothing: it
  runs the model forward over each object's element series and thresholds the residuals. Chronos-Bolt
  is explicitly CPU-fast (designed for cheap inference); a small Chronos (`small`/`base`) or
  `timesfm-1.0-200m` forecasts an object's multi-year daily series in well under a second on a
  commodity GPU and is tractable on CPU. **Zero GPU-hours, < few GB memory.** The v0.3 baseline can
  therefore *ship zero-shot first* — the cheapest possible path, and the one this spike most wants to
  confirm is open.
- **Light fine-tune = within the V7 budget.** Specializing the quiet-dynamics prior to the
  satellite-element domain (full fine-tune of a `small`/`base` checkpoint, or LoRA adapters on a
  larger one) over the v0.2-scale window set (`O(10⁵)` windows ⇒ `T ≈ 10⁹` tokens) is
  `6 · N · T ≈ 3 × 10¹⁷` FLOP for `N ≈ 5 × 10⁷` — the **same hours-on-one-GPU band** V7 measured for
  the transformer baseline (Free T4 overnight $0; L4/A10G ~6–9 h ~$5–9; RTX 4090 ~2–4 h ~$1–3). Peak
  memory for a ≤ 200M-param model on the D11 window length is **under 16 GB** on a ≤ 24 GB card (LoRA
  far less); no multi-GPU. A full v0.3 foundation run (zero-shot eval + a light fine-tune + a small
  sweep) is **< ~1 GPU-day and < ~$50**, $0 on the free tiers — inside the V7 envelope.

As in V7, this **pins the budget the runs are held to**, not the runs themselves: the measured
zero-shot recall, fine-tune wall-clock, and peak memory are recorded on the v0.3 checkpoint's model
card (D8) against the V7 acceptance gate (< ~8 GPU-h, < ~16 GB).

---

## Part D — Chronos vs TimesFM

The residual recipe (Part A) is **forecaster-agnostic** — any model that returns a predicted value
plus a predictive scale fills the slot — so the choice is which to *lead* the v0.3 baseline with, not
an exclusive bet. **Lead with Chronos; keep TimesFM in the extra as the second entry.** The deciding
factors, in order:

1. **Native probabilistic forecast.** Chronos samples token sequences into empirical predictive
   **quantiles** at every checkpoint size, out of the box. Standardizing the residual by a predictive
   interval, and turning that interval into a *calibrated* `confidence` (D6) for the v0.3
   uncertainty-calibration deliverable, is then direct. TimesFM is point-forecast by default; quantile
   heads arrive only with 2.0. **Edge: Chronos** — the predictive interval is the crux of the residual
   recipe and the calibration deliverable.
2. **Cheap inference for the zero-shot path.** Chronos-Bolt is built for fast, CPU-friendly inference,
   which keeps the zero-shot baseline runnable on the same free tiers the V7 budget targets (and
   echoes V7's "the scored board needs no GPU"). **Edge: Chronos.**
3. **Checkpoint-size granularity.** Chronos spans 8M → 710M (plus the Bolt line), so the baseline can
   be sized to the budget and the data; TimesFM offers 200M / 500M. **Edge: Chronos.**
4. **Licence / maturity.** Both Apache-2.0 (Part B); both have fine-tuning paths. **Tie**, slight edge
   to the more granular `chronos-forecasting` tooling.

TimesFM stays wired (it brings a longer native context and a strong independent zero-shot prior), so
v0.3 can report a TimesFM entry on the same recipe at near-zero extra cost — a useful cross-check, not
a parallel implementation.

---

## Recommendation (→ D14)

- **D14.1 — recipe.** **Forecast-residual thresholding**, per orbit class, feeding the unchanged D4
  matcher / D5 inversion / D6 schema / D7 scorer. The foundation model replaces only the classical
  detector's hand-built quiet-dynamics prior with a learned conditional forecast; its predictive
  interval feeds the v0.3 uncertainty calibration. Proven against the real scorer.
- **D14.2 — licence.** Both families publish **Apache-2.0** checkpoints — redistributable fine-tunes,
  permissive runtime dep. **Pin the checkpoint id + revision and confirm its Apache-2.0 card at
  ingest** before publishing a fine-tune (D9: fine-tunes inherit the base licence).
- **D14.3 — budget.** **Zero-shot is inference-only (zero GPU-hours)** and a **light fine-tune fits the
  V7 single-GPU envelope** (< ~8 GPU-h, < ~16 GB, < ~$50 / $0 on free tiers). Ship zero-shot first;
  fine-tune is optional polish. Measured figures land on the model card against the V7 gate (D8).
- **D14.4 — model choice.** **Chronos leads** the v0.3 baseline (native probabilistic quantiles →
  clean residual standardization + calibrated confidence; CPU-cheap Bolt inference; tiny→large size
  range). **TimesFM stays in the `[foundation]` extra** as a drop-in second entry on the same recipe.
- **D14.5 — `[foundation]` extra contract.** `[foundation] = chronos-forecasting, timesfm` (both
  Apache-2.0) stays an **optional extra the base install excludes** (D10 decoupling). The v0.3
  `detectors/foundation.py` imports them **lazily**, so the base install, the CLI, and the default
  test suite never pull the torch-heavy stack; foundation tests run only in a dedicated CI job with the
  extra installed (`importorskip`). Checkpoints are fetched from the HF Hub at runtime (the base
  `huggingface_hub` dep), not vendored; fine-tune checkpoints ship with a model card (D8).

## Reproducibility

`v6_foundation_residual_proof.py` is stdlib + numpy + the installed package, deterministic (seeded
noise, fictional ids, rounded figures), and reuses the shipped `benchmark.scoring` scorer — so the
mechanism it asserts is a property of the code the v0.3 detector will feed, not of a mock. Run it with
`python docs/design/spikes/v6_foundation_residual_proof.py` (or `uv run`); two invocations produce
byte-identical output. The budget figures are an estimate from `6·N·T` and public single-GPU pricing
(reproducible as arithmetic, per V7), confirmed or revised by the measured numbers on the v0.3 model
card. The licence findings are read from the published checkpoint cards and re-confirmed at ingest.

## Caveats / open items

- **The proof validates the wiring, not the foundation model's accuracy.** The stand-in forecaster
  shows the residual → per-class-threshold → canonical-record → real-scorer contract holds; it does
  **not** establish Chronos/TimesFM zero-shot recall on real, noisy element series. That is the v0.3
  baseline's job to measure (and the classical detector's real-TLE wall — the ~0.54 above-floor
  ceiling on noisy 1990s–2000s missions — is the bar the learned prior exists to beat; clean,
  well-tracked objects already reach literature-level recall).
- **Licence is per-revision.** Both families are Apache-2.0 today; the v0.3 build pins and re-confirms
  the exact checkpoint card at ingest (D14.2) — a model card can change between revisions.
- **The budget is order-of-magnitude**, not measured: `6·N·T` ignores the patch/quadratic-attention
  constant (negligible at the D11 window length), and small-model effective throughput is empirical.
  The V7 acceptance gate (D14.3) is where the estimate is confirmed.
- **Predictive-interval calibration is assumed, not yet measured.** Chronos's quantiles are the input
  to the v0.3 uncertainty-calibration deliverable; whether they are well-calibrated *on element
  series* (vs. needing temperature scaling / conformal adjustment) is that deliverable's question, not
  this spike's.
- **Irregular cadence into a foundation model.** Both models assume a roughly regular sampling grid;
  the D11 encoding (time-encoded deltas / daily regularization, no interpolation) is the contract the
  forecaster consumes — confirm the chosen model ingests it without the interpolation D11 rejected.
- Ratify D14 when the v0.3 foundation-model baseline is implemented against it.

## References

- Charter §3 prerequisite validation **V6**; §2 v0.3 deliverable; the `[foundation]` optional extra.
- [`v3-detectability-floor.md`](v3-detectability-floor.md) — the per-class detectability floor (D4)
  the residual threshold expresses in standardized-residual units.
- [`v4-dv-inversion.md`](v4-dv-inversion.md) — the D5 Δv/type inversion the residual detector reuses
  unchanged to attribute each detection.
- [`v5-irregular-sampling-encoding.md`](v5-irregular-sampling-encoding.md) — the D11 input encoding the
  forecaster consumes, and the timing-only "cheating floor" the residual detector stays above.
- [`v7-leaderboard-integrity-and-compute-budget.md`](v7-leaderboard-integrity-and-compute-budget.md) —
  the `6·N·T` single-GPU budget framing and acceptance gate (D12) reused here.
- `benchmark/scoring.py`, `benchmark/matching.py`, `benchmark/metrics.py` — the deterministic scorer
  the proof reuses unchanged. `schema.py` — the canonical `Maneuver` record the detector emits.
- Chronos (`chronos-forecasting`, `amazon/chronos-t5-*`, `amazon/chronos-bolt-*`) and TimesFM
  (`timesfm`, `google/timesfm-1.0-200m`, `google/timesfm-2.0-500m`) — the candidate Apache-2.0
  forecasters and their Hub checkpoints.
