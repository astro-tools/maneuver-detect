# V7 spike — leaderboard integrity + single-GPU compute budget

**Status:** findings + recommendation, feeding **D12** (leaderboard integrity + training budget) and
gating the v0.2 public leaderboard Space and the neural-baseline training. Two questions are settled
here so neither has to be re-decided downstream: **can a scored benchmark be hosted publicly without
leaking the held-out labels or being overfit by probing**, and **what concrete single-GPU budget do
the v0.2 baselines need** (confirming the charter's "trainable on a single GPU in hours" claim).
Hosting/anti-overfitting are settled by a dry-run of the real scorer against hidden labels; the
budget by an order-of-magnitude estimate the baseline runs will measure against.

## Question

The v0.2 deliverable is a public, scored leaderboard. Three things must be true before building it:

1. **Hosting is feasible.** A Hugging Face Space (Gradio) can host a scored benchmark with the
   **test labels kept hidden**, accept a predictions submission, run the deterministic scorer, and
   post a ranked result.
2. **It is safe.** A concrete anti-overfitting policy — hidden labels, rate-limited submissions, and
   a submission format that cannot exfiltrate labels — with a decided submission cadence.
3. **The baselines are cheap enough.** A pinned single-GPU training budget (where, GPU class,
   wall-clock, rough cost) for the ~10M-parameter transformer and the BiLSTM.

---

## Part A — Leaderboard hosting & integrity

### A.1 The scoring leg needs no GPU

The scorer is pure element-arithmetic: match predictions to labels under the D4 tolerance, then
count per class (`benchmark/matching.py` + `benchmark/metrics.py`). It runs in milliseconds on a CPU
and is already byte-deterministic (D8). So the **leaderboard Space runs on the free Hugging Face CPU
tier** — the only GPU spend in the whole project is *offline baseline training* (Part C), never the
hosted board. This is the load-bearing simplification: hosting cost is ~zero and the integrity story
reduces to "what does the Space return, and what can a submitter send".

### A.2 Architecture

- **Front end:** a Gradio Space (free CPU tier). A user uploads a `predictions.json` — a JSON array
  of canonical maneuver records (`schema.py`), exactly what `read_predictions` parses.
- **Hidden test set:** the held-out labels + per-object exposure live as a **Space secret / private
  HF Dataset**, readable only by the Space at runtime. They are never in the Space's public files,
  the model repo, or any response.
- **Scoring:** the Space calls the shipped `score()` and returns **only aggregate metrics** — the
  headline above-floor recall per class at the operating point, plus the published D11 timing-only
  "cheating floor". The internal per-label match table is never serialised into a response.
- **Board:** submissions (user id, UTC timestamp, score) are appended to a public HF Dataset and
  rendered sorted by headline recall. The board is reproducible because the scorer is deterministic.

### A.3 The two integrity surfaces

A hidden-label leaderboard can leak the answer key two ways, and the design closes both:

- **The response.** If the Space returned the per-label match table (which label each detection
  hit), an attacker reads the labels directly. The response is therefore a strict **subset** of the
  score report — aggregate recall/precision only, no per-label structure.
- **The submission.** If the submission channel accepted arbitrary input, it could carry a query.
  The fixed-schema parser admits **only** canonical maneuver records and rejects everything else, so
  a submission can express nothing but "here are my predicted maneuvers".

---

## Part B — Anti-overfitting policy

Hidden labels stop *direct* reads; they do not stop **probing** — the classic public-leaderboard
attack where a competitor submits repeatedly and overfits to the feedback. Four mechanisms bound it:

1. **Hidden test labels** (A.2) — never shipped; only on the Space.
2. **Public / private split** — the live board scores a **public** test subset; the **final ranking
   is recomputed on a held-back private subset, revealed only at the v0.2 release** (the standard
   Kaggle pattern). Probing can overfit the public subset but never touches the private one, so it
   cannot win the final ranking.
3. **Rate limit** — **5 scored submissions per user per UTC day**, keyed to the Hugging Face user
   id. Enough for honest iteration; slow enough that systematic probing is both glacial and visible
   as anomalous submission volume.
4. **Fixed-schema, aggregate-only round trip** (A.3) — a submission cannot smuggle a query and a
   response cannot return a label.

**Submission cadence (the decided knob):** **5 / user / UTC day** on the public split; the **private
split is scored once, at release** (and on an explicit, logged final-eval request). The number is a
Space configuration constant, revisable per release without touching the protocol.

### Why probing is bounded — quantified

The strongest probe is a **single-detection oracle**: submit one detection at one candidate gap and
watch whether the headline recall rises. A rise means that gap (within the D4 ±1-adjacent-gap
tolerance) holds an above-floor label. This *does* work — the proof confirms the oracle exists — but
it costs **one submission per candidate gap**. With `G` candidate gaps and `R` scored
submissions/user/day, recovering the public split takes `ceil(G / R)` submission-days, is detectable
as anomalous volume, and the D4 tolerance only localises each label to a ~3-gap window, not a single
gap. The private split is never in the loop, so even a fully-probed public board does not move the
final ranking.

---

## Part C — Single-GPU compute budget

Both baselines are small, and the labelled dataset is modest, so training is **hours on one
commodity GPU** — the charter claim holds with wide margin. The estimate is order-of-magnitude, from
the standard `≈ 6 · N · T` training-FLOP rule (forward + backward, `N` parameters, `T` tokens seen):

- **Model size.** Transformer `N ≈ 10⁷`; BiLSTM `N ≈ 1–3 × 10⁶`.
- **Data.** Windowed per-object element series (D11: `W = 64` tokens, stride `< W`). A v0.2 training
  set of a few hundred objects over multi-year history is `O(10⁵)` windows ⇒ `O(10⁶)` tokens/epoch;
  a few hundred epochs ⇒ `T ≈ 10⁹` tokens seen.
- **Transformer compute.** `6 · 10⁷ · 10⁹ ≈ 6 × 10¹⁷` FLOP ≈ 0.6 EFLOP for a full run.

Against realistic single-GPU effective throughput on a *small* model (memory/launch-bound, well
below peak):

| GPU (≤ 24 GB) | effective TFLOP/s (small model) | transformer full run | rough cost |
|---|---|---|---|
| Free T4 (Colab / Kaggle) | ~8–12 | ~14–20 h (overnight) | $0 |
| L4 / A10G (cloud single-GPU) | ~20–30 | ~6–9 h | ~$0.7–1.0/h ⇒ ~$5–9 |
| RTX 4090 (rented) | ~40–80 | ~2–4 h | ~$0.3–0.7/h ⇒ ~$1–3 |

The BiLSTM has far fewer FLOP (smaller `N`); even with its weaker GPU utilisation (sequential
recurrence) it lands in the same 1–4 h band. Peak memory for a 10M-param model on `W = 64` sequences
is **well under 16 GB**, so a 16–24 GB card suffices — no multi-GPU, no gradient checkpointing.

**Pinned budget.** Train **offline on a single ≤ 24 GB GPU** (free Colab/Kaggle T4, or a rented
RTX 4090 / L4 / A10G); push the checkpoint + model card to the HF Hub. Each baseline trains in
**< ~8 GPU-hours**; a full reproducible v0.2 run — both baselines, a modest hyperparameter sweep
(~8–12 configs), and the final checkpoints — is **< ~1 GPU-day wall-clock and < ~$50**, or $0 on the
free tiers overnight. The hosted leaderboard adds no GPU cost (Part A.1).

**Acceptance gate (measured at baseline-training time).** The charter's "single GPU in hours" claim
is confirmed when each baseline's *measured* wall-clock on a single ≤ 24 GB GPU is **< ~8 h** at
**< ~16 GB** peak memory. The real figures (throughput, epochs to converge, final memory) are
recorded on each checkpoint's model card (D8) — this spike pins the budget the runs are held to, not
the runs themselves.

---

## Proof — hidden-label scoring dry-run

[`v7_leaderboard_scoring_proof.py`](v7_leaderboard_scoring_proof.py) (stdlib + the installed package;
no network, no GPU; deterministic, fictional catalogue ids per the V1/D2 no-redistribution practice)
runs the Space's scoring endpoint against a hidden held-out set and asserts the four integrity
properties, then runs the probing attack end-to-end to bound it. Verbatim output:

```
V7 — leaderboard hidden-label scoring dry-run (real scorer)
================================================================

[1] Honest submission → public result the Space would return:
{
  "operating_point_fa_per_sat_year": 1.0,
  "headline_recall_above_floor": {
    "LEO": 1.0,
    "MEO": null,
    "GEO": 0.0
  },
  "timing_only_floor_auc": {
    "LEO": 0.62,
    "GEO": 0.68
  }
}

[2] Integrity checks (all assert-backed, passed):
    - held-out label epochs absent from the payload
    - response is aggregate-only (no per-label match table)
    - submission channel rejected 3/3 non-prediction payloads
    - scoring is byte-deterministic across runs (D8)

[3] Single-detection probing attack, bounded:
{
  "candidate_gaps": 60,
  "positive_oracle_signals": 15,
  "rate_per_user_per_day": 5,
  "submission_days_to_exfiltrate_public_split": 12
}

    => exfiltrating the public split needs 60 probes = 12 days at 5/user/day,
       is anomalous-volume detectable, and never touches the private split.
```

What the run shows:

- **The public result carries no labels.** The returned payload is aggregate-only; the held-out
  label epochs do not appear in it (asserted), and the per-label match table the scorer computes
  internally is never serialised.
- **The submission channel can't carry a query.** Three non-prediction payloads (missing canonical
  fields, a JSON object, a query string) are all rejected by the fixed-schema reader before scoring.
- **Scoring is deterministic** (D8) — the same submission scores identically across runs and across
  separate process invocations, so the board and the private final-eval are reproducible.
- **The probing oracle is real but expensive.** 60 candidate gaps yield 15 positive signals — the 5
  above-floor labels each smeared across their ±1 tolerance window (the below-floor label correctly
  produces none). Recovering the public split costs one probe per gap: **12 days at 5/user/day**, on
  a board that logs every submission, and the private split stays untouched.

## Recommendation (→ D12)

- **D12.1 — hosting.** **Gradio Space, free CPU tier.** Scoring is CPU-cheap and deterministic, so
  the board needs no GPU; hidden labels live as a Space secret / private HF Dataset; the public
  board persists to a public HF Dataset.
- **D12.2 — integrity.** **Aggregate-only responses** (no per-label match table) over a
  **fixed-schema submission** (`read_predictions` rejects non-prediction payloads). The two leak
  surfaces — response and submission — are both closed; proven against the real scorer.
- **D12.3 — anti-overfitting.** **Hidden test labels + a public/private split** (private scored once
  at release) + a **5/user/UTC-day rate limit** on the public split. Probing recovers only the
  public split, at one submission per candidate gap (≈ `ceil(G/R)` days), detectably; the private
  split decides the final ranking. **Cadence: 5/user/day public, private at release.**
- **D12.4 — compute budget.** **Train offline on one ≤ 24 GB GPU** (free Colab/Kaggle T4 or a rented
  RTX 4090 / L4 / A10G); each baseline **< ~8 GPU-h**, a full v0.2 run (both baselines + a small
  sweep + finals) **< ~1 GPU-day and < ~$50**, $0 on the free tiers. Confirmed by an acceptance gate
  measured on the model cards at training time.

## Reproducibility

`v7_leaderboard_scoring_proof.py` is stdlib + the installed package, deterministic (no RNG, fixed
synthetic data), and reuses the shipped `benchmark.scoring` scorer — so the integrity properties are
properties of the code that will run in the Space, not of a mock. Run it with
`python docs/design/spikes/v7_leaderboard_scoring_proof.py` (or `uv run`); two invocations produce
byte-identical output. The compute table is an estimate from `6·N·T` and public single-GPU pricing —
reproducible as arithmetic, not as a benchmark; the measured numbers land on the baseline model
cards.

## Caveats / open items

- The compute budget is **order-of-magnitude**, not measured: `6·N·T` ignores attention's quadratic
  term (negligible at `W = 64`), and effective throughput on small models is empirical. The
  acceptance gate (D12.4) is where the estimate is confirmed or revised.
- The dataset-size input (`O(10⁵)` windows) is the v0.2 expectation; the dataset-growth pass may move
  it, but a 10× larger set is still hours on one GPU (linear in `T`).
- The proof's probing bound uses a daily gap grid; a real catalogue's irregular cadence changes the
  candidate-gap count but not the *one-probe-per-gap* economics or the public/private firewall.
- The HF free-tier limits (CPU Space quotas, Dataset size) are adequate for a deterministic CPU
  scorer + a small label set, but are vendor terms to re-confirm at build time.
- Ratify D12 when the leaderboard Space is implemented against it.

## References

- Charter §3 prerequisite validation **V7**; §4 leaderboard deliverable.
- [`../benchmark-protocol.md`](../benchmark-protocol.md) §8 (submission / held-out labels /
  rate-limiting — the v0.1 protocol forward-references V7 as the place this is settled).
- [`v1-dataset-redistribution.md`](v1-dataset-redistribution.md) — the no-redistribution practice
  (D2) the proof follows by using only fictional catalogue ids.
- [`v5-irregular-sampling-encoding.md`](v5-irregular-sampling-encoding.md) — the timing-only
  "cheating floor" (D11) the public response publishes as context.
- `benchmark/scoring.py`, `benchmark/matching.py`, `benchmark/metrics.py` — the deterministic scorer
  the Space reuses unchanged.
- Hugging Face Spaces (Gradio, CPU tier) and Datasets — the hosting + persistence substrate.
