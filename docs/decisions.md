# Design decisions (D1–D13)

The frozen decision record. It consolidates the prerequisite analysis (the dataset-redistribution,
label-source, detectability-floor, Δv-inversion, irregular-sampling, and leaderboard-integrity studies) and
the project charter into the decisions that fix the shape of the dataset, the benchmark, and the library
contract. **D1–D10 fix the v0.1 surface; D11–D13 the v0.2 additions** — the learned baselines, the public
leaderboard, and the dataset-growth pass. Each decision states the call and its rationale; the implementable
benchmark contract is on the [benchmark protocol](benchmark.md) page, and the output schema and Δv inversion
on the [schema reference](schema.md) page.

The record is **frozen by release** — each release's surface matches it, and any change is a version bump with
a documented rationale.

---

## D1 — Package layout + stack

Package `maneuver_detect`, Python ≥ 3.10, Hatchling, `py.typed`:

- the public surface — `detect`, the `datasets` accessor, and the canonical maneuver type;
- `data/` — the catalogue fetchers (CelesTrak, Space-Track), elset cleaning, and series assembly;
- `labels/` — one module per label source plus the epoch-to-gap labeller;
- `features/` — mean-element feature engineering;
- `detectors/` — the classical reference detector in v0.1; learned detectors arrive in later releases;
- `benchmark/` — splits, the matching rule, the metrics, and the scorer;
- `physics.py` — the Δv inversion; `cli.py` — the `maneuver-detect` command line.

Stack: numpy / pandas; sgp4; astropy (internal only); and the PyTorch / Lightning modelling stack and the
Hugging Face Hub / `datasets` libraries, on which the learned baselines and the Hub-distributed artifacts
build. The time-series foundation-model stack is an optional `[foundation]` extra so the base install stays
light. Dev tooling: pytest, pytest-cov, ruff, mypy, mkdocs-material, mkdocstrings.

## D2 — Dataset distribution — recipe-first hybrid

Publish operator **labels**, a **pinned reconstruction recipe** (the fetch code, the NORAD catalogue, the
per-object date ranges and query parameters, and a per-series SHA-256 content-hash manifest), and
**directly-shipped data only where it is openly licensed**. **Raw Space-Track data — and analysis derived
from it — is never redistributed**: the Space-Track User Agreement reaches derived analysis, and while TLEs
are public-domain U.S.-Government works, the terms of use bind redistribution. Multi-year training history
therefore comes from Space-Track *via the recipe*, with each user reconstructing it from their own account.
Reconstruction is **byte-deterministic**. A recipe entry's epoch window scopes **both** the series fetch and
that object's maneuver labels, so the committed label set is a function of the whole recipe, not the full
announced history.

## D3 — Label sources + class scope

Sources: **DORIS/IDS** maneuver files (LEO altimetry, *with Δv* — the Δv-labelled core), **ILRS** maneuver
history (which links to the same DORIS/IDS files), and **GPS NANUs** of the `FCSTDV` type (MEO,
U.S.-Government public domain, epoch-only; SVN/PRN resolved to NORAD via the CelesTrak crosswalk). The
Shorten benchmark is a **development-only cross-check** (unlicensed, not redistributed). SpotGEO is rejected
(optical object detection, not maneuver labels). **Class scope: LEO** (primary, Δv-labelled) **+ MEO**
(epoch-only) **+ GEO** (best-effort, epoch-only); **HEO is deferred** — sparse, a later addition.

## D4 — Labelling granularity + matching tolerance + detectability floor

A **label is the inter-elset gap** that brackets a maneuver epoch (per-gap, not per-second), because a
maneuver is observable only as a discontinuity between consecutive elsets. The **detection-matching tolerance
is the labelled gap plus one adjacent gap on each side (≈ ±2 days)**, set by the ~1-day median TLE cadence.
The **detectability floor** (in-track Δv) is **~cm/s in LEO and ~0.05–0.15 m/s in GEO**, calibrated per
object/class because it is TLE-quality-dependent; MEO is analytical and HEO not applicable. The false-positive
unit is **false alarms per satellite-year**, with a primary operating point of **1 FA/sat-year**, reported as
a curve. Detection **must be multi-element** — mean motion alone catches only a small fraction of maneuvers —
and the benchmark scores precision/recall at a fixed false-alarm rate over the above-floor population.

## D5 — Δv inversion + type rule + tolerance

The inversion reads a Δv back out of the mean-element step across the gap: **vis-viva** for the in-track
component (from the semi-major-axis change), the **Gauss** relations for the cross-track component (from the
inclination and node change), and the residual eccentricity change for the weakly-observable radial
component. The reported magnitude is `|Δv|` and the **type is the dominant component** — in-track ↔ Δa,
cross-track ↔ Δi / ΔΩ, radial ↔ Δe. **Element steps are detrended first** to remove secular drift (notably
the J2 nodal regression), which would otherwise read as a large spurious cross-track Δv. The implementation
uses the **full Gauss variational equations**, not the linearised circular form. Δv is reported only above
the floor, to within about ±25%; radial-dominated maneuvers are reported low-confidence.

## D6 — Canonical output schema + Detector interface

The canonical maneuver record (and the DataFrame `detect` returns) is **`epoch`** (UTC), **`confidence`**
(calibrated, `[0, 1]`), **`type`** (in-track / cross-track / radial), **`delta_v_estimate`** (m/s), plus the
provenance **`norad_id`**, **`elset_epoch_before`**, **`elset_epoch_after`** (the bounding epochs of the
inter-elset gap). This is frozen as the library contract. A **`Detector` interface** returns that schema,
**`detect(history, model=...)`** dispatches by name, and the **`datasets`** accessor exposes
`tle_history(...)`. v0.1 ships the classical detector behind `detect()`; learned models arrive later behind
the same interface.

## D7 — Benchmark protocol

Per the [benchmark protocol](benchmark.md): **leak-free splits by satellite and time** (no satellite and no
overlapping time window shared across train/val/test), seeded and byte-stable; the **matching rule** (the D4
tolerance); the metric — **precision/recall at a fixed false-alarm rate per class over the above-floor
population**, plus per-class type confusion; and a **deterministic scorer**. The protocol, the splits, and the
schema are frozen by release.

## D8 — Reproducibility / versioning

Seeded, byte-stable splits; a **pinned, reconstructable dataset with a content-hash manifest** (D2); the
dataset and checkpoints **versioned in lockstep**, each checkpoint carrying a **model card** (training data,
splits, metrics, intended use); and a deterministic scorer that reproduces the reported baseline numbers from
committed prediction files.

## D9 — Licensing

**Code is MIT** (the org convention). **Authored dataset artifacts** — the label mapping, the splits, the
manifests, the recipe, and features derived from open data — are **CC-BY-4.0**; openly-licensed pass-through
data keeps its upstream licence; **raw Space-Track data is not redistributed**. **Model weights** are MIT or
CC-BY-4.0, with foundation-model fine-tunes inheriting their base licence. All **runtime dependencies are
permissive**. Because the label sources are open or U.S.-Government public domain, the dataset licence is not
forced restrictive; the development-only Shorten labels stay out of the distribution.

## D10 — Decoupling guarantee

**GMAT-free** — no GMAT dependency at runtime, in tests, or in the build. The foundation-model stack is an
optional **`[foundation]`** extra the base install excludes. The deferred charter studies — irregular-sampling
model input, foundation-model applicability, and leaderboard integrity / compute budget — belong to later
milestones, not v0.1.

---

## D11 — Irregular-sampling input encoding

The frozen encoding the learned baselines consume from the unevenly-sampled element series:
**time-encoded element deltas, no interpolation.** Resampling to a fixed grid is rejected — interpolating
across a gap fabricates values on the very interval a maneuver lives in and roughly **halves above-floor
recall**. Element deltas are fed **signed** (the non-linear model recovers the burn magnitude itself, and the
sign carries direction for the D5 type classification); timing is carried as a bounded `Δt` (time2vec)
block. Secular drift (the J2 nodal regression and apsidal precession) is removed by a two-sided local-linear
fit before the delta, angles are carried as the eccentricity vector plus an unwrapped node, and
normalisation is per-class robust (median/IQR) on **train-split statistics only**. `Δt` stays in the input
because step-rate and detrending need it; it carries only a modest, structural correlation with the label, so
the benchmark **reports a timing-only baseline as the "cheating floor"** a submission must beat (D7) and keeps
the headline metric as recall over the above-floor population (D4).

## D12 — Leaderboard integrity and compute budget

The public leaderboard is a **Gradio Space on the free Hugging Face CPU tier** — scoring is pure
element-arithmetic (the D4 matching and per-class counts), CPU-cheap and deterministic (D8), so the board
needs no GPU; the only GPU spend in the project is offline baseline training. Submissions are a
`predictions.json` of canonical maneuver records (the [schema](schema.md)), and two surfaces keep it safe:
the response is **aggregate-only** (per-class above-floor recall at the operating point plus the timing-only
floor, never the per-label match table) and the submission is **fixed-schema** (the reader rejects any
non-prediction payload, so a submission cannot carry a query). A courtesy **rate limit of 5 scored
submissions per user per UTC day** guards against floods. **Compute budget:** both baselines are small
(transformer ≈ 10⁷ parameters, BiLSTM ≈ 1–3 × 10⁶) on an `O(10⁵)`-window set, so a full v0.2 run — both
baselines, a small sweep, and the finals — is **under ~1 GPU-day and trains in hours on a single ≤ 24 GB
GPU**, no multi-GPU; a wall-clock and peak-memory acceptance gate is recorded on each checkpoint's model
card.

**Amendment — v0.2 ships a reproducibility board, not a hidden-label competition.** The integrity design
assumed hidden test labels, but the open dataset (D8/D9) publishes the full answer key:
`dataset/v0.2/labels.json` commits every label and `splits.json` marks the `test` objects, and git history
makes that irretractable. The hidden-label firewall is therefore unbuildable on the v0.2 test set and is
**dropped** — the public/private-subset split is removed, and the aggregate-only response and the rate limit
are kept as **courtesy / abuse guards, not integrity guarantees**. The board is a **reproducibility /
convenience board** on the public splits (see the [leaderboard guide](leaderboard.md)); a true hidden-label
competition would need a separate, never-committed forward holdout and is deferred. One D2 consequence: the
scorer's matching windows are real elset epochs (derived Space-Track data, not redistributed), so the board's
scoring fixture is built offline from a credentialed reconstruction and supplied as private deploy-time data,
not committed.

## D13 — Expanded label sources for v0.2 dataset growth

The v0.2 dataset-growth pass extends the D3 source set, resolving the "no public GEO maneuver-label source"
gap and the open question on non-GPS GNSS notices:

- **MEO — add Galileo NAGUs** (`PLN_MANV`): a second, independent MEO operator beyond GPS — public, no-auth,
  machine-ingestible, GSAT → NORAD via the CelesTrak Galileo crosswalk, epoch-only. The terms are an
  attribution-required reuse grant (© EU), so the labels are redistribution-clean and **shipped**. **GLONASS
  is excluded** — its terms cap public reproduction more tightly than Space-Track and fail D2.
- **GEO — self-labelled longitude-shift.** No openly-licensed GEO maneuver-label file exists to
  redistribute, so v0.2 labels GEO best-effort from **longitude-shift inspection of the reconstructed
  series**; GEO stays epoch-only. The operator-announced BeiDou feed remains a recipe-first option (the D2
  pattern — ship the fetch recipe and parser, never redistributed labels), not shipped data.
- **Scope and licence otherwise unchanged (D3 / D9):** LEO primary and Δv-labelled, HEO deferred; attribution
  stacks per source, and no new redistribution restriction attaches.

---

*Derived from the v0.1 and v0.2 prerequisite studies and the project charter.*
