# maneuver-detect — v0.1 design decisions (D1–D10)

The frozen decision record for v0.1, consolidating the prerequisite spikes (V1–V4, under
[`spikes/`](spikes/)) and the project charter. **No implementation lands until it matches this record.**
Each decision states the call and its rationale/source. The detailed, implementable benchmark contract
lives in [`benchmark-protocol.md`](benchmark-protocol.md).

Status as of the design freeze: V1–V4 complete; D2–D5 are lifted from the merged spike records, D1 and
D6–D10 from the charter and the spike implications.

---

## D1 — Package & repo layout + stack pins

Package `src/maneuver_detect/`, Python ≥ 3.10, Hatchling, `py.typed`:

- `__init__.py` — public surface: `detect`, the `datasets` accessor, the canonical maneuver type.
- `data/` — `celestrak.py`, `spacetrack.py`, `tracss.py` (CC0 source, D2), `clean.py`, `history.py`.
- `labels/` — one module per source (D3) + `labeller.py`.
- `features/` — mean-element feature engineering.
- `detectors/` — `classical.py` (v0.1); `bilstm.py`, `transformer.py` (v0.2); `foundation.py` (v0.3).
- `benchmark/` — `splits.py`, `matching.py`, `metrics.py`, the scorer.
- `physics.py` — the Δv inversion (D5). `cli.py` — `maneuver-detect detect`.

Stack: numpy / pandas; sgp4; astropy (internal only); torch + lightning (v0.2 models);
huggingface_hub + datasets; gradio (v0.2 leaderboard). Optional **`[foundation]`** extra
(chronos / timesfm, v0.3). Dev: pytest, pytest-cov, ruff, mypy, mkdocs-material, mkdocstrings.

## D2 — Dataset distribution model — recipe-first hybrid (V1)

Publish: operator **labels** + a **pinned reconstruction recipe** (fetch code + NORAD catalogue +
per-object date ranges/query params + a per-series **SHA-256 content-hash manifest**) + **directly-shipped
data only where openly licensed** — TraCSS **CC0-1.0** OMM (current/go-forward) and the
redistribution-clean **pre-2022 McDowell archive**. **Never** redistribute raw Space-Track data *or its
analysis* (the Space-Track User Agreement / 10 U.S.C. §2274(c)(2) reaches derived analysis; TLEs are
public-domain U.S.-Government works but the ToU binds). Multi-year *training history* comes from
Space-Track via the recipe; the CC0 layer grows as TraCSS coverage deepens. Reconstruction is
byte-deterministic (proven in V1). *(Residual: confirm TraCSS public reads need no registration — at #10.)*

## D3 — Label sources + v0.1 class scope (V2)

Sources: **DORIS/IDS** maneuver files (LEO altimetry, *with Δv* — the Δv-labelled set), **ILRS**
maneuver predictions, **GPS NANUs FCSTDV** (MEO, US-Government public domain, epoch-only; SVN/PRN →
NORAD via the CelesTrak crosswalk). The **Shorten benchmark** is a **dev-only cross-check** (unlicensed).
**SpotGEO rejected** (optical object-detection, not maneuver labels). **v0.1 class scope: LEO (primary,
Δv-labelled) + MEO (epoch-only) + GEO (best-effort, epoch-only); HEO deferred** (sparse; a later add).

## D4 — Labelling granularity + matching tolerance + detectability floor (V3)

A **label is the inter-elset gap** bracketing a maneuver epoch (per-gap, not per-second). **Detection-
matching tolerance = the labelled gap ±1 adjacent gap (≈ ±2 days)** (TLE cadence ~1-day median).
**Detectability floor (in-track Δv): ~cm/s (LEO), ~0.05–0.15 m/s (GEO)**, calibrated **per object/class**
(TLE-quality-dependent); MEO analytical; HEO N/A. **FPR unit = false-alarms per satellite-year**,
operating point **1 FA/sat-yr**, reported as a curve. Detection **must be multi-element** (mean-motion
alone catches only ~5–15%); the benchmark scores **P/R at fixed FPR over the above-floor population**.

## D5 — Δv inversion + type rule + tolerance (V4)

Inversion: **vis-viva** in-track (from Δa) + **Gauss** cross-track (from Δi, ΔΩ) + radial from the
residual eccentricity change (weakly observable); `|Δv|` and `type = argmax component`. **Type rule:**
in-track ↔ Δa (mean motion), cross-track ↔ Δi / ΔΩ, radial ↔ Δe. **Element jumps must be detrended**
(remove secular drift, esp. J2 nodal regression) before inversion. **Δv tolerance ~±25% above the floor;
none reported below; radial low-confidence.** `physics.py` (#14) implements the **full Gauss VOP** (not
just the linearised circular form). Quantitative accuracy-vs-published-Δv is validated at **#10** against
the DORIS burn Δv.

## D6 — Canonical output schema + Detector interface

Canonical maneuver record / DataFrame columns: **`epoch`** (UTC), **`confidence`** (calibrated, [0, 1]),
**`type`** (in-track / cross-track / radial), **`delta_v_estimate`** (m/s), plus **provenance**
(`norad_id`, the bounding elset epochs). Frozen as the library contract. A **`Detector` ABC** returns the
canonical schema; **`detect(history, model=...)`** dispatches; the **`datasets`** accessor exposes
`tle_history(...)`. v0.1 ships the classical detector behind `detect()`; learned models arrive v0.2.

## D7 — Benchmark protocol

Per [`benchmark-protocol.md`](benchmark-protocol.md): **leak-free splits by satellite + time** (no
satellite and no overlapping time window across train/val/test), seeded + byte-stable; the **matching
rule** (D4 tolerance); the metric **P/R at a fixed FA/sat-year per class over the above-floor population**
+ per-class type confusion; a **deterministic scorer**; frozen by release. That spec is the contract
#12/#13 implement and #18 publishes.

## D8 — Reproducibility / versioning

Seeded, byte-stable splits; a **pinned, reconstructable dataset + content-hash manifest** (D2); dataset
and checkpoints **versioned in lockstep**, each checkpoint with a **model card** (training data, splits,
metrics, intended use); a deterministic scorer that reproduces the reported baseline numbers from
committed prediction files.

## D9 — Licensing

**Code MIT** (org convention). **Authored dataset artifacts** (label mapping, splits, manifests, the
recipe, features derived from open data) **CC-BY-4.0**; pass-through TraCSS data stays **CC0**; **raw
Space-Track is not redistributed**. **Model weights** MIT or CC-BY-4.0; foundation-model fine-tunes
inherit their base (Chronos / TimesFM Apache-2.0). All **runtime deps permissive**. The label sources are
open / US-gov public domain (V2), so the dataset licence is **not forced restrictive**; the Shorten labels
are unlicensed → dev-only.

## D10 — Decoupling guarantee

**GMAT-free** — no GMAT / gmatpy at runtime, in tests, or as a dependency; no setup-gmat. The
foundation-model stack is an optional **`[foundation]`** extra (the base install excludes it). The charter
validations **V5 (irregular-sampling model input), V6 (foundation-model applicability), and V7 (leaderboard
integrity + compute budget) are deferred to their v0.2 / v0.3 milestones** — not v0.1.

---

*Sources: [`spikes/v1-dataset-redistribution.md`](spikes/v1-dataset-redistribution.md),
[`spikes/v2-label-sources.md`](spikes/v2-label-sources.md),
[`spikes/v3-detectability-floor.md`](spikes/v3-detectability-floor.md),
[`spikes/v4-dv-inversion.md`](spikes/v4-dv-inversion.md), and the project charter.*
