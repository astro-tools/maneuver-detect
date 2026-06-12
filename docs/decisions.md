# Design decisions (D1–D10)

The frozen decision record for v0.1. It consolidates the prerequisite analysis (the dataset-redistribution,
label-source, detectability-floor, and Δv-inversion studies) and the project charter into ten decisions
that fix the shape of the dataset, the benchmark, and the library contract. Each decision states the call
and its rationale; the implementable benchmark contract is on the [benchmark protocol](benchmark.md) page,
and the output schema and Δv inversion on the [schema reference](schema.md) page.

The record is **frozen by release** — the v0.1 surface matches it, and any change is a version bump with a
documented rationale.

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

*Derived from the v0.1 prerequisite studies and the project charter.*
