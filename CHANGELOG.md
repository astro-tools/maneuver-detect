# Changelog

All notable changes to maneuver-detect are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-15

A foundation-model baseline, a new scored orbit class with operator-announced labels, and a meaningful
confidence column. v0.3 adds a zero-shot forecast-residual detector built on a pretrained time-series
model, grows the dataset with executed-Δv operator feeds that add an IGSO class and move GEO from
self-labelled to operator-announced, and bakes uncertainty calibration into every published model so the
confidence is calibrated per orbit class. The classical detector remains the in-package default; the
learned and foundation baselines are the entries every new method is measured against.

### Added

- **Foundation-model baseline** — a forecast-residual detector that replaces the classical detector's
  hand-built quiet-dynamics prior with a pretrained time-series model, selectable via
  `detect(history, model="chronos-residual")`. It forecasts each object's mean-element series,
  standardises the residual by a robust per-object MAD, and thresholds it per orbit class, reusing the
  matcher, the Δv inversion, the schema, and the scorer unchanged. The shipped backend is Chronos
  (`amazon/chronos-bolt-small`); it ships zero-shot — trains nothing and runs on the free CPU/GPU tiers —
  behind the optional `[foundation]` extra, with the checkpoint pulled from the Hub at runtime.
- **IGSO class + operator-announced label sources** — QZSS Operational History Information files
  (executed orbit-maintenance Δv) add a new scored IGSO class and, with NOAA GOES navigation summaries,
  move GEO from self-labelled to operator-announced; the Galileo NAGU back-catalogue thickens MEO. HEO is
  reserved as an empty class (no ingestible operator feed exists for the regime). `DATASET_VERSION → 0.3.0`;
  the v0.1 and v0.2 partitions stay pinned and byte-stable. Attribution stacks per source (NOAA public
  domain, QZSS CC-BY-4.0, Galileo © EU).
- **Uncertainty calibration** — reliability diagrams, temperature scaling, and split-conformal intervals as
  model-agnostic machinery, plus published per-class operating points. Each published detector now carries a
  calibrator fit on the val split only, frozen into its bundle, so a loaded model emits calibrated confidence
  with no calibration data at inference; per-class reliability curves and operating points are rendered onto
  the model cards.
- **Class-balanced selection objective + per-class detection thresholds** — checkpoint selection on a
  class-balanced criterion and per-class detection thresholds, so the dense GEO/IGSO classes no longer swamp
  recall on the sparse ones.

### Changed

- The `[foundation]` extra (`chronos-forecasting`) is now exercised by the foundation baseline; it and the
  base modelling stack remain permissively licensed (Apache-2.0 checkpoints), and the extra stays optional.
- `ScoreReport.to_json` gains a per-class `operating_point_confidence` field — additive (every prior field
  byte-for-byte unchanged) but a change to the frozen scorer artifact, so it is a v0.3-boundary change.
- All published baselines (the classical detector, the BiLSTM and transformer learned models, and the
  Chronos foundation baseline) were retrained and re-evaluated on the v0.3 (IGSO) dataset under the bumped
  score protocol, with the train/test leak in the Chronos fine-tune corrected to the train split.

Ships the frozen v0.3 design decisions D14–D17 alongside the v0.1/v0.2 set.

## [0.2.0] - 2026-06-13

Learned baselines and the public benchmark. v0.2 adds two trained detectors, distributes the
dataset and checkpoints through the Hugging Face Hub, stands up a public leaderboard on a frozen
held-out split, and grows the dataset to GEO and new label sources with class-stratified and
temporal split variants. The classical detector remains the in-package default; the learned models
are the first entries every new method is measured against.

### Added

- **Learned baselines** — a BiLSTM sequence detector and a ~10M-parameter transformer detector,
  both trained on the real-data benchmark splits and selectable via
  `detect(history, model="bilstm-base" | "transformer-base")`. A Lightning training harness with
  early stopping, best-checkpoint restore, and checkpoint selection on val-split benchmark recall
  (not BCE val-loss).
- **Irregular-sampling feature layer** — mean-element feature engineering with an explicit
  irregular-sampling encoding (D11), so the sequence models consume the non-uniform TLE cadence
  directly.
- **Hugging Face Hub distribution** — `detect(model=...)` resolves checkpoints from the Hub, pinned
  in lockstep to `v{DATASET_VERSION}`; dataset and per-model publishers
  (`maneuver-detect dataset publish`, `maneuver-detect models publish`) generate dataset and model
  cards, including per-class test metrics read from the checkpoint.
- **Public leaderboard** — a Hugging Face Space serving the held-out test split with a submission
  path and an integrity + single-GPU compute-budget protocol (D12), so external methods are scored
  on the same frozen partition.
- **Dataset-growth pass** — GEO objects (self-labelled longitude-shift) and Galileo MEO, with new
  licence-clean label sources (D13); the v0.2 partition re-frozen.
- **Split variants** — a class-stratified, leak-free, byte-stable packing mode and a
  temporal-holdout split (novel-satellite × novel-era) with a leak-free meta-test, plus per-class
  confidence intervals on recall and precision.

### Changed

- The forward-wired `torch` / `lightning` / `huggingface_hub` / `datasets` base dependencies are now
  exercised by the learned baselines and Hub loading; all remain permissively licensed and the
  `[foundation]` extra stays optional.

Ships the frozen v0.2 design decisions D11–D13 alongside the v0.1 set.

## [0.1.0] - 2026-06-11

The first release of maneuver-detect: a curated, reconstructable, labelled dataset built from public
TLE history and operator maneuver announcements; a classical reference detector with a
vis-viva / Gauss-variational Δv inversion; and a frozen, leak-free benchmark protocol — the shared,
citable baseline a new detection method can be measured against on the same splits. The load-bearing
pieces are the dataset and the benchmark; the detector is the baseline every learned model must beat.

### Added

- **`detect()` entry point and the canonical maneuver schema** — `detect(history, model=...)` returns
  a DataFrame of detected maneuvers, each with a detection epoch, a calibrated confidence, a maneuver
  type (radial / in-track / cross-track), a Δv estimate, and provenance. `Maneuver` and `ManeuverType`
  define the canonical record; a `Detector` interface with `available_models()` / `get_detector()`
  lets a new method plug into the same surface.
- **Classical reference detector** — time-aware Holt smoothing plus a multi-element jump rule,
  robustified to literature-level precision/recall on real TLE history (per-type floors, daily
  regularisation, transient rejection). It ships in the package and is the baseline every learned
  model must beat.
- **Δv inversion and maneuver-type classification** — the vis-viva energy change combined with the
  Gauss variational equations to resolve a radial / in-track (along-track) / cross-track Δv from a
  mean-element step and classify the maneuver type.
- **Data layer** — CelesTrak and Space-Track fetchers with an on-disk cache and rate-limit
  discipline, elset cleaning, and per-NORAD mean-element time-series assembly from the cleaned
  history.
- **Labels** — operator-announcement ingest (the open DORIS/IDS `man.txt` files and the CelesTrak GPS
  NANU archive) and an epoch-to-elset-gap labeller that maps an announced maneuver to the affected
  elset interval.
- **Reconstructable dataset** — a pinned recipe (`recipe.json`) plus a content-hash manifest
  (`manifest.json`) and parsed labels (`labels.json`), rebuilt locally with
  `maneuver-detect dataset build` and verifiable byte-for-byte. The raw Space-Track element history is
  never redistributed; only the recipe, the labels, and the per-series digests ship.
- **Benchmark protocol** — leak-free train / val / test splits (by satellite and time), the
  detection-matching rule, the per-class metric (precision / recall at a fixed false-alarm rate), and
  a deterministic scorer that reproduces the reported numbers from a committed predictions file.
- **`maneuver-detect` CLI** — `detect` (on a NORAD catalogue id fetched live, or a local TLE file)
  and `dataset build` (reconstruct and verify the dataset).
- **Packaging and docs** — a typed (PEP 561) MIT package on Python 3.10, 3.11, and 3.12 with only
  permissive runtime dependencies; the time-series foundation-model stack stays behind the optional
  `[foundation]` extra. Ships with a published documentation site and the frozen v0.1 design
  decisions (D1–D10).

[Unreleased]: https://github.com/astro-tools/maneuver-detect/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/astro-tools/maneuver-detect/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/astro-tools/maneuver-detect/releases/tag/v0.1.0
