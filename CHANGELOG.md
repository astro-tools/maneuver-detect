# Changelog

All notable changes to maneuver-detect are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/astro-tools/maneuver-detect/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/astro-tools/maneuver-detect/releases/tag/v0.1.0
