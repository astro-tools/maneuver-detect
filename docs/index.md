# maneuver-detect

Open dataset, models, and benchmark for detecting orbital maneuvers from public TLE history.

maneuver-detect takes a satellite's public TLE history and returns a DataFrame of detected
maneuvers — each with a detection epoch, a calibrated confidence, a maneuver type
(in-track / cross-track / radial), and a Δv estimate. It ships a curated, reconstructable,
labelled dataset, a classical reference detector with a vis-viva / Gauss-variational Δv
inversion, and a frozen, leak-free benchmark so a new detection method can be measured against
prior work on the same splits.

## Why

Detecting maneuvers from public TLEs is a long-running space-situational-awareness problem, but
every paper rebuilds its own dataset, cleaning pipeline, detector, and evaluation — so results
are not comparable and the data is rarely published. maneuver-detect provides the missing shared
piece: an open, citable dataset and a reproducible benchmark, with a classical baseline every
learned model must beat and a deterministic scorer that reproduces the published numbers.

## What it is not

- Not a maneuver *predictor* — it detects maneuvers that have already happened, not future ones.
- Not a real-time or streaming pipeline — it is batch: a TLE history in, a maneuver DataFrame out.
- Not a new propagator or orbit-determination engine — it consumes SGP4 mean elements and the
  small inversions the Δv estimate needs.
- Not a general time-series-anomaly framework — the detectors are maneuver detectors on orbital
  element series.
- Not a cross-catalog correlation or object-association tool — it works one catalogued object's
  history at a time, not the multi-sensor tracking problem.
- Closed or commercial data is out of scope — only publicly redistributable data is used.

## Explore

- **[Getting started](getting-started.md)** — install, run the detector on a NORAD id, read the result.
- **[Dataset and label sources](dataset.md)** — what is in the dataset, how it is distributed, and each
  source's terms.
- **[Benchmark protocol](benchmark.md)** — the splits, the matching rule, and the metric a method is scored on.
- **[Output schema and Δv inversion](schema.md)** — the columns `detect` returns and the physics behind them.
- **[Design decisions](decisions.md)** — the frozen v0.1 decision record (D1–D10).
- **[API reference](api.md)** — the full public surface.

See the
[changelog](https://github.com/astro-tools/maneuver-detect/blob/main/CHANGELOG.md) for
released functionality.
