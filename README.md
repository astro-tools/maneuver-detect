# maneuver-detect

[![CI](https://github.com/astro-tools/maneuver-detect/actions/workflows/ci.yml/badge.svg)](https://github.com/astro-tools/maneuver-detect/actions/workflows/ci.yml)
[![Docs](https://github.com/astro-tools/maneuver-detect/actions/workflows/docs.yml/badge.svg)](https://astro-tools.github.io/maneuver-detect/)
[![PyPI](https://img.shields.io/pypi/v/maneuver-detect.svg)](https://pypi.org/project/maneuver-detect/)
[![Python versions](https://img.shields.io/pypi/pyversions/maneuver-detect.svg)](https://pypi.org/project/maneuver-detect/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Open dataset, models, and benchmark for detecting orbital maneuvers from public TLE history.

maneuver-detect takes a satellite's public TLE history and returns a DataFrame of detected
maneuvers — each with a detection epoch, a calibrated confidence, a maneuver type
(in-track / cross-track / radial), and a Δv estimate. It ships a curated, reconstructable,
labelled dataset built from public catalog data and operator maneuver announcements; a classical
reference detector with a vis-viva / Gauss-variational Δv inversion; and a frozen, leak-free
benchmark protocol so a new detection method can be measured against prior work on the same
splits. See the [changelog](CHANGELOG.md) for released functionality.

## What this is

Detecting maneuvers from public TLEs is a long-running space-situational-awareness problem, but
the open ecosystem has no shared answer to it: every paper rebuilds its own dataset, cleaning
pipeline, detector, and evaluation, so results are not comparable and the data is rarely
published. maneuver-detect provides the missing shared piece — an open, citable dataset and a
reproducible benchmark. The load-bearing engineering is the dataset and the benchmark protocol
(leak-free splits, the detection-matching rule, the metric, and the physics of the Δv inversion),
not the model code, which is deliberately small and standard. The classical reference detector is
the baseline every learned model must beat.

## Quick start

```python
from maneuver_detect import detect, datasets

history = datasets.tle_history(norad_id=25544, start="2024-01-01")
maneuvers = detect(history, model="classical")
# DataFrame columns: epoch, confidence, type, delta_v_estimate, plus provenance
```

From the command line, on a NORAD id (fetched live) or a local TLE file:

```bash
maneuver-detect detect 25544
```

## Dataset, models, and leaderboard

- **Dataset** — a curated, labelled dataset built from public TLE history (CelesTrak, Space-Track,
  TraCSS) and operator maneuver announcements, distributed per the source-data terms as a pinned,
  byte-deterministic reconstruction recipe plus the openly-licensed data layer.
- **Model checkpoints** — the classical baseline, and the learned baselines as they land, each
  with a model card documenting training data, splits, and metrics.
- **Leaderboard** — a public leaderboard with frozen train / val / test splits and held-out test
  labels, so submitted methods get directly comparable scores.

The dataset and checkpoints are distributed through the Hugging Face Hub and the leaderboard runs
on a Hugging Face Space; both are versioned in lockstep with each release.

## Installation

```bash
pip install maneuver-detect
```

The base install carries only permissive dependencies. It includes the PyTorch / Lightning
modelling stack and the Hugging Face Hub / `datasets` libraries — the learned baselines and the
Hub-distributed dataset and checkpoints build on them — so a GPU is needed only to *train* new
baselines, never to install the package or run the classical detector. The optional time-series
foundation-model baseline lives behind the `[foundation]` extra:

```bash
pip install "maneuver-detect[foundation]"
```

maneuver-detect supports Python 3.10, 3.11, and 3.12.

## What this is not

- **Not a maneuver predictor.** It detects maneuvers that have already happened; forecasting
  future maneuvers is a different problem, deliberately out of scope.
- **Not real-time or streaming.** It is batch — a TLE history in, a maneuver DataFrame out.
- **Not a new propagator or orbit-determination engine.** It consumes SGP4 mean elements and the
  small inversions the Δv estimate requires; it does not do precise propagation.
- **Not a general time-series-anomaly framework.** The detectors are maneuver detectors on
  orbital element series, not a reusable anomaly library.
- **No closed or commercial data.** Only publicly available TLEs and publicly released maneuver
  labels are used; redistribution-restricted commercial SSA products are excluded.

## Documentation

Full documentation is at
[astro-tools.github.io/maneuver-detect](https://astro-tools.github.io/maneuver-detect/) — getting
started, the dataset and label-source reference, the benchmark protocol, the output schema and
Δv-inversion reference, and the API reference.

## Development

```bash
git clone https://github.com/astro-tools/maneuver-detect.git
cd maneuver-detect
uv sync --all-groups
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and local checks. The frozen v0.1 design
decisions live in [`docs/design/`](docs/design/).

## License

MIT — see [LICENSE](LICENSE).
