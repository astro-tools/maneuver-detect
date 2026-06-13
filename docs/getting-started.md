# Getting started

## Installation

```bash
pip install maneuver-detect
```

The base install carries only permissive dependencies. It includes the PyTorch / Lightning modelling stack
and the Hugging Face Hub / `datasets` libraries — the learned baselines and the Hub-distributed artifacts
build on them — so a GPU is needed only to *train* new baselines, never to install the package or run the
classical detector. The optional time-series foundation-model baseline lives behind the `[foundation]` extra:

```bash
pip install "maneuver-detect[foundation]"
```

maneuver-detect supports Python 3.10, 3.11, and 3.12.

## Detect maneuvers for a satellite

Hand `detect` a per-object mean-element TLE history and read back the detected maneuvers:

```python
from maneuver_detect import detect, datasets

history = datasets.tle_history(norad_id=25544, start="2024-01-01")
maneuvers = detect(history, model="classical")
```

Each row of the returned DataFrame is one detected maneuver: the detection `epoch` (UTC), a calibrated
`confidence` in `[0, 1]`, the maneuver `type` (in-track / cross-track / radial), a `delta_v_estimate` in m/s,
and the provenance (`norad_id` and the bounding elset epochs). `"classical"` is the default model; the learned
`"bilstm-base"` and `"transformer-base"` models load their checkpoints from the Hugging Face Hub on first use
(see [Models and the Hub](models.md)). The [output schema reference](schema.md) documents every column. See the
[detect-on-a-NORAD-id example](https://github.com/astro-tools/maneuver-detect/blob/main/examples/detect_norad.py)
for a runnable version.

## Get a TLE history

`datasets.tle_history` is the per-object accessor — the cleaned mean-element series the detector consumes:

```python
from maneuver_detect import datasets

# Full available history, from the credentialled Space-Track archive (the default source):
history = datasets.tle_history(norad_id=25544)

# A bounded window:
history = datasets.tle_history(norad_id=25544, start="2024-01-01", end="2024-06-30")

# The no-auth current elset from CelesTrak (a single point — refresh, not history):
latest = datasets.tle_history(norad_id=25544, source="celestrak")
```

The multi-epoch history needed to *detect* a maneuver comes from **Space-Track** (`source="spacetrack"`, the
default), which needs a free account — set `SPACETRACK_USERNAME` and `SPACETRACK_PASSWORD` in the environment.
CelesTrak serves only the latest elset, so it refreshes a series go-forward but cannot supply history on its
own. See the [dataset reference](dataset.md) for each source's terms.

## From the command line

The `detect` subcommand mirrors the API for a one-shot detection on a NORAD id (its history fetched live) or a
local TLE file:

```bash
maneuver-detect detect 25544                       # a NORAD catalogue id
maneuver-detect detect path/to/history.tle         # a local TLE file
```

The target is read as a local file when it names one, otherwise as a NORAD id. Useful flags:

| Flag | Effect |
|---|---|
| `--model` | Detector to run (default: the classical reference detector). |
| `--source` | Catalogue source for a NORAD-id fetch: `spacetrack` (credentialled history, default) or `celestrak` (no-auth current elset). Ignored for a file. |
| `--start` / `--end` | ISO-8601 epoch bounds, e.g. `--start 2024-01-01 --end 2024-06-30`. |
| `--format` | `table` (default, human), `csv`, or `json`. |
| `--output, -o` | Write to a file instead of stdout. |

The CLI is a thin shell over the API, so it produces the same result as the equivalent `detect()` call. Output
stays ASCII (the Δv column prints as `delta_v_estimate`), so it is safe on a Windows console.

## Reconstruct the dataset

The labelled dataset is distributed as a pinned **reconstruction recipe**, not a data blob (the raw Space-Track
history is not redistributable). Rebuild it locally from your own Space-Track account:

```bash
export SPACETRACK_USERNAME=you@example.com
export SPACETRACK_PASSWORD=...
maneuver-detect dataset build --out dataset/
```

This fetches each catalogue object's series and the open maneuver-label files, reconstructs the labelled
dataset, and writes `recipe.json`, `labels.json`, and a content-hash `manifest.json` — the raw series is never
written. The [dataset reference](dataset.md) covers the distribution model, the catalogue, and the source
terms in full.

## Where to next

- The [dataset and label sources](dataset.md) — what is in the dataset and the terms of every source.
- [Models and the Hub](models.md) — the learned detectors, how `detect(model=…)` loads from the Hub, and caching.
- The [benchmark protocol](benchmark.md) — the splits, the matching rule, and the metric a method is scored on.
- The [output schema and Δv inversion](schema.md) — the columns `detect` returns and the physics behind them.
- The [API reference](api.md) — the full public surface.
