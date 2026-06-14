# Leaderboard

The [public leaderboard](https://huggingface.co/spaces/astro-tools/maneuver-detect-leaderboard) runs
the package's shipped [deterministic scorer](benchmark.md#scorer) on the **frozen v0.2 test split** and
ranks your method against the classical, BiLSTM, and transformer baselines. You upload a
`predictions.json`; it returns your per-class above-floor recall at the operating point. It is the
hosted, zero-setup way to get a number directly comparable to prior work — the same number the
[benchmark protocol](benchmark.md) defines.

## What kind of board this is

The v0.2 test labels are part of the public CC-BY-4.0 dataset (`dataset/v0.2/labels.json` and
`splits.json` commit the labels and mark the `test` objects), so the answer key is already published.
The board is therefore a **reproducibility / convenience board**: it hosts the canonical scorer on the
splits everyone already has, so a new method gets a directly comparable score without standing up the
scorer locally.

It is **not** a hidden-label competition. The aggregate-only response and the per-user rate limit are
courtesy and abuse guards, not integrity guarantees — the labels are public either way. A true
hidden-label competition needs a never-committed forward holdout, which the open dataset cannot provide;
that is deferred to a later release. See
[D12](decisions.md#d12-leaderboard-integrity-and-compute-budget) for the rationale.

## How to submit

### 1. Reconstruct the v0.2 objects

Run your detector over the same objects the test split scores. Reconstruct their element series from
**your own** Space-Track account — the recipe is pinned in `dataset/v0.2/recipe.json`, and the
content-hash manifest verifies the rebuild byte-for-byte:

```bash
export SPACETRACK_USERNAME=you@example.com
export SPACETRACK_PASSWORD=...
maneuver-detect dataset build --out dataset/
```

The labels and splits are public; only the raw element history is reconstructed locally (the recipe-first
distribution model — see the [dataset reference](dataset.md)).

### 2. Produce a `predictions.json`

Run your method and serialise its detections to a JSON array of [canonical maneuver
records](schema.md) — exactly what the package's `read_predictions` parses. The package's own helpers
do this: `from_frame` turns a `detect` result into canonical records, and `predictions_to_json` writes
the file (sorted keys, ISO-8601 epochs):

```python
from pathlib import Path

from maneuver_detect import detect, datasets
from maneuver_detect.benchmark import predictions_to_json
from maneuver_detect.schema import from_frame

maneuvers = []
for norad_id in test_objects:                       # the test-split objects from splits.json
    history = datasets.tle_history(norad_id=norad_id)
    maneuvers.extend(from_frame(detect(history)))   # your own detector in place of detect()

Path("predictions.json").write_text(predictions_to_json(maneuvers))
```

Each record carries the schema fields — `epoch`, `confidence`, `type`, `delta_v_estimate`, and the
provenance (`norad_id`, `elset_epoch_before`, `elset_epoch_after`). A submission can express **nothing but
predictions**: the fixed-schema reader rejects any other payload, so a submission cannot smuggle a query
for the labels.

### 3. Upload and read your score

On the [Space](https://huggingface.co/spaces/astro-tools/maneuver-detect-leaderboard), upload your
`predictions.json`, enter a method name and your Hugging Face user id, and submit. The board scores it
and adds your row.

## What you get back

The response is **aggregate-only** — per class, never per label:

- **Above-floor recall and precision** at the primary operating point (1 false alarm / satellite-year),
  the [headline metric](benchmark.md#metric).
- The published **timing-only "cheating floor"** — the score a detector reaches from inter-elset timing
  alone, with no element signal. A method that only learned the sampling cadence lands here; a real
  detector must beat it.

No per-label match table is ever returned, so the board exposes no label a submission did not already
have from the public dataset.

## Limits and etiquette

- **Rate limit:** up to **5 scored submissions per Hugging Face user per UTC day**, keyed to your user
  id. It is a courtesy guard against accidental floods, not an integrity control.
- **Space-Track terms:** you reconstruct the series under your own account; nothing about a submission
  ships or proxies raw Space-Track data — only your predictions.

## Reproduce it locally

You do not need the Space to get a comparable number — the same deterministic scorer ships in the
package (`maneuver_detect.benchmark`) and reproduces the board's numbers byte-for-byte from a predictions
file. The [benchmark protocol](benchmark.md#scorer) shows the `read_predictions` → `score` call, and the
[reproduce-the-baseline example](https://github.com/astro-tools/maneuver-detect/blob/main/examples/reproduce_baseline.py)
runs the local scorer end to end on a synthetic labelled series.
