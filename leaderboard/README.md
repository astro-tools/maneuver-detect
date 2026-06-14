---
title: maneuver-detect leaderboard
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
---

# maneuver-detect leaderboard

A Hugging Face Space that scores a maneuver-detection method on the **frozen v0.2 test split** with the
package's shipped deterministic scorer, and ranks it against the classical, BiLSTM, and transformer
baselines. Upload a `predictions.json` (a JSON array of canonical maneuver records) and get the
per-class above-floor recall at the operating point, plus the published timing-only "cheating floor".

This directory is both the Space (`app.py` + `requirements.txt`) and the offline tooling that prepares
its data (`build_fixture.py`). The server-side logic lives in the package
(`maneuver_detect.leaderboard`); the app is a thin Gradio front end over it.

## What kind of board this is

The v0.2 test labels are part of the public CC-BY-4.0 dataset (`dataset/v0.2/labels.json` + `splits.json`),
so the answer key is already published. This board is therefore a **reproducibility / convenience**
board: it runs the canonical scorer on the splits everyone already has, so a new method gets a directly
comparable number without standing up the scorer locally. It is **not** a hidden-label competition — the
rate limit and the aggregate-only response are courtesy / abuse guards, not integrity guarantees. (A true
hidden-label competition would need a never-committed forward holdout; that is deferred — see the D12
amendment in `docs/design/decisions.md`.)

## The private bundle (and why it is not committed)

The Space scores against a **bundle** it never exposes:

```
fixture.json        the held-out ScoringFixture — test-split labels + exposure + the timing floor
seeds/<name>.json   each baseline's test-split predictions, re-scored on the board as a seed entry
```

The fixture's matching windows are real elset (TLE) epochs — derived Space-Track data the recipe-first
dataset deliberately does not redistribute (D2). So the bundle is **private deploy-time data**, not a repo
artifact. Build it from a credentialed reconstruction and upload it to a **private** Hugging Face Dataset
the Space can read:

```bash
export SPACETRACK_USERNAME='you@example.com'
export SPACETRACK_PASSWORD='your-space-track-password'
python leaderboard/build_fixture.py --out leaderboard-bundle
# then upload ./leaderboard-bundle/ to a private HF Dataset (do not commit it)
```

Because the labels it encodes are already public, the bundle leaks nothing new — it is private only to
honour D2. Anyone with Space-Track access can rebuild a byte-identical bundle, so the board stays
reproducible.

## Configuring the Space

The app resolves the bundle from one of two environment variables:

| Variable | Meaning |
|---|---|
| `LEADERBOARD_BUNDLE_DIR` | Local path to a bundle directory — for running the app locally. |
| `LEADERBOARD_BUNDLE_REPO` | A private HF Dataset id to download the bundle from at startup. |
| `HF_TOKEN` | A read token for that private Dataset (set as a Space secret). |

Run it locally against a freshly built bundle:

```bash
LEADERBOARD_BUNDLE_DIR=./leaderboard-bundle python leaderboard/app.py
```

## How to submit

1. Run your detector over the v0.2 objects (reconstruct the element series from your own Space-Track
   access — the recipe is in `dataset/v0.2/recipe.json`).
2. Write your detections to a `predictions.json` — a JSON array of canonical maneuver records, exactly
   what the package's `read_predictions` parses.
3. Upload it on the Space, enter a name / Hugging Face user id, and submit. The board updates with your
   per-class recall.

A submission can express nothing but predictions: the fixed-schema reader rejects anything else, and the
response is aggregate-only, so no label is ever returned.
