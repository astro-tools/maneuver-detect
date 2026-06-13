# Models and the Hugging Face Hub

The package ships the classical reference detector in-tree, and distributes the learned baselines and
the labelled dataset through the [Hugging Face Hub](https://huggingface.co/astro-tools). Nothing
heavy is fetched at install time: a learned checkpoint and the dataset are pulled on first use,
CPU-only, and cached on disk.

## Detectors

| Model | Where it lives | Notes |
|---|---|---|
| `classical` | In the package | Rule-based reference detector; the default, no download. |
| `bilstm-base` | [`astro-tools/maneuver-detect-bilstm-base`](https://huggingface.co/astro-tools/maneuver-detect-bilstm-base) | Learned BiLSTM; checkpoint pulled from the Hub. |
| `transformer-base` | [`astro-tools/maneuver-detect-transformer-base`](https://huggingface.co/astro-tools/maneuver-detect-transformer-base) | Learned ~10M-parameter transformer; checkpoint pulled from the Hub. |

A learned model localises maneuvers in the element series; the same vis-viva / Gauss physics
inversion the classical detector uses then recovers the Δv magnitude and type for each detection.

## Run a learned model

Select it by name — the checkpoint is fetched from the Hub the first time the detector runs, then
served from the on-disk cache:

```python
from maneuver_detect import detect, datasets

history = datasets.tle_history(norad_id=25544, start="2024-01-01")
maneuvers = detect(history, model="bilstm-base")     # or "transformer-base"
```

or from the command line:

```bash
maneuver-detect detect 25544 --model transformer-base
```

Inference is CPU-only — a GPU is needed only to *train* a new baseline, never to run one. The
checkpoint bundle carries the network weights together with the frozen train-split normaliser and the
windowing parameters, so a download reproduces the exact training-time inference pipeline.

### Caching, offline use, and a local checkpoint

- Downloads are cached by `huggingface_hub` under its hub cache (`HF_HOME`, by default the XDG cache
  dir). Set `HF_HUB_OFFLINE=1` to force the cache and never hit the network.
- A given library version pins the checkpoints (and the dataset) to a release revision, so it loads
  the artifacts it was released in lockstep with. Set `MANEUVER_DETECT_HUB_REVISION` (e.g. to `main`)
  to load from a different revision — useful for validating a published artifact before its release
  tag exists.
- To run a locally-trained checkpoint instead of the Hub one, point the matching environment variable
  at the bundle: `MANEUVER_DETECT_BILSTM_CHECKPOINT` or `MANEUVER_DETECT_TRANSFORMER_CHECKPOINT`. The
  resolution order is an explicitly-passed bundle, then that env var, then the Hub.

## The dataset on the Hub

The labelled dataset (the reconstruction recipe, the labels, the manifest, and the splits — never the
raw element series) is published to [`astro-tools/maneuver-detect`](https://huggingface.co/datasets/astro-tools/maneuver-detect)
and downloadable on first use:

```python
from maneuver_detect import datasets

recipe = datasets.load_recipe()        # the pinned reconstruction recipe
manifest = datasets.load_manifest()    # the content-hash manifest
labels = datasets.load_labels()        # the parsed operator labels
local_dir = datasets.fetch_dataset()   # the whole dataset snapshot directory
```

You still reconstruct the element series locally from your own Space-Track account
(`datasets.reconstruct`, or `maneuver-detect dataset build`) — the recipe and manifest make that
byte-for-byte verifiable. See the [dataset reference](dataset.md) for the distribution model and the
source terms.

## Publishing (maintainers)

Publishing is split by what each artifact needs.

**Dataset** — the artifacts are committed and deterministic, so a release publishes them
automatically. The release workflow pushes `dataset/v<minor>/` and a generated dataset card to the
Hub dataset repo on each `v*` tag, authenticated with an `HF_TOKEN` repository secret. It can also be
run by hand:

```bash
HF_TOKEN=... maneuver-detect dataset publish        # defaults to the current dataset version
```

**Checkpoints** — a checkpoint is GPU-trained offline, so it is published from the training
environment (which has the weights), not from CI:

```bash
HF_TOKEN=... maneuver-detect models publish bilstm-base ./bilstm-base.pt
HF_TOKEN=... maneuver-detect models publish transformer-base ./transformer-base.pt
```

Each model card is generated from the bundle's own provenance — the training data version, the
measured metrics, the architecture, and the training cost — so the documented numbers cannot drift
from the weights they describe. The dataset and the checkpoints carry the same version tag, kept in
lockstep with the library release.
