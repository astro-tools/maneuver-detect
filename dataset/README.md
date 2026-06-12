# maneuver-detect dataset — recipe-first distribution

This directory holds the **distributable** form of the labelled maneuver-detection dataset, one
subdirectory per version (currently `v0.2/`). The dataset is published as a *recipe*, not as raw
catalogue data: each user reconstructs the element-series locally from their own catalogue access,
and a content-hash manifest makes that reconstruction verifiable bit-for-bit.

## What ships here

For each version `vX.Y/`:

| File | What it is | Source terms |
|---|---|---|
| `recipe.json` | The pinned catalogue — every object's NORAD id, orbit class, label source, and the per-object fetch parameters. | Public reference facts. |
| `labels.json` | The parsed maneuver labels (epoch / window / type / Δv). | DORIS/IDS open data; GPS NANUs are US-Government public domain; Galileo NAGUs are GSC reuse-with-attribution (© EU); GEO labels are self-derived (authored). |
| `manifest.json` | One SHA-256 per reconstructed series — the integrity check. | A one-way digest; carries no element data. |
| `splits.json` | The frozen, leak-free train/val/test partition — present once a version's benchmark split is frozen. | Authored. |

The **raw element-series is never shipped.** Space-Track's terms of use reach derived analysis, so
the multi-year history is re-fetched locally from each user's own account rather than redistributed
(the recipe-first model). The labels, the recipe parameters, and the per-series digests are all
either openly licensed or carry no usable catalogue data, so they ship directly.

`recipe.json` is committed (a serialisation of the pinned in-code catalogue), as is `splits.json`
once the version's benchmark split is frozen. `labels.json` and `manifest.json` are produced by a
reconstruction run (below), because computing a real content hash requires fetching the real series.

## Class scope

- **LEO** — the DORIS/IDS satellites that publish a `man.txt` maneuver file: the altimetry missions
  (the Δv-labelled core) and the SPOT satellites.
- **MEO** — two operator constellations: the **GPS** satellites (labels from the FCSTDV "forecast
  delta-V" NANU notices) and, from v0.2, the **Galileo** satellites (labels from the `PLN_MANV` NAGU
  notices). Both are epoch-only.
- **GEO** — from v0.2, actively station-kept geostationary satellites; with no public GEO operator
  maneuver feed, their labels are **self-derived** from the element series by longitude-drift
  inspection (best-effort, epoch-only — see `maneuver_detect.labels.longitude_shift`).

## Reconstructing / verifying

The series come from Space-Track, which needs an account. Set credentials in the environment and
run the build:

```bash
export SPACETRACK_USERNAME='you@example.com'
export SPACETRACK_PASSWORD='…'
uv run maneuver-detect dataset build --out dataset/v0.2   # --recipe-version defaults to 0.2.0
```

This fetches each catalogue object's mean-element history from Space-Track (cached and rate-limited),
crawls the open DORIS `man.txt` files, the CelesTrak NANU archive, and the GSC Galileo NAGU archive
for the operator labels, derives the GEO labels from the reconstructed series, and writes
`recipe.json`, `labels.json`, and `manifest.json`. It is a long run — the label archives are crawled
file-by-file at a polite rate, in addition to the per-object Space-Track fetch. Re-running on the same
recipe reproduces identical hashes, so a mismatch against the committed `manifest.json` means the
reconstruction diverged. Pass `--recipe-version 0.1.0` to rebuild the frozen v0.1 set instead.

The GPS NANU and Galileo NAGU archives are crawled over a year window — `--nanu-start-year`
(default 2016) and `--nanu-end-year` (default: the current year) — so `labels.json` is a snapshot of
that window. `splits.json` is regenerated from `labels.json` by `make_splits` (no series needed).

The raw fetched series is held only in the local cache; it is never written into this directory.

## Licence

The **authored dataset artifacts** — the recipe (`recipe.json`), the parsed label mapping
(`labels.json`), the splits (`splits.json`), and the content-hash manifest (`manifest.json`) — are
released under **[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)**. The underlying label
sources pass through under their own terms: the DORIS/IDS `man.txt` maneuver files are open data, the
GPS NANU notices are US-Government public domain, and the Galileo NAGU notices are reused from the GSC
with attribution (**© EU**); the GEO labels are self-derived (authored). The **raw Space-Track element
history is not redistributed** under any licence — it is re-fetched locally from each user's own
account under Space-Track's terms (the recipe-first model above). No model weights ship in this
release.

