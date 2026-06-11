# maneuver-detect v0.1 dataset — recipe-first distribution

This directory holds the **distributable** form of the labelled maneuver-detection dataset. The
dataset is published as a *recipe*, not as raw catalogue data: each user reconstructs the
element-series locally from their own catalogue access, and a content-hash manifest makes that
reconstruction verifiable bit-for-bit.

## What ships here

| File | What it is | Source terms |
|---|---|---|
| `v0.1/recipe.json` | The pinned catalogue — every object's NORAD id, orbit class, label source, and the per-object fetch parameters. | Public reference facts. |
| `v0.1/labels.json` | The parsed operator maneuver labels (epoch / window / type / Δv). | DORIS/IDS open data; GPS NANUs are US-Government public domain. |
| `v0.1/manifest.json` | One SHA-256 per reconstructed series — the integrity check. | A one-way digest; carries no element data. |

The **raw element-series is never shipped.** Space-Track's terms of use reach derived analysis, so
the multi-year history is re-fetched locally from each user's own account rather than redistributed
(the recipe-first model). The labels, the recipe parameters, and the per-series digests are all
either openly licensed or carry no usable catalogue data, so they ship directly.

`recipe.json` is committed (it is a serialisation of the pinned in-code catalogue). `labels.json`
and `manifest.json` are produced by a reconstruction run (below), because computing a real content
hash requires fetching the real series.

## Class scope

- **LEO** — the DORIS/IDS satellites that publish a `man.txt` maneuver file: the altimetry missions
  (the Δv-labelled core) and the SPOT satellites.
- **MEO** — the operational GPS constellation; labels are the FCSTDV ("forecast delta-V") notices
  from the CelesTrak NANU archive (epoch-only).
- **GEO** — deferred (no public GEO maneuver-label file source).

## Reconstructing / verifying

The series come from Space-Track, which needs an account. Set credentials in the environment and
run the build:

```bash
export SPACETRACK_USERNAME='you@example.com'
export SPACETRACK_PASSWORD='…'
uv run maneuver-detect dataset build --out dataset/v0.1
```

This fetches each catalogue object's mean-element history from Space-Track (cached and rate-limited),
crawls the open DORIS `man.txt` files and the CelesTrak NANU archive for the maneuver labels,
reconstructs the labelled dataset, and writes `recipe.json`, `labels.json`, and `manifest.json`.
It is a long run — the GPS label archive is crawled file-by-file at a polite rate, in addition to
the per-object Space-Track fetch. Re-running on the same recipe reproduces identical hashes, so a
mismatch against the committed `manifest.json` means the reconstruction diverged.

The GPS NANU archive is crawled over a year window — `--nanu-start-year` (default 2016) and
`--nanu-end-year` (default: the current year) — so `labels.json` is a snapshot of that window.

The raw fetched series is held only in the local cache; it is never written into this directory.
