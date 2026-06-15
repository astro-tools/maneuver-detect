# maneuver-detect dataset — recipe-first distribution

This directory holds the **distributable** form of the labelled maneuver-detection dataset, one
subdirectory per version (currently `v0.3/`). The dataset is published as a *recipe*, not as raw
catalogue data: each user reconstructs the element-series locally from their own catalogue access,
and a content-hash manifest makes that reconstruction verifiable bit-for-bit.

## What ships here

For each version `vX.Y/`:

| File | What it is | Source terms |
|---|---|---|
| `recipe.json` | The pinned catalogue — every object's NORAD id, orbit class, label source, and the per-object fetch parameters. | Public reference facts. |
| `labels.json` | The parsed maneuver labels (epoch / window / type / Δv). | DORIS/IDS open data; GPS NANUs + NOAA GOES navsum are US-Government public domain; Galileo NAGUs are GSC reuse-with-attribution (© EU); QZSS OHI is reuse-with-attribution (Quasi-Zenith Satellite System website, CC-BY-4.0); self-labelled GEO/HEO are authored. |
| `manifest.json` | One SHA-256 per reconstructed series — the integrity check. | A one-way digest; carries no element data. |
| `splits.json` | The frozen, leak-free **temporal-holdout** train/val/test partition — novel satellites scored in novel eras (the timeline cut into three guard-separated bands, each a disjoint object set). Present once a version's benchmark split is frozen. | Authored. |

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
  delta-V" NANU notices) and the **Galileo** satellites (labels from the `PLN_MANV` NAGU notices).
  Both are epoch-only.
- **GEO** — geostationary satellites. The **GOES** weather satellites carry operator-announced labels
  from the NOAA OSPO navigation summary, and the equatorial **QZSS** satellites (QZS-3/6) carry the
  operator-Δv QZSS OHI labels; the **Meteosat/Himawari** satellites have no public operator feed, so
  their labels are **self-derived** by longitude-drift inspection (best-effort, epoch-only — see
  `maneuver_detect.labels.longitude_shift`).
- **IGSO** — the inclined/eccentric-geosynchronous **QZSS** satellites (QZS-2/4/1R), labelled from
  the Cabinet Office of Japan's Operational History Information (OHI) files — the only surveyed
  operator feed that ships an executed Δv (see `maneuver_detect.labels.qzss_ohi`).
- **HEO** — high-eccentricity apogee/perigee-control objects (XMM-Newton, INTEGRAL, TESS). No public
  operator feed covers HEO, so their labels are **self-derived** from the element series by
  energy/eccentricity-step inspection (best-effort, epoch-only — see `maneuver_detect.labels.heo_self`).

## Reconstructing / verifying

The series come from Space-Track, which needs an account. Set credentials in the environment and
run the build:

```bash
export SPACETRACK_USERNAME='you@example.com'
export SPACETRACK_PASSWORD='…'
uv run maneuver-detect dataset build --out dataset/v0.3
```

This fetches each catalogue object's mean-element history from Space-Track (cached and rate-limited),
crawls the open DORIS `man.txt` files, the CelesTrak NANU archive, the GSC Galileo NAGU archive, the
QZSS OHI files, and the NOAA GOES navigation summaries (via the Internet Archive, for the maneuver
history of those live-state files) for the operator labels, derives the self-labelled GEO and HEO
labels from the reconstructed series, and writes `recipe.json`, `labels.json`, and `manifest.json`.
It is a long run — the label archives are crawled file-by-file at a polite rate, in addition to the
per-object Space-Track fetch. Fetched label files are cached on disk (the immutable operator
notices effectively forever), so re-running re-downloads only the changing archive indexes and
append-only files — a re-run makes almost no requests to the label providers. Re-running on the same
recipe reproduces identical hashes, so a mismatch against the committed `manifest.json` means the
reconstruction diverged.

The GPS NANU and Galileo NAGU archives are crawled over a year window — `--nanu-start-year`
(default 2016) and `--nanu-end-year` (default: the current year) — so `labels.json` is a snapshot of
that window. `splits.json` is regenerated from `labels.json` by `make_temporal_split` (no series
needed); the dense GEO labels collapse the plain satellite-overlap split into one component, so the
frozen v0.2 partition is the temporal-holdout variant, which stays leak-free in both dimensions.

The raw fetched series is held only in the local cache; it is never written into this directory.

## Licence

The **authored dataset artifacts** — the recipe (`recipe.json`), the parsed label mapping
(`labels.json`), the splits (`splits.json`), and the content-hash manifest (`manifest.json`) — are
released under **[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)**. The underlying label
sources pass through under their own terms: the DORIS/IDS `man.txt` maneuver files are open data, the
GPS NANU notices and the NOAA GOES navigation summaries are US-Government public domain, the Galileo
NAGU notices are reused from the GSC with attribution (**© EU**), and the QZSS OHI files are reused
with attribution (**Source: Quasi-Zenith Satellite System website**, CC-BY-4.0); the self-labelled
GEO and HEO epochs are authored. The **raw Space-Track element history is not redistributed** under
any licence — it is re-fetched locally from each user's own account under Space-Track's terms (the
recipe-first model above). No model weights ship in this release.

