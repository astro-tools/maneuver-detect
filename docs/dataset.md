# Dataset and label sources

The dataset is the load-bearing piece of the project: a curated, labelled set of satellites whose maneuvers
are known from public operator announcements, paired with the mean-element TLE history those maneuvers show up
in. This page documents how it is distributed, what is in it, and the terms of every source it draws on.

## How the dataset is distributed — recipe-first

The raw TLE history a detector trains on comes from Space-Track, whose terms of use **do not permit
redistributing the raw data or analysis derived from it**. So the dataset is not shipped as a blob. Instead it
is published as a **pinned reconstruction recipe**:

- the **operator labels** (the maneuver epochs, and Δv where the source provides it);
- a **recipe** — the exact set of objects, each one's orbit class and label source, and the per-object
  catalogue source and epoch window to fetch;
- a **content-hash manifest** — a SHA-256 digest per reconstructed series.

You reconstruct the dataset locally from **your own** Space-Track account, and the manifest verifies that what
you rebuilt matches the pinned dataset **byte-for-byte**. Nothing about the recipe carries Space-Track data —
only the parameters needed to re-fetch it — which is what keeps the distribution model compliant. Because each
recipe entry's epoch window scopes *both* the series and that object's labels, the committed label set is a
function of the whole recipe, not the full announced history.

Build it with the CLI (Space-Track credentials in the environment):

```bash
export SPACETRACK_USERNAME=you@example.com
export SPACETRACK_PASSWORD=...
maneuver-detect dataset build --out dataset/
# writes recipe.json, labels.json, and manifest.json
```

The recipe, labels, manifest, and splits are also published to the
[Hugging Face Hub](https://huggingface.co/datasets/astro-tools/maneuver-detect) and downloadable on
first use — `datasets.load_recipe()`, `datasets.load_manifest()`, `datasets.load_labels()`, or
`datasets.fetch_dataset()` for the whole snapshot. You still reconstruct the element series locally
from your own account; the manifest verifies it. See [Models and the Hub](models.md).

## What is in the v0.1 dataset

Two orbit classes, matching the v0.1 label scope:

- **LEO — fifteen altimetry satellites** that publish a DORIS/IDS maneuver file and have a confident NORAD id:
  the altimetry missions (TOPEX/Poseidon, the Jason series, Envisat, CryoSat-2, SARAL, HY-2A, the Sentinel-3
  and Sentinel-6 satellites) and the SPOT imaging satellites. This is the **Δv-labelled core** — the LEO
  sources carry burn magnitudes, not just epochs.
- **MEO — the operational GPS constellation** (space-vehicle number, broadcast PRN, and NORAD id), sourced and
  cross-checked against the CelesTrak GPS catalogue. This table doubles as the SVN→NORAD crosswalk the GPS
  label parser needs. The GPS labels are **epoch-only** — no Δv, no direction.

**GEO is deferred.** There is no public GEO maneuver-label file, so a GEO object would carry an unlabelled
series; it is out of v0.1 (best-effort in scope, but with nothing to label against). HEO is deferred for the
same scarcity reason.

The catalogue is a **pinned snapshot**: a satellite's SVN→NORAD mapping is fixed for its lifetime, but
constellation membership and PRN-slot assignments drift, so each dataset version captures the set as it stood
at sourcing time.

## Catalogue (series) sources

| Source | Auth | What it provides | Terms |
|---|---|---|---|
| **Space-Track** (`gp_history`) | Free account required | The multi-year back-element archive — the training history | The [User Agreement](https://www.space-track.org/documentation#user_agree) and API *Rules of Behaviour* prohibit redistributing raw data **and analysis derived from it**, and ask that you cache locally, avoid repeat queries for the same data, and respect the query-rate limits. The library logs in once per build, caches, and rate-limits accordingly; it **never ships or proxies your credentials or the raw series**. |
| **CelesTrak** (`gp.php`) | None | The **current** GP elset for an object | Subject to CelesTrak's soft ~100 MB/day per-IP cap and a one-download-per-update policy; the library honours both with an on-disk cache and an `If-Modified-Since` conditional GET. CelesTrak serves only the latest elset, so it is the no-credential way to refresh a series go-forward, not a history source. |

## Label sources

| Source | Class | Δv? | Terms |
|---|---|---|---|
| **DORIS / IDS** maneuver files | LEO | **Yes** | Publicly distributed by the [International DORIS Service](https://ids-doris.org/) (mirrored at NASA's CDDIS) under the IDS data use policy — freely available for research, with citation/registration requested. One fixed-format `man.txt` file per altimetry satellite records each maneuver's window and per-axis ΔV. |
| **ILRS** maneuver history | LEO | Yes | The International Laser Ranging Service publishes no separate maneuver-file format; its maneuver *history* links to the same DORIS/IDS files, so that one parser covers both services' quantitative labels. |
| **GPS NANUs** (`FCSTDV`) | MEO | No | Notices Advisory to Navstar Users, published by the U.S. Coast Guard [NAVCEN](https://www.navcen.uscg.gov/) as **public-domain U.S.-Government** text. The `FCSTDV` ("Forecast Delta-V") notice gives a scheduled-maneuver window but no Δv magnitude or direction, so GPS labels are epoch-only. |

Two sources are deliberately **excluded** from the distribution: the **Shorten maneuver benchmark** is used
only as a development-time cross-check (it is unlicensed, so it is never redistributed), and **SpotGEO** is an
optical object-detection dataset, not a maneuver-label source.

## Licensing

- **Code** is MIT.
- **Authored dataset artifacts** — the label mapping, the splits, the manifests, the recipe, and features
  derived from open data — are **CC-BY-4.0**.
- **Openly-licensed pass-through data** keeps its upstream licence; **raw Space-Track data is never
  redistributed**.

Because every label source is open or U.S.-Government public domain, the dataset licence is not forced
restrictive. See the [design decisions](decisions.md) for the full rationale.
