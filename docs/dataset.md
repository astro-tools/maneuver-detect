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

## What is in the v0.3 dataset

Four scored orbit classes — v0.3 adds the **IGSO** class and brings operator-announced labels to GEO,
breaking the v0.2 self-label circularity:

- **LEO — the altimetry and imaging satellites** that publish a DORIS/IDS maneuver file and have a confident
  NORAD id: the altimetry missions (TOPEX/Poseidon, the Jason series, Envisat, CryoSat-2, SARAL, HY-2A, the
  Sentinel-3 and Sentinel-6 satellites) and the SPOT imaging satellites. This is the **Δv-labelled core** —
  the LEO sources carry burn magnitudes, not just epochs.
- **MEO — the GPS and Galileo constellations**, two independent operators. GPS objects (space-vehicle number,
  broadcast PRN, and NORAD id) are sourced and cross-checked against the CelesTrak GPS catalogue, which
  doubles as the SVN→NORAD crosswalk the GPS label parser needs; Galileo objects (GSAT id and SVID) resolve to
  NORAD via the CelesTrak Galileo crosswalk — v0.3 crawls the full Galileo NAGU back-catalogue to thicken the
  class. MEO labels are **epoch-only** — no Δv, no direction.
- **GEO — operator-announced and best-effort.** v0.3 adds two operator feeds: **NOAA GOES** maneuver epochs
  (from the OSPO navigation summary) and the equatorial **QZSS** satellites QZS-3/6, whose Operational History
  Information files carry an executed Δv with a north-south / east-west burn marker. The remaining GEO objects
  keep the v0.2 **longitude-shift self-labelling** of the reconstructed series, since no openly-licensed GEO
  maneuver-label file covers them. GEO labels are **epoch-only**, except the QZSS GEO satellites, which carry
  an operator Δv.
- **IGSO — the new v0.3 scored class.** The inclined, slightly-eccentric Quasi-Zenith satellites QZS-2/4/1R
  (e ≈ 0.075, i ≈ 37–44°), labelled from the same QZSS Operational History Information files. These carry an
  **executed Δv** — magnitude only, because the IGSO files omit the GEO files' burn-direction marker, so no
  type is fabricated — making IGSO the second Δv-labelled class after the LEO core.

**HEO is a reserved class with no objects in v0.3.** No ingestible operator maneuver feed exists for the
high-eccentricity regime — even credentialed — and self-labelling the noisy deep-space TLEs is
perturbation-dominated, so it does not rescue the class. The class member, its detectability-floor entry, and
the self-label deriver are retained for a future source.

The catalogue is a **pinned snapshot**: a satellite's SVN/GSAT/QZS→NORAD mapping is fixed for its lifetime, but
constellation membership and slot assignments drift, so each dataset version captures the set as it stood at
sourcing time. v0.3 partitions the objects by a **frozen temporal-holdout split** — novel satellites scored in
novel eras — with per-split, per-class counts shipped alongside; the [benchmark protocol](benchmark.md)
documents the split contract.

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
| **Galileo NAGUs** (`PLN_MANV`) | MEO | No | Notice Advisory to Galileo Users, published by the [EU Agency for the Space Programme](https://www.gsc-europa.eu/) as machine-ingestible per-notice text under an **attribution-required reuse grant (© EU)** — redistribution-clean, so the labels are shipped. The `PLN_MANV` notice gives a scheduled-maneuver window; GSAT id and SVID resolve to NORAD via the CelesTrak Galileo crosswalk. Epoch-only. |
| **QZSS OHI files** (`ohi-qzs*.txt`) | IGSO + GEO | **Yes** | Per-satellite *Operational History Information* files published by the [Cabinet Office of Japan](https://qzss.go.jp/) under a reuse-with-attribution grant (CC-BY-4.0, *"Source: Quasi-Zenith Satellite System website"*) — redistribution-clean, so the labels are shipped. The only surveyed operator feed besides DORIS that ships an executed Δv. The equatorial QZS-3/6 (GEO) files carry a north-south / east-west burn marker used directly as the operator's maneuver type; the inclined QZS-2/4/1R (IGSO) files omit it, so those labels carry the frame-invariant `|Δv|` magnitude only. Clustered station-keeping burns collapse to one event. |
| **NOAA GOES** navigation summary (`navsum.txt`) | GEO | No | The [NOAA OSPO](https://www.ospo.noaa.gov/) navigation summary names each GOES bird's last-maneuver day, as US-Government **public-domain** text — shipped. It is a live-state file (latest maneuver only, day-of-year granularity), so the maneuver *history* is rebuilt by replaying its Internet-Archive snapshots and deduplicating the epochs. Epoch-only. |
| **Self-labelled longitude shift** | GEO | No | An **authored CC-BY-4.0 artifact**, not pass-through data: GEO station-keeping and relocation maneuvers are inferred in-house from longitude drift in the reconstructed element series, for the GEO objects no operator feed covers. Epoch-only. |

Several sources are deliberately **excluded** from the distribution: the **Shorten maneuver benchmark** is used
only as a development-time cross-check (it is unlicensed, so it is never redistributed); **SpotGEO** is an
optical object-detection dataset, not a maneuver-label source; **GLONASS** notices are excluded because their
terms cap public reproduction more tightly than Space-Track; **EUMETSAT** GEO notices are login- and
JavaScript-gated under a restrictive data policy; and the operator-announced **BeiDou** feed (re-confirmed
uncrawlable in v0.3) carries no open licence, so at most it is a recipe-first option (ship the fetch recipe and
parser, never the labels) rather than shipped data.

## Licensing

- **Code** is MIT.
- **Authored dataset artifacts** — the label mapping, the splits, the manifests, the recipe, and features
  derived from open data — are **CC-BY-4.0**.
- **Openly-licensed pass-through data** keeps its upstream licence; **raw Space-Track data is never
  redistributed**.

Because every shipped label source is open, U.S.-Government public domain (the GPS NANUs and the NOAA GOES
summary), an attribution-required reuse grant (© EU for Galileo NAGUs, and CC-BY-4.0 *"Source: Quasi-Zenith
Satellite System website"* for the QZSS OHI files), or authored in-house (the self-labelled GEO maneuvers), the
dataset licence is not forced restrictive — attribution stacks per source, but no redistribution restriction
attaches. See the [design decisions](decisions.md) for the full rationale.
