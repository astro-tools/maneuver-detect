# V2 spike — public maneuver-label source survey + coverage per class

**Status:** findings + recommendation, feeding **D3** (label sources + v0.1 class scope) and informing
**D9** (the dataset licence inherits the label-source terms). Engineering survey of public sources and
their terms — **not legal advice**; the org owner ratifies at the design freeze.

## Question

What public, machine-ingestible, licence-clean maneuver-label sources actually exist; how much labelled
coverage is obtainable per orbit class (LEO / MEO / GEO / HEO); which classes are viable for the v0.1
benchmark; and does ESA Kelvins **SpotGEO** yield usable maneuver labels?

## Method

Surveyed candidate sources by web research, recording for each: orbit class, fields (epoch / type / Δv),
access (auth?), licence/terms, and obtainable coverage. Prototyped a no-auth ingest of the cleanest
source (GPS NANUs) into the normalised label record to confirm machine-ingestibility.

## Source survey

| Source | Class | Fields | Δv? | Access | Licence / terms | Coverage |
|---|---|---|---|---|---|---|
| **DORIS / IDS maneuver files** (CDDIS + IDS FTP) | LEO (altimetry/DORIS) | epoch + burn info | **often yes** | CDDIS (some products via free NASA Earthdata login; IDS FTP open) | NASA / IDS open data policy | Envisat, TOPEX, Jason-1/2/3, SPOT, Sentinel-3A/3B, Sentinel-6, SARAL, HY-2A, CryoSat-2 — ~dozen sats, multi-year |
| **ILRS maneuver predictions** | LEO (laser targets) | epoch windows | no | open (ilrs.gsfc.nasa.gov) | ILRS open data policy | laser-ranging targets, overlaps the altimetry set |
| **GPS NANUs — FCSTDV** | MEO | epoch window (start/stop) | no | NAVCEN, **no-auth**, archive to 1997 | **US-Government public domain** | ~31 active GPS SVs, frequent station-keeping; carries SVN/PRN (NORAD via CelesTrak crosswalk) |
| Other GNSS notices (Galileo NAGUs, GLONASS, BeiDou) | MEO | epoch | no | varies (some public) | varies | survey/defer — availability uneven |
| **Shorten benchmark** (`dpshorten/TLE_observation_benchmark_dataset`) | GEO + LEO | epoch-only (ISO8601, YAML) | no | GitHub, open | **NO LICENSE file — unclear** | 15 sats: 5 GEO (Fengyun-2D/2E/2F/2H, 4A), 8 LEO altimetry, 2 high-LEO |
| GEO longitude-shift inspection (literature method) | GEO | epoch / Δlongitude | no | derived/self-labelled | n/a | best-effort, per-object |
| CSpOC / Space-Track | — | — | — | restricted | **redistribution-restricted (see V1)** | no standard public maneuver-label feed; not a label source here |
| **ESA Kelvins SpotGEO** | — | — | — | open dataset | CC BY-NC-SA | **not a maneuver-label source — see below** |

### SpotGEO — rejected (decided, not assumed)

SpotGEO is an **optical object-detection** challenge: 6400 five-frame grayscale image sequences (32,000
PNGs) from a low-cost ground telescope; the ground truth is **pixel coordinates of GEO/near-GEO objects**.
Orbital maneuvers appear only as a *nuisance factor* ("in rare cases, orbital manoeuvres conducted by an
active GEO satellite"). It contains **no maneuver epochs / labels** and is therefore **out of scope** as a
label source. (The charter explicitly flagged this to decide rather than assume.)

### The Shorten benchmark — precedent, cross-check, licence caveat

`dpshorten/TLE_observation_benchmark_dataset` (arXiv:2212.08662, *J. Spacecraft & Rockets* 2023) is the
closest existing artifact: **15 satellites with high-quality ground-truth `manoeuvre_timestamps`** (ISO8601,
in per-object YAML; maneuvers correspond to large mean-motion changes). It is valuable as **prior art, a
cross-validation oracle, and a possible bootstrap**, but it has **no LICENSE file** (same gap as the
SpaceTrack-TimeSeries precedent from V1), so its labels cannot be redistributed by us without clarifying
the licence. Treat it as a **dev-only cross-check** unless/until the licence is confirmed; self-source the
shipped labels from the open primary sources above.

## Per-class coverage and the v0.1 class scope (→ D3)

- **LEO — IN (primary class).** The altimetry/DORIS satellites (DORIS/IDS files, ILRS) give the
  highest-quality labels, **with Δv/burn information** — the richest ground truth and the basis of the
  literature's standard validation pairs (TOPEX 22076, Envisat 27386). This is the Δv-labelled set.
- **MEO — IN.** GPS NANUs (FCSTDV) are public-domain, no-auth, high-volume (decades × ~31 SVs), and
  cleanly machine-ingestible — **epoch-only** (no Δv).
- **GEO — best-effort IN (epoch-only).** Coverable via the Shorten Fengyun set (licence permitting) and
  the literature's longitude-shift inspection method; labels are weaker and **carry no Δv**. Include if it
  clears a minimum count at the label-layer stage; otherwise defer.
- **HEO — DEFER from v0.1.** True HEO (Molniya/Tundra/GTO) public maneuver logs are sparse. (Note: the
  Shorten set tags Sentinel-6/TOPEX as "HEO", but those are ~1336 km **high-LEO**, not true HEO.)

**Recommendation:** v0.1 scopes **LEO (primary, Δv-labelled) + MEO (epoch-only) + GEO (best-effort,
epoch-only); HEO deferred.** The **Δv-labelled subset is LEO/altimetry (DORIS)** — exactly the validation
set V4 uses for the Δv-inversion check, so V2 and V4 line up.

## D9 (licensing) implication

The viable label sources are predominantly **open / US-Government public domain** — GPS NANUs (USCG public
domain), DORIS/IDS and CDDIS (NASA/IDS open data policy), ILRS (open). So **the label sources do not force
the dataset licence to be restrictive**: D9's CC-BY-4.0 for authored artifacts remains viable, with
per-source attribution. **Caveat:** the Shorten benchmark is unlicensed — do not redistribute its labels
without clarification; and confirm the CDDIS/IDS and ILRS data policies before finalising D9.

## Machine-ingestibility proof

The cleanest source (GPS NANUs, no-auth, public domain) is shown to ingest into the normalised label
record `(norad_id, epoch, type, delta_v, source, ...)` by
[`v2_label_ingest_proof.py`](v2_label_ingest_proof.py) (stdlib only):

- **Part A (offline, reproducible):** two embedded FCSTDV NANUs are parsed into normalised records —
  SVN/PRN resolved to a NORAD id via a crosswalk where known (SVN62 → 36585; SVN74 → unmapped → `None`),
  JDAY + Zulu converted to ISO8601 UTC (JDAY 086/2025 → `2025-03-27`), `delta_v` left `None` (NANUs give
  no magnitude). JSON-serialised + SHA-256, identical across two separate processes:

  ```
  057b5f9762bea125f647d89d5fe744ee23b76b05ab0669b1d975fb0c7fc08e28
  ```

- **Part B (best-effort, no-auth):** a live NAVCEN `current_nanu.nnu` fetch succeeds (the no-auth ingest
  leg works); the fetched data is not written to disk or committed.

NANUs are US-Government public domain, so embedding samples is redistribution-clean (unlike Space-Track
data — see V1). The proof confirms the **format → normalised record** path and its determinism; full
volume counts come from a real pull during the label-layer work.

## Open items

- **Confirm the DORIS/IDS maneuver-file format and access** (which products are open vs. need a free
  Earthdata login; whether they carry Δv per maneuver) — during the label-layer work (#10).
- **Clarify the Shorten benchmark licence** before any redistribution of its labels; until then treat it
  as a dev-only cross-validation oracle.
- **Quantify exact per-class counts** via a real pull at #10 — the table's coverage is source-level, not a
  head count yet.
- **Firm up the GEO labelling approach** (Shorten vs. longitude-shift inspection) and confirm the
  **SVN/PRN → NORAD crosswalk** source (CelesTrak GPS catalogue).
- Ratify D3 (and the D9 licence interaction) at the design freeze.

## References

- GPS NANUs — US Coast Guard NAVCEN: <https://www.navcen.uscg.gov/nanu-abbreviations-and-descriptions>;
  current file <https://www.navcen.uscg.gov/sites/default/files/gps/nanu/current_nanu.nnu>; CelesTrak NANU
  types/templates <https://celestrak.org/GPS/NANU/description.php>.
- DORIS maneuver files — NASA CDDIS DORIS data center
  <https://cddis.nasa.gov/Data_and_Derived_Products/DORIS/>; International DORIS Service
  <https://ids-doris.org/>.
- ILRS maneuver predictions — <https://ilrs.gsfc.nasa.gov/data_and_products/predictions/maneuver.html>.
- Aviso+ altimetry missions (orbit maintenance) — <https://www.aviso.altimetry.fr/en/missions.html>.
- Shorten et al., "Wide-scale Monitoring of Satellite Lifetimes: Pitfalls and a Benchmark Dataset" —
  <https://arxiv.org/abs/2212.08662>; dataset <https://github.com/dpshorten/TLE_observation_benchmark_dataset>.
- ESA Kelvins SpotGEO (optical object detection, *not* maneuver labels) —
  <https://kelvins.esa.int/spot-the-geo-satellites/>.
- GEO maneuver detection / classification prior art — Roberts (AMOS 2021); Kelecy (AMOS 2007).
