# V1 spike — dataset redistribution & licensing model

**Status:** findings + recommendation, feeding decisions **D2** (dataset distribution model) and
**D9** (dataset / weight licensing) at the design freeze. Engineering assessment of public terms of
use — **not legal advice**; the org owner ratifies the final call.

## Question

Can a TLE-derived, labelled maneuver-detection dataset be published openly, given the terms attached
to its source data — and if so, in what form? The roadmap and charter flag this as the load-bearing
prerequisite: it gates the dataset deliverable and the Hugging Face Hub release.

## Findings

### 1. The constraint is statutory + contractual, not copyright

Two-line element sets are produced by the U.S. Space Force's 18th/19th Space Defense Squadron and are
U.S. Government work product — facts, not creative works. Under 17 U.S.C. §105 (no copyright in U.S.
Government works) and the idea/fact doctrine, **TLEs carry no copyright**. So the limit on
redistribution is *not* a copyright licence — it is the **terms of use** that ride on the channel the
data came through.

### 2. Space-Track (the catalogue source) — redistribution is restricted, including derived analysis

Space-Track.org is operated under the DoD SSA-sharing authority (**10 U.S.C. §2274**). Its User
Agreement (cited to §2274(c)(2)) binds every account holder:

> "The User agrees not to transfer any data or technical information received from this website, or
> other U.S. Government source, **including the analysis of data**, to any other entity without prior
> express approval."

Two consequences that shape the decision:

- Redistributing **raw** Space-Track TLEs publicly is not permitted without express approval.
- The clause explicitly reaches **"the analysis of data"** — so a *derived-feature* product built from
  Space-Track data does **not** escape the restriction merely by being derived. This rules out
  "ship derived features" as a compliance shortcut for Space-Track-sourced data.

The underlying statute (10 U.S.C. §2274) also notes that a **basic level of SSA data — the publicly
releasable portion of the DoD catalogue — is provided free of direct user fees**, which is the hook
the open-data successor (below) hangs on.

### 3. CelesTrak — an authorized redistributor, but no downstream open licence

CelesTrak (Dr. T.S. Kelso) holds a standing Air Force authorization to redistribute Space-Track TLEs
and Space Situation Report data ("until superseded by formal updated documentation signed out by
either the AFSPC/A3 or 14 AF/CC"). That authorization covers *CelesTrak's own* redistribution; the
site publishes **no open licence granting downstream users the right to re-redistribute**, emphasises
Space-Track as the primary source, and enforces a one-download-per-update rate discipline. So CelesTrak
is an excellent **fetch** source (no-auth, current GP), but **sourcing a redistributable dataset from
CelesTrak rests on an authorization granted to CelesTrak, not to us** — too thin a basis to ship raw
TLEs under.

### 4. TraCSS (Office of Space Commerce) — basic SSA data under CC0-1.0 (the open path)

The civil SSA mission is transitioning from DoD to NOAA's **Office of Space Commerce** under Space
Policy Directive-3, via the **Traffic Coordination System for Space (TraCSS)** (planned production
release January 2026). Its data policy states:

> "With extremely limited exceptions, TraCSS data and data from satellite owner/operators will be made
> publicly available under an open data license (**CC0-1.0**)" — "free of charge in partnership with
> commercial cloud providers."

CC0-1.0 is a public-domain dedication: **freely redistributable by anyone**, and the **public elements
of the TraCSS Cat are openly queryable and downloadable** — structured JSON, via the TraCSS website and
a machine-to-machine API, and the catalogue publishes **OMM (Orbit Mean-Element Messages,
TraCSS-Spec-004)**, i.e. exactly the mean elements this project consumes. Owner/operator *registration*
(open since May 2026) is for receiving services, not for reading the public catalogue.

**But TraCSS is a current / go-forward catalogue, not a deep historical archive.** The FAQ confirms
Space-Track "will not be turned off" and runs "in parallel", and the multi-year back-element archive
(Space-Track's `gp_history` — 138M+ elsets) is **not** something TraCSS provides today. So TraCSS is
the clean open source for **current / go-forward** mean elements — and the path to migrate toward — but
the **multi-year history that training needs still comes from Space-Track**, which is exactly what the
reconstruction recipe is for. (Residual: confirm public catalogue *reads* need no registration/rate
gating — verified during the data-layer work against the TraCSS-Spec docs / a direct OSC query.)

### 4b. A redistribution-clean (but stale) historical source: the McDowell archive

Jonathan McDowell's *Historical TLE Orbital Elements* archive (planet4589.org) is notable for being
**explicitly redistribution-clean**: "All the TLE data on this site originating from US government
sources was obtained from other public sources, or else from the GSFC OIG site **under agreements that
did not restrict redistribution of the data**" (Russian Vympel data excluded per their restrictions).
The catch: it was last meaningfully updated ~April 2022, "does not contain recent element sets", and
covers catalogue ranges up to ~52200 — so it is a **directly-redistributable but stale and incomplete**
historical slice, useful for seeding older-epoch coverage, not a substitute for the live catalogue. No
explicit licence is stated (worth confirming before relying on it).

### 5. Precedent

The most directly comparable recent work, **SpaceTrack-TimeSeries** (Shanghai Jiao Tong Univ., 2026),
publishes a curated TLE + ephemeris dataset on **Figshare** with the **crawling/processing code on
GitHub** "enabling full reproducibility", frames it as a *derived/processed* product, asserts
Space-Track ToU compliance, and **states no explicit licence**. It shows that academic groups do ship
TLE-derived datasets — but the cleanest ones lead with a **reproducible fetch/processing pipeline**,
and the licensing is left ambiguous (a gap we should not copy).

## Options evaluated

| Option | Verdict |
|---|---|
| **(a) Labels + pinned reconstruction recipe** — users re-fetch from their own account and re-derive locally; we ship no raw catalogue data | **Recommended core.** Unambiguously compliant: we never transfer Space-Track data or its analysis. Each user operates under their own Space-Track agreement. |
| **(b) Derived non-reconstructable features** as a *compliance escape* for Space-Track-sourced data | **Rejected as an escape.** The User Agreement covers "the analysis of data", so features derived from Space-Track data are still restricted. (Derived features remain fine when their *source* is openly licensed — see (c)/TraCSS.) |
| **(c) Directly ship data sourced from an open licence** (TraCSS CC0-1.0; the McDowell archive; operator-published ephemerides) | **Recommended opportunistic layer.** Whatever is sourced under CC0/open terms ships directly. Bounded today: TraCSS CC0 is current/go-forward (not deep history); the McDowell archive is redistribution-clean but pre-2022 and incomplete — so the directly-shippable *historical* layer is currently thin, and multi-year history comes via the recipe. |

## Recommendation

### D2 — dataset distribution model: **recipe-first hybrid**

1. **Publish, in-repo / on the Hub:**
   - the operator-sourced maneuver **labels** (licensed per their own sources — V2 / D3 settles this);
   - a **pinned reconstruction recipe** — the fetch code, the exact NORAD-ID catalogue, per-object
     date ranges and query parameters, and a **per-series content-hash manifest** (e.g. SHA-256 over
     the canonical mean-element series) so a reconstruction can be *verified* bit-for-bit;
   - **directly-shippable data only where its source is openly licensed** — current/go-forward mean
     elements from **TraCSS (CC0-1.0, OMM)**, the **redistribution-clean pre-2022 McDowell archive**
     (caveated: stale, incomplete), and operator-published ephemerides — plus splits and features
     derived from those open sources.
2. **Do not redistribute** raw Space-Track TLEs, or features/analysis derived from Space-Track data.
   Users reconstruct locally from their own Space-Track account (or from CelesTrak / TraCSS).
3. **Grow the directly-shipped (download, no-reconstruction) layer as open coverage matures.** TraCSS
   CC0 covers current/go-forward epochs today; as its historical depth grows, progressively more of the
   dataset becomes a plain CC0 download rather than a Space-Track reconstruction. The multi-year
   *history* training needs comes from Space-Track via the recipe until then.

This makes the dataset reproducible and citable *without* a redistribution-rights blocker, and lets the
open (CC0) fraction grow over time. It is also why the benchmark's **splits, matching rule, and
content-hash manifest are the load-bearing published artifacts** — they make a reconstructed dataset
verifiably identical to the one the baselines were trained on.

### D9 — licensing

- **Code:** MIT (org convention).
- **Authored dataset artifacts** (label mapping, splits, manifests, the recipe, and features derived
  from openly-licensed sources): **CC-BY-4.0** (attribution; standard for a citable benchmark). CC0 is
  a viable alternative if maximal reuse is preferred — the freeze picks.
- **Pass-through open data** (TraCSS CC0) stays CC0; per-source provenance and terms are documented on
  the dataset card.
- **Model weights / checkpoints:** MIT or CC-BY-4.0; foundation-model fine-tunes inherit their base
  licence (Chronos / TimesFM are Apache-2.0 — compatible).
- **Raw Space-Track TLEs:** not redistributed.
- The dataset licence is **subject to the label-source licences** surfaced by V2 — if a label source
  imposes a share-alike or non-commercial term, D9 narrows accordingly.

## Reconstruction-determinism proof

The recipe-first model only works if reconstruction is **byte-deterministic** — re-running the recipe
on the same pinned input must yield an identical series, so the published hash manifest is a real
integrity check. Demonstrated by [`v1_reconstruct_proof.py`](v1_reconstruct_proof.py) (stdlib only):

- **Part A (offline, reproducible):** a pinned 8-point *synthetic* elset series (fictional catalogue id
  90001, with an injected mean-motion step) is parsed → mean-element series → canonically serialised →
  SHA-256, twice. Both runs, and two *separate* process invocations, produce the identical digest:

  ```
  c406654b90af3de5ee637b3f4d51345ea1595ac37fda1d2bfa4198c4439bbf19
  ```

- **Part B (best-effort, no-auth):** a live CelesTrak GP fetch for CATNR 25544 (ISS) parses to sensible
  mean elements (n ≈ 15.49 rev/day), confirming the no-auth **fetch leg** works. The fetched data is
  never written to disk or committed.

Synthetic data is used deliberately — the proof needs *determinism of the derivation*, not real
catalogue data, and using synthetic elsets keeps the repo free of any redistributed TLEs (the
recommendation, practised). Full **Space-Track historical** reconstruction needs the user's own
credentials and is validated when the data layer lands.

## ToS-compatibility confirmation

- **Recipe-first (labels + reconstruction):** compliant — no transfer of Space-Track data or its
  analysis; each user fetches under their own agreement.
- **TraCSS-sourced data:** redistributable under CC0-1.0.
- **Label sources:** out of scope here — their licences are surveyed by V2 and feed D3 / D9.

## Open items

- **TraCSS historical depth + access — verified (this spike).** The public TraCSS Cat is openly
  queryable/downloadable (CC0-1.0, JSON + machine-to-machine API, publishing OMM), but it is a
  **current / go-forward** catalogue, **not** a deep historical archive — Space-Track's `gp_history`
  (138M+ elsets) remains the multi-year source, fetched via the recipe. Residual to confirm during
  data-layer work: that public catalogue *reads* need no registration/rate gating (per the TraCSS-Spec
  docs / a direct OSC query), and the McDowell archive's licence status for its pre-2022 slice.
- **Confirm the dataset licence against the V2 label-source licences** before finalising D9.
- Ratify D2 / D9 at the design freeze.

## References

- Space-Track User Agreement & documentation — <https://www.space-track.org/documentation> (User
  Agreement; cites 10 U.S.C. §2274(c)(2)).
- 10 U.S.C. §2274 (DoD SSA data sharing; basic data free of user fees).
- CelesTrak system notices / redistribution authorization — <https://celestrak.org/NORAD/elements/notice.php>;
  GP data — <https://celestrak.org/NORAD/documentation/gp-data-formats.php>.
- TraCSS Data & Information Policy (CC0-1.0) — Office of Space Commerce,
  <https://space.commerce.gov/traffic-coordination-system-for-space-tracss/tracss-user-agreement-data-policy/>;
  FAQ (Space-Track runs "in parallel", not turned off) —
  <https://space.commerce.gov/traffic-coordination-system-for-space-tracss/tracss-frequently-asked-questions/>;
  OMM catalogue format — TraCSS-Spec-004,
  <https://space.commerce.gov/wp-content/uploads/2026/01/TraCSS-Spec-004-v1.2_OMM.pdf>.
- Historical TLE archive (redistribution-clean, pre-2022) — J. McDowell, Jonathan's Space Pages,
  <https://planet4589.org/space/ele.html>.
- 17 U.S.C. §105 — no copyright in U.S. Government works.
- Precedent: "SpaceTrack-TimeSeries" — <https://arxiv.org/abs/2506.13034> (Figshare data + GitHub
  reconstruction code).
