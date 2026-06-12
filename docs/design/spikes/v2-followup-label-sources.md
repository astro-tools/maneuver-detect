# V2 follow-up spike — licence-clean GEO + non-GPS GNSS maneuver-label sources

**Status:** findings + recommendation, extending the V2 survey
([`v2-label-sources.md`](v2-label-sources.md)) and feeding **D13** (the ratified v0.2
label-source set), with a knock-on to **D3** (class scope) and **D9** (licence inheritance). This is
the source survey the v0.2 dataset-growth pass is blocked on. Engineering survey of public sources and
their terms — **not legal advice**; the org owner ratifies the set.

## Question

The v0.1 set is **DORIS/IDS-LEO + GPS-NANU-MEO only**, with **no GEO label path** (the catalogue notes
"there is no public GEO maneuver-label file source"). Two gaps drive v0.2 growth:

1. **GEO station-keeping notices** — is there a public, machine-ingestible, redistribution-clean source
   of GEO maneuver epochs (operator notices / longitude-slot logs / a licensed benchmark)?
2. **Non-GPS GNSS notices** — Galileo / GLONASS / BeiDou: which are public + machine-ingestible +
   licence-clean, and what coverage do they add beyond the GPS NANUs?

For each candidate: orbit class, fields (epoch / type / Δv), access (auth?), licence/terms, obtainable
coverage — the V2 table shape — then a no-auth ingest proof for the cleanest new source.

## Method

Surveyed each provider's operator-notice channel and its terms of use; confirmed the on-disk notice
format from a real notice file per source; checked each against the **D2 redistribution model** (ship
*labels* only where openly licensed, else ship a *reconstruction recipe*, never redistribute terms-bound
data). Prototyped a no-auth ingest of the cleanest new source (Galileo NAGUs) into the normalised label
record to confirm machine-ingestibility ([`v2_followup_label_ingest_proof.py`](v2_followup_label_ingest_proof.py)).

## Source survey

| Source | Class | Fields | Δv? | Access | Licence / terms | Coverage |
|---|---|---|---|---|---|---|
| **Galileo NAGU — `PLN_MANV`** (GSC) | MEO | start/end UTC + GSAT id + SVID | no | gsc-europa.eu, **no-auth**, per-notice `.txt`, archive to 2013 | **EU reuse — reproduction/use authorised with attribution (© EU)** | ~28 active GSAT SVs, frequent planned maneuvers; second independent MEO operator |
| **GLONASS NAGU** (IAC / TsNIIMash) | MEO | slot + window | no | glonass-iac.ru portal + FTP archive, no-auth | **NOT open — public reproduction capped at 150 chars w/o consent (see below)** | ~24 slots; redistribution-blocked |
| **BeiDou NABU** (CSNO TARC) | **GEO + IGSO** + MEO | start/end UTC + satellite (names GEO) + type | no | csno-tarc.cn, **no-auth**, per-notice `.zip` at a stable URL | **no open licence stated (Chinese-gov); terms unclear** | BDS-3 3×GEO + 3×IGSO + 24×MEO; GEO/IGSO maneuver often |
| **NOAA GOES** alert messages (OSPO) | GEO | free-text, date-time | no | ospo.noaa.gov, no-auth | **US-Government public domain** | few GOES; SK maneuvers buried in admin free-text, sparse |
| GEO longitude-shift inspection (literature) | GEO | epoch / Δlongitude | no | derived/self-labelled | n/a (self-generated) | best-effort, per-object (unchanged from V2) |
| Shorten Fengyun GEO set | GEO | epoch-only | no | GitHub, open | **NO LICENSE — dev-only** (unchanged from V2) | 5 GEO (Fengyun) |

### Galileo NAGU — the clean win (MEO)

NAGUs are published by the European GNSS Service Centre as flat `KEY: value` `.txt` notices at a stable
per-notice URL, with an active/archived listing and ~40 typed categories. The maneuver type is
**`PLN_MANV`** ("planned activity affecting the attitude and/or orbit"), carrying `START`/`END DATE
EVENT (UTC)` already in calendar UTC, the `SATELLITE AFFECTED` (GSAT id) and `SPACE VEHICLE ID`. A real
notice (NAGU 2025001, GSAT0102):

```
NAGU TYPE: PLN_MANV
NAGU NUMBER: 2025001
START DATE EVENT (UTC): 2025-01-22 06:00
END DATE EVENT (UTC): 2025-02-01 23:05
SATELLITE AFFECTED: GSAT0102
SPACE VEHICLE ID: 12
```

**Terms:** the GSC site permits "downloading, reproduction and use of all the materials … provided the
source is acknowledged" (© EU). That is an **attribution-required reuse grant** — redistribution-clean,
not a no-redistribution clause. It clears the D2 model directly (ship the labels with the © EU
attribution) and sits comfortably under D9's CC-BY-4.0 for authored artifacts (attribution stacks per
source). Galileo is **epoch-only** (no Δv), like the GPS NANUs, and adds a **second, independent MEO
operator** — different maneuver cadence/strategy than GPS — roughly doubling the MEO label population.

### GLONASS NAGU — blocked on terms (decided, not assumed)

The IAC (glonass-iac.ru, owner JSC TsNIIMash) publishes constellation-status messages and a Notice
Advisory to GLONASS Users, but its **"Rules for the use of information"** are **not an open licence**:
public internet reproduction without consent is capped at **150 printed characters** (§2.1.3), and
"making available to the public … in a volume greater than" that "without the consent of the Information
Owner is prohibited" (§2.2). A redistributed maneuver-label set built from GLONASS notices would far
exceed that. This is **more restrictive than Space-Track** (which D2 already declines to redistribute),
so GLONASS is **out** as a shipped or recipe source for the public benchmark — **no-go on terms**,
independent of format. (Per §2.1.1 an individual may still self-fetch for personal research; that is not
a project distribution path.)

### BeiDou NABU — the GEO path, via recipe-reconstruction

CSNO's Test & Assessment Research Center publishes Notices Advisory to BDS Users as `.zip` files at a
**stable URL pattern** (`…/data/upload/nabu/NABU-YYYY-NNNN.zip`), in a flat `KEY  value` format that
mirrors the Galileo NAGU (`NABU TYPE`, `START`/`END DATE EVENT (UTC)`, `SATELLITE INFORMATION`),
**UTC-precise**, and — crucially — it **names GEO satellites explicitly** (e.g. "BDS-3 GEO-01"). BDS-3
flies 3 GEO + 3 IGSO satellites that station-keep frequently, so NABU is the **only operator-announced,
machine-ingestible GEO maneuver-notice feed found** — the gap V1 said had no source.

Two caveats keep it conditional, not clean:

- **Terms:** csno-tarc.cn states **no open licence**; redistribution rights are unclear (Chinese-gov
  data). So BeiDou labels are **not shipped** — they are handled by the **D2 recipe-first model exactly
  as Space-Track is**: publish the *reconstruction recipe* (the stable `.zip` URL pattern + the parser +
  a content-hash manifest), and the user self-fetches. That makes a sizable GEO/IGSO label path viable
  **without redistributing** terms-bound data.
- **Maneuver type string to confirm:** the surveyed sample was a PRN reallocation
  (`RNSS_GNR_PRN_REALLOCATE`); the exact `NABU TYPE` value for an orbit maneuver is **to confirm at
  ingest** (the taxonomy, UTC epochs, and GEO coverage are proven; the one string is not). The literature
  independently confirms BDS GEO/IGSO maneuvers are frequent and announced.

Note the BeiDou GEO/IGSO maneuver epochs that appear in the academic literature are **researcher-derived
from observations**, not taken from NABU — that is the same self-labelled class as longitude-shift
inspection, not an operator source.

### NOAA GOES — clean but weak (GEO)

NOAA OSPO satellite messages are **US-Government public domain** (redistribution-clean, like the GPS
NANUs), and GOES are GEO. But station-keeping/drift maneuvers are **buried in free-text administrative
messages**, not a structured maneuver-epoch feed, and coverage is a handful of GOES with infrequent SK.
Usable as a clean *supplementary* GEO source, not a primary one.

## Per-class go/no-go (→ D13)

- **MEO — GO: add Galileo NAGU.** Clean (attribution-required), no-auth, machine-ingestible `.txt`,
  `PLN_MANV` epoch windows; ~doubles MEO and adds a second operator. **GLONASS: NO-GO** (terms). This is
  a real, redistributable growth increment available now.
- **GEO — conditional GO via BeiDou NABU (recipe-reconstructed).** A machine-ingestible, UTC-precise,
  GEO/IGSO operator feed exists; ship it as a **recipe** (not redistributed labels) per D2, pending the
  maneuver-type-string confirmation. **Fallbacks:** public-domain NOAA GOES (sparse) and self-labelled
  longitude-shift inspection on reconstructed GEO series (no redistribution issue — self-generated).
  Shorten Fengyun stays **dev-only**. GEO remains **epoch-only** (no Δv).
- **LEO / HEO — unchanged from D3** (LEO primary Δv-labelled; HEO deferred).

**Net for sizing:** v0.2 growth gets (a) a clean MEO increment (Galileo, redistributable) and (b) a
concrete GEO path (BeiDou NABU, recipe-reconstructed). Neither blocks on a "licensed benchmark" that does
not exist — confirmed: **no openly-licensed GEO maneuver benchmark is redistributable**.

## D2 redistribution-model compliance

| Source | Shipped how (D2) | Compliant? |
|---|---|---|
| Galileo NAGU | **labels shipped** + © EU attribution | yes — open w/ attribution |
| BeiDou NABU | **recipe only** (URL pattern + parser + hash manifest); not redistributed | yes — Space-Track pattern |
| NOAA GOES | labels shippable (public domain) | yes — public domain |
| GLONASS NAGU | neither (terms forbid public reproduction > 150 chars) | **no — excluded** |
| Shorten Fengyun | dev-only cross-check, never shipped | yes — unchanged |

Nothing recommended here ships data that fails D2.

## D9 (licensing) implication

The added redistributable sources are **open-with-attribution (Galileo, © EU)** and **US-Government
public domain (NOAA)** — neither forces the dataset licence restrictive, so **D9's CC-BY-4.0 for authored
artifacts holds**, with attribution stacked per source (USCG public domain for NANUs, NASA/IDS for DORIS,
© EU for Galileo, NOAA for GOES). BeiDou contributes only a recipe (no redistributed bytes), so it adds
no licence obligation to shipped artifacts beyond crediting CSNO as the upstream the recipe fetches.
GLONASS is excluded, so its restrictive terms never touch the dataset.

## Machine-ingestibility proof

The cleanest new source (Galileo NAGUs, no-auth, attribution-clean) is shown to ingest into the normalised
label record `(norad_id, epoch, type, delta_v, source, ...)` by
[`v2_followup_label_ingest_proof.py`](v2_followup_label_ingest_proof.py) (stdlib only):

- **Part A (offline, reproducible):** two embedded `PLN_MANV` NAGUs (NAGU 2025001 verbatim + one in the
  same `.txt` format) are parsed into normalised records — `SATELLITE AFFECTED` resolved to a NORAD id
  via a GSAT→NORAD crosswalk where known (GSAT0102 → 37847; GSAT0220 → unmapped → `None`), UTC windows
  carried straight through (no day-of-year conversion, unlike NANUs), `delta_v` left `None`. JSON-serialised
  + SHA-256, identical across two runs:

  ```
  a61a4f74fbbf796917bbbe27642be18f8f40bc677e00e0448664fb590a1fc030
  ```

- **Part B (best-effort, no-auth):** a live GSC NAGU page fetch succeeds (the no-auth ingest leg works);
  the fetched data is not written to disk or committed.

NAGU 2025001 is reproduced under the GSC reuse terms with © EU attribution. The proof confirms the
**format → normalised record** path and its determinism; full volume counts come from a real pull during
the growth work.

## Open items

- **Confirm the BeiDou `NABU TYPE` value for an orbit maneuver** (the surveyed sample was a PRN
  reallocation) and the GEO/IGSO satellite → NORAD/PRN crosswalk source (CelesTrak BeiDou catalogue / IGS
  MGEX) — at the growth/ingest stage.
- **Pin the Galileo NAGU archive listing access** (active + archived) and the GSAT→NORAD crosswalk
  (CelesTrak Galileo catalogue) for the full pull.
- **Confirm the NOAA OSPO message channel** that carries GOES SK/drift maneuvers and whether epochs are
  recoverable at useful precision — decide whether GOES is worth wiring vs. self-labelled longitude-shift.
- Ratify **D13** (and the D9 attribution stacking) with the org owner.

## References

- Galileo NAGUs — European GNSS Service Centre: per-notice pages
  `https://www.gsc-europa.eu/notice-advisory-to-galileo-users-nagu-<YYYYNNN>`; NAGU info / archive
  <https://www.gsc-europa.eu/system-service-status/nagu-information>; GSC terms of use
  <https://www.gsc-europa.eu/terms-of-use>.
- GLONASS — Information and Analysis Center (IAC / JSC TsNIIMash) <https://glonass-iac.ru/en/>; rules for
  the use of information <https://glonass-iac.ru/about/policy/>.
- BeiDou NABU — CSNO Test & Assessment Research Center <https://www.csno-tarc.cn/en/support/announcement>;
  notice `.zip` pattern `https://www.csno-tarc.cn/data/upload/nabu/NABU-<YYYY>-<NNNN>.zip`.
- NOAA GOES — Office of Satellite and Product Operations satellite messages
  <https://www.ospo.noaa.gov/operations/messages.html>.
- BeiDou GEO/IGSO maneuver prior art (observation-derived epochs) — e.g. Qiao et al., "A Method to
  Determine BeiDou GEO/IGSO Orbital Maneuver Time Periods"; Huang et al., orbit-maneuver detection from
  orbital elements for BeiDou GEO/IGSO satellites.
- V2 survey — [`v2-label-sources.md`](v2-label-sources.md); original NANU ingest proof —
  [`v2_label_ingest_proof.py`](v2_label_ingest_proof.py).
