# V2 follow-up #2 spike — HEO + IGSO + GEO-de-circularising maneuver-label sources

**Status:** findings + recommendation, extending the V2 survey
([`v2-label-sources.md`](v2-label-sources.md)) and its first follow-up
([`v2-followup-label-sources.md`](v2-followup-label-sources.md)), feeding **D15** (the ratified v0.3
label-source set + the IGSO class addition, HEO deferred), with knock-ons to **D3** (class scope), **D4**
(detectability floor), and **D9** (licence inheritance). This is the source survey the v0.3
dataset-growth pass is blocked on. Engineering survey of public sources and their terms — **not legal
advice**; the org owner ratifies the set.

## Question

The v0.2 set is **DORIS/IDS-LEO + GPS/Galileo-MEO + self-labelled-GEO**, with two honest coverage
caveats recorded at v0.2 close and one missing class:

1. **GEO labels are self-derived (circular).** The D13 GEO plan was a BeiDou-NABU *operator* recipe,
   but that feed proved un-crawlable headless, so v0.2 GEO falls back to self-labelled longitude-shift
   inspection — partly circular (labels derived from the same series the detector reads), so reported
   separately and not folded into headline recall. Is there now a genuine operator-announced GEO feed?
2. **MEO is thin, especially Galileo** (the second operator contributes a handful of `PLN_MANV`
   notices). Can the MEO/near-GEO labelled count be raised with more years or another GNSS operator?
3. **No HEO class.** The v0.1/v0.2 datasets omit the high-eccentricity apogee/perigee-control regime.
   Is there a licence-clean, headless-ingestible HEO maneuver-label source so HEO can be *scored*?

For each candidate: orbit class, fields (epoch / type / Δv), access (auth?), licence/terms, and
obtainable coverage — the V2 table shape — then the modelling decisions the new sources force.

## Method

Surveyed each provider's operator-notice channel and confirmed the on-disk format from a **real
fetched file** per source (not from memory); checked each against the **D2 redistribution model**
(ship *labels* only where openly licensed, else ship a *reconstruction recipe*, never redistribute
terms-bound data); and confirmed headless-ingestibility (a deterministic HTTP fetch + parse, no
JS-only portal, no signed/session-bound API). Companion proof:
[`v2_followup2_label_ingest_proof.py`](v2_followup2_label_ingest_proof.py).

## Source survey

| Source | Class | Fields | Δv? | Access | Licence / terms | Coverage |
|---|---|---|---|---|---|---|
| **QZSS OHI** (Cabinet Office of Japan) | **IGSO + GEO** | start/end UTC + duration + Δv vector | **yes** | qzss.go.jp, **no-auth**, per-satellite `.txt` (`ohi-qzsN.txt`) | **CC-BY-4.0** ("Source: Quasi-Zenith Satellite System website") | 5 sats (QZS-2/4/1R IGSO + QZS-3/6 GEO), ~2–4 maneuver campaigns/sat/yr since 2017 |
| **NOAA GOES navsum** (NOAA OSPO) | GEO | last-maneuver day (`yy/ddd`) | no | ospo.noaa.gov, **no-auth**, live `navsum.txt`; history via Internet Archive | **US-Government public domain** | 4 GOES birds; one latest-maneuver epoch per snapshot, history by replaying archive snapshots |
| **Galileo NAGU — `PLN_MANV`** (GSC) | MEO | start/end UTC + GSAT id | no | gsc-europa.eu, no-auth, per-notice `.txt`, archive to 2016 | © EU reuse-with-attribution | full back-catalogue; Galileo station-keeps rarely, so the count stays modest |
| ~~Self-labelled HEO~~ (element series) | HEO | epoch / Δa, Δe step | no | derived/self-labelled | n/a (authored) | **rejected** — perturbation-dominated on noisy deep-space TLEs (XMM 213/26 yr vs ~1–2 real/yr); no maneuver/noise separation |
| BeiDou NABU (CSNO TARC) | GEO/IGSO/MEO | — | — | csno-tarc.cn **JS-only SPA**, no server-rendered notices; no maneuver semantics | no open licence | **EXCLUDED** — uncrawlable + not a maneuver feed |
| GLONASS (IAC/TsNIIMash) | MEO | — | — | glonass-iac.ru | reproduction capped at **150 chars** | **EXCLUDED** — terms forbid even a recipe |
| EUMETSAT (Meteosat) notices | GEO | minute-precise window | no | login/JS-gated mailing list | restrictive data policy | **EXCLUDED** — not headlessly ingestible + redistribution-restricted |
| Science HEO (XMM/INTEGRAL/TESS) operator notices | HEO | — | — | ESA/NASA **prose pages only** | n/a | **EXCLUDED** — no machine-ingestible feed; maneuvers rare |

### QZSS OHI — the clean win (IGSO + operator-Δv GEO)

The Cabinet Office publishes a per-satellite *Operational History Information* file at a stable URL
(`https://qzss.go.jp/en/technical/qzssinfo/.../ohi-qzsN.txt`). It is a flat text file with
`#+SECTION` / `#-SECTION` delimited blocks; the `#+SATELLITE/MANEUVER` block lists the executed burns:

```
#+SATELLITE/MANEUVER
#DATE TIME START(UTC),END(UTC),DURATION,DVX(m/s),DVY(m/s),DVZ(m/s)
2017-11-15 11:03:31,2017-11-15 11:05:50,00:02:19,-2.325,0.004,0.032
```

This is the only surveyed operator feed that ships an **executed Δv** (not merely an outage window).
**Terms:** the QZSS website permits reuse under CC-BY-4.0 with the acknowledgement "Source:
Quasi-Zenith Satellite System website" — redistribution-clean, so the **labels are shipped** (D2).
QZS-2/4/1R are inclined/eccentric geosynchronous (e≈0.075, i≈37–44°) → a **new IGSO class**; QZS-3/6
are equatorial → **GEO** (operator-Δv, non-circular).

Two file layouts exist, found at ingest: the IGSO files (QZS-2/4/1R) are 6-column
(`…,DURATION,DVX,DVY,DVZ`), while the GEO files (QZS-3/6) add an explicit `NS/EW` marker column
(`…,DURATION,NS/EW,DVX,DVY,DVZ`). Two modelling decisions the parser makes (and D15 ratifies):

- **Type from the operator `NS/EW` marker where the file gives one.** The GEO files mark each burn
  `NS` (north-south = inclination control → cross-track) or `EW` (east-west = longitude control →
  in-track) — the operator's own classification, used directly. The IGSO files omit the marker, and
  the raw `DVX/DVY/DVZ` axes carry **no documented reference frame**, so those labels are
  magnitude-only (`maneuver_type = None`) rather than fabricating a split. Either way the
  frame-invariant `|Δv| = ‖(DVX, DVY, DVZ)‖` is kept.
- **Burns collapsed into events.** A station-keeping campaign appears as a cluster of burns hours
  apart (a two-impulse ± pair, or a multi-burn inclination campaign), then the next campaign is
  weeks-to-months later. Consecutive burns within a 2-day gap are collapsed into one event (the D4
  "one label = one operator maneuver event" granularity), the event Δv being the **sum of the burns'
  magnitudes** — robust to the ± cancellation a vector sum suffers on a two-impulse correction.

### NOAA GOES navsum — operator GEO truth (breaks the circularity)

NOAA OSPO's `navsum.txt` is a per-spacecraft navigation summary whose Comments footer states each
GOES bird's last-maneuver day at `yy/ddd` granularity:

```
Spacecraft :                                  GOES-16
Comments:
Fuel and oxidizer remaining are estimates after the last maneuver on 26/159.
```

NOAA content is **US-Government public domain** → labels shipped. The one wrinkle: `navsum.txt` is a
**live-state** file reporting only the *latest* maneuver, so a maneuver *history* is built by replaying
the file's **Internet-Archive snapshots** (the CDX API lists content-distinct snapshots; each is
fetched verbatim with the `id_` modifier and parsed) and deduplicating the `(satellite, day)` epochs.
The GOES birds move from self-labelled to **operator-announced** (epoch-only); Meteosat/Himawari, with
no public feed, stay self-labelled.

### HEO — no usable source exists; the class is deferred

The headline negative result: **no machine-ingestible HEO orbit-maneuver source exists — not even
credentialed.** The first survey ruled out licence-clean/headless feeds; a second pass relaxed the
constraint to *any* credentialed/registration-gated source and still came up empty. Verified dead
ends: science HEO (XMM-Newton, INTEGRAL) maneuvers are documented only in ESA/NASA **prose/PDF**;
**ESA SPICE SPKs** and archive auxiliary data are continuous **ephemeris** (re-deriving maneuvers is
circular); **Space-Track**'s `maneuver` class is operator-panel *predicted* notices (wrong scope,
won't include science HEO); **ESA DISCOS** has no maneuver entity; academic sets (MaDDG,
SpaceTrack-TimeSeries) are synthetic or unverified/NC. The only ingestible exception — **TESS
`QUALITY`-flag reaction-wheel desaturations** (MAST FITS) — is attitude-control momentum dumps
(~100+/yr, TESS-only), not orbit-control burns.

**Self-labelling does not rescue HEO either.** An energy/eccentricity-step deriver
(`heo_self.derive_heo_labels`, the GEO longitude-shift analogue) was implemented and run on the
credentialed reconstruction: on the noisy deep-space HEO TLEs it is **perturbation-dominated**, not
maneuver-driven — XMM-Newton gave 213 "maneuvers" over 26 yr, INTEGRAL 340 over 24 yr (vs. ~1–2 real
maneuvers/yr), TESS TLEs are too noisy to use (median per-gap Δa ≈ 2821 km), and there is no clean
maneuver-vs-noise separation at any threshold (luni-solar perturbations near perigee produce real
`a`/`e` swings indistinguishable from burns in TLEs).

**Conclusion: HEO is deferred** — a reserved `OrbitClass` member with **no objects** in v0.3. The
deriver, the enum member, and the floor entry are retained for a future source. **IGSO (QZSS
operator-Δv) is the v0.3 new scored class instead.** (Spektr-RG was also ruled out as an HEO object —
it orbits Sun-Earth L2, which SGP4 cannot model.)

## Recommendation (ratified as D15)

1. **IGSO + GEO:** ship **QZSS OHI** (operator-Δv; the highest-quality new artifact). QZS-2/4/1R →
   IGSO, QZS-3/6 → GEO.
2. **GEO circularity:** add **NOAA GOES navsum** operator epochs (history via the Internet Archive) for
   the GOES birds; Meteosat/Himawari stay self-labelled.
3. **MEO:** crawl the **Galileo NAGU** back-catalogue (2016→present); modest gain (Galileo rarely
   station-keeps), so QZSS is the real near-GEO thickener.
4. **HEO:** **deferred** — no machine-ingestible maneuver source exists (even credentialed) and
   self-labelling measured as perturbation noise on the credentialed run, so HEO ships as a reserved
   class with no objects; IGSO (QZSS operator-Δv) is the new scored class instead.
5. **Dead ends:** BeiDou (uncrawlable + no maneuver semantics), GLONASS (terms), EUMETSAT (gated), and
   true-HEO operator feeds (none) — documented, not papered over.

All four shipped/derived sources clear the D2 model; the licences stack per source under the
CC-BY-4.0 authored artifacts (D9). The v0.3 dataset is a lockstep version bump (D8) with the leak-free
+ class-stratified splits re-frozen and the per-class Wilson-CI scorer extended to the new classes.
