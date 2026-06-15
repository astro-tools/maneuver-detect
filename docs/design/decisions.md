# maneuver-detect — design decisions (D1–D17)

The frozen decision record, consolidating the prerequisite spikes (under [`spikes/`](spikes/)) and the
project charter. **No implementation lands until it matches this record.** Each decision states the call
and its rationale/source. The detailed, implementable benchmark contract lives in
[`benchmark-protocol.md`](benchmark-protocol.md).

**D1–D10** are the v0.1 freeze (V1–V4): D2–D5 lifted from the merged spike records, D1 and D6–D10 from
the charter and the spike implications. **D11** is the first v0.2 decision — the irregular-sampling input
contract from V5, gating the learned baselines. **D12** settles V7 — leaderboard integrity and the
single-GPU training budget — gating the public leaderboard Space and the baseline training. **D13**
extends the label-source set for the v0.2 dataset growth (the V2 follow-up survey). **D14** is the first
v0.3 decision — the foundation-model baseline (Chronos/TimesFM forecast-residual detector) and the
`[foundation]` extra contract, from V6, gating the v0.3 baseline detector. **D15** is the second v0.3
decision — the expanded label-source set and the new IGSO orbit class (HEO reserved but deferred) for the v0.3
dataset-growth pass (the V2-survey follow-up #2), extending D3/D4/D9/D13. **D16** settles the V7 follow-up — the
hidden-label competition track (a never-committed forward holdout) the D12 amendment deferred — gating the
competition-board build. **D17** is the v0.3 score-protocol bump — the per-class operating point persisted into
the report JSON and confidence calibration applied to the published baselines, folding in the publish half of
the uncertainty-calibration work, extending D7/D8/D11.

---

## D1 — Package & repo layout + stack pins

Package `src/maneuver_detect/`, Python ≥ 3.10, Hatchling, `py.typed`:

- `__init__.py` — public surface: `detect`, the `datasets` accessor, the canonical maneuver type.
- `data/` — `celestrak.py`, `spacetrack.py`, `tracss.py` (CC0 source, D2), `clean.py`, `history.py`.
- `labels/` — one module per source (D3) + `labeller.py`.
- `features/` — mean-element feature engineering.
- `detectors/` — `classical.py` (v0.1); `bilstm.py`, `transformer.py` (v0.2); `foundation.py` (v0.3).
- `benchmark/` — `splits.py`, `matching.py`, `metrics.py`, the scorer.
- `physics.py` — the Δv inversion (D5). `cli.py` — `maneuver-detect detect`.

Stack: numpy / pandas; sgp4; astropy (internal only); torch + lightning (v0.2 models);
huggingface_hub + datasets; gradio (v0.2 leaderboard). Optional **`[foundation]`** extra
(chronos, v0.3 — TimesFM was wired then removed; see D14). Dev: pytest, pytest-cov, ruff, mypy,
mkdocs-material, mkdocstrings.

## D2 — Dataset distribution model — recipe-first hybrid (V1)

Publish: operator **labels** + a **pinned reconstruction recipe** (fetch code + NORAD catalogue +
per-object date ranges/query params + a per-series **SHA-256 content-hash manifest**) + **directly-shipped
data only where openly licensed** — TraCSS **CC0-1.0** OMM (current/go-forward) and the
redistribution-clean **pre-2022 McDowell archive**. **Never** redistribute raw Space-Track data *or its
analysis* (the Space-Track User Agreement / 10 U.S.C. §2274(c)(2) reaches derived analysis; TLEs are
public-domain U.S.-Government works but the ToU binds). Multi-year *training history* comes from
Space-Track via the recipe; the CC0 layer grows as TraCSS coverage deepens. Reconstruction is
byte-deterministic (proven in V1). A recipe entry's epoch window scopes **both** the series fetch and
that object's maneuver labels, so the committed label set is a function of the whole recipe (a
different window yields a different label set), not the full announced history.
*(Residual: confirm TraCSS public reads need no registration — at #10.)*

## D3 — Label sources + v0.1 class scope (V2)

Sources: **DORIS/IDS** maneuver files (LEO altimetry, *with Δv* — the Δv-labelled set), **ILRS**
maneuver predictions, **GPS NANUs FCSTDV** (MEO, US-Government public domain, epoch-only; SVN/PRN →
NORAD via the CelesTrak crosswalk). The **Shorten benchmark** is a **dev-only cross-check** (unlicensed).
**SpotGEO rejected** (optical object-detection, not maneuver labels). **v0.1 class scope: LEO (primary,
Δv-labelled) + MEO (epoch-only) + GEO (best-effort, epoch-only); HEO deferred** (sparse; a later add).
*v0.2 extends this source set — see **D13** (Galileo NAGU added for MEO; a BeiDou-NABU recipe path for
GEO; GLONASS excluded on terms).*

## D4 — Labelling granularity + matching tolerance + detectability floor (V3)

A **label is the inter-elset gap** bracketing a maneuver epoch (per-gap, not per-second). **Detection-
matching tolerance = the labelled gap ±1 adjacent gap (≈ ±2 days)** (TLE cadence ~1-day median).
**Detectability floor (in-track Δv): ~cm/s (LEO), ~0.05–0.15 m/s (GEO)**, calibrated **per object/class**
(TLE-quality-dependent); MEO analytical; HEO N/A. **FPR unit = false-alarms per satellite-year**,
operating point **1 FA/sat-yr**, reported as a curve. Detection **must be multi-element** (mean-motion
alone catches only ~5–15%); the benchmark scores **P/R at fixed FPR over the above-floor population**.

## D5 — Δv inversion + type rule + tolerance (V4)

Inversion: **vis-viva** in-track (from Δa) + **Gauss** cross-track (from Δi, ΔΩ) + radial from the
residual eccentricity change (weakly observable); `|Δv|` and `type = argmax component`. **Type rule:**
in-track ↔ Δa (mean motion), cross-track ↔ Δi / ΔΩ, radial ↔ Δe. **Element jumps must be detrended**
(remove secular drift, esp. J2 nodal regression) before inversion. **Δv tolerance ~±25% above the floor;
none reported below; radial low-confidence.** `physics.py` (#14) implements the **full Gauss VOP** (not
just the linearised circular form). Quantitative accuracy-vs-published-Δv is validated at **#10** against
the DORIS burn Δv.

## D6 — Canonical output schema + Detector interface

Canonical maneuver record / DataFrame columns: **`epoch`** (UTC), **`confidence`** (calibrated, [0, 1]),
**`type`** (in-track / cross-track / radial), **`delta_v_estimate`** (m/s), plus **provenance**
(`norad_id`, the bounding elset epochs). Frozen as the library contract. A **`Detector` ABC** returns the
canonical schema; **`detect(history, model=...)`** dispatches; the **`datasets`** accessor exposes
`tle_history(...)`. v0.1 ships the classical detector behind `detect()`; learned models arrive v0.2.

## D7 — Benchmark protocol

Per [`benchmark-protocol.md`](benchmark-protocol.md): **leak-free splits by satellite + time** (no
satellite and no overlapping time window across train/val/test), seeded + byte-stable; the **matching
rule** (D4 tolerance); the metric **P/R at a fixed FA/sat-year per class over the above-floor population**
+ per-class type confusion; a **deterministic scorer**; frozen by release. That spec is the contract
#12/#13 implement and #18 publishes.

## D8 — Reproducibility / versioning

Seeded, byte-stable splits; a **pinned, reconstructable dataset + content-hash manifest** (D2); dataset
and checkpoints **versioned in lockstep**, each checkpoint with a **model card** (training data, splits,
metrics, intended use); a deterministic scorer that reproduces the reported baseline numbers from
committed prediction files.

## D9 — Licensing

**Code MIT** (org convention). **Authored dataset artifacts** (label mapping, splits, manifests, the
recipe, features derived from open data) **CC-BY-4.0**; pass-through TraCSS data stays **CC0**; **raw
Space-Track is not redistributed**. **Model weights** MIT or CC-BY-4.0; foundation-model fine-tunes
inherit their base (Chronos / TimesFM Apache-2.0). All **runtime deps permissive**. The label sources are
open / US-gov public domain (V2), so the dataset licence is **not forced restrictive**; the Shorten labels
are unlicensed → dev-only. *v0.2 (D13) adds two attribution obligations that stack per source under the
CC-BY-4.0 artifacts: **© EU** for Galileo NAGUs and NOAA credit for any GOES labels; GLONASS is excluded
on terms so its restrictions never attach.*

## D10 — Decoupling guarantee

**GMAT-free** — no GMAT / gmatpy at runtime, in tests, or as a dependency; no setup-gmat. The
foundation-model stack is an optional **`[foundation]`** extra (the base install excludes it). The charter
validation **V6 (foundation-model applicability) is deferred to its v0.3 milestone**; **V5
(irregular-sampling model input) and V7 (leaderboard integrity + compute budget) are resolved at v0.2 by
D11 and D12.**

---

## D11 — Irregular-sampling input contract (V5) — *v0.2*

The frozen encoding the feature layer emits and the BiLSTM/transformer baselines consume, from the V5
comparison of three candidates on real labelled element series (LEO + GEO). **Encoding =
time-encoded deltas, no interpolation.** Resample-to-grid is rejected — interpolating across the gap
fabricates values on the very interval the maneuver lives in and **halves above-floor recall** (LEO
0.86→0.74, GEO 0.44→0.19). A continuous-time recurrence is rejected for the baselines (no measured signal
or leak advantage over deltas at the baseline tier); its **bounded `Δt` (time2vec) representation** is
adopted as the deltas encoding's timing block. **Leak is handled at the protocol level, not the
encoding:** `Δt` stays in the input (needed for step-rate + secular detrending) and carries only a modest,
**structural** correlation with the label (timing-alone AUC ≈0.62 LEO / 0.68 GEO — not trivially
separable, and rank-AUC-invariant so not removable by clipping/time2vec/daily-regularization); the
benchmark therefore **reports a timing-only baseline as the "cheating floor" submissions must beat** and
keeps the headline metric as **recall over the above-floor population** (D4/D7), where the element signal
dominates. **Element deltas are fed signed** — the maneuver signal is in the step *magnitude* (a burn raises or
lowers `a`/`i`/Ω, so a linear readout on signed deltas separates nothing; it is `|Δ|` that scores), but
the non-linear model recovers magnitude itself and the sign carries burn direction for the D5 Δv-type
classification. **Normalisation:** per-class robust (median/IQR) per channel, train-split statistics only;
secular drift (J2 nodal regression / apsidal precession) removed by a two-sided local-linear fit before
the delta (computed once per object series, not per gap); angles carried as the eccentricity vector
`(h, k)` + unwrapped Ω (no wrap). The full tensor
contract (channels, masking, windowing, shape) is specified in
[`spikes/v5-irregular-sampling-encoding.md`](spikes/v5-irregular-sampling-encoding.md) — the feature layer
implements it verbatim.

---

## D12 — Leaderboard integrity + single-GPU compute budget (V7) — *v0.2*

The public leaderboard can be hosted **safely** and the v0.2 baselines are **cheap to train**, from
the V7 dry-run of the real scorer against hidden labels plus a `6·N·T` compute estimate.
**Hosting:** a **Gradio Space on the free Hugging Face CPU tier** — scoring is pure element-arithmetic
(D4 matching + per-class counts), CPU-cheap and deterministic (D8), so the board needs **no GPU**; the
only GPU spend in the project is offline baseline training. Hidden test labels + exposure live as a
**Space secret / private HF Dataset**, never in any public file or response; submissions are
`predictions.json` (canonical maneuver records, `schema.py`); the board persists to a public HF
Dataset, ranked by headline recall. **Integrity** closes both leak surfaces: the **response is
aggregate-only** (per-class above-floor recall at the operating point + the published D11 timing-only
"cheating floor" — never the per-label match table), and the **submission is fixed-schema**
(`read_predictions` rejects any non-prediction payload, so a submission cannot carry a query) — both
proven against the shipped scorer. **Anti-overfitting:** **hidden labels + a public/private split**
(the live board scores a public subset; the **final ranking is recomputed on a held-back private
subset revealed only at release**, the Kaggle pattern) + a **rate limit of 5 scored submissions per
user per UTC day** on the public split, keyed to the HF user id. A single-detection probing oracle is
real but costs one submission per candidate gap (≈ `ceil(G/R)` submission-days, anomalous-volume
detectable) and only ever touches the public split — the private split decides the ranking.
**Submission cadence: 5/user/day public, private scored once at release.** **Compute budget:** both
baselines are small (transformer `N ≈ 10⁷`, BiLSTM `N ≈ 1–3 × 10⁶`) on an `O(10⁵)`-window set, so a
full run is `≈ 6 × 10¹⁷` FLOP — **hours on one ≤ 24 GB GPU** (< ~8 GPU-h each; < ~16 GB peak; no
multi-GPU). **Train offline** on a free Colab/Kaggle T4 or a rented RTX 4090 / L4 / A10G and push the
checkpoint + model card to the Hub; a full v0.2 run (both baselines + a small sweep + finals) is
**< ~1 GPU-day and < ~$50**, $0 on the free tiers — confirming the charter's "single GPU in hours"
claim. The numbers are an order-of-magnitude estimate with a measured **acceptance gate** (wall-clock
< ~8 h, peak < ~16 GB) recorded on each checkpoint's model card (D8) at training time. The full
hosting/integrity/budget rationale is in
[`spikes/v7-leaderboard-integrity-and-compute-budget.md`](spikes/v7-leaderboard-integrity-and-compute-budget.md).

**Amendment — v0.2 ships a reproducibility board, not a hidden-label competition (the open dataset).**
D12.3's hidden-label firewall is **not achievable on the v0.2 test set, and is dropped.** The v0.2
dataset publishes the full answer key: `dataset/v0.2/labels.json` commits every label (epoch / type / Δv
/ window) and `splits.json` marks the `test` objects, so the held-out labels are public and — being in
git history — irretractable. The V7 spike's hidden-label assumption was never reconciled with the
open-dataset choice (D8/D9: all labels public, CC-BY-4.0, committed for reproducibility); the open
dataset wins. The v0.2 leaderboard is therefore a **reproducibility / convenience board** on the public
splits — it hosts the shipped deterministic scorer so a method gets a directly comparable number on
identical splits. The **public/private-subset firewall (D12.3, point 2) is removed** — it cannot exist
on a public test set — and the **aggregate-only response and the 5/user/UTC-day rate limit are kept as
courtesy / abuse guards, not integrity guarantees.** A true hidden-label competition would need a
separate, never-committed forward/rolling holdout sourced from data never added to the public dataset;
the design of that track is **settled by the V7 follow-up (D16)** — a forward holdout keyed to the release
cadence — with the board itself a follow-up build. One D2 consequence carries into the build: the scorer's
matching windows are real elset epochs (derived Space-Track data the dataset does not redistribute), so
the leaderboard's scoring fixture is **not committed** — it is built offline from a credentialed
reconstruction and supplied to the Space as private deploy-time data. **D12.1 / D12.2 / D12.4** (hosting,
the aggregate-only + fixed-schema integrity surfaces, compute budget) are unchanged.

---

## D13 — Expanded label-source set for v0.2 dataset growth (V2 follow-up) — *v0.2*

The ratified source set the v0.2 dataset-growth pass draws from, extending **D3** from the V2 follow-up
survey of GEO + non-GPS-GNSS sources. Resolves the "no public GEO maneuver-label source" gap and the V2
"survey/defer" open item on other GNSS notices.

- **MEO — add Galileo NAGUs (`PLN_MANV`).** Public, no-auth, machine-ingestible (per-notice `.txt` with
  `START`/`END DATE EVENT (UTC)` + GSAT id + SVID); GSAT → NORAD via the CelesTrak Galileo crosswalk.
  Terms are an **attribution-required reuse grant (© EU)** — redistribution-clean, so the **labels are
  shipped** (D2). Adds a **second, independent MEO operator** beyond GPS (~28 GSAT SVs), epoch-only (no
  Δv). **GLONASS is excluded:** the IAC/TsNIIMash "Rules for the use of information" cap public internet
  reproduction at **150 characters** without consent — more restrictive than Space-Track, fails D2.
- **GEO — best-effort GO via a BeiDou-NABU recipe.** CSNO's NABUs are machine-ingestible (per-notice
  `.zip` at a stable URL), UTC-precise, and **name GEO/IGSO satellites** — the only operator-announced
  GEO maneuver feed. csno-tarc.cn states **no open licence**, so BeiDou is handled by the **D2
  recipe-first model** (ship the fetch recipe + parser + hash manifest, never the redistributed labels —
  the Space-Track pattern), not shipped data. The exact `NABU TYPE` string for an orbit maneuver is
  **confirmed at ingest** (the surveyed sample was a PRN reallocation). **Fallbacks:** public-domain NOAA
  GOES messages (clean but sparse, free-text) and self-labelled longitude-shift inspection on
  reconstructed GEO series; Shorten Fengyun stays dev-only. GEO stays **epoch-only**.
- **Class scope (D3) unchanged otherwise:** LEO primary (Δv-labelled), HEO deferred. **Licence (D9):**
  attribution stacks per source (© EU for Galileo, NOAA for GOES); no new restriction attaches (GLONASS
  excluded; BeiDou contributes a recipe, not bytes). **No openly-licensed GEO maneuver *benchmark*
  exists** to redistribute — the BeiDou recipe + self-labelled fallbacks are the GEO path.

Detail in [`spikes/v2-followup-label-sources.md`](spikes/v2-followup-label-sources.md).

---

## D14 — Foundation-model baseline + the `[foundation]` extra contract (V6) — *v0.3*

The v0.3 foundation-model baseline is a **forecast-residual detector** built on a pretrained
time-series model, from the V6 dry-run of the recipe against the real scorer plus the licence and
single-GPU budget findings. It resolves the charter's V6 prerequisite ("can a foundation model be
turned into a maneuver detector") and pins the model choice and the optional-extra contract.

- **Recipe — forecast-residual thresholding.** Forecast each object's mean-element series (the D11
  encoding) with the pretrained model, standardize the residual by the model's predictive interval,
  and threshold it **per orbit class** (the D4 detectability floor in residual units), then NMS per
  object. The model replaces **only** the classical detector's hand-built quiet-dynamics prior with a
  learned conditional forecast; the **D4** matcher, **D5** Δv/type inversion, **D6** schema, and
  **D7** scorer are reused unchanged. The predictive interval feeds the v0.3 uncertainty calibration
  (it is what makes the D6 `confidence` column meaningful). Proven end-to-end against the shipped
  scorer in V6.
- **Model choice — Chronos leads, TimesFM is the second entry.** **Chronos** (lead: `chronos-bolt-*` /
  `chronos-t5-*`) is chosen for the baseline: native probabilistic quantiles give clean residual
  standardization and a calibrated confidence, Bolt inference is CPU-cheap, and the 8M–710M size range
  fits any budget. **TimesFM** (`timesfm-1.0-200m` / `2.0-500m`) stays wired as a drop-in second entry
  on the same forecaster-agnostic recipe — a cross-check at near-zero extra cost, not a parallel build.
  *(Amended at v0.3 implementation — TimesFM removed; see the ratification note below.)*
- **Licence.** Both families publish **Apache-2.0** checkpoints, so a fine-tune is redistributable
  (D9: fine-tunes inherit the base) and the dep is permissive. The build **pins the checkpoint id +
  revision and confirms its Apache-2.0 card at ingest** before publishing a fine-tune (a licence is a
  per-revision property) — the "confirm at ingest" discipline D13 used for BeiDou.
- **Budget (reuses V7).** **Zero-shot residual detection trains nothing** (inference only — zero
  GPU-hours, runnable on the free CPU/T4 tiers), so the baseline ships zero-shot first. A **light
  fine-tune** (full small/base checkpoint or LoRA) over the v0.2-scale window set is `≈ 3 × 10¹⁷` FLOP
  — the same hours-on-one-≤24 GB-GPU band V7 pinned (< ~8 GPU-h, < ~16 GB, < ~$50 / $0 on free tiers).
  Measured figures land on the v0.3 model card against the V7 acceptance gate (D8).
- **`[foundation]` extra contract.** `[foundation] = chronos-forecasting, timesfm` (both Apache-2.0)
  stays an **optional extra the base install excludes** (D1/D10 decoupling — no torch-foundation stack
  at base install, in the CLI, or in the default test suite). `detectors/foundation.py` imports them
  **lazily**; foundation tests run only in a dedicated CI job with the extra installed
  (`importorskip`). Checkpoints are fetched from the HF Hub at runtime (the base `huggingface_hub`
  dep), not vendored; fine-tune checkpoints ship a model card (D8). *(Amended at v0.3 implementation
  — the shipped extra is `[foundation] = chronos-forecasting` only; TimesFM removed, below.)*

**Ratified (amended) — implemented in the v0.3 baseline, Chronos only.** `detectors/foundation.py`
ships **one** registered detector, `chronos-residual`, on the shared forecaster-agnostic
forecast-residual pipeline (a `Forecaster` protocol, so a further backend can drop in later), reusing
the D4 matcher / D5 inversion / D6 schema / D7 scorer verbatim. A `FoundationBundle`
(`models/foundation.py`) — separate from the torch-network `ModelBundle`, so the v0.2 baselines are
untouched — pins the backend, the Hub checkpoint id (`amazon/chronos-bolt-small`) and its revision,
the rolling context length, the calibrated per-class thresholds, and an optional fine-tune
`state_dict`; the offline driver provides zero-shot assembly, val-split threshold calibration through
the shared benchmark, held-out scoring that back-fills the model card, and a light Chronos fine-tune
(GPU when present, else CPU). The extra is exercised by a dedicated `importorskip` CI job; the
minimal-install job still asserts it is absent from the base install.

Three implementation findings amended the spike's plan:

- **Standardise by a robust per-object MAD, not the model's per-gap predictive interval.** The
  predictive interval D14.1 leaned on (and D14.4 cited as Chronos's edge) **collapses toward zero on
  confident gaps**, blowing ordinary gaps into spurious infinite-z spikes — measured on real series,
  it is unusable for per-gap thresholding (the open question V6 flagged about interval calibration on
  element series, now answered). The detector standardises the forecast residual by the centred MAD
  of the residuals instead (the V6 stand-in's stabiliser); the calibrated interval is left to the
  uncertainty-calibration deliverable. On the v0.2 test split `chronos-residual` then reaches **LEO
  ≈0.49 / MEO ≈0.54** above-floor recall at 1 FA/sat-year (beating the classical reference and the
  v0.2 transformer on those classes); **GEO ≈0.08** stays the hard class (tiny station-keeping steps),
  as for every detector.
- **TimesFM removed.** The wired-in second entry was dropped: its zero-shot forecast is **not robust
  on real noisy sub-daily LEO** (LEO recall 0.000 — sustained forecast biases that a transient filter
  cannot remove, where Chronos stays solid), so it would ship a worse-than-classical card. This is
  exactly why D14.4 made Chronos the lead; revisiting TimesFM (e.g. with a fine-tune) is a possible
  future entry, not a v0.3 deliverable. The shipped extra is therefore `[foundation] =
  chronos-forecasting`.
- **Zero-shot ships; the fine-tune is measured but not the default.** The leak-free light fine-tune
  (train-split objects only — the earlier wiring leaked val/test, fixed under #103; 200 steps on
  `chronos-bolt-small`) was run: **LEO ≈0.57 / MEO ≈0.46 / GEO ≈0.03**. It lifts the LEO core but
  lowers GEO and *pooled* recall versus zero-shot (LEO/MEO-dominated train windows), for a ~190 MB
  bundle vs the 3.7 KB zero-shot one. So the v0.3 baseline ships **zero-shot**, and the fine-tune
  stays the optional polish D14.3 framed it as.

The calibrated cutoff is frozen across classes for now (genuine per-class thresholds are a later
deliverable). Publishing the bundle + card to the Hub and seeding the leaderboard is the offline
credentialed step the driver feeds.

Detail in [`spikes/v6-foundation-model-applicability.md`](spikes/v6-foundation-model-applicability.md).

---

## D15 — Expanded label-source set + IGSO/HEO classes for v0.3 dataset growth (V2 follow-up #2) — *v0.3*

The ratified source set and class additions the v0.3 dataset-growth pass draws from, extending **D13**
(itself the v0.2 extension of **D3**) from a second V2-survey follow-up, and resolving the v0.2
coverage caveats: the GEO labels were self-derived (circular) and the second MEO operator (Galileo)
was thin. Engineering survey of public sources and their terms — **not legal advice**; the org owner
ratifies the set. Every source was verified by a real headless fetch.

- **IGSO + GEO — add QZSS via the OHI files.** The Cabinet Office of Japan publishes, per
  Quasi-Zenith satellite, an *Operational History Information* file (`ohi-qzsN.txt`) carrying the
  executed orbit-maintenance maneuvers with a **Δv vector** — the only surveyed operator feed that
  ships a real executed Δv, not just an outage window. Headless-fetchable (static `.txt`), terms are a
  reuse-with-attribution grant (CC-BY-4.0, "Source: Quasi-Zenith Satellite System website") →
  **labels are shipped** (D2). QZS-2/4/1R are inclined/eccentric geosynchronous (e≈0.075, i≈37–44°) —
  a **new IGSO class**; QZS-3/6 are equatorial → **GEO** (operator-Δv, breaking the GEO
  self-label circularity for those objects). Two modelling choices (detail in the spike): the **GEO**
  OHI files (QZS-3/6) carry an explicit **`NS/EW`** burn marker (north-south = inclination control →
  cross-track, east-west = longitude control → in-track) used directly as the operator's own type,
  while the **IGSO** files omit it and the raw `DVX/DVY/DVZ` frame is undocumented, so those labels
  carry the frame-invariant **|Δv| magnitude only** (`maneuver_type = None`) rather than a fabricated
  split; and clustered burns (a station-keeping campaign) are **collapsed into one event** (D4
  granularity), the event Δv being the sum of the burns' magnitudes and its type the dominant burn's
  marker.
- **GEO — add NOAA GOES operator epochs.** The NOAA OSPO navigation summary (`navsum.txt`) names each
  GOES bird's last-maneuver day; it is US-Government **public domain** → labels shipped. It is a
  **live-state** file (latest maneuver only, day-of-year granularity), so the maneuver *history* is
  built by replaying its **Internet-Archive snapshots** (CDX-listed, content-distinct) and
  deduplicating the epochs. The GOES birds therefore move from self-labelled to **operator-announced**
  (epoch-only), the second half of breaking the GEO circularity caveat. Meteosat/Himawari (no public
  feed) stay self-labelled.
- **MEO — Galileo back-catalogue.** The same NAGU `PLN_MANV` feed (D13), crawled over the full
  2016→present window, is the MEO-thickening lever. Galileo genuinely station-keeps rarely, so the
  realized count stays modest — the richer GNSS thickening is the QZSS operator-Δv set above.
- **HEO — deferred (a reserved class with no objects in v0.3).** **No** ingestible operator maneuver
  feed exists for the high-eccentricity regime — *not even credentialed* (the licence/headless
  constraint was relaxed and re-surveyed): science-HEO (XMM-Newton, INTEGRAL) maneuvers are
  documented only in prose/PDF; ESA SPICE SPKs and archive auxiliary data are continuous ephemeris
  (re-deriving maneuvers from them is circular); Space-Track's `maneuver` class is operator-panel
  *predicted* notices (wrong scope, won't include science HEO); ESA DISCOS has no maneuver entity;
  academic sets are synthetic or unverified. The only ingestible exception, TESS `QUALITY`-flag
  reaction-wheel desaturations, is attitude-control momentum dumps (~100+/yr, TESS-only), not
  orbit-control burns. **And self-labelling does not rescue it:** the credentialed reconstruction
  measured the energy/eccentricity-step self-labeller as **perturbation-dominated** on the noisy
  deep-space HEO TLEs (XMM 213 "maneuvers"/26 yr, INTEGRAL 340/24 yr vs. ~1–2 real/yr; TESS TLEs
  too noisy to use), with no clean maneuver/noise separation. So HEO ships as a **reserved class with
  no objects**; the `OrbitClass.HEO` member, the detectability-floor entry, and the
  `labels.heo_self` deriver are retained for a future source. **IGSO (QZSS operator-Δv) is therefore
  the v0.3 new scored class.**
- **Dead ends (re-confirmed).** **BeiDou** NABU stays uncrawlable (JS-only SPA, no maneuver
  semantics); **GLONASS** stays excluded on terms (150-character reproduction cap); **EUMETSAT** GEO
  notices are real but login/JS-gated with a restrictive data policy. No high-e HEO operator feed
  exists — the self-labelling conclusion is a finding, not a gap papered over.
- **Class scope (D3) + floor (D4).** The taxonomy carries five members — LEO, MEO, GEO, **IGSO**,
  **HEO** — but only four are populated in v0.3 (IGSO is the new scored class; HEO is reserved/empty,
  above). The runtime `orbit_class_of` (semi-major-axis only) is **unchanged** — it returns
  LEO/MEO/GEO, so a detector buckets an IGSO object as the nearest coarse class for its working
  floor/normalisation; the benchmark scores by the **pinned** dataset class, so IGSO is genuinely
  scored. The detectability-floor table gains an IGSO entry (≈GEO, geosynchronous) and an HEO
  analytical placeholder (kept for the reserved class). An eccentricity-aware classifier + per-class
  IGSO normalisation statistics are a later refinement, not this pass.
- **Licence (D9).** Attribution stacks per source under the CC-BY-4.0 authored artifacts: NOAA GOES is
  US-gov public domain; QZSS adds "Source: Quasi-Zenith Satellite System website" (CC-BY-4.0); Galileo
  © EU carries forward. No new restriction attaches (BeiDou/GLONASS/EUMETSAT excluded). Each source's
  licence is **confirmed at ingest** (the D13 discipline).
- **Versioning (D8).** A lockstep **v0.3** dataset bump: the recipe, labels, manifest, and re-frozen
  splits version to 0.3.0; the leak-free + class-stratified split construction and the per-class
  Wilson-CI scorer are class-generic, so the new IGSO class flows through reporting automatically.
- **Public/private boundary (coordination with the competition track, D16).** All v0.3 labels are
  committed **public**; none are held back. The hidden-label competition (the V7 follow-up, **D16**) draws
  a **forward holdout** — maneuvers with epoch *after* the v0.3 freeze, reconstructed from the same ongoing
  operator feeds (D2) — which is disjoint from the historical labels committed here, so nothing need be
  private and no competition label can leak into `labels.json` / `splits.json`. The new operator feeds
  (QZSS OHI, NOAA GOES) also make a forward GEO/IGSO holdout viable, settling the "thin forward GEO" risk
  in D16.

Detail in [`spikes/v2-followup2-heo-igso-sources.md`](spikes/v2-followup2-heo-igso-sources.md).

---

## D16 — Hidden-label competition track via a never-committed forward holdout (V7 follow-up) — *v0.3+*

The design of the true hidden-label competition the **D12 amendment** deferred — settling the V7 spike's
retained firewall analysis against a forward holdout, from a follow-up dry-run of the *real* leaderboard
service. It resolves the three open questions (holdout source + the GEO problem, one-time cutoff vs.
rolling, reveal cadence) and shows the firewall the open v0.2 answer key made unbuildable holds on a
never-committed holdout. The **board itself is a follow-up build** (a second "competition" board on the
existing Space, the holdout-fixture builder, the per-release refresh) — this decision fixes its shape, not
its code.

- **The holdout (D16.1).** Maneuvers with **epoch strictly after the public dataset's freeze**,
  reconstructed from the same operator feeds via the **D2 recipe**, **never committed** to `labels.json` /
  `splits.json`. Disjointness is **temporal** — a single `epoch > freeze` cut, auditable and
  satellite-agnostic (the same object may maneuver on both sides) — the boundary D15 reserved, made
  precise. The fixture is built offline from a credentialed reconstruction and supplied as private
  deploy-time data (D2); unlike the v0.2 reproducibility fixture, the **labels themselves are private**.
- **Rolling, keyed to the release cadence (D16.2).** A one-time forward cutoff is rejected (it ages out as
  the next release publishes past it). The private holdout is always "after the current freeze"; at each
  release the matured window's labels are **revealed** (folded into the next public dataset) and a fresh
  forward window becomes the new private holdout. **Reveal cadence = per release** — the Kaggle
  private-leaderboard-at-deadline pattern with the deadline at each release; the refresh is one offline
  fixture build per release cut, not a standing service.
- **The firewall, restored as integrity (D16.3).** All four D12.3 mechanisms return — hidden labels +
  public/private subset split (private scored once at the release reveal) + the **5/user/UTC-day rate
  limit, now an integrity bound** (probing leaks hidden labels) + the aggregate-only / fixed-schema round
  trip (D12.2, unchanged). The V7 probing bound carries over: a single-detection oracle recovers only the
  public subset (`ceil(G/R)` submission-days, anomalous-volume detectable) and never the private subset
  that decides the ranking — proven against the real service.
- **Thin classes scored, not dropped (D16.4).** LEO (Δv-labelled) and MEO carry the competition from
  launch; GEO/IGSO are populated for real by the **D15** operator feeds (QZSS OHI operator-Δv, NOAA GOES
  epochs) — breaking the v0.2 self-label circularity — but thin, so they are scored per-class with their
  honest small-`N` Wilson intervals (sharpening across reveal cycles). BeiDou NABU stays unavailable.
- **Gating (D8).** The implementation is gated on the first post-freeze window existing — i.e. after a v0.3
  release freezes the public dataset to define `epoch > freeze`. Ratify D16 when the board is built against
  it (the V7 → D12 discipline).

Detail in [`spikes/v7-followup-hidden-label-competition.md`](spikes/v7-followup-hidden-label-competition.md).

---

## D17 — v0.3 score-protocol bump: per-class operating point in the report JSON + calibrated confidence — *v0.3*

The publish half of the uncertainty-calibration work, applied to the real v0.3 baselines: make the
``confidence`` column mean what it says, and persist the per-class operating point the calibration publishes.
The calibration *machinery* (reliability diagnostics, temperature scaling, split-conformal) landed earlier as
offline, synthetic-tested model-agnostic code; this decision fixes how it is **applied and shipped**, and the
one byte-stable-artifact (D8) boundary change it forces.

- **The protocol bump (D17.1).** ``ScoreReport.to_json`` gains a per-class ``operating_point_confidence`` —
  the confidence cut admitted within the false-alarm budget at the headline operating point (the D7 curve's
  per-class point), ``None`` when no detection is admitted. It is **additive** (every prior field is byte-for-
  byte unchanged; sorted-key canonical JSON places it deterministically) but it **changes the frozen artifact**,
  so it is a v0.3-boundary change, not a v0.2 patch: the v0.2 report deliberately kept this value **in-memory
  only** (an ``ClassMetrics`` convenience), and the release-frozen v0.2 ``scores.json`` snapshot is left as it
  was. The committed scorer golden is regenerated at this boundary.
- **Confidence calibration is baked into the artifact, not re-fit at load (D17.2).** Each published detector
  carries a calibrator (temperature scaling, plus a split-conformal predictor for prediction sets) **fit on the
  val split only** — never the test labels — and frozen into its bundle alongside the weights/thresholds, so a
  loaded detector emits **calibrated** confidence with no calibration data at inference (the same
  card-cannot-drift-from-weights discipline as D8). The emitted scalar ``confidence`` is the temperature-
  calibrated probability; conformal rides along for the reliability/operating-point reporting (a prediction set
  is not a scalar). Old bundles without a calibrator load unchanged (the back-compatible ``None``).
- **Reliability + per-class operating points are published, not asserted (D17.3).** The per-detector, per-class
  reliability curve (binned predicted-vs-empirical) and the calibrated per-class operating point are recorded
  into the bundle from the same val-split run and rendered onto the generated model card and the benchmark docs.
  Consistent with the recipe-first / no-committed-real-data convention, the **committed** docs carry the
  methodology and a bundle→diagram render helper rather than real-data figures; the figures themselves are
  rendered at the credentialed release-cut run and uploaded to the Hub cards.
- **The leaderboard tolerates the field by ignoring it (D17.4).** The leaderboard re-scores submissions through
  the same scorer (so the field is present in the report it computes) but its public response is a strict
  aggregate subset — headline above-floor recall per class, the operating point, and the timing floor — so the
  new field neither leaks into a response nor changes scoring. The submission reader's fixed-schema integrity
  surface (D12) is unchanged: it guards the *predictions* file, not the report JSON.
- **Scope (D8).** This decision is the *application* boundary; the calibration mechanism and the class-balanced
  selection objective / per-class detection thresholds landed separately. Per-class detection-threshold
  refinement for the foundation residual gate is a later follow-up (it keeps one global gate here).

---

*Sources: [`spikes/v1-dataset-redistribution.md`](spikes/v1-dataset-redistribution.md),
[`spikes/v2-label-sources.md`](spikes/v2-label-sources.md),
[`spikes/v2-followup-label-sources.md`](spikes/v2-followup-label-sources.md),
[`spikes/v2-followup2-heo-igso-sources.md`](spikes/v2-followup2-heo-igso-sources.md),
[`spikes/v3-detectability-floor.md`](spikes/v3-detectability-floor.md),
[`spikes/v4-dv-inversion.md`](spikes/v4-dv-inversion.md),
[`spikes/v5-irregular-sampling-encoding.md`](spikes/v5-irregular-sampling-encoding.md),
[`spikes/v6-foundation-model-applicability.md`](spikes/v6-foundation-model-applicability.md),
[`spikes/v7-leaderboard-integrity-and-compute-budget.md`](spikes/v7-leaderboard-integrity-and-compute-budget.md),
[`spikes/v7-followup-hidden-label-competition.md`](spikes/v7-followup-hidden-label-competition.md),
and the project charter.*
