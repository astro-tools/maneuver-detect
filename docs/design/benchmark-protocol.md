# maneuver-detect — benchmark protocol (v0.1)

The implementable contract the benchmark code (#12 splits, #13 matching/metrics/scorer) builds and the
docs (#18) publish. Frozen by release; changes require a version bump. Grounded in the V3 detectability
analysis and the [`decisions.md`](decisions.md) record (D4, D7, D8).

## 1. Task

Given a per-object **mean-element TLE time series**, detect orbital maneuvers and return the canonical
schema (D6): `epoch`, `confidence`, `type` (in-track / cross-track / radial), `delta_v_estimate`,
provenance. A maneuver is observable only as a discontinuity **between consecutive elsets**, so the
prediction target is **per inter-elset gap**, not a continuous timestamp.

## 2. Population and classes

- **Classes scored: LEO, MEO, GEO** (HEO deferred — D3). Each object is assigned its class from its mean
  motion.
- **Above-floor population.** Maneuvers below the per-object/class **detectability floor** (D4: ~cm/s
  LEO, ~0.05–0.15 m/s GEO, MEO analytical) are physically undetectable from TLEs by *any* method. The
  **primary metric is computed over the above-floor population**; full-population recall is reported as a
  secondary (lower-bound) figure. The floor is calibrated per object/class, not a single global constant.

## 3. Splits (#12)

- **By satellite and time window**, **leak-free**: no satellite and no overlapping time window appears in
  more than one of train / val / test.
- **Seeded and byte-stable**: identical across runs and platforms under the fixed seed; serialised and
  **frozen by release**.
- Per-split, per-class object and maneuver counts are reported.

## 4. Detection-matching rule (#13)

- A predicted detection **matches** a labelled maneuver if it falls in the **labelled inter-elset gap or
  within ±1 adjacent gap (≈ ±2 days)** (D4).
- **One-to-one assignment**: each label is matched by at most one detection and vice versa (greedy
  assignment by descending detection confidence within the tolerance — the standard detection protocol);
  unmatched detections are false positives, unmatched (above-floor) labels are false negatives.
- Unit-tested on hand-constructed near-boundary cases.

## 5. Metric (#13)

- **Precision and recall at a fixed false-alarm rate per class**, the FPR unit being **false-alarms per
  satellite-year** (D4). **Primary operating point: 1 FA/sat-year.** Report a **P/R curve over a sweep**
  (0.3 / 1 / 3 FA/sat-year).
- **Headline number: recall @ 1 FA/sat-year over the above-floor population, per class.**
- **Per-class sampling confidence intervals.** Recall and precision are reported with a Wilson score
  interval (default 95%, configurable) on the metric estimate — closed-form and byte-stable, so a
  per-class number from few test objects is read with its uncertainty. This is the uncertainty of the
  *estimate*, not a calibration of the detector's per-detection `confidence` (the latter is v0.3 work).
- **Per-class type confusion** (in-track / cross-track / radial) over matched above-floor detections.
- **Δv error** where Δv ground truth exists (LEO/altimetry via DORIS, #10): reported per class, expected
  within ~±25% above the floor (D5); not scored below the floor or for radial-dominated maneuvers.

## 6. Scorer (#13)

Deterministic: from a predictions file + held-out labels → scores, identical across runs/platforms.
Reproduces the reported baseline numbers from committed prediction files (a CI check).

## 7. Reproducibility (D8)

Seeded byte-stable splits; the pinned dataset recipe + content-hash manifest (D2); the dataset, splits,
and checkpoints versioned in lockstep with each release; model cards per checkpoint.

## 8. Submission (v0.2 leaderboard)

The v0.2 leaderboard hosts the shipped deterministic scorer on the **public** test split as a
**reproducibility / convenience board** (the **D12 amendment**): the test labels are part of the public
CC-BY-4.0 dataset, so the board exists to give a method a directly comparable score on identical splits,
not to hide an answer key. A submission is a `predictions.json` (a JSON array of canonical maneuver
records); the response is **aggregate-only** (per-class above-floor recall at the operating point plus
the published timing-only floor — never a per-label match table), and submissions are **rate-limited**
as a courtesy / abuse guard. The scorer's matching windows are derived elset epochs the dataset does not
redistribute (D2), so the Space's scoring fixture is built offline and supplied as private deploy-time
data rather than committed. A true hidden-label competition (a never-committed forward holdout, scored
behind the restored D12.3 firewall) is **settled by the V7 follow-up (D16)** — keyed to the release
cadence; its second board is a follow-up build gated on a v0.3 freeze. Settled by **V7 (v0.2)** as
amended (and **D16** for the competition track); out of scope for v0.1, which freezes the protocol and
ships the local scorer.

## 9. Versioning

The protocol, the splits, and the output schema are **frozen by release**; any change is a version bump
with a documented rationale.
