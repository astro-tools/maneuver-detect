# Benchmark protocol

The written contract a detection method is measured against. It fixes the task, the population, the splits,
the matching rule, the metric, and the scorer, so that two methods evaluated under it produce **directly
comparable** numbers. The protocol, the splits, and the [output schema](schema.md) are **frozen by release**:
any change is a version bump with a documented rationale.

## Task

Given a per-object **mean-element TLE time series**, detect orbital maneuvers and return the
[canonical schema](schema.md): `epoch`, `confidence`, `type`, `delta_v_estimate`, and provenance. A maneuver
is observable only as a discontinuity **between consecutive elsets**, so the prediction target is **the
inter-elset gap**, not a continuous timestamp.

## Population and classes

- **Classes scored: LEO, MEO, GEO** (HEO is deferred). Each object is assigned its class from its mean motion.
- **Above-floor population.** A maneuver below the per-object **detectability floor** (≈ cm/s in LEO,
  ≈ 0.05–0.15 m/s in GEO, analytical for MEO) is physically undetectable from TLEs by *any* method. The
  **primary metric is computed over the above-floor population**; full-population recall is reported as a
  secondary, lower-bound figure. The floor is calibrated per object/class, not a single global constant.

## Splits

- **By satellite and time window**, and **leak-free**: no satellite and no overlapping time window appears in
  more than one of train / val / test.
- **Seeded and byte-stable**: identical across runs and platforms under the fixed seed; serialised and
  **frozen by release**.
- Per-split, per-class object and maneuver counts are reported alongside the splits.

## Detection-matching rule

- A predicted detection **matches** a labelled maneuver when it falls in the **labelled inter-elset gap, or
  within one adjacent gap on either side (≈ ±2 days)**.
- **One-to-one assignment.** Each label is matched by at most one detection and each detection by at most one
  label, assigned greedily by descending detection confidence within the tolerance — the standard detection
  protocol. Unmatched detections are false positives; unmatched above-floor labels are false negatives.

## Metric

- **Precision and recall at a fixed false-alarm rate per class.** The false-positive unit is **false alarms
  per satellite-year**. The **primary operating point is 1 FA/sat-year**, and a **P/R curve** is reported over
  a sweep (0.3 / 1 / 3 FA/sat-year).
- **Headline number: recall at 1 FA/sat-year over the above-floor population, per class.**
- **Per-class confidence intervals.** Recall and precision carry a Wilson score confidence interval (95%
  by default), so a per-class number estimated from few test objects is read with its sampling
  uncertainty rather than as a point fact — the interval of the *estimate*, distinct from a detector's
  per-detection `confidence`.
- **Per-class type confusion** (in-track / cross-track / radial) over the matched above-floor detections.
- **Δv error** where Δv ground truth exists (LEO altimetry, via DORIS): reported per class, expected within
  about ±25% above the floor; not scored below the floor or for radial-dominated maneuvers.

## Scorer

The scorer is **deterministic**: a predictions file plus the held-out labels go in, the score report comes
out, identical across runs and platforms. The report serialises to canonical JSON — sorted keys, ISO-8601 UTC
epochs, shortest-round-trip floats — so the same predictions reproduce the same numbers **byte-for-byte**.
Reproducing the reported baseline numbers from committed prediction files is a continuous-integration check.

The scorer lives in `maneuver_detect.benchmark`:

```python
from maneuver_detect.benchmark import read_predictions, score

predictions = read_predictions("predictions.json")   # canonical maneuver records
report = score(predictions, labels)                   # held-out, per-object labels
print(report.headline())                              # recall over the above-floor population, per class
```

The [reproduce-the-baseline example](https://github.com/astro-tools/maneuver-detect/blob/main/examples/reproduce_baseline.py)
runs this end to end on a synthetic labelled series.

## Reproducibility

Seeded, byte-stable splits; the pinned dataset recipe and content-hash manifest; the dataset, the splits, and
the checkpoints versioned in lockstep with each release; and a model card per checkpoint. See the
[dataset reference](dataset.md) and the [design decisions](decisions.md).

## Submitting a method

v0.1 ships the **local scorer**: reconstruct the dataset, run your detector to produce a predictions file,
and score it against the labels with the snippet above to get numbers directly comparable to the classical
baseline. A hosted leaderboard with held-out test labels and rate-limited submissions is planned for a later
release — until then the protocol on this page is the shared contract, and the local scorer reproduces the
published numbers.
