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

## Confidence calibration

A detector's per-detection `confidence` is meaningful only if it matches the empirical hit-rate: among
detections at confidence ~`p`, about a fraction `p` should be true positives. Calibration measures and corrects
that, **fit on the validation split only** so the test labels are never touched.

- **Reliability diagrams** bin the detections by confidence and plot the mean predicted confidence against the
  empirical precision in each bin; the **expected calibration error** (ECE) and **Brier score** summarise the
  gap. A perfectly calibrated detector sits on the diagonal.
- **Temperature scaling** rescales the confidence — `sigmoid(logit(confidence) / T)` for a single `T` fit on
  the val split — so an over-confident detector (`T > 1`) is softened toward its realised hit-rate.
- **Conformal prediction** turns a confidence into a maneuver / false-alarm prediction set with a marginal
  coverage guarantee (at least `1 - alpha`), calibrated on the val split.
- **Per-class operating points.** Each class's operating point is the confidence cut at which its false-alarm
  rate reaches the target (1 FA/sat-year) — `ClassMetrics.operating_point_confidence`, published per class so a
  consumer knows the confidence threshold the headline recall is read at.

The `(confidence, outcome)` pairs a calibrator is fit on come from the **same** matching the scorer uses, on
the val split:

```python
from maneuver_detect.calibration import CalibratedDetector, expected_calibration_error
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.models.evaluate import calibration_samples_on_val, fit_temperature_on_val

# Fit one temperature on the detector's val-split detections (the test labels are untouched).
temperature = fit_temperature_on_val(detector, series_by_norad, labels, split)

# Wrap any detector so it emits calibrated confidence — the classical reference included.
calibrated = CalibratedDetector(detector, temperature)

# Reliability per class, from the val samples.
samples = calibration_samples_on_val(detector, series_by_norad, labels, split)
leo = samples[OrbitClass.LEO]
ece = expected_calibration_error(leo.confidences, leo.outcomes)
```

Calibration only rescales the confidence column — it does not change which gaps a detector fires on, just how
confident it says it is. The reliability diagrams and per-class operating points for each shipped detector are
published with the versioned models.

## Scorer

The scorer is **deterministic**: a predictions file plus the held-out labels go in, the score report comes
out, identical across runs and platforms. The report serialises to canonical JSON — sorted keys, ISO-8601 UTC
epochs, shortest-round-trip floats — so the same predictions reproduce the same numbers **byte-for-byte**.
Reproducing the reported baseline numbers from committed prediction files is a continuous-integration check.

The scorer lives in `maneuver_detect.benchmark`. It takes the predictions, the held-out above-floor test
labels, and each object's exposure, and returns a per-class report:

```python
from pathlib import Path

from maneuver_detect.benchmark import read_predictions, score

predictions = read_predictions(Path("predictions.json").read_text())  # canonical maneuver records
report = score(predictions, labels, exposure)   # held-out ScoredLabels + per-object ObjectExposure
print(report.headline())                        # above-floor recall at the operating point, per class
```

`read_predictions` parses the canonical predictions **text** (not a path); `labels` are the held-out
`ScoredLabel`s and `exposure` the per-object `ObjectExposure` (observation-years — the satellite-year
denominator of the false-alarm rate). The
[reproduce-the-baseline example](https://github.com/astro-tools/maneuver-detect/blob/main/examples/reproduce_baseline.py)
constructs all three and runs the scorer end to end on a synthetic labelled series.

## Reproducibility

Seeded, byte-stable splits; the pinned dataset recipe and content-hash manifest; the dataset, the splits, and
the checkpoints versioned in lockstep with each release; and a model card per checkpoint. See the
[dataset reference](dataset.md) and the [design decisions](decisions.md).

## Submitting a method

Two paths run the **same deterministic scorer** and produce the **same numbers**:

- **The local scorer** — reconstruct the dataset, run your detector to produce a predictions file, and score
  it against the labels with the snippet above. Numbers directly comparable to the baselines, with no account
  beyond your own Space-Track access.
- **The public [leaderboard](leaderboard.md)** — upload your `predictions.json` to the hosted Hugging Face
  Space and read your per-class above-floor recall, ranked against the classical, BiLSTM, and transformer
  baselines on the frozen v0.2 test split.

The protocol on this page is the shared contract behind both. The [leaderboard guide](leaderboard.md) walks a
submission end to end.
