"""Uncertainty calibration — make the ``confidence`` column mean what it says.

A detector emits a per-detection ``confidence`` in ``[0, 1]``; calibration makes that number match
the empirical hit-rate, so that among detections at confidence ~``p`` a fraction ~``p`` are true
positives. This module is the model-agnostic machinery for that:

* **Reliability diagnostics** — :func:`reliability_curve` (binned predicted-vs-empirical),
  :func:`expected_calibration_error`, and :func:`brier_score`.
* **A post-hoc calibrator** — :class:`TemperatureScaling`, a one-parameter map fit on held-out data
  that rescales the confidence so it is reliable.
* **A conformal predictor** — :class:`ConformalPredictor`, split-conformal maneuver/false-alarm
  prediction sets with a marginal coverage guarantee.
* **A wrapper** — :class:`CalibratedDetector`, which applies a fitted calibrator to *any* detector's
  confidence output (the classical reference included, which carries no checkpoint).

Everything is fit on the **val** split only — never the test labels — so the reported reliability is
a genuine held-out estimate. The (confidence, outcome) pairs a calibrator is fit on are produced by
:func:`maneuver_detect.models.evaluate.calibration_samples_on_val`, which runs the *same* benchmark
matching the scorer uses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from maneuver_detect.detectors.base import Detector

__all__ = [
    "FALSE_ALARM",
    "MANEUVER",
    "BundledCalibration",
    "CalibratedDetector",
    "CalibrationSamples",
    "Calibrator",
    "ConformalPredictor",
    "ReliabilityBin",
    "ReliabilityCurve",
    "TemperatureScaling",
    "apply_calibration",
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
]

FloatArray = npt.NDArray[np.float64]

#: The two outcomes a detection's binary calibration is scored against — the true label is a real
#: maneuver (an above-floor true positive) or a false alarm.
MANEUVER = "maneuver"
FALSE_ALARM = "false_alarm"

#: Keeps :func:`_logit` finite when a confidence sits exactly at 0 or 1.
_EPS = 1e-6


def _as_pairs(confidences: npt.ArrayLike, outcomes: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Coerce, validate, and pair ``(confidences, outcomes)`` — the shared input guard."""
    conf = np.asarray(confidences, dtype=np.float64)
    out = np.asarray(outcomes, dtype=np.float64)
    if conf.shape != out.shape:
        raise ValueError(
            f"confidences and outcomes must have the same shape, got {conf.shape} and {out.shape}"
        )
    if conf.ndim != 1:
        raise ValueError(f"confidences and outcomes must be 1-D, got {conf.ndim}-D")
    # A NaN slips past the range check below (``nan < 0`` and ``nan > 1`` are both False), so guard
    # it explicitly — an unfiltered NaN confidence silently poisons ECE/Brier and biases a fitted
    # temperature or conformal quantile to NaN with no error.
    if conf.size and not np.isfinite(conf).all():
        raise ValueError("confidences must be finite")
    if conf.size and (conf.min() < 0.0 or conf.max() > 1.0):
        raise ValueError("confidences must lie in [0, 1]")
    if out.size and not np.isin(out, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0.0 (false alarm) or 1.0 (true positive)")
    return conf, out


def _logit(p: FloatArray) -> FloatArray:
    clamped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.asarray(np.log(clamped / (1.0 - clamped)), dtype=np.float64)


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


@dataclass(frozen=True)
class CalibrationSamples:
    """The ``(confidence, outcome)`` pairs a calibrator is fit / measured on for one population.

    ``confidences`` are the detector's emitted ``[0, 1]`` confidences and ``outcomes`` the matched
    benchmark verdict per detection — ``1.0`` for an above-floor true positive, ``0.0`` for a false
    alarm (below-floor matches are excluded, mirroring the benchmark's precision). Produced on the
    val split by :func:`maneuver_detect.models.evaluate.calibration_samples_on_val`.
    """

    confidences: FloatArray
    outcomes: FloatArray

    def __len__(self) -> int:
        return int(self.confidences.shape[0])


@dataclass(frozen=True)
class ReliabilityBin:
    """One confidence bin of a reliability diagram — predicted vs. empirical for its detections."""

    lo: float
    hi: float
    count: int
    mean_confidence: float | None
    empirical_precision: float | None


@dataclass(frozen=True)
class ReliabilityCurve:
    """The binned reliability diagram — predicted confidence vs. empirical precision per bin."""

    bins: tuple[ReliabilityBin, ...]

    def populated(self) -> tuple[ReliabilityBin, ...]:
        """The bins that hold at least one detection (the points a diagram actually plots)."""
        return tuple(b for b in self.bins if b.count > 0)


def reliability_curve(
    confidences: npt.ArrayLike, outcomes: npt.ArrayLike, *, n_bins: int = 10
) -> ReliabilityCurve:
    """Bin the detections by confidence and report predicted vs. empirical precision per bin.

    Splits ``[0, 1]`` into ``n_bins`` equal-width bins; each :class:`ReliabilityBin` carries its
    detection count, mean predicted confidence, and empirical precision (the true-positive share).
    A perfectly calibrated detector has ``mean_confidence == empirical_precision`` in every bin.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    conf, out = _as_pairs(confidences, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        bins.append(
            ReliabilityBin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                count=count,
                mean_confidence=float(conf[mask].mean()) if count else None,
                empirical_precision=float(out[mask].mean()) if count else None,
            )
        )
    return ReliabilityCurve(bins=tuple(bins))


def expected_calibration_error(
    confidences: npt.ArrayLike, outcomes: npt.ArrayLike, *, n_bins: int = 10
) -> float:
    """The count-weighted mean gap between predicted confidence and empirical precision (ECE).

    ``0.0`` for a perfectly calibrated detector; larger means the stated confidence drifts further
    from the realised hit-rate. An empty sample scores ``0.0``.
    """
    conf, out = _as_pairs(confidences, outcomes)
    if conf.size == 0:
        return 0.0
    total = conf.size
    ece = 0.0
    for b in reliability_curve(conf, out, n_bins=n_bins).bins:
        if b.count and b.mean_confidence is not None and b.empirical_precision is not None:
            ece += (b.count / total) * abs(b.mean_confidence - b.empirical_precision)
    return ece


def brier_score(confidences: npt.ArrayLike, outcomes: npt.ArrayLike) -> float:
    """Mean squared error between confidence and outcome — a strictly proper calibration score.

    Lower is better; an empty sample scores ``0.0``.
    """
    conf, out = _as_pairs(confidences, outcomes)
    if conf.size == 0:
        return 0.0
    return float(np.mean((conf - out) ** 2))


@runtime_checkable
class Calibrator(Protocol):
    """A fitted post-hoc map from raw to calibrated confidence (applied by a wrapper detector)."""

    def transform(self, confidences: npt.ArrayLike) -> FloatArray:
        """Map raw ``[0, 1]`` confidences to calibrated ``[0, 1]`` confidences."""
        ...


@dataclass(frozen=True)
class TemperatureScaling:
    """Post-hoc temperature scaling: ``calibrated = sigmoid(logit(confidence) / T)``.

    A single positive scalar ``T`` fit on held-out (val) data by minimising the binary
    cross-entropy of the rescaled confidences against the outcomes. ``T > 1`` softens an
    over-confident detector toward the base rate, ``T < 1`` sharpens an under-confident one, and
    ``T == 1`` is the identity. The cross-entropy is convex in ``w = 1/T``, so a few Newton steps
    converge; ``T`` is clamped to ``t_bounds`` so a near-separable val sample cannot send it to 0 or
    infinity. Fit on the val split only — never the test labels.
    """

    temperature: float

    @classmethod
    def fit(
        cls,
        confidences: npt.ArrayLike,
        outcomes: npt.ArrayLike,
        *,
        max_iter: int = 100,
        tol: float = 1e-9,
        t_bounds: tuple[float, float] = (0.05, 20.0),
    ) -> TemperatureScaling:
        """Fit the temperature on ``(confidences, outcomes)`` (raises on an empty sample)."""
        conf, out = _as_pairs(confidences, outcomes)
        if conf.size == 0:
            raise ValueError("cannot fit temperature on an empty sample")
        z = _logit(conf)
        # Newton's method on w = 1/T for the convex BCE of sigmoid(w * z) against the outcomes.
        w = 1.0
        for _ in range(max_iter):
            p = _sigmoid(w * z)
            grad = float(np.sum(z * (p - out)))
            hess = float(np.sum(z * z * p * (1.0 - p)))
            if hess <= 1e-12:
                break
            step = grad / hess
            w -= step
            if abs(step) < tol:
                break
        lo, hi = t_bounds
        temperature = min(max(1.0 / w, lo), hi) if w > 0.0 else hi
        return cls(temperature=float(temperature))

    def transform(self, confidences: npt.ArrayLike) -> FloatArray:
        """Rescale ``confidences`` through the fitted temperature, staying in ``[0, 1]``."""
        conf = np.asarray(confidences, dtype=np.float64)
        if conf.size and not np.isfinite(conf).all():
            raise ValueError("confidences must be finite")
        if conf.size and (conf.min() < 0.0 or conf.max() > 1.0):
            raise ValueError("confidences must lie in [0, 1]")
        return _sigmoid(_logit(conf) / self.temperature)


@dataclass(frozen=True)
class ConformalPredictor:
    """Split-conformal maneuver/false-alarm prediction sets with marginal coverage >= ``1 - alpha``.

    Calibrated on held-out (val) outcomes by the LAC rule: a detection's non-conformity score is
    ``1 - p(true label)`` with ``p(MANEUVER) = confidence``, and ``q`` is the
    ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of the val scores. The prediction set for a
    new confidence is ``{label : p(label) >= 1 - q}`` — a subset of ``{MANEUVER, FALSE_ALARM}`` that
    contains the truth with probability at least ``1 - alpha`` under exchangeability. Where the
    quantile rank exceeds the sample, ``q`` saturates to ``1`` and the set always covers (it returns
    both labels). Fit on the val split only.
    """

    q: float
    alpha: float

    @classmethod
    def fit(
        cls, confidences: npt.ArrayLike, outcomes: npt.ArrayLike, *, alpha: float = 0.1
    ) -> ConformalPredictor:
        """Fit the conformal quantile at error level ``alpha`` (raises on empty / out-of-range)."""
        conf, out = _as_pairs(confidences, outcomes)
        if conf.size == 0:
            raise ValueError("cannot fit a conformal predictor on an empty sample")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        # Score of the true label: confidence for a maneuver (outcome 1), else 1 - confidence.
        p_true = np.where(out == 1.0, conf, 1.0 - conf)
        scores = np.sort(1.0 - p_true)
        n = conf.size
        rank = math.ceil((n + 1) * (1.0 - alpha))
        q = 1.0 if rank > n else float(scores[rank - 1])
        return cls(q=q, alpha=alpha)

    def predict_set(self, confidence: float) -> frozenset[str]:
        """The conformal prediction set for a single ``confidence`` (a subset of the two labels)."""
        labels: set[str] = set()
        if confidence >= 1.0 - self.q:
            labels.add(MANEUVER)
        if (1.0 - confidence) >= 1.0 - self.q:
            labels.add(FALSE_ALARM)
        return frozenset(labels)

    def covers(self, confidence: float, outcome: float) -> bool:
        """Whether the prediction set for ``confidence`` contains the true ``outcome``'s label."""
        truth = MANEUVER if outcome == 1.0 else FALSE_ALARM
        return truth in self.predict_set(confidence)


def apply_calibration(frame: pd.DataFrame, calibrator: Calibrator) -> pd.DataFrame:
    """Return ``frame`` with its ``confidence`` column mapped through ``calibrator`` (clamped).

    The single place a fitted calibrator is applied to a detector's canonical maneuver frame: an
    empty frame passes through untouched, otherwise the ``confidence`` column is remapped (clamped to
    ``[0, 1]``) and every other column — schema, dtypes, row order — is preserved. Shared by
    :class:`CalibratedDetector` and the published detectors that carry a baked-in calibrator, so
    inference applies calibration identically however the calibrator was supplied.
    """
    if frame.empty:
        return frame
    calibrated = calibrator.transform(frame["confidence"].to_numpy(dtype=np.float64))
    out: pd.DataFrame = frame.copy()
    out["confidence"] = np.clip(calibrated, 0.0, 1.0)
    return out


class CalibratedDetector(Detector):
    """Wrap a detector so its emitted ``confidence`` is passed through a fitted :class:`Calibrator`.

    Model-agnostic: the inner detector localises and inverts as usual, then every detection's
    ``confidence`` is mapped through ``calibrator`` (clamped to ``[0, 1]``) before the canonical
    frame is returned. This is how the classical reference — which carries no checkpoint to freeze a
    calibrator into — gets calibrated too. The schema, dtypes, and row order are preserved.
    """

    name: ClassVar[str] = "calibrated"

    def __init__(self, inner: Detector, calibrator: Calibrator) -> None:
        self.inner = inner
        self.calibrator = calibrator

    def detect(self, history: pd.DataFrame) -> pd.DataFrame:
        """Run the inner detector and return its frame with calibrated ``confidence``."""
        return apply_calibration(self.inner.detect(history), self.calibrator)


@dataclass(frozen=True)
class BundledCalibration:
    """The fitted calibration baked into a published detector bundle — val-fit, shipped (D17).

    Everything a published detector needs to emit **calibrated** confidence with no calibration data
    at inference, plus what its model card and the benchmark docs render:

    * ``temperature`` — the post-hoc :class:`TemperatureScaling` the detector applies to its emitted
      confidence (a single pooled scalar fit across classes).
    * ``conformal_q`` / ``conformal_alpha`` — the split-conformal predictor, for prediction-set
      reporting (a prediction set is not a scalar, so it rides alongside the emitted confidence).
    * ``reliability`` — the per-orbit-class reliability curve of the **calibrated** confidence (the
      data a per-class reliability diagram plots), keyed by orbit-class value.
    * ``ece`` — the per-orbit-class expected calibration error of the calibrated confidence, a scalar
      calibration-quality summary the card reports.

    Everything is fit on the **val** split only (never the test labels). Stored in a bundle's
    ``calibration`` slot and round-tripped as a plain dict, so an old bundle without one loads as
    ``None`` and behaves exactly as before.
    """

    temperature: float
    conformal_q: float
    conformal_alpha: float
    reliability: dict[str, ReliabilityCurve]
    ece: dict[str, float]

    def temperature_scaling(self) -> TemperatureScaling:
        """The fitted post-hoc calibrator the published detector applies to its confidence."""
        return TemperatureScaling(temperature=self.temperature)

    def conformal_predictor(self) -> ConformalPredictor:
        """The fitted split-conformal predictor, for prediction-set / coverage reporting."""
        return ConformalPredictor(q=self.conformal_q, alpha=self.conformal_alpha)

    @classmethod
    def fit(
        cls,
        samples: Mapping[str, CalibrationSamples],
        *,
        alpha: float = 0.1,
        n_bins: int = 10,
    ) -> BundledCalibration:
        """Fit the bundled calibration from per-orbit-class val ``(confidence, outcome)`` samples.

        Pools every class's samples to fit the single temperature and the conformal predictor (the
        per-detector calibrator), then measures the per-class reliability and ECE on the **calibrated**
        confidences — the curve the published detector's emitted confidence actually follows. Raises
        :class:`ValueError` when no class carries a matched detection to calibrate on.
        """
        pooled_conf = (
            np.concatenate([s.confidences for s in samples.values()])
            if samples
            else np.asarray([], dtype=np.float64)
        )
        pooled_out = (
            np.concatenate([s.outcomes for s in samples.values()])
            if samples
            else np.asarray([], dtype=np.float64)
        )
        if pooled_conf.size == 0:
            raise ValueError("no matched detections on the val split to calibrate on")
        temperature = TemperatureScaling.fit(pooled_conf, pooled_out)
        conformal = ConformalPredictor.fit(pooled_conf, pooled_out, alpha=alpha)
        reliability: dict[str, ReliabilityCurve] = {}
        ece: dict[str, float] = {}
        for key, sample in samples.items():
            calibrated = (
                temperature.transform(sample.confidences) if len(sample) else sample.confidences
            )
            reliability[key] = reliability_curve(calibrated, sample.outcomes, n_bins=n_bins)
            ece[key] = expected_calibration_error(calibrated, sample.outcomes, n_bins=n_bins)
        return cls(
            temperature=temperature.temperature,
            conformal_q=conformal.q,
            conformal_alpha=conformal.alpha,
            reliability=reliability,
            ece=ece,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict of scalars/lists, for the bundle's :func:`torch.save` payload."""
        return {
            "temperature": self.temperature,
            "conformal_q": self.conformal_q,
            "conformal_alpha": self.conformal_alpha,
            "reliability": {
                key: [_bin_to_dict(b) for b in curve.bins]
                for key, curve in self.reliability.items()
            },
            "ece": dict(self.ece),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BundledCalibration:
        """Reconstruct from :meth:`to_dict` (the inverse used by the bundle loaders)."""
        return cls(
            temperature=float(data["temperature"]),
            conformal_q=float(data["conformal_q"]),
            conformal_alpha=float(data["conformal_alpha"]),
            reliability={
                str(key): ReliabilityCurve(bins=tuple(_bin_from_dict(b) for b in bins))
                for key, bins in data.get("reliability", {}).items()
            },
            ece={str(key): float(value) for key, value in data.get("ece", {}).items()},
        )


def _bin_to_dict(b: ReliabilityBin) -> dict[str, Any]:
    return {
        "lo": b.lo,
        "hi": b.hi,
        "count": b.count,
        "mean_confidence": b.mean_confidence,
        "empirical_precision": b.empirical_precision,
    }


def _bin_from_dict(data: Mapping[str, Any]) -> ReliabilityBin:
    return ReliabilityBin(
        lo=float(data["lo"]),
        hi=float(data["hi"]),
        count=int(data["count"]),
        mean_confidence=None if data["mean_confidence"] is None else float(data["mean_confidence"]),
        empirical_precision=(
            None if data["empirical_precision"] is None else float(data["empirical_precision"])
        ),
    )
