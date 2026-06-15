"""Tests for the calibration machinery — reliability, temperature, conformal, and the wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maneuver_detect.calibration import (
    FALSE_ALARM,
    MANEUVER,
    CalibratedDetector,
    ConformalPredictor,
    TemperatureScaling,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from maneuver_detect.detectors.base import Detector
from maneuver_detect.schema import COLUMNS, Maneuver, ManeuverType, empty_frame, to_frame

# --- reliability diagnostics -------------------------------------------------------------------


def test_reliability_curve_bins_counts_and_populated() -> None:
    curve = reliability_curve([0.05, 0.15, 0.15], [0.0, 1.0, 0.0], n_bins=10)
    assert len(curve.bins) == 10
    first, second = curve.bins[0], curve.bins[1]
    assert first.count == 1 and first.mean_confidence == pytest.approx(0.05)
    assert first.empirical_precision == pytest.approx(0.0)
    assert second.count == 2 and second.mean_confidence == pytest.approx(0.15)
    assert second.empirical_precision == pytest.approx(0.5)
    # Empty bins carry None and are dropped by populated().
    assert curve.bins[9].count == 0 and curve.bins[9].mean_confidence is None
    assert len(curve.populated()) == 2


def test_ece_is_zero_when_confidence_matches_precision() -> None:
    # All in one bin at 0.5 with a 50% hit-rate -> predicted == empirical -> ECE 0.
    confidences = [0.5] * 100
    outcomes = [1.0, 0.0] * 50
    assert expected_calibration_error(confidences, outcomes) == pytest.approx(0.0)


def test_ece_is_large_when_badly_miscalibrated() -> None:
    # Confidently 0.9 but never right -> ECE ~ 0.9.
    assert expected_calibration_error([0.9] * 100, [0.0] * 100) == pytest.approx(0.9)


def test_brier_score_known_value() -> None:
    assert brier_score([0.8, 0.2], [1.0, 0.0]) == pytest.approx(0.04)


def test_empty_sample_scores_zero() -> None:
    assert expected_calibration_error([], []) == 0.0
    assert brier_score([], []) == 0.0


@pytest.mark.parametrize(
    ("confidences", "outcomes", "match"),
    [
        ([0.5, 0.5], [1.0], "same shape"),
        ([1.5], [1.0], r"\[0, 1\]"),
        ([0.5], [2.0], "0.0 .* or 1.0"),
    ],
)
def test_input_guard_rejects_bad_pairs(
    confidences: list[float], outcomes: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        reliability_curve(confidences, outcomes)


# --- temperature scaling -----------------------------------------------------------------------


def _overconfident_sample(n: int, sharpen: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic (confidence, outcome) set whose confidence is over-sharpened by ``sharpen``."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    outcomes = (rng.uniform(0.0, 1.0, n) < p).astype(np.float64)
    logit = np.log(p / (1.0 - p))
    confidences = 1.0 / (1.0 + np.exp(-logit / sharpen))  # sharpen < 1 -> over-confident
    return confidences, outcomes


def test_temperature_scaling_reduces_ece() -> None:
    confidences, outcomes = _overconfident_sample(4000, sharpen=0.5, seed=0)
    before = expected_calibration_error(confidences, outcomes)
    temperature = TemperatureScaling.fit(confidences, outcomes)
    after = expected_calibration_error(temperature.transform(confidences), outcomes)
    assert temperature.temperature > 1.0  # softens an over-confident detector
    assert after < before / 2.0  # markedly more reliable


def test_temperature_near_identity_when_already_calibrated() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 1.0, 4000)
    outcomes = (rng.uniform(0.0, 1.0, 4000) < p).astype(np.float64)
    temperature = TemperatureScaling.fit(p, outcomes)
    assert temperature.temperature == pytest.approx(1.0, abs=0.2)


def test_temperature_transform_is_monotonic_and_in_range() -> None:
    confidences = np.linspace(0.0, 1.0, 50)
    out = TemperatureScaling(temperature=2.5).transform(confidences)
    assert np.all(np.diff(out) >= 0.0)  # order-preserving
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_temperature_fit_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        TemperatureScaling.fit([], [])


# --- split conformal ---------------------------------------------------------------------------


def test_conformal_covers_at_least_one_minus_alpha() -> None:
    confidences, outcomes = _overconfident_sample(8000, sharpen=0.6, seed=2)
    alpha = 0.1
    predictor = ConformalPredictor.fit(confidences[:4000], outcomes[:4000], alpha=alpha)
    covered = np.mean(
        [predictor.covers(c, o) for c, o in zip(confidences[4000:], outcomes[4000:], strict=True)]
    )
    assert covered >= 1.0 - alpha - 0.03  # marginal coverage guarantee (small finite-sample slack)


def test_conformal_smaller_alpha_gives_wider_sets() -> None:
    confidences, outcomes = _overconfident_sample(4000, sharpen=0.6, seed=3)
    loose = ConformalPredictor.fit(confidences, outcomes, alpha=0.2)
    tight = ConformalPredictor.fit(confidences, outcomes, alpha=0.02)
    assert tight.q >= loose.q  # a stricter error level admits more into each set
    # A very confident detection is a maneuver-only set; an ambiguous one keeps both labels.
    assert tight.predict_set(0.999) == frozenset({MANEUVER})
    assert tight.predict_set(0.5) == frozenset({MANEUVER, FALSE_ALARM})


def test_conformal_rejects_empty_and_bad_alpha() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        ConformalPredictor.fit([], [], alpha=0.1)
    with pytest.raises(ValueError, match="alpha must lie"):
        ConformalPredictor.fit([0.5], [1.0], alpha=1.5)


# --- the calibrated-detector wrapper -----------------------------------------------------------


class _FixedDetector(Detector):
    """A detector that emits a fixed list of confidences, for exercising the wrapper."""

    name = "fixed"

    def __init__(self, confidences: list[float]) -> None:
        self._confidences = confidences

    def detect(self, history: pd.DataFrame) -> pd.DataFrame:
        if not self._confidences:
            return empty_frame()
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        return to_frame(
            [
                Maneuver(
                    epoch=ts,
                    confidence=c,
                    type=ManeuverType.IN_TRACK,
                    delta_v_estimate=1.0,
                    norad_id=1,
                    elset_epoch_before=ts,
                    elset_epoch_after=ts,
                )
                for c in self._confidences
            ]
        )


def test_calibrated_detector_transforms_confidence_and_keeps_schema() -> None:
    inner = _FixedDetector([0.9, 0.1])
    temperature = TemperatureScaling(temperature=2.0)
    out = CalibratedDetector(inner, temperature).detect(pd.DataFrame())
    assert list(out.columns) == list(COLUMNS)
    expected = temperature.transform(np.array([0.9, 0.1]))
    assert out["confidence"].to_numpy() == pytest.approx(expected)
    assert out["confidence"].min() >= 0.0 and out["confidence"].max() <= 1.0


def test_calibrated_detector_passes_through_empty_frame() -> None:
    out = CalibratedDetector(_FixedDetector([]), TemperatureScaling(2.0)).detect(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == list(COLUMNS)
