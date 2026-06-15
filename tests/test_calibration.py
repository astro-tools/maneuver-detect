"""Tests for the calibration machinery — reliability, temperature, conformal, and the wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maneuver_detect.calibration import (
    FALSE_ALARM,
    MANEUVER,
    BundledCalibration,
    CalibratedDetector,
    CalibrationSamples,
    ConformalPredictor,
    TemperatureScaling,
    apply_calibration,
    brier_score,
    expected_calibration_error,
    format_reliability_curve,
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
        ([float("nan"), 0.5], [1.0, 0.0], "finite"),  # a NaN must not slip past the range check
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


def test_temperature_transform_rejects_non_finite() -> None:
    # A NaN confidence must not pass through transform un-flagged (it would emit a NaN confidence).
    with pytest.raises(ValueError, match="finite"):
        TemperatureScaling(temperature=2.0).transform([0.5, float("nan")])


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


# --- the bundled, serialisable calibration (D17) ------------------------------------------------


def _val_samples() -> dict[str, CalibrationSamples]:
    """Per-orbit-class val samples — a rich LEO class and an empty (sparse) IGSO class."""
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.0, 1.0, size=64)
    # An over-confident detector: a detection fires true a bit below its stated confidence.
    outcome = (rng.uniform(0.0, 1.0, size=64) < conf * 0.7).astype(np.float64)
    return {
        "LEO": CalibrationSamples(conf, outcome),
        "IGSO": CalibrationSamples(
            np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        ),
    }


def test_bundled_calibration_fit_pools_and_measures_per_class() -> None:
    cal = BundledCalibration.fit(_val_samples(), alpha=0.1, n_bins=10)
    assert cal.temperature > 0.0
    assert 0.0 <= cal.conformal_q <= 1.0
    assert cal.conformal_alpha == 0.1
    # Reliability + ECE are present for every class given, including the empty one.
    assert set(cal.reliability) == {"LEO", "IGSO"}
    assert set(cal.ece) == {"LEO", "IGSO"}
    # An empty class yields an all-empty reliability curve and a zero ECE (sparse IGSO rides on).
    assert cal.ece["IGSO"] == 0.0
    assert not cal.reliability["IGSO"].populated()
    assert cal.reliability["LEO"].populated()


def test_bundled_calibration_dict_round_trips() -> None:
    cal = BundledCalibration.fit(_val_samples())
    assert BundledCalibration.from_dict(cal.to_dict()) == cal


def test_bundled_calibration_fit_rejects_all_empty() -> None:
    empty = {"LEO": CalibrationSamples(np.asarray([]), np.asarray([]))}
    with pytest.raises(ValueError, match="no matched detections"):
        BundledCalibration.fit(empty)


def test_bundled_calibration_temperature_scaling_is_the_baked_calibrator() -> None:
    cal = BundledCalibration.fit(_val_samples())
    assert cal.temperature_scaling().temperature == cal.temperature


def test_bundled_calibration_falls_back_to_identity_when_no_ece_gain() -> None:
    # A perfectly-calibrated, separable val sample (raw ECE already 0): no temperature can reduce
    # it, so the do-no-harm guard ships identity (T = 1) rather than a confidence-distorting fit.
    conf = np.array([0.0, 0.0, 1.0, 1.0])
    outcome = np.array([0.0, 0.0, 1.0, 1.0])
    cal = BundledCalibration.fit({"LEO": CalibrationSamples(conf, outcome)})
    assert cal.temperature == 1.0


def test_bundled_calibration_is_never_worse_calibrated_than_raw() -> None:
    # The shipped calibration's pooled ECE never exceeds the raw confidence's (the do-no-harm
    # invariant), whichever way the guard resolves.
    samples = _val_samples()
    cal = BundledCalibration.fit(samples)
    pooled_conf = np.concatenate([s.confidences for s in samples.values()])
    pooled_out = np.concatenate([s.outcomes for s in samples.values()])
    raw_ece = expected_calibration_error(pooled_conf, pooled_out)
    cal_ece = expected_calibration_error(
        cal.temperature_scaling().transform(pooled_conf), pooled_out
    )
    assert cal_ece <= raw_ece + 1e-12


def test_apply_calibration_remaps_confidence_and_keeps_schema() -> None:
    frame = _FixedDetector([0.9, 0.1, 0.5]).detect(pd.DataFrame())
    temperature = TemperatureScaling(temperature=2.0)
    out = apply_calibration(frame, temperature)
    assert list(out.columns) == list(COLUMNS)
    assert out["confidence"].to_numpy() == pytest.approx(
        temperature.transform(frame["confidence"].to_numpy())
    )
    # Every non-confidence column is preserved unchanged.
    for column in COLUMNS:
        if column != "confidence":
            assert out[column].tolist() == frame[column].tolist()


def test_apply_calibration_passes_through_empty_frame() -> None:
    empty = _FixedDetector([]).detect(pd.DataFrame())
    out = apply_calibration(empty, TemperatureScaling(2.0))
    assert out.empty
    assert list(out.columns) == list(COLUMNS)


def test_format_reliability_curve_renders_populated_bins() -> None:
    # Two low-confidence false alarms (bin 0) and two high-confidence hits (bin 9): two populated
    # bins; the eight empty bins between them are omitted.
    curve = reliability_curve(
        np.array([0.02, 0.08, 0.92, 0.98]), np.array([0.0, 0.0, 1.0, 1.0]), n_bins=10
    )
    rendered = format_reliability_curve(curve)
    lines = rendered.splitlines()
    assert lines[0] == "| bin | n | predicted | empirical |"
    assert len(lines) == 4  # header + separator + exactly the two populated bins
    assert "| [0.0, 0.1) | 2 |" in rendered
    assert "| [0.9, 1.0) | 2 |" in rendered


def test_format_reliability_curve_handles_an_empty_curve() -> None:
    empty = reliability_curve(np.array([]), np.array([]), n_bins=10)
    assert format_reliability_curve(empty) == "_(no binned detections)_"
