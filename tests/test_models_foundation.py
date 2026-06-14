"""Tests for the foundation bundle and its offline driver — round-trip, calibration, scoring.

The calibration / scoring drivers wrap the tested ``models.evaluate`` path, so they are exercised
here with the deterministic drift-continuation **stand-in** forecaster (no ``[foundation]`` extra)
on a synthetic temporal split — the same construction the evaluate tests use.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import torch

from _synthetic import Burn, synthetic_series
from maneuver_detect.benchmark import ScoreReport, SplitName, TemporalSplit
from maneuver_detect.detectors.foundation import DriftContinuationForecaster
from maneuver_detect.errors import ManeuverDetectError
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.models.foundation import (
    FOUNDATION_DEFAULTS,
    FOUNDATION_THRESHOLD_SWEEP,
    FoundationBundle,
    calibrate_and_score,
    calibrate_thresholds,
    load_foundation_bundle,
    save_foundation_bundle,
    score_bundle,
    zero_shot_bundle,
)

_CUT1 = datetime(2024, 8, 1, tzinfo=timezone.utc)
_CUT2 = datetime(2025, 8, 1, tzinfo=timezone.utc)
_GUARD = timedelta(days=7)


def _label(frame: pd.DataFrame, gap_index: int, dv: float) -> ManeuverLabel:
    epochs = list(frame["epoch"])
    midpoint = epochs[gap_index - 1] + (epochs[gap_index] - epochs[gap_index - 1]) / 2
    return ManeuverLabel(
        norad_id=int(frame["norad_id"].iloc[0]),
        epoch=midpoint.to_pydatetime(),
        window_start=epochs[gap_index - 1].to_pydatetime(),
        window_end=epochs[gap_index].to_pydatetime(),
        source="SYNTHETIC",
        source_ref=f"{int(frame['norad_id'].iloc[0])}-{gap_index}",
        orbit_class=OrbitClass.LEO,
        maneuver_type=None,
        delta_v=dv,
    )


def _split(**members: frozenset[int]) -> TemporalSplit:
    return TemporalSplit(
        dataset_version="test",
        seed=0,
        cut1=_CUT1,
        cut2=_CUT2,
        guard=_GUARD,
        train=members.get("train", frozenset()),
        val=members.get("val", frozenset()),
        test=members.get("test", frozenset()),
    )


def test_bundle_round_trips_through_disk(tmp_path: Path) -> None:
    bundle = FoundationBundle(
        backend="chronos",
        checkpoint_id="amazon/chronos-bolt-small",
        revision="abc123",
        context_length=64,
        class_thresholds={"LEO": 4.5, "MEO": 4.0, "GEO": 3.5},
        finetune_state={"w": torch.ones(3)},
        metadata={"mode": "fine-tuned", "dataset_version": "0.2.0"},
    )
    path = tmp_path / "chronos-residual.pt"
    save_foundation_bundle(bundle, path)
    loaded = load_foundation_bundle(path)

    assert loaded.backend == "chronos"
    assert loaded.checkpoint_id == "amazon/chronos-bolt-small"
    assert loaded.revision == "abc123"
    assert loaded.context_length == 64
    assert loaded.class_thresholds == {"LEO": 4.5, "MEO": 4.0, "GEO": 3.5}
    assert loaded.finetune_state is not None
    assert torch.equal(loaded.finetune_state["w"], torch.ones(3))
    assert loaded.metadata["mode"] == "fine-tuned"


def test_load_rejects_a_non_bundle(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(ManeuverDetectError, match="is not a bundle"):
        load_foundation_bundle(path)


def test_load_rejects_a_truncated_bundle(tmp_path: Path) -> None:
    path = tmp_path / "partial.pt"
    torch.save({"backend": "chronos"}, path)  # missing checkpoint_id / revision / ...
    with pytest.raises(ManeuverDetectError, match="missing required keys"):
        load_foundation_bundle(path)


def test_zero_shot_bundle_defaults_from_backend() -> None:
    bundle = zero_shot_bundle("chronos", revision="rev-1")
    assert bundle.backend == "chronos"
    assert bundle.checkpoint_id == FOUNDATION_DEFAULTS["chronos"].checkpoint_id
    assert bundle.context_length == FOUNDATION_DEFAULTS["chronos"].context_length
    assert bundle.revision == "rev-1"
    assert bundle.finetune_state is None
    assert bundle.metadata["mode"] == "zero-shot"
    assert bundle.class_thresholds == {}


def test_zero_shot_bundle_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown foundation backend"):
        zero_shot_bundle("not-a-model")


def test_calibrate_thresholds_tunes_and_freezes_per_class() -> None:
    frame = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(400, "in_track_ms", 4.0),))
    labels = [_label(frame, 400, 4.0)]
    split = _split(val=frozenset({1}))
    bundle = zero_shot_bundle("chronos")

    calibrated, tuning = calibrate_thresholds(
        bundle,
        {1: frame},
        labels,
        split,
        forecaster=DriftContinuationForecaster(),
    )

    assert tuning.threshold in FOUNDATION_THRESHOLD_SWEEP
    # The chosen residual-z cutoff is frozen into every class (per-class refinement is a later
    # deliverable); the recall and the full sweep ride along on the metadata.
    assert set(calibrated.class_thresholds) == {oc.value for oc in OrbitClass}
    assert set(calibrated.class_thresholds.values()) == {tuning.threshold}
    assert calibrated.metadata["calibration"]["recall"] == pytest.approx(tuning.recall)
    assert 0.0 <= tuning.recall <= 1.0


def test_score_bundle_records_the_test_report() -> None:
    frame = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(700, "in_track_ms", 4.0),))
    labels = [_label(frame, 700, 4.0)]
    split = _split(test=frozenset({1}))
    bundle = zero_shot_bundle("chronos", class_thresholds={"LEO": 4.0, "MEO": 4.0, "GEO": 4.0})

    scored, report = score_bundle(
        bundle,
        {1: frame},
        labels,
        split,
        forecaster=DriftContinuationForecaster(),
        partition=SplitName.TEST,
    )

    assert isinstance(report, ScoreReport)
    assert OrbitClass.LEO in report.per_class
    # The report is serialised onto the bundle in the shape the model card consumes.
    recorded = scored.metadata["test_report"]
    assert isinstance(recorded["per_class"], dict)
    assert "LEO" in recorded["per_class"]


def test_calibrate_and_score_runs_the_full_flow() -> None:
    # The end-to-end driver: assemble a zero-shot bundle, calibrate on val (object 1), score on test
    # (object 2). Exercised with the stand-in forecaster, so it needs no [foundation] extra.
    val_frame = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(400, "in_track_ms", 4.0),))
    test_frame = synthetic_series(norad_id=2, seed=1, n=900, burns=(Burn(700, "in_track_ms", 4.0),))
    labels = [_label(val_frame, 400, 4.0), _label(test_frame, 700, 4.0)]
    split = _split(val=frozenset({1}), test=frozenset({2}))

    bundle, report = calibrate_and_score(
        "chronos",
        {1: val_frame, 2: test_frame},
        labels,
        split,
        forecaster=DriftContinuationForecaster(),
    )

    assert isinstance(report, ScoreReport)
    # The bundle came out calibrated (val) and scored (test): both rounds rode through.
    assert set(bundle.class_thresholds) == {oc.value for oc in OrbitClass}
    assert "calibration" in bundle.metadata
    assert isinstance(bundle.metadata["test_report"]["per_class"], dict)


def test_finetune_trains_on_the_train_split_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Leak guard: the fine-tune must see only the train-split objects, never the held-out val/test
    # objects it is then scored against. Capture what reaches finetune_chronos.
    import maneuver_detect.models.foundation as mf

    captured: dict[str, set[int]] = {}

    def fake_finetune(bundle, series_by_norad, **kwargs):  # type: ignore[no-untyped-def]
        captured["ids"] = set(series_by_norad)
        return bundle

    monkeypatch.setattr(mf, "finetune_chronos", fake_finetune)

    train = synthetic_series(norad_id=1, seed=0, n=900, burns=(Burn(100, "in_track_ms", 4.0),))
    val = synthetic_series(norad_id=2, seed=1, n=900, burns=(Burn(400, "in_track_ms", 4.0),))
    test = synthetic_series(norad_id=3, seed=2, n=900, burns=(Burn(700, "in_track_ms", 4.0),))
    labels = [_label(train, 100, 4.0), _label(val, 400, 4.0), _label(test, 700, 4.0)]
    split = _split(train=frozenset({1}), val=frozenset({2}), test=frozenset({3}))

    calibrate_and_score(
        "chronos",
        {1: train, 2: val, 3: test},
        labels,
        split,
        finetune=True,
        forecaster=DriftContinuationForecaster(),
    )

    assert captured["ids"] == {1}  # only the train-split object — not the val/test objects
