"""Tests for the labelled-series data module: encoding, target alignment, and train-only fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch

from _synthetic import Burn, object_series, synthetic_series
from maneuver_detect.benchmark.splits import Split
from maneuver_detect.features.channels import N_CHANNELS, build_channels
from maneuver_detect.features.normalize import ClassNormaliser
from maneuver_detect.labels.labeller import LabelledInterval, label_series
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.models.datamodule import (
    ElementSeriesDataModule,
    objects_from_labelled_dataset,
    per_token_target,
)


def test_per_token_target_marks_the_elset_after_each_gap() -> None:
    burns = (Burn(30, "in_track_ms", 2.0), Burn(70, "cross_track_ms", 2.5))
    frame = synthetic_series(norad_id=1, seed=0, burns=burns)
    channels = build_channels(frame)
    epochs = [pd.Timestamp(frame["epoch"].iloc[burn.gap_index]) for burn in burns]

    target = per_token_target(channels, epochs)

    assert target.dtype == np.float32
    assert target.sum() == 2.0
    assert set(np.flatnonzero(target).tolist()) == {30, 70}


def test_per_token_target_skips_unalignable_epochs() -> None:
    channels = build_channels(synthetic_series(norad_id=1, seed=0))
    # An epoch outside the series span aligns to no token and is dropped, never mis-assigned.
    target = per_token_target(channels, [pd.Timestamp("2000-01-01T00:00:00", tz="UTC")])
    assert target.sum() == 0.0


def test_per_token_target_aligns_under_microsecond_epoch_resolution() -> None:
    # build_channels is resolution-agnostic, so a non-canonical microsecond-resolution frame yields
    # microsecond token epochs. per_token_target keys tokens by nanosecond, so those epochs still
    # align — without that, every target would be silently dropped and the model would train on an
    # all-negative label set.
    burns = (Burn(30, "in_track_ms", 2.0), Burn(70, "cross_track_ms", 2.5))
    coarse = synthetic_series(norad_id=1, seed=0, burns=burns)
    coarse["epoch"] = coarse["epoch"].dt.as_unit("us")
    epochs = [pd.Timestamp(coarse["epoch"].iloc[burn.gap_index]) for burn in burns]

    channels = build_channels(coarse)
    assert channels.epochs.dtype == np.dtype("datetime64[us]")  # the microsecond path is exercised

    target = per_token_target(channels, epochs)
    assert set(np.flatnonzero(target).tolist()) == {30, 70}
    assert target.sum() == 2.0


def test_setup_builds_window_tensors_and_fits_train_only_normaliser() -> None:
    train = [
        object_series(norad_id=1, seed=1, burns=(Burn(40, "in_track_ms", 3.0),)),
        object_series(norad_id=2, seed=2, burns=(Burn(55, "cross_track_ms", 3.0),)),
    ]
    val = [object_series(norad_id=3, seed=3, burns=(Burn(50, "in_track_ms", 3.0),))]

    dm = ElementSeriesDataModule(train, val, window=32, stride=16, batch_size=8)
    dm.setup()

    assert isinstance(dm.normaliser, ClassNormaliser)
    assert OrbitClass.LEO in dm.normaliser.classes

    features, validity, target = next(iter(dm.train_dataloader()))
    assert features.shape[1:] == (32, N_CHANNELS)
    assert features.dtype.is_floating_point
    assert validity.dtype == torch.bool
    assert target.shape == validity.shape
    assert dm.positive_weight() > 1.0  # maneuver gaps are rare → positives up-weighted


def test_val_dataloader_empty_without_val_objects() -> None:
    dm = ElementSeriesDataModule(
        [object_series(norad_id=1, seed=1, burns=(Burn(40, "in_track_ms", 3.0),))],
        window=32,
        stride=16,
    )
    dm.setup()
    # No validation set: a valid but empty loader (Lightning rejects a None), and the flag is False.
    assert dm.has_validation is False
    assert len(dm.val_dataloader().dataset) == 0  # type: ignore[arg-type]


def test_setup_rejects_empty_training_set() -> None:
    with pytest.raises(ValueError, match="at least one training object"):
        ElementSeriesDataModule([]).setup()


def test_objects_from_labelled_dataset_slices_by_split() -> None:
    @dataclass
    class _Obj:
        norad_id: int
        series: pd.DataFrame
        intervals: list[LabelledInterval]

    @dataclass
    class _Dataset:
        objects: list[_Obj]

    frame = synthetic_series(norad_id=10, seed=0, burns=(Burn(40, "in_track_ms", 3.0),))
    epochs = list(frame["epoch"])
    midpoint = epochs[39] + (epochs[40] - epochs[39]) / 2
    labels = [
        ManeuverLabel(
            norad_id=10,
            epoch=midpoint.to_pydatetime(),
            window_start=epochs[39].to_pydatetime(),
            window_end=epochs[40].to_pydatetime(),
            source="SYNTHETIC",
            source_ref="10-40",
            orbit_class=OrbitClass.LEO,
        )
    ]
    intervals = label_series(frame, labels).intervals
    dataset = _Dataset(objects=[_Obj(norad_id=10, series=frame, intervals=intervals)])
    split = Split(
        dataset_version="test",
        seed=0,
        ratios=(0.7, 0.15, 0.15),
        train=frozenset({10}),
        val=frozenset(),
        test=frozenset(),
    )

    sliced = objects_from_labelled_dataset(dataset, split)
    assert [o.norad_id for o in sliced["train"]] == [10]
    assert sliced["val"] == []
    assert sliced["train"][0].maneuver_epochs == (intervals[0].elset_epoch_after,)
