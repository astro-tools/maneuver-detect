"""The Lightning data module — labelled element series → windowed training tensors, leak-free.

:class:`ElementSeriesDataModule` turns per-object labelled mean-element series into the
``(features, validity, target)`` window batches the training loop consumes, reusing the frozen
feature layer end to end: :func:`~maneuver_detect.features.build_channels` →
:class:`~maneuver_detect.features.ClassNormaliser` (fit on the **train split only**) →
:func:`~maneuver_detect.features.make_windows`. The per-token target — "the gap into this token
holds a maneuver" — is derived here from each object's labelled intervals and windowed in lockstep
with the features, the one place a label meets the otherwise label-free feature layer.

The module takes already-split object lists (train / val), so the leak-free split construction
(:mod:`maneuver_detect.benchmark.splits`) stays the benchmark's concern, not the harness's;
:func:`objects_from_labelled_dataset` is the adapter that slices a reconstructed
:class:`~maneuver_detect.datasets.LabelledDataset` by a :class:`~maneuver_detect.benchmark.Split`.
The normaliser is fit on the train objects at :meth:`setup` and exposed for freezing into the
checkpoint, so inference standardises by the same statistics (D11.3).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import lightning as L
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from maneuver_detect.features.channels import N_CHANNELS, RawChannels, build_channels
from maneuver_detect.features.normalize import ClassNormaliser
from maneuver_detect.features.windows import STRIDE, WINDOW, make_windows

__all__ = ["ElementSeriesDataModule", "ObjectSeries", "objects_from_labelled_dataset"]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectSeries:
    """One object's training unit — its mean-element series and its maneuver-gap epochs.

    Attributes:
        norad_id: NORAD id of the object.
        series: The cleaned mean-element series
            (:data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS`).
        maneuver_epochs: The ``elset_epoch_after`` of every labelled maneuver gap — the epoch of the
            token the per-gap target attaches to (a maneuver in gap ``[t_i, t_{i+1})`` targets the
            token at ``t_{i+1}``). Timezone-aware UTC.
    """

    norad_id: int
    series: pd.DataFrame
    maneuver_epochs: tuple[pd.Timestamp, ...]


def _naive_utc_ns(timestamp: pd.Timestamp) -> int:
    """The UTC nanosecond value of ``timestamp`` with any timezone normalised away (UTC assumed)."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.to_datetime64().astype("datetime64[ns]").view("int64"))


def per_token_target(
    channels: RawChannels, maneuver_epochs: Sequence[pd.Timestamp]
) -> npt.NDArray[np.float32]:
    """The per-token target aligned to ``channels`` tokens — 1.0 at each maneuver-gap token.

    Each maneuver epoch is the ``elset_epoch_after`` of a labelled gap; the token whose epoch
    matches it is marked positive. An epoch with no matching token (e.g. an elset dropped as
    non-finite by the feature layer) is skipped and counted, never silently mis-aligned.
    """
    target = np.zeros(channels.n_tokens, dtype=np.float32)
    if not maneuver_epochs:
        return target
    # Key tokens by their UTC-nanosecond value. ``channels.epochs`` carries whatever datetime64
    # resolution the source frame had; normalise to ns so the keys match ``_naive_utc_ns`` (always
    # ns). Without this, a non-canonical us-resolution frame would mis-key every token and silently
    # drop all targets, training the model on an all-negative label set.
    epochs_ns = channels.epochs.astype("datetime64[ns]")
    token_index = {int(value.view("int64")): i for i, value in enumerate(epochs_ns)}
    missed = 0
    for epoch in maneuver_epochs:
        index = token_index.get(_naive_utc_ns(epoch))
        if index is None:
            missed += 1
            continue
        target[index] = 1.0
    if missed:
        _logger.warning(
            "object %d: %d maneuver epoch(s) did not align to a token and were dropped",
            channels.norad_id,
            missed,
        )
    return target


@dataclass(frozen=True)
class _ObjectWindows:
    """The windowed feature / validity / target tensors of one object, before concatenation."""

    features: npt.NDArray[np.float32]
    validity: npt.NDArray[np.bool_]
    target: npt.NDArray[np.float32]


class ElementSeriesDataModule(L.LightningDataModule):
    """Windowed training tensors from labelled element series, with a train-only normaliser.

    Pass the train and validation object lists (already split). :meth:`setup` fits the per-class
    normaliser on the **train** objects and encodes every object into overlapping windows; the
    normaliser is then available as :attr:`normaliser` to freeze into the checkpoint. ``batch_size``
    and the window geometry (``window`` / ``stride``, defaulting to the frozen feature-layer values)
    are training knobs; the encoding itself is the frozen contract.
    """

    def __init__(
        self,
        train_objects: Sequence[ObjectSeries],
        val_objects: Sequence[ObjectSeries] = (),
        *,
        window: int = WINDOW,
        stride: int = STRIDE,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self._train_objects = list(train_objects)
        self._val_objects = list(val_objects)
        self.window = window
        self.stride = stride
        self.batch_size = batch_size
        self.normaliser: ClassNormaliser | None = None
        self._train: TensorDataset | None = None
        self._val: TensorDataset | None = None
        self._pos_weight = 1.0

    def setup(self, stage: str | None = None) -> None:
        """Fit the train-split normaliser and window every object into training tensors."""
        if not self._train_objects:
            raise ValueError("the data module needs at least one training object")
        train_channels = [build_channels(obj.series) for obj in self._train_objects]
        self.normaliser = ClassNormaliser.fit(train_channels)

        train_windows = [
            self._object_windows(obj, channels)
            for obj, channels in zip(self._train_objects, train_channels, strict=True)
        ]
        self._train = self._dataset(train_windows)
        self._pos_weight = _positive_weight(train_windows)

        if self._val_objects:
            val_windows = [
                self._object_windows(obj, build_channels(obj.series)) for obj in self._val_objects
            ]
            self._val = self._dataset(val_windows)

    def _object_windows(self, obj: ObjectSeries, channels: RawChannels) -> _ObjectWindows:
        assert self.normaliser is not None  # set in setup before any object is windowed
        matrix = self.normaliser.transform(channels)
        target = per_token_target(channels, obj.maneuver_epochs)
        windows = make_windows(matrix, window=self.window, stride=self.stride, target=target)
        assert windows.target is not None  # a target was supplied, so it is windowed
        return _ObjectWindows(
            features=windows.features, validity=windows.validity, target=windows.target
        )

    @staticmethod
    def _dataset(windows: Sequence[_ObjectWindows]) -> TensorDataset:
        features = torch.from_numpy(np.concatenate([w.features for w in windows], axis=0))
        validity = torch.from_numpy(np.concatenate([w.validity for w in windows], axis=0))
        target = torch.from_numpy(np.concatenate([w.target for w in windows], axis=0))
        return TensorDataset(features, validity, target)

    def positive_weight(self) -> float:
        """The train-split negative/positive token ratio — the default BCE ``pos_weight``."""
        return self._pos_weight

    @property
    def has_validation(self) -> bool:
        """Whether a validation set was supplied (the training entrypoint disables val if not)."""
        return bool(self._val_objects)

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        assert self._train is not None  # setup() populates it
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        # Lightning rejects a ``None`` here, so an absent validation set yields an empty loader the
        # training entrypoint never iterates (it sets ``limit_val_batches=0`` when there is none).
        dataset = self._val if self._val is not None else _empty_dataset(self.window)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


def _empty_dataset(window: int) -> TensorDataset:
    """A zero-row dataset with the right tensor ranks — a valid, never-iterated empty val loader."""
    return TensorDataset(
        torch.empty(0, window, N_CHANNELS, dtype=torch.float32),
        torch.empty(0, window, dtype=torch.bool),
        torch.empty(0, window, dtype=torch.float32),
    )


def _positive_weight(windows: Sequence[_ObjectWindows]) -> float:
    """Negative/positive ratio over the valid tokens of ``windows`` (>= 1.0), for the BCE weight."""
    positives = 0.0
    valid = 0.0
    for window in windows:
        mask = window.validity.astype(np.float64)
        valid += float(mask.sum())
        positives += float((window.target.astype(np.float64) * mask).sum())
    if positives <= 0.0:
        return 1.0
    negatives = valid - positives
    return max(negatives / positives, 1.0)


def objects_from_labelled_dataset(dataset: object, split: object) -> dict[str, list[ObjectSeries]]:
    """Slice a reconstructed labelled dataset into per-split :class:`ObjectSeries` lists.

    ``dataset`` is a :class:`~maneuver_detect.datasets.LabelledDataset` and ``split`` a
    :class:`~maneuver_detect.benchmark.Split` (typed loosely to keep the heavy ``datasets`` import
    out of the harness import path). Returns ``{"train": [...], "val": [...], "test": [...]}``;
    objects in no split are dropped. Each object's maneuver epochs are the ``elset_epoch_after`` of
    its labelled intervals.
    """
    from maneuver_detect.benchmark.splits import SplitName

    membership = split.by_norad()  # type: ignore[attr-defined]
    out: dict[str, list[ObjectSeries]] = {name.value: [] for name in SplitName}
    for obj in dataset.objects:  # type: ignore[attr-defined]
        name = membership.get(obj.norad_id)
        if name is None:
            continue
        out[name.value].append(
            ObjectSeries(
                norad_id=obj.norad_id,
                series=obj.series,
                maneuver_epochs=tuple(iv.elset_epoch_after for iv in obj.intervals),
            )
        )
    return out
