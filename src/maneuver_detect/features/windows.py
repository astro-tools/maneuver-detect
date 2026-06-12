"""Sliding-window extraction — the ``(batch, W, channels)`` tensors the sequence models train on.

The V5 contract trains on sliding windows of ``W`` consecutive tokens (a few months at the daily
cadence), stride ``< W`` so every inter-elset gap appears with bidirectional context. A window never
crosses a satellite boundary — :func:`make_windows` operates on **one object's**
:class:`~maneuver_detect.features.normalize.FeatureMatrix`, so the guarantee holds by construction
(the caller builds one matrix per ``norad_id``). Windows that run off the end of the series are
zero-padded and the padding is marked invalid, so a short series still yields one window and the
model can mask the padding.

A per-token ``target`` (the per-gap maneuver label, computed elsewhere from the labels) may be
windowed **in lockstep** with the features, but it is never *computed* here — the feature layer
stays label-free. The features a window carries do not depend on the target at all: passing a
different target changes only the returned ``target`` tensor, never the features (the leak-free
boundary the issue DoD asserts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from maneuver_detect.features.normalize import FeatureMatrix

__all__ = ["STRIDE", "WINDOW", "WindowedTensors", "make_windows"]

Float32Array = npt.NDArray[np.float32]

#: The default window length ``W`` (consecutive tokens / elsets) — ~two months at daily cadence.
WINDOW = 64

#: The default stride between window starts (``< WINDOW``, so windows overlap and every gap is seen
#: with both-sided context).
STRIDE = 32


@dataclass(frozen=True, eq=False)
class WindowedTensors:
    """A batch of fixed-length windows over one object's feature series.

    Attributes:
        norad_id: The object the windows were extracted from.
        features: The ``(n_windows, window, n_channels)`` ``float32`` feature tensor.
        validity: The ``(n_windows, window)`` boolean mask — ``True`` for a real token, ``False``
            for end-of-series padding.
        target: The ``(n_windows, window)`` ``float32`` per-token target windowed in lockstep, or
            ``None`` if no target was supplied. Padding positions are zero.
        window: The window length ``W`` the batch was cut at.
        stride: The stride between window starts.
    """

    norad_id: int
    features: Float32Array
    validity: npt.NDArray[np.bool_]
    target: Float32Array | None
    window: int
    stride: int

    @property
    def n_windows(self) -> int:
        """The number of windows (the batch dimension)."""
        return int(self.features.shape[0])


def make_windows(
    matrix: FeatureMatrix,
    *,
    window: int = WINDOW,
    stride: int = STRIDE,
    target: npt.NDArray[Any] | None = None,
) -> WindowedTensors:
    """Cut ``matrix`` into overlapping windows of ``window`` tokens at ``stride``.

    Each window starts at ``0, stride, 2·stride, …`` while the start is still within the series, so
    the last real token is always covered; a window running past the end is zero-padded with its
    padding marked invalid. If ``target`` is given (a per-token array aligned to ``matrix``'s
    tokens), it is windowed identically and returned cast to ``float32`` (padding zero); the feature
    windows are unaffected by it.

    Raises:
        ValueError: if ``window < 1``, ``stride`` is not in ``[1, window]``, or ``target`` is given
            but its length does not match the token count.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    if not 1 <= stride <= window:
        raise ValueError(f"stride must be in [1, window], got {stride} (window={window})")

    n_tokens = matrix.n_tokens
    n_channels = matrix.features.shape[1]
    target_array: Float32Array | None = None
    if target is not None:
        target_flat = np.asarray(target, dtype=np.float32).reshape(-1)
        if target_flat.shape[0] != n_tokens:
            raise ValueError(
                f"target length {target_flat.shape[0]} does not match the {n_tokens} tokens"
            )
        target_array = target_flat

    starts = list(range(0, n_tokens, stride))
    n_windows = len(starts)

    features = np.zeros((n_windows, window, n_channels), dtype=np.float32)
    validity = np.zeros((n_windows, window), dtype=np.bool_)
    targets = np.zeros((n_windows, window), dtype=np.float32) if target_array is not None else None

    for row, start in enumerate(starts):
        real = min(window, n_tokens - start)
        features[row, :real] = matrix.features[start : start + real]
        validity[row, :real] = matrix.validity[start : start + real]
        if targets is not None and target_array is not None:
            targets[row, :real] = target_array[start : start + real]

    return WindowedTensors(
        norad_id=matrix.norad_id,
        features=features,
        validity=validity,
        target=targets,
        window=window,
        stride=stride,
    )
