"""Mean-element feature engineering — the V5 / D11 irregular-sampling encoding the models consume.

The feature layer turns a cleaned per-object mean-element series into the frozen tensor contract the
v0.2 sequence models train on, in three composable steps that keep a clean, **label-free** boundary
(it takes the element series and returns model-input tensors + masks; it never sees a split or a
label, so no feature can leak the target):

1. :func:`build_channels` — one object's series → the raw per-token channel matrix
   (:class:`RawChannels`): element levels, secular-detrended residuals, signed deltas across each
   gap, the ``time2vec`` timing block, and the validity / saturation mask bits.
2. :class:`ClassNormaliser` — per-class robust (median / IQR) standardisation of the element
   channels, fit on the **train split only** (the caller's contract), yielding a float32
   :class:`FeatureMatrix`.
3. :func:`make_windows` — the matrix → the ``(batch, window, channels)`` tensors, with an optional
   per-token target windowed in lockstep but never computed here.

:func:`encode_history` is the convenience that runs step 1 over a multi-object history, grouping by
``norad_id`` — the natural input to :meth:`ClassNormaliser.fit`.
"""

from __future__ import annotations

import pandas as pd

from maneuver_detect.features.channels import (
    BASE_CHANNELS,
    CHANNEL_NAMES,
    CLIP_CAP_DAYS,
    DETREND_HALFWIDTH,
    N_CHANNELS,
    N_ELEMENT_CHANNELS,
    TIME2VEC_PERIODS_DAYS,
    TIME2VEC_SCALE_DAYS,
    RawChannels,
    build_channels,
)
from maneuver_detect.features.normalize import ClassNormaliser, FeatureMatrix
from maneuver_detect.features.windows import STRIDE, WINDOW, WindowedTensors, make_windows

__all__ = [
    "BASE_CHANNELS",
    "CHANNEL_NAMES",
    "CLIP_CAP_DAYS",
    "DETREND_HALFWIDTH",
    "N_CHANNELS",
    "N_ELEMENT_CHANNELS",
    "STRIDE",
    "TIME2VEC_PERIODS_DAYS",
    "TIME2VEC_SCALE_DAYS",
    "WINDOW",
    "ClassNormaliser",
    "FeatureMatrix",
    "RawChannels",
    "WindowedTensors",
    "build_channels",
    "encode_history",
    "make_windows",
]


def encode_history(history: pd.DataFrame) -> list[RawChannels]:
    """Encode a (possibly multi-object) mean-element ``history`` into per-object channels.

    Groups ``history`` by ``norad_id`` and runs :func:`build_channels` on each object, returning the
    channels in ascending ``norad_id`` order — the collection :meth:`ClassNormaliser.fit` pools to
    estimate the per-class statistics. An empty history yields an empty list.
    """
    if history.empty:
        return []
    return [build_channels(group) for _, group in history.groupby("norad_id", sort=True)]
