"""Per-class robust normalisation of the element channels — the train-only standardiser (D11.3).

The element channels (the level / residual / delta blocks of :class:`RawChannels`) live on
wildly different scales — kilometres for ``a``, arc-seconds for the node — and differ by orbit
class (a LEO drag make-up and a GEO east-west burn are different regimes). :class:`ClassNormaliser`
standardises them **robustly** (median / inter-quartile range, immune to the maneuver outliers) and
**per class**, exactly as D11.3 froze it.

The leak-free contract — statistics fit on the **train split only** — is the *caller's* to honour:
this layer never sees split membership, it only fits on whatever series it is handed. The training
harness fits a normaliser on its train-split series and applies the same one to val / test, so the
held-out objects are standardised by training-set statistics and nothing leaks. The bounded timing
and mask columns are passed through unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from maneuver_detect.features.channels import (
    CHANNEL_NAMES,
    N_ELEMENT_CHANNELS,
    RawChannels,
)
from maneuver_detect.labels.record import OrbitClass

__all__ = ["ClassNormaliser", "FeatureMatrix"]

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

# A floor on the inter-quartile range so a degenerate (constant) channel cannot divide-by-zero; a
# real value then reads as overwhelmingly large and a constant channel as zero, both correct.
_IQR_FLOOR = 1e-12


@dataclass(frozen=True, eq=False)
class FeatureMatrix:
    """A normalised per-token feature matrix ready for windowing — the float32 model input.

    Attributes:
        norad_id: The object the features were built for.
        orbit_class: The object's orbit class (the normalisation statistics it was standardised by).
        epochs: The token epochs (naive UTC ``datetime64[ns]``), one per row, in order.
        features: The ``(n_tokens, N_CHANNELS)`` standardised channel matrix, ``float32``, in
            :data:`~maneuver_detect.features.channels.CHANNEL_NAMES` order.
        validity: A ``(n_tokens,)`` boolean — every real token is valid; windowing flips padded
            positions to ``False``.
        channel_names: The column names of :attr:`features`.
    """

    norad_id: int
    orbit_class: OrbitClass
    epochs: npt.NDArray[np.datetime64]
    features: Float32Array
    validity: npt.NDArray[np.bool_]
    channel_names: tuple[str, ...] = CHANNEL_NAMES

    @property
    def n_tokens(self) -> int:
        """The number of tokens (rows)."""
        return int(self.features.shape[0])


@dataclass(frozen=True, eq=False)
class ClassNormaliser:
    """Per-class robust (median / IQR) standardiser for the element channels.

    Built by :meth:`fit` over a collection of :class:`RawChannels` (the **train split**, the
    caller's responsibility) and applied by :meth:`transform`. The fitted statistics are
    serialisable (:meth:`to_dict` / :meth:`from_dict`) so a normaliser freezes alongside a model
    checkpoint and reproduces the same standardisation later.

    Attributes:
        medians: Per-class median of each element channel (``{class: (N_ELEMENT_CHANNELS,)}``).
        iqrs: Per-class inter-quartile range of each element channel, floored away from zero.
    """

    medians: dict[OrbitClass, FloatArray]
    iqrs: dict[OrbitClass, FloatArray]

    @classmethod
    def fit(cls, channels: Iterable[RawChannels]) -> ClassNormaliser:
        """Fit per-class median / IQR statistics from ``channels`` (intended: the train split).

        The element block of every series is pooled by orbit class and the per-column median and
        inter-quartile range are taken over all of that class's tokens. A class with no tokens
        contributes no statistics (and :meth:`transform` then refuses it).

        Raises:
            ValueError: if ``channels`` yields no tokens at all.
        """
        blocks: dict[OrbitClass, list[FloatArray]] = {}
        for series in channels:
            if series.n_tokens == 0:
                continue
            blocks.setdefault(series.orbit_class, []).append(series.element_block())
        if not blocks:
            raise ValueError("cannot fit a normaliser on an empty set of channels")

        medians: dict[OrbitClass, FloatArray] = {}
        iqrs: dict[OrbitClass, FloatArray] = {}
        for orbit_class, parts in blocks.items():
            pooled = np.vstack(parts)
            q25, q50, q75 = np.percentile(pooled, [25.0, 50.0, 75.0], axis=0)
            medians[orbit_class] = np.asarray(q50, dtype=np.float64)
            iqrs[orbit_class] = np.maximum(np.asarray(q75 - q25, dtype=np.float64), _IQR_FLOOR)
        return cls(medians=medians, iqrs=iqrs)

    @property
    def classes(self) -> frozenset[OrbitClass]:
        """The orbit classes the normaliser carries statistics for."""
        return frozenset(self.medians)

    def transform(self, channels: RawChannels) -> FeatureMatrix:
        """Standardise the element block of ``channels`` by its class statistics → a float32 matrix.

        The element columns are mapped to ``(x - median) / IQR`` for the series' orbit class; the
        timing and mask columns (bounded by construction) pass through unchanged. The result is cast
        to ``float32`` (the frozen tensor dtype) with an all-valid token mask.

        Raises:
            ValueError: if the normaliser was not fit on the series' orbit class.
        """
        if channels.orbit_class not in self.medians:
            raise ValueError(
                f"normaliser has no statistics for class {channels.orbit_class.value}; "
                f"fit covered {sorted(c.value for c in self.classes)}"
            )
        median = self.medians[channels.orbit_class]
        iqr = self.iqrs[channels.orbit_class]

        out = channels.matrix.astype(np.float64, copy=True)
        out[:, :N_ELEMENT_CHANNELS] = (out[:, :N_ELEMENT_CHANNELS] - median) / iqr
        features = out.astype(np.float32)
        validity = np.ones(channels.n_tokens, dtype=np.bool_)
        return FeatureMatrix(
            norad_id=channels.norad_id,
            orbit_class=channels.orbit_class,
            epochs=channels.epochs,
            features=features,
            validity=validity,
            channel_names=channels.channel_names,
        )

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        """Serialise the fitted statistics to a plain, JSON-stable dict (to freeze with a model)."""
        return {
            "medians": {c.value: self.medians[c].tolist() for c in sorted(self.medians, key=str)},
            "iqrs": {c.value: self.iqrs[c].tolist() for c in sorted(self.iqrs, key=str)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, list[float]]]) -> ClassNormaliser:
        """Reconstruct a normaliser from :meth:`to_dict` output."""
        medians = {
            OrbitClass(name): np.asarray(values, dtype=np.float64)
            for name, values in data["medians"].items()
        }
        iqrs = {
            OrbitClass(name): np.asarray(values, dtype=np.float64)
            for name, values in data["iqrs"].items()
        }
        return cls(medians=medians, iqrs=iqrs)
