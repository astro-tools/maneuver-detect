"""The BiLSTM learned detector — localise gaps with the model, invert the physics for Δv/type.

``BiLstmDetector`` is the inference side of the learned baseline: it loads a trained checkpoint
bundle, runs the bidirectional LSTM over the V5-encoded series on **CPU**, and turns the per-token
maneuver probabilities into the canonical maneuver schema. The division of labour is the fixed one —
**the model localises, the physics inverts**: the network decides *which* inter-elset gaps hold a
maneuver and with what confidence, and each detected gap is then handed to the same
:func:`maneuver_detect.physics.invert` Gauss inversion the classical detector uses to recover the Δv
magnitude and the maneuver type. The per-type detectability floor that gates the reported Δv is
reused verbatim from the classical detector's calibration, so a learned and a rule-based detection
are gated identically (D4/D5).

The detector registers under ``"bilstm-base"``. It needs a trained checkpoint: construct it with a
bundle (or a path), or set the :data:`CHECKPOINT_ENV` environment variable to a bundle path so the
no-argument construction the registry uses (``detect(history, model="bilstm-base")``) finds one.
Without a checkpoint it raises a clear error — downloading a published checkpoint from the Hub is a
later milestone. The heavy ``torch`` / model imports are deferred to construction time, so importing
the package (or using the classical detector) never pays for them.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from maneuver_detect.detectors.base import Detector
from maneuver_detect.detectors.classical import ClassicalDetector
from maneuver_detect.features.channels import BASE_CHANNELS, RawChannels, build_channels
from maneuver_detect.features.windows import make_windows
from maneuver_detect.physics import ElementStep, Inversion, Orbit, invert, local_step
from maneuver_detect.schema import COLUMNS, Maneuver, empty_frame, to_frame

if TYPE_CHECKING:
    from torch import nn

    from maneuver_detect.features.normalize import ClassNormaliser
    from maneuver_detect.models.checkpoint import ModelBundle

__all__ = ["CHECKPOINT_ENV", "BiLstmDetector"]

FloatArray = npt.NDArray[np.float64]

#: Environment variable naming a checkpoint-bundle path the no-argument detector loads, so the
#: ``detect(history, model="bilstm-base")`` dispatch path can find a trained model without the
#: caller threading one through. A published-checkpoint default (Hub) arrives in a later release.
CHECKPOINT_ENV = "MANEUVER_DETECT_BILSTM_CHECKPOINT"

#: Samples per side for the two-sided local-linear step fit and the reference-orbit median window
#: the physics inversion linearises about (the same role as the classical detector's ``window``).
_STEP_WINDOW = 4

_BASE_INDEX = {name: index for index, name in enumerate(BASE_CHANNELS)}


class BiLstmDetector(Detector):
    """Learned BiLSTM detector — per-gap localisation by the model, Δv/type by the physics.

    Construct with a trained checkpoint (a :class:`~maneuver_detect.models.checkpoint.ModelBundle`
    or a path to one); the no-argument construction the registry uses falls back to the
    :data:`CHECKPOINT_ENV` path, and raises from :meth:`detect` if neither is available.
    ``threshold`` overrides the bundle's default per-gap maneuver-probability threshold.
    """

    name = "bilstm-base"

    def __init__(
        self,
        checkpoint: ModelBundle | str | Path | None = None,
        *,
        threshold: float | None = None,
    ) -> None:
        if checkpoint is None:
            env_path = os.environ.get(CHECKPOINT_ENV)
            checkpoint = env_path if env_path else None

        self._network: nn.Module | None = None
        self._normaliser: ClassNormaliser | None = None
        self._window = 0
        self._stride = 0
        self._threshold = 0.5

        if checkpoint is not None:
            self._load(checkpoint, threshold)

    def _load(self, checkpoint: ModelBundle | str | Path, threshold: float | None) -> None:
        # Deferred so importing the package (or the classical detector) never imports torch.
        from maneuver_detect.features.normalize import ClassNormaliser
        from maneuver_detect.models.checkpoint import ModelBundle, build_network, load_bundle

        bundle = checkpoint if isinstance(checkpoint, ModelBundle) else load_bundle(checkpoint)
        self._network = build_network(bundle)
        self._normaliser = ClassNormaliser.from_dict(bundle.normaliser)
        self._window = bundle.window
        self._stride = bundle.stride
        self._threshold = bundle.threshold if threshold is None else threshold

    @property
    def is_loaded(self) -> bool:
        """Whether a trained checkpoint is loaded (``detect`` works only when it is)."""
        return self._network is not None

    def detect(self, history: pd.DataFrame) -> pd.DataFrame:
        """Detect maneuvers in ``history`` and return the canonical maneuver DataFrame.

        ``history`` is a mean-element series; a frame with multiple objects is grouped by
        ``norad_id`` and each object detected independently, with rows returned sorted by
        ``(norad_id, epoch)``. Raises :class:`ValueError` if no trained checkpoint is loaded.
        """
        if self._network is None or self._normaliser is None:
            raise ValueError(
                f"the {self.name!r} detector needs a trained checkpoint; construct "
                f"BiLstmDetector(checkpoint=...) or set ${CHECKPOINT_ENV} (Hub-published "
                "checkpoints arrive in a later release)"
            )
        if history.empty:
            return empty_frame()

        maneuvers: list[Maneuver] = []
        for _, group in history.groupby("norad_id", sort=True):
            maneuvers.extend(self._detect_object(group.sort_values("epoch")))

        frame = to_frame(maneuvers)
        ordered = frame.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
        result: pd.DataFrame = ordered[list(COLUMNS)]
        return result

    def _detect_object(self, series: pd.DataFrame) -> list[Maneuver]:
        channels = build_channels(series)
        token_prob = self._token_probabilities(channels)
        gaps = _detected_gaps(token_prob, self._threshold)
        if not gaps:
            return []

        elements = _AlignedElements.from_channels(channels)
        floors = ClassicalDetector().floor_for(series)
        norad_id = channels.norad_id

        maneuvers: list[Maneuver] = []
        for gap in gaps:
            inversion = elements.invert_gap(gap)
            before = elements.epoch_utc(gap - 1)
            after = elements.epoch_utc(gap)
            maneuvers.append(
                Maneuver(
                    epoch=pd.Timestamp(before + (after - before) / 2),
                    confidence=float(token_prob[gap]),
                    type=inversion.maneuver_type,
                    delta_v_estimate=inversion.delta_v_estimate(floors[inversion.maneuver_type]),
                    norad_id=norad_id,
                    elset_epoch_before=before,
                    elset_epoch_after=after,
                )
            )
        return maneuvers

    def _token_probabilities(self, channels: RawChannels) -> FloatArray:
        """Per-token maneuver probability over the series, averaged across overlapping windows."""
        import torch

        assert self._normaliser is not None and self._network is not None
        matrix = self._normaliser.transform(channels)
        windows = make_windows(matrix, window=self._window, stride=self._stride)
        with torch.no_grad():
            logits = self._network(torch.from_numpy(windows.features))
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)

        n_tokens = channels.n_tokens
        prob_sum = np.zeros(n_tokens, dtype=np.float64)
        count = np.zeros(n_tokens, dtype=np.float64)
        for row, start in enumerate(range(0, n_tokens, self._stride)):
            real = min(self._window, n_tokens - start)
            prob_sum[start : start + real] += probs[row, :real]
            count[start : start + real] += 1.0
        return prob_sum / np.maximum(count, 1.0)


def _detected_gaps(token_prob: FloatArray, threshold: float) -> list[int]:
    """Gap indices the model fires on — each contiguous above-threshold run reduced to its peak.

    Token ``i`` (for ``i >= 1``) scores the gap ``[i-1, i]``. A maneuver can light up a short run of
    adjacent tokens; collapsing each run to its argmax yields one detection per maneuver (the
    learned analogue of the classical detector's non-maximum suppression), within the D4 tolerance.
    """
    n = token_prob.shape[0]
    gaps: list[int] = []
    i = 1
    while i < n:
        if token_prob[i] >= threshold:
            j = i
            while j + 1 < n and token_prob[j + 1] >= threshold:
                j += 1
            peak = i + int(np.argmax(token_prob[i : j + 1]))
            gaps.append(peak)
            i = j + 1
        else:
            i += 1
    return gaps


class _AlignedElements:
    """The per-token physical elements recovered from a series' channel matrix, for the inversion.

    The level block of :class:`RawChannels` carries the un-normalised element values, so the
    physical series the Gauss inversion needs — ``a`` (km), ``e``, inclination and node (rad) — is
    read off it, perfectly aligned to the tokens the model scored (no re-cleaning, no index drift).
    ``h`` / ``k`` (the eccentricity vector) recover the argument of perigee for the reference orbit.
    """

    __slots__ = (
        "a_km",
        "argp_rad",
        "cos_i",
        "e",
        "epochs_utc",
        "h",
        "inc_rad",
        "k",
        "raan_rad",
        "sin_i",
        "t_days",
    )

    def __init__(self, channels: RawChannels) -> None:
        matrix = channels.matrix
        self.a_km: FloatArray = matrix[:, _BASE_INDEX["a"]].astype(np.float64)
        self.e: FloatArray = matrix[:, _BASE_INDEX["e"]].astype(np.float64)
        self.sin_i: FloatArray = matrix[:, _BASE_INDEX["sin_i"]].astype(np.float64)
        self.cos_i: FloatArray = matrix[:, _BASE_INDEX["cos_i"]].astype(np.float64)
        self.h: FloatArray = matrix[:, _BASE_INDEX["h"]].astype(np.float64)
        self.k: FloatArray = matrix[:, _BASE_INDEX["k"]].astype(np.float64)
        raan_deg: FloatArray = matrix[:, _BASE_INDEX["raan"]].astype(np.float64)
        self.inc_rad: FloatArray = np.arctan2(self.sin_i, self.cos_i)
        self.raan_rad: FloatArray = np.deg2rad(raan_deg)
        epoch_index = pd.DatetimeIndex(channels.epochs).tz_localize("UTC")
        self.epochs_utc: list[pd.Timestamp] = list(epoch_index)
        origin = epoch_index[0]
        self.t_days: list[float] = [
            (epoch - origin).total_seconds() / 86400.0 for epoch in epoch_index
        ]

    @classmethod
    def from_channels(cls, channels: RawChannels) -> _AlignedElements:
        return cls(channels)

    def epoch_utc(self, index: int) -> pd.Timestamp:
        return self.epochs_utc[index]

    def invert_gap(self, gap: int) -> Inversion:
        """The Gauss inversion of the detrended element step across the gap into token ``gap``."""
        n = len(self.epochs_utc)
        width = min(_STEP_WINDOW, gap, n - gap)
        step = ElementStep(
            delta_a_km=self._step(self.a_km, gap, width),
            delta_eccentricity=self._step(self.e, gap, width),
            delta_inclination_rad=self._step(self.inc_rad, gap, width),
            delta_raan_rad=self._step(self.raan_rad, gap, width),
        )
        return invert(step, self._reference_orbit(gap))

    def _step(self, values: FloatArray, gap: int, width: int) -> float:
        """The detrended two-sided step across the gap, or a plain difference near a series end."""
        if width >= 2:
            return local_step(self.t_days, values.tolist(), gap, window=width)
        return float(values[gap] - values[gap - 1])

    def _reference_orbit(self, gap: int) -> Orbit:
        """The pre-gap reference orbit the inversion linearises about, from a local median."""
        lo = max(0, gap - _STEP_WINDOW)
        eccentricity = min(max(float(np.median(self.e[lo:gap])), 0.0), 0.999_999)
        inclination = min(
            max(
                math.atan2(
                    float(np.median(self.sin_i[lo:gap])), float(np.median(self.cos_i[lo:gap]))
                ),
                0.0,
            ),
            math.pi,
        )
        arg_perigee = math.atan2(float(np.median(self.k[lo:gap])), float(np.median(self.h[lo:gap])))
        return Orbit(
            semi_major_axis_km=float(np.median(self.a_km[lo:gap])),
            eccentricity=eccentricity,
            inclination_rad=inclination,
            arg_perigee_rad=arg_perigee,
        )
