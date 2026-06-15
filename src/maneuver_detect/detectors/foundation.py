"""Foundation-model forecast-residual detector — the Chronos v0.3 baseline.

The v0.3 baseline (decision **D14**) turns a pretrained time-series foundation model into a maneuver
detector by **forecast-residual thresholding**, the recipe V6 proved against the shipped scorer. A
foundation model replaces **one** component of the classical detector — the hand-built
quiet-dynamics
prior — with a learned conditional forecast; everything downstream is reused verbatim:

* **Forecast** each object's mean-element series (the two well-observed trigger channels the
  classical detector fires on: the semi-major axis ``a`` for in-track and the inclination for
  cross-track) one-step-ahead and rolling, getting a predicted next value per token.
* **Residual → score.** Standardize the forecast residual ``realised - forecast`` by a **robust
  per-object scale** (the centred MAD of the residuals), not the model's per-gap predictive interval
  — V6 measured the interval collapsing toward zero on confident gaps, which blows an ordinary gap
  up into a spurious spike, whereas the per-object MAD is stable (the calibrated interval is left to
  the uncertainty-calibration deliverable). The per-token score is the largest standardized residual
  across the trigger channels; a maneuver steps an element the quiet-trained forecaster cannot
  anticipate, so its residual spikes while quiet gaps stay near zero.
* **Per-class threshold + NMS.** Threshold the score **per orbit class** — the D4 detectability
floor
  expressed in standardized-residual units — and collapse each contiguous above-threshold run to its
  peak (the same :func:`~maneuver_detect.detectors.learned._detected_gaps` non-maximum suppression
  the
  v0.2 learned baselines use).
* **Emit canonical records.** Each surviving gap becomes a canonical
  :class:`~maneuver_detect.schema.Maneuver` via the unchanged D5 Gauss inversion
  (:class:`~maneuver_detect.detectors.learned._AlignedElements`): the model forecasts, the physics
  inverts. The confidence is read off the residual magnitude; the Δv is gated by the same per-type
  detectability floor the classical detector calibrates.

The foundation stack is the optional **`[foundation]`** extra (``chronos-forecasting``, Apache-2.0 —
D14.5). Its imports are deferred to construction, so importing the package, or using the classical /
v0.2 learned detectors, never pulls the foundation stack. The ``"chronos-residual"`` detector
registers on this recipe; the pipeline is forecaster-agnostic (a :class:`Forecaster` protocol), so a
further backend can drop in without touching it. The forecaster is resolved in this order: an
explicit :class:`Forecaster` (or :class:`~maneuver_detect.models.foundation.FoundationBundle`)
passed to the constructor, then the ``$…_CHECKPOINT`` environment variable, then — for the no-arg
construction the registry uses — the Hub-published bundle, fetched on first use.

(D14.4 also wired a TimesFM second entry, but its zero-shot forecast proved unreliable on real
noisy LEO element series — sustained forecast biases that no transient filter removes, where Chronos
stays robust — so the v0.3 baseline ships Chronos only; see the D14 ratification note.)
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from maneuver_detect.detectors.base import Detector
from maneuver_detect.detectors.classical import ClassicalDetector
from maneuver_detect.detectors.learned import _AlignedElements, _detected_gaps
from maneuver_detect.features.channels import build_channels
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import COLUMNS, Maneuver, empty_frame, to_frame

if TYPE_CHECKING:
    from maneuver_detect.features.channels import RawChannels
    from maneuver_detect.models.foundation import FoundationBundle

__all__ = [
    "CHRONOS_CHECKPOINT_ENV",
    "ChronosResidualDetector",
    "DriftContinuationForecaster",
    "Forecast",
    "Forecaster",
    "_ForecastResidualDetector",
]

FloatArray = npt.NDArray[np.float64]

_logger = logging.getLogger(__name__)

#: Environment variable naming a local foundation-bundle path the no-argument detector loads, so the
#: ``detect(history, model=<name>)`` dispatch path can use a calibrated bundle without the caller
#: threading one through. Unset, the no-argument detector falls back to the Hub-published bundle.
CHRONOS_CHECKPOINT_ENV = "MANEUVER_DETECT_CHRONOS_CHECKPOINT"

#: A floor on the per-object robust scale, in each channel's own unit (km for ``a``, rad for the
#: inclination), guarding the standardized residual against a degenerate near-constant quiet series
#: (whose MAD would be zero, making every gap a spurious infinite-z spike). Far below any real
#: channel's residual MAD, so it never bites a normally-noisy series.
_SCALE_FLOOR = {"a": 1e-6, "inc": 1e-9}

#: The default residual-z detection threshold a detector uses when neither a per-class calibration
#: nor an explicit override is supplied — a sane operating point in standardized-residual units (a
#: gap whose forecast residual is this many robust scales out is flagged). The calibrated per-class
#: thresholds in a published bundle replace it; the offline driver tunes it on the val split.
_DEFAULT_THRESHOLD = 4.0


class Forecast:
    """A forecaster's one-step-ahead prediction over a single channel series.

    ``mean[i]`` is the value predicted for token ``i`` from the tokens before it, and ``scale[i]``
    the
    predictive scale (a robust spread, e.g. half the central predictive interval) at that token.
    Token
    ``0`` has no history to forecast from, so its entries are not used. Both arrays are aligned to
    the
    channel series the forecaster was given, one entry per token.
    """

    __slots__ = ("mean", "scale")

    def __init__(self, mean: FloatArray, scale: FloatArray) -> None:
        self.mean = mean
        self.scale = scale


@runtime_checkable
class Forecaster(Protocol):
    """A one-step-ahead forecaster of a single mean-element channel series.

    The one slot the foundation model fills: given a channel's value series (e.g. the semi-major
    axis over the object's tokens), return the rolling one-step-ahead forecast and a predictive
    scale
    per token (:class:`Forecast`). The forecast-residual detector standardizes the realised series
    by
    this forecast and thresholds the residual; any model returning a value and a scale fits the slot
    (Chronos in the extra, a deterministic stand-in in the tests).
    """

    def forecast(self, series: FloatArray) -> Forecast:
        """Return the rolling one-step-ahead forecast and predictive scale for ``series``."""
        ...


class DriftContinuationForecaster:
    """A deterministic, dependency-free reference forecaster — robust drift continuation.

    The quiet-dynamics forecast is ``last value + a secular drift``, the drift the **median** of the
    series' inter-token first-differences and the predictive scale their robust **MAD**
    (``·1.4826``),
    both over the whole series. A handful of maneuver steps are outliers the median / MAD shrug off,
    so the scale is stable and a maneuver lands as one large standardized residual on its gap. This
    is
    the V6 spike's stand-in: it proves the residual → per-class-threshold → canonical-record
    contract
    without any model weights, and a foundation model only raises forecast quality on the same
    contract. Used as the detector's reference / test forecaster; the registered detector fills the
    slot with Chronos instead.
    """

    def __init__(self, *, scale_floor: float = 0.0) -> None:
        self._scale_floor = scale_floor

    def forecast(self, series: FloatArray) -> Forecast:
        n = series.shape[0]
        mean = np.full(n, np.nan, dtype=np.float64)
        scale = np.ones(n, dtype=np.float64)
        if n < 2:
            return Forecast(mean=mean, scale=scale)
        diffs = np.diff(series)
        drift = float(np.median(diffs))
        robust = 1.4826 * float(np.median(np.abs(diffs - drift)))
        spread = max(robust, self._scale_floor, 1e-12)
        mean[1:] = series[:-1] + drift
        scale[1:] = spread
        return Forecast(mean=mean, scale=scale)


class _CachingForecaster:
    """Memoise an inner forecaster by channel-series content — for the val threshold sweep.

    The val-split threshold tuner rebuilds the detector and re-runs detection once per candidate
    threshold, but the forecast is **threshold-independent** (the threshold only gates the residual
    afterwards), so a bare forecaster would re-forecast the whole val set N times. Wrapping it here
    forecasts each ``(object, channel)`` series exactly once and reuses it across the sweep — the
    forecast is the expensive part on CPU. Keyed by the series bytes (the same series is rebuilt per
    detect call but its content is identical). Each real forecast logs, so progress is observable.
    """

    def __init__(self, inner: Forecaster) -> None:
        self._inner = inner
        self._cache: dict[bytes, Forecast] = {}

    def forecast(self, series: FloatArray) -> Forecast:
        key = series.tobytes()
        hit = self._cache.get(key)
        if hit is None:
            _logger.info(
                "forecasting a %d-token channel (%d cached so far)",
                series.shape[0],
                len(self._cache),
            )
            hit = self._inner.forecast(series)
            self._cache[key] = hit
        return hit


class _ForecastResidualDetector(Detector):
    """Shared base for the forecast-residual detectors — the model forecasts, the physics inverts.

    Carries the whole forecaster-agnostic pipeline (forecast the trigger channels, standardize the
    residual, threshold per orbit class, non-maximum-suppress, emit the canonical schema via the D5
    inversion); a concrete detector only pins its registry :attr:`~Detector.name`, its
    checkpoint-path environment variable, and the foundation backend it expects. Construct with an
    explicit :class:`Forecaster` (with per-class thresholds), a
    :class:`~maneuver_detect.models.foundation.FoundationBundle` (or a path to one), or nothing —
    the
    no-argument construction the registry uses falls back to :attr:`checkpoint_env` and then the
    Hub-published bundle, fetched on first :meth:`detect`. ``threshold`` overrides the per-class
    operating point with a single residual-z cutoff applied to every class (what the offline
    val-split threshold tuner sweeps).
    """

    #: The foundation backend a bundle for this detector must carry (``"chronos"``).
    backend: ClassVar[str]
    #: The environment variable naming a local bundle path the no-argument detector loads.
    checkpoint_env: ClassVar[str]

    def __init__(
        self,
        bundle: FoundationBundle | str | Path | None = None,
        *,
        forecaster: Forecaster | None = None,
        class_thresholds: dict[str, float] | None = None,
        threshold: float | None = None,
    ) -> None:
        self._forecaster: Forecaster | None = forecaster
        self._class_thresholds: dict[str, float] = dict(class_thresholds or {})
        self._threshold_override = threshold
        self._context_length: int | None = None

        if bundle is None and forecaster is None:
            env_path = os.environ.get(self.checkpoint_env)
            bundle = env_path if env_path else None
        # No explicit forecaster, no bundle, no env path: fall back to the Hub-published bundle,
        # fetched lazily on the first detect() call (construction stays network-free — the order is
        # explicit forecaster → bundle → $…_CHECKPOINT → Hub).
        self._hub_pending = forecaster is None and bundle is None

        if bundle is not None:
            self._load(bundle)

    def _load(self, bundle: FoundationBundle | str | Path) -> None:
        """Build the forecaster and adopt the per-class thresholds from ``bundle`` (or its path)."""
        from maneuver_detect.models.foundation import FoundationBundle, load_foundation_bundle

        loaded = bundle if isinstance(bundle, FoundationBundle) else load_foundation_bundle(bundle)
        if loaded.backend != self.backend:
            raise ValueError(
                f"{type(self).__name__} expects a {self.backend!r} bundle, got {loaded.backend!r}"
            )
        self._forecaster = build_forecaster(loaded)
        self._context_length = loaded.context_length
        if not self._class_thresholds:
            self._class_thresholds = dict(loaded.class_thresholds)

    def _load_from_hub(self) -> None:
        """Fetch this detector's Hub-published bundle and load it (cached on disk)."""
        from maneuver_detect import hub

        path = hub.checkpoint_path(self.name)
        self._load(path)
        self._hub_pending = False

    @property
    def is_loaded(self) -> bool:
        """Whether a forecaster is resolved (``detect`` works only when it is)."""
        return self._forecaster is not None

    def _threshold_for(self, orbit_class: OrbitClass) -> float:
        """The residual-z threshold for ``orbit_class`` — the override, else the per-class value."""
        if self._threshold_override is not None:
            return self._threshold_override
        return self._class_thresholds.get(orbit_class.value, _DEFAULT_THRESHOLD)

    def detect(self, history: pd.DataFrame) -> pd.DataFrame:
        """Detect maneuvers in ``history`` and return the canonical maneuver DataFrame.

        ``history`` is a mean-element series; a frame with multiple objects is grouped by
        ``norad_id`` and each object detected independently, with rows returned sorted by
        ``(norad_id, epoch)``. The first call with no local forecaster fetches the detector's
        Hub-published bundle (cached on disk). Raises
        :class:`~maneuver_detect.hub.HubError` if that fetch fails and :class:`ValueError` if the
        detector has no forecaster and none can be resolved.
        """
        if self._forecaster is None and self._hub_pending:
            self._load_from_hub()
        if self._forecaster is None:
            raise ValueError(
                f"the {self.name!r} detector needs a calibrated bundle; construct "
                f"{type(self).__name__}(bundle=...), set ${self.checkpoint_env} to a local bundle, "
                "or publish one to the Hub"
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
        if channels.n_tokens < 2:
            return []
        elements = _AlignedElements.from_channels(channels)
        score = self._residual_score(channels, elements)
        threshold = self._threshold_for(channels.orbit_class)
        gaps = _detected_gaps(score, threshold)
        if not gaps:
            return []

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
                    confidence=_confidence(float(score[gap]), threshold),
                    type=inversion.maneuver_type,
                    delta_v_estimate=inversion.delta_v_estimate(floors[inversion.maneuver_type]),
                    norad_id=norad_id,
                    elset_epoch_before=before,
                    elset_epoch_after=after,
                )
            )
        return maneuvers

    def _residual_score(self, channels: RawChannels, elements: _AlignedElements) -> FloatArray:
        """Per-token score: the largest robustly-standardized forecast residual across the channels.

        Forecasts the semi-major axis (in-track) and the inclination (cross-track) — the two
        well-observed channels the classical detector triggers on — and standardizes the forecast
        residual by a **robust per-object scale** (the centred MAD of the residuals), *not* the
        model's per-gap predictive interval. V6 measured the reason: the interval collapses toward
        zero on confident gaps, which blows an ordinary gap's residual up into a spurious spike and
        buries the maneuver, whereas the per-object MAD is stable — a handful of maneuver residuals
        are outliers it shrugs off (the V6 stand-in's standardisation; the calibrated predictive
        interval is left to the uncertainty-calibration deliverable). Reduces to one score per token
        (the max across channels). Token ``0`` scores ``0`` (no gap precedes it); token ``i`` scores
        the inter-elset gap ``[i-1, i]``, the convention :func:`_detected_gaps` thresholds.
        """
        n = channels.n_tokens
        score = np.zeros(n, dtype=np.float64)
        for series, floor in (
            (elements.a_km, _SCALE_FLOOR["a"]),
            (elements.inc_rad, _SCALE_FLOOR["inc"]),
        ):
            forecast = self._forecaster.forecast(series)  # type: ignore[union-attr]
            residual = series - forecast.mean  # residual[0] is undefined (no forecast at token 0)
            body = residual[1:]
            # Centre and scale on the *finite* residuals only. A single non-finite forecast value
            # (a forecaster edge case, or a cleaned series with an interior gap) would otherwise
            # poison the per-object median/MAD into NaN, drive the scale to NaN, and zero this
            # channel's score for every token — silently suppressing real detections. Instead a
            # non-finite token simply scores 0 (it cannot trigger a detection) while the rest of
            # the channel scores normally.
            finite = np.isfinite(body)
            if not finite.any():
                continue
            center = float(np.median(body[finite]))
            mad = 1.4826 * float(np.median(np.abs(body[finite] - center)))
            scale = max(mad, floor)
            z = np.zeros(n, dtype=np.float64)
            z[1:][finite] = np.abs(body[finite] - center) / scale
            score = np.maximum(score, z)
        return score


def _confidence(z: float, threshold: float) -> float:
    """Map a standardized-residual peak to a calibrated detection confidence in ``[0, 1)``.

    ``1 - exp(-z / threshold)`` is monotonic in the residual magnitude, hits ``1 - 1/e ≈ 0.63`` at
    the
    threshold, and saturates towards ``1`` for a far-out spike — a bounded, threshold-relative
    confidence the canonical schema accepts and the leaderboard ranks by quantile (the V6 mapping).
    """
    return float(1.0 - math.exp(-z / threshold)) if threshold > 0.0 else 1.0


def build_forecaster(bundle: FoundationBundle) -> Forecaster:
    """Build the forecaster a :class:`FoundationBundle` describes (lazy; GPU when present).

    Dispatches on the bundle's ``backend`` tag — currently the Chronos backend, imported lazily from
    the optional ``[foundation]`` extra so the base install never pays for it. The dispatch stays
    open for additional backends; an unknown one raises :class:`ValueError`.
    """
    if bundle.backend == "chronos":
        return _build_chronos_forecaster(bundle)
    raise ValueError(f"unknown foundation backend {bundle.backend!r}; supported: 'chronos'")


def _rolling_contexts(series: FloatArray, context_length: int) -> list[FloatArray]:
    """The one-step-ahead context windows: for each token ``i ≥ 1``, the up-to-``context_length``
    preceding values the forecaster conditions on to predict token ``i``."""
    contexts: list[FloatArray] = []
    for i in range(1, series.shape[0]):
        lo = max(0, i - context_length)
        contexts.append(series[lo:i])
    return contexts


def _build_chronos_forecaster(bundle: FoundationBundle) -> Forecaster:  # pragma: no cover
    """Build the Chronos forecaster from a bundle (needs the ``[foundation]`` extra)."""
    from maneuver_detect.detectors._chronos import ChronosForecaster

    return ChronosForecaster(
        checkpoint_id=bundle.checkpoint_id,
        revision=bundle.revision,
        context_length=bundle.context_length,
        finetune_state=bundle.finetune_state,
    )


class ChronosResidualDetector(_ForecastResidualDetector):
    """Chronos forecast-residual detector — the v0.3 foundation baseline (D14.4).

    Chronos brings a broad pretrained prior over time-series shape and is robust on real, noisy
    element series. All the inference machinery is the shared forecaster-agnostic pipeline in
    :class:`_ForecastResidualDetector`; this class only pins the registry name, the checkpoint
    environment variable, and the ``"chronos"`` backend.
    """

    name = "chronos-residual"
    backend = "chronos"
    checkpoint_env: ClassVar[str] = CHRONOS_CHECKPOINT_ENV
