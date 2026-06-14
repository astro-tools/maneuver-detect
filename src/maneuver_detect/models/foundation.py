"""The foundation-model bundle and its offline driver — calibrate, score, fine-tune (D14).

A forecast-residual detector (:mod:`maneuver_detect.detectors.foundation`) is reproduced elsewhere
only if it loads the **same** pretrained forecaster, the **same** calibrated per-class residual-z
thresholds, and any **light fine-tune** it was scored with. A pinned checkpoint id alone is not
enough, so the :class:`FoundationBundle` ships all three as one file: the backend, the Hub
checkpoint id and its **pinned revision** (the exact Apache-2.0-confirmed revision, D14.2), the
rolling forecast context length, the per-class thresholds, an optional fine-tune ``state_dict``, and
provenance metadata. It is kept deliberately separate from the torch-network
:class:`~maneuver_detect.models.checkpoint.ModelBundle` (a foundation forecaster is a pretrained
model fetched by id, not a network this package builds), so the v0.2 baselines are untouched.

The driver is the **offline** half (D14.3 — zero-shot is inference-only; a light fine-tune fits the
V7 single-GPU envelope): :func:`zero_shot_bundle` assembles a no-fine-tune bundle,
:func:`calibrate_thresholds` tunes the residual-z operating point on the val split through the same
benchmark the leaderboard uses, :func:`score_bundle` scores a variant on the held-out test split and
records the metrics onto the bundle (the model card reads them), and :func:`finetune_chronos`
specializes the Chronos quiet-dynamics prior to the element-series domain. The heavy
``chronos`` / ``torch`` work is deferred so importing this module never pulls the foundation stack.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from maneuver_detect.datasets.catalogue import DATASET_VERSION
from maneuver_detect.errors import ManeuverDetectError
from maneuver_detect.models.evaluate import (
    ThresholdTuning,
    score_on_temporal_split,
    tune_threshold_on_val,
)

if TYPE_CHECKING:
    import pandas as pd

    from maneuver_detect.benchmark import ScoreReport, SplitName, TemporalSplit
    from maneuver_detect.detectors.foundation import Forecaster, _ForecastResidualDetector
    from maneuver_detect.labels.record import ManeuverLabel

__all__ = [
    "FOUNDATION_DEFAULTS",
    "FOUNDATION_THRESHOLD_SWEEP",
    "FoundationBundle",
    "FoundationDefault",
    "calibrate_and_score",
    "calibrate_thresholds",
    "finetune_chronos",
    "load_foundation_bundle",
    "save_foundation_bundle",
    "score_bundle",
    "zero_shot_bundle",
]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoundationDefault:
    """The default Hub checkpoint a backend leads with, sized for the V7 single-GPU budget."""

    checkpoint_id: str
    context_length: int


#: Per-backend default checkpoints (Apache-2.0, D14.2/D14.4). Chronos-Bolt is the CPU-fast lead and
#: the v0.3 baseline backend. The *revision* is pinned per build at ingest (a model card can change
#: between revisions — D14.2), so it is supplied to :func:`zero_shot_bundle` rather than baked here;
#: ``"main"`` is only the unpinned fallback the offline ingest replaces.
FOUNDATION_DEFAULTS: dict[str, FoundationDefault] = {
    "chronos": FoundationDefault(checkpoint_id="amazon/chronos-bolt-small", context_length=64),
}

#: The residual-z detection thresholds :func:`calibrate_thresholds` sweeps — standardized-residual
#: units (how many robust scales out a gap must be to fire), not the ``[0, 1]`` probabilities the
#: learned baselines tune. Spans the V3/D4 per-class detectability band, low enough (down to 1.0)
#: that the tuner is never floored by the sweep — the cached forecast makes extra candidates free.
FOUNDATION_THRESHOLD_SWEEP: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)

# Everything the detector needs to reproduce a foundation variant; a bundle missing any is
# malformed.
_REQUIRED_KEYS = (
    "backend",
    "checkpoint_id",
    "revision",
    "context_length",
    "class_thresholds",
)


@dataclass(frozen=True)
class FoundationBundle:
    """A calibrated foundation forecast-residual detector — everything inference needs.

    Attributes:
        backend: The forecaster family — ``"chronos"`` (the v0.3 baseline backend).
        checkpoint_id: The Hub checkpoint the forecaster loads (e.g. ``amazon/chronos-bolt-small``).
        revision: The **pinned** checkpoint revision, confirmed Apache-2.0 at ingest (D14.2).
        context_length: The rolling one-step-ahead forecast context — the tokens of history the
            forecaster conditions on per step.
        class_thresholds: Per-orbit-class residual-z detection threshold (``OrbitClass`` value →
            cutoff) — the calibrated operating point, the D4 floor in standardized-residual units.
        finetune_state: An optional light fine-tune ``state_dict`` loaded onto the forecaster's
            model; ``None`` for the zero-shot variant (inference-only, no weights — D14.3).
        metadata: Free-form provenance (dataset version, the held-out ``test_report``, the measured
            single-GPU cost) the model card is generated from, so the card cannot drift from the
            weights.
    """

    backend: str
    checkpoint_id: str
    revision: str
    context_length: int
    class_thresholds: dict[str, float]
    finetune_state: dict[str, torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def save_foundation_bundle(bundle: FoundationBundle, path: str | Path) -> None:
    """Serialise ``bundle`` to ``path`` with :func:`torch.save` (the fine-tune carries tensors)."""
    payload = {
        "backend": bundle.backend,
        "checkpoint_id": bundle.checkpoint_id,
        "revision": bundle.revision,
        "context_length": bundle.context_length,
        "class_thresholds": bundle.class_thresholds,
        "finetune_state": bundle.finetune_state,
        "metadata": bundle.metadata,
    }
    torch.save(payload, Path(path))


def load_foundation_bundle(path: str | Path, *, map_location: str = "cpu") -> FoundationBundle:
    """Load a :class:`FoundationBundle` saved by :func:`save_foundation_bundle` (CPU by default).

    Raises :class:`~maneuver_detect.errors.ManeuverDetectError` if the file is not a bundle dict or
    is missing any key inference needs, so a truncated or version-mismatched Hub artifact surfaces
    as a clear error naming the path and the missing fields, not a bare ``KeyError``.
    """
    payload: dict[str, Any] = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ManeuverDetectError(
            f"foundation bundle at {Path(path)} is not a bundle "
            f"(expected a dict, got {type(payload).__name__})"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise ManeuverDetectError(
            f"foundation bundle at {Path(path)} is missing required keys: {missing}"
        )
    return FoundationBundle(
        backend=str(payload["backend"]),
        checkpoint_id=str(payload["checkpoint_id"]),
        revision=str(payload["revision"]),
        context_length=int(payload["context_length"]),
        class_thresholds=dict(payload["class_thresholds"]),
        finetune_state=payload.get("finetune_state"),
        metadata=payload.get("metadata", {}),
    )


def zero_shot_bundle(
    backend: str,
    *,
    revision: str = "main",
    checkpoint_id: str | None = None,
    context_length: int | None = None,
    class_thresholds: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FoundationBundle:
    """Assemble a **zero-shot** foundation bundle (no fine-tune — inference only, D14.3).

    Defaults the checkpoint id and context length from :data:`FOUNDATION_DEFAULTS` for ``backend``.
    ``revision`` should be the exact Apache-2.0-confirmed checkpoint revision pinned at ingest
    (D14.2); ``class_thresholds`` is the calibrated operating point (see
    :func:`calibrate_thresholds`)
    or empty to start. Raises :class:`ValueError` for an unknown backend.
    """
    if backend not in FOUNDATION_DEFAULTS:
        raise ValueError(
            f"unknown foundation backend {backend!r}; supported: {sorted(FOUNDATION_DEFAULTS)}"
        )
    default = FOUNDATION_DEFAULTS[backend]
    base_metadata: dict[str, Any] = {"dataset_version": DATASET_VERSION, "mode": "zero-shot"}
    base_metadata.update(metadata or {})
    return FoundationBundle(
        backend=backend,
        checkpoint_id=checkpoint_id if checkpoint_id is not None else default.checkpoint_id,
        revision=revision,
        context_length=context_length if context_length is not None else default.context_length,
        class_thresholds=dict(class_thresholds or {}),
        finetune_state=None,
        metadata=base_metadata,
    )


def calibrate_thresholds(
    bundle: FoundationBundle,
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    forecaster: Forecaster | None = None,
    candidates: Sequence[float] = FOUNDATION_THRESHOLD_SWEEP,
    partition: SplitName | None = None,
) -> tuple[FoundationBundle, ThresholdTuning]:
    """Tune the residual-z operating point on the val split and return the recalibrated bundle.

    Sweeps ``candidates`` through the same benchmark the leaderboard uses — era-scoped labels,
    in-era
    detections, per-type floor, recall at the fixed false-alarm budget — via
    :func:`~maneuver_detect.models.evaluate.tune_threshold_on_val`, picking the threshold with the
    best pooled above-floor recall. The chosen cutoff is frozen into every class of the returned
    bundle's :attr:`~FoundationBundle.class_thresholds` (per-class refinement is a later
    deliverable); the recall and the full sweep ride along on the metadata for provenance.
    """
    from maneuver_detect.benchmark import SplitName
    from maneuver_detect.detectors.foundation import _CachingForecaster

    resolved = forecaster if forecaster is not None else build_forecaster_for(bundle)
    # The sweep re-runs detection per candidate threshold, but the forecast is
    # threshold-independent, so cache it once across all candidates (the forecast is the slow part).
    cached = _CachingForecaster(resolved)

    def make_detector(threshold: float) -> _ForecastResidualDetector:
        return _bundle_detector(bundle, cached, threshold=threshold)

    tuning = tune_threshold_on_val(
        make_detector,
        series_by_norad,
        labels,
        split,
        candidates=candidates,
        partition=partition if partition is not None else SplitName.VAL,
    )
    from maneuver_detect.labels.record import OrbitClass

    thresholds = {orbit_class.value: tuning.threshold for orbit_class in OrbitClass}
    metadata = dict(bundle.metadata)
    metadata["calibration"] = {"recall": tuning.recall, "by_threshold": dict(tuning.by_threshold)}
    return replace(bundle, class_thresholds=thresholds, metadata=metadata), tuning


def score_bundle(
    bundle: FoundationBundle,
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    forecaster: Forecaster | None = None,
    partition: SplitName | None = None,
) -> tuple[FoundationBundle, ScoreReport]:
    """Score the bundle's detector on the held-out test split and record the metrics on the bundle.

    Runs :func:`~maneuver_detect.models.evaluate.score_on_temporal_split` (the detector-agnostic,
    leak-free path the v0.2 model cards came from) and stores the report under
    ``metadata["test_report"]`` so the foundation model card documents measured, not asserted,
    performance. Returns the metrics-bearing bundle and the report.
    """
    from maneuver_detect.benchmark import SplitName

    resolved = forecaster if forecaster is not None else build_forecaster_for(bundle)
    detector = _bundle_detector(bundle, resolved, threshold=None)
    report = score_on_temporal_split(
        detector,
        series_by_norad,
        labels,
        split,
        partition=partition if partition is not None else SplitName.TEST,
    )
    metadata = dict(bundle.metadata)
    metadata["test_report"] = json.loads(report.to_json())
    return replace(bundle, metadata=metadata), report


def calibrate_and_score(
    backend: str,
    series_by_norad: Mapping[int, pd.DataFrame],
    labels: Sequence[ManeuverLabel],
    split: TemporalSplit,
    *,
    revision: str = "main",
    checkpoint_id: str | None = None,
    context_length: int | None = None,
    finetune: bool = False,
    finetune_steps: int = 200,
    candidates: Sequence[float] = FOUNDATION_THRESHOLD_SWEEP,
    forecaster: Forecaster | None = None,
) -> tuple[FoundationBundle, ScoreReport]:
    """Assemble, optionally fine-tune, calibrate on val, and score on test — one offline run.

    The end-to-end offline driver behind ``maneuver-detect models calibrate-foundation`` and
    ``examples/calibrate_foundation_real.py``: build the zero-shot bundle, optionally apply a light
    Chronos fine-tune (``finetune=True``, GPU), tune the residual-z operating point on the **val**
    split, then score the result on the held-out **test** split and record the per-class metrics
    onto the bundle's model-card provenance. Returns the metrics-bearing bundle and the test report.
    Pass ``forecaster`` to reuse one pre-built (or stand-in) model across the run — for the tests
    to exercise the orchestration without the ``[foundation]`` extra; otherwise the forecaster is
    built from the bundle (and rebuilt after a fine-tune so the scored model is the fine-tuned one).
    """
    bundle = zero_shot_bundle(
        backend, revision=revision, checkpoint_id=checkpoint_id, context_length=context_length
    )
    if finetune:
        _logger.info("fine-tuning the %s forecaster (%d steps)...", backend, finetune_steps)
        bundle = finetune_chronos(bundle, series_by_norad, max_steps=finetune_steps)
    resolved = forecaster if forecaster is not None else build_forecaster_for(bundle)
    _logger.info(
        "calibrating the residual-z threshold on the val split (%d candidates)...",
        len(tuple(candidates)),
    )
    bundle, tuning = calibrate_thresholds(
        bundle, series_by_norad, labels, split, forecaster=resolved, candidates=candidates
    )
    _logger.info(
        "calibrated threshold %.2f (val recall %.3f); scoring on the test split...",
        tuning.threshold,
        tuning.recall,
    )
    return score_bundle(bundle, series_by_norad, labels, split, forecaster=resolved)


def build_forecaster_for(bundle: FoundationBundle) -> Forecaster:
    """Build the (lazy; GPU when present) forecaster for ``bundle`` — a thin driver re-export."""
    from maneuver_detect.detectors.foundation import build_forecaster

    return build_forecaster(bundle)


def _bundle_detector(
    bundle: FoundationBundle, forecaster: Forecaster, *, threshold: float | None
) -> _ForecastResidualDetector:
    from maneuver_detect.detectors.foundation import ChronosResidualDetector

    detector_cls = {"chronos": ChronosResidualDetector}[bundle.backend]
    return detector_cls(
        forecaster=forecaster, class_thresholds=bundle.class_thresholds, threshold=threshold
    )


def finetune_chronos(
    bundle: FoundationBundle,
    series_by_norad: Mapping[int, pd.DataFrame],
    *,
    max_steps: int = 200,
    learning_rate: float = 1e-4,
    seed: int = 0,
) -> FoundationBundle:  # pragma: no cover
    """Light fine-tune of the Chronos quiet-dynamics prior on the element series (offline, GPU).

    Specializes the pretrained forecaster to the satellite-element domain with a short AdamW pass
    over the objects' element series — the *optional polish* on top of the zero-shot baseline
    (D14.3), sized to the V7 single-GPU envelope. Needs the ``[foundation]`` extra (and,
    realistically, a GPU); delegated to :mod:`maneuver_detect.detectors._chronos`, which owns the
    Chronos training surface. Returns a new bundle carrying the fine-tune state and the recorded
    fine-tune mode.
    """
    if bundle.backend != "chronos":
        raise ValueError(f"finetune_chronos needs a 'chronos' bundle, got {bundle.backend!r}")
    from maneuver_detect.detectors._chronos import finetune_chronos_model

    state, cost = finetune_chronos_model(
        checkpoint_id=bundle.checkpoint_id,
        revision=bundle.revision,
        context_length=bundle.context_length,
        series=[series for series in series_by_norad.values()],
        max_steps=max_steps,
        learning_rate=learning_rate,
        seed=seed,
    )
    metadata = dict(bundle.metadata)
    metadata["mode"] = "fine-tuned"
    metadata["finetune_cost"] = cost
    return replace(bundle, finetune_state=state, metadata=metadata)
