"""Publish a foundation forecast-residual bundle and its generated model card to the Hub (D14).

The v0.3 foundation baselines are calibrated offline (a zero-shot threshold calibration, optionally
a
light fine-tune — D14.3) and published from that environment, not by CI. This module is their
publishing path, mirroring :mod:`maneuver_detect.models.publish` for the torch baselines: the
**model
card is generated from the bundle's own provenance**
(:attr:`~maneuver_detect.models.foundation.FoundationBundle.metadata`), so the documented
checkpoint,
licence, and metrics cannot drift from the calibrated thresholds they describe (D8). Publishing
**refuses a bundle with no recorded held-out test metrics** (a blank-metrics card) unless explicitly
allowed. The Hub repo id, revision tag, and auth path come from :mod:`maneuver_detect.hub`; a
foundation bundle is a :class:`FoundationBundle`, not the torch
:class:`~maneuver_detect.models.checkpoint.ModelBundle`, so it has its own loader and card.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from maneuver_detect import hub
from maneuver_detect.datasets.catalogue import DATASET_VERSION
from maneuver_detect.models.publish import _provenance_table, _test_report_table

if TYPE_CHECKING:
    from maneuver_detect.models.foundation import FoundationBundle

__all__ = ["FOUNDATION_MODEL_NAMES", "build_foundation_card", "publish_foundation"]

#: The registered foundation detector names this publisher handles (the ``*-residual`` baseline).
#: The CLI routes a ``models publish`` for one of these here rather than to the torch publisher.
FOUNDATION_MODEL_NAMES = frozenset({"chronos-residual"})


def publish_foundation(
    name: str,
    bundle_path: str | Path,
    *,
    token: str | None = None,
    version: str = DATASET_VERSION,
    allow_unscored: bool = False,
) -> str:
    """Upload the foundation bundle ``bundle_path`` and its generated model card to the Hub.

    ``name`` is the registered detector name (``"chronos-residual"``); it
    selects the target Hub model repo (:data:`maneuver_detect.hub.MODELS`). The bundle is loaded
    first (validating it and supplying the card's provenance), then the ``.pt`` and a ``README.md``
    model card are uploaded and the lockstep ``v{version}`` tag is moved onto them. ``token`` is the
    HF write token (falls back to ``$HF_TOKEN`` / a prior login). Returns the model repo id.

    Refuses a bundle with no recorded held-out test metrics — whose card would document no measured
    performance — unless ``allow_unscored`` is set; score it first (``score_bundle`` back-fills the
    metrics) so the published card carries real numbers. Raises
    :class:`~maneuver_detect.hub.HubError` for an unknown ``name`` or an unscored bundle published
    without ``allow_unscored``.
    """
    from maneuver_detect.models.foundation import load_foundation_bundle

    if name not in hub.MODELS:
        raise hub.HubError(f"unknown model {name!r}; publishable models: {sorted(hub.MODELS)}")
    model = hub.MODELS[name]

    bundle = load_foundation_bundle(bundle_path)
    if not allow_unscored and not _has_test_metrics(bundle):
        raise hub.HubError(
            f"foundation bundle {bundle_path} has no recorded held-out test metrics, so its model "
            "card would document no measured performance; score it first (score_bundle), or pass "
            "allow_unscored=True (--allow-unscored) to publish it anyway"
        )
    card = build_foundation_card(bundle, name, version=version)

    api = hub.hf_api(token)
    api.create_repo(model.repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(bundle_path),
        path_in_repo=model.filename,
        repo_id=model.repo_id,
        repo_type="model",
        commit_message=f"Publish {name} v{version}",
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=model.repo_id,
        repo_type="model",
        commit_message=f"Model card for {name} v{version}",
    )
    hub.move_tag(api, model.repo_id, repo_type="model", version=version)
    return model.repo_id


def _has_test_metrics(bundle: FoundationBundle) -> bool:
    """Whether the bundle carries a held-out test report with at least one per-class metric."""
    report = bundle.metadata.get("test_report")
    return (
        isinstance(report, dict)
        and isinstance(report.get("per_class"), dict)
        and bool(report["per_class"])
    )


def _thresholds_table(class_thresholds: dict[str, float]) -> str:
    """The calibrated per-class residual-z operating point as a markdown table (or a note)."""
    if not class_thresholds:
        return "_No per-class thresholds were calibrated into this bundle._"
    order = {"LEO": 0, "MEO": 1, "GEO": 2, "IGSO": 3, "HEO": 4}
    rows = "\n".join(
        f"| {key} | {class_thresholds[key]:.2f} |"
        for key in sorted(class_thresholds, key=lambda k: (order.get(k, 99), k))
    )
    return f"| Class | Residual-z threshold |\n|---|---|\n{rows}"


def build_foundation_card(
    bundle: FoundationBundle, name: str, *, version: str = DATASET_VERSION
) -> str:
    """Generate the Hugging Face model card (``README.md``) for a foundation ``bundle``.

    The card carries HF frontmatter (license, library, tags, the linked dataset), documents how to
    load and run the detector, the forecast-residual recipe and the **pinned** Apache-2.0 forecaster
    it builds on, the calibrated per-class operating point, the measured metrics, the intended use
    and
    its limits, and the full provenance — all read off the bundle so the card cannot drift.
    """
    metadata = dict(bundle.metadata)
    test_report = metadata.pop("test_report", None)
    eval_block = _test_report_table(test_report) if isinstance(test_report, dict) else ""
    dataset_version = str(metadata.get("dataset_version", version))
    mode = str(metadata.get("mode", "zero-shot"))
    finetuned = "a light fine-tune of" if bundle.finetune_state is not None else "the pretrained"

    return f"""---
license: mit
library_name: maneuver-detect
tags:
- orbital-mechanics
- maneuver-detection
- space-situational-awareness
- time-series
- satellite
- foundation-model
datasets:
- {hub.DATASET_REPO}
metrics:
- recall
- precision
---

# {name}

A foundation-model **forecast-residual** maneuver detector for
[`maneuver-detect`](https://github.com/astro-tools/maneuver-detect). It forecasts a satellite's
mean-element series with {finetuned} `{bundle.checkpoint_id}` (the `{bundle.backend}` backend),
standardises the forecast residual, and flags the inter-elset gaps where the realised series departs
from the forecast beyond a per-orbit-class threshold; the same vis-viva / Gauss physics inversion
the
classical detector uses then recovers the Δv magnitude and maneuver type for each detection (the
model forecasts, the physics inverts).

## How to use

The bundle is fetched from this repo automatically on first use — cached on disk, no weights at
install time; inference is CPU-capable and uses a GPU when one is present. The forecaster needs the
optional `[foundation]` extra:

```python
# pip install "maneuver-detect[foundation]"
from maneuver_detect import detect, datasets

history = datasets.tle_history(norad_id=25544, start="2024-01-01")
maneuvers = detect(history, model="{name}")
# DataFrame columns: epoch, confidence, type, delta_v_estimate, plus provenance
```

## Model description

- **Recipe:** forecast-residual thresholding — a pretrained time-series foundation model replaces
the
  classical detector's hand-built quiet-dynamics prior with a learned conditional forecast; the
  standardised residual is thresholded per orbit class (the detectability floor in residual units),
  non-maximum-suppressed, and inverted for Δv/type.
- **Forecaster:** `{bundle.checkpoint_id}` (`{bundle.backend}`), revision `{bundle.revision}`,
  rolling one-step context {bundle.context_length}. Mode: **{mode}**.
- **Licence:** the forecaster checkpoint is **Apache-2.0**, confirmed at the pinned revision; a
  fine-tune inherits that licence.
- **Inference:** CPU-capable; the forecaster is fetched from the Hub at runtime, not vendored.

### Calibrated operating point

Per-orbit-class detection threshold in standardized-residual units:

{_thresholds_table(bundle.class_thresholds)}

## Training data

Calibrated and scored on the `maneuver-detect` labelled dataset **v{dataset_version}**
([`{hub.DATASET_REPO}`](https://huggingface.co/datasets/{hub.DATASET_REPO})), versioned in lockstep
with this bundle. The dataset is distributed recipe-first (operator labels + a pinned reconstruction
recipe + a content-hash manifest; the raw Space-Track element history is never redistributed) and
partitioned by the frozen, leak-free temporal-holdout splits — novel satellites scored in novel
eras.
Zero-shot uses no training data beyond the forecaster's own pretraining; a fine-tuned variant
specialises the quiet-dynamics prior on the training split only.

## Evaluation

{eval_block}The benchmark scores precision/recall at a fixed false-alarm rate per orbit class over
the above-floor population, with per-class type confusion, via the deterministic scorer. Performance
is sharply data-quality-stratified: well-tracked modern satellites reach literature-level recall,
while noisy historical series are bounded by the TLE detectability floor.

## Intended use and limitations

- **Use:** post-hoc detection of orbital maneuvers from public TLE history for space-situational-
  awareness research and as a reproducible benchmark baseline.
- **Not** a maneuver *predictor* (it detects maneuvers that already happened), not real-time, and
  not an orbit-determination engine.
- **Detectability floor:** maneuvers below the per-object TLE detectability floor are not reported;
  recall on noisy historical series is fundamentally limited by TLE data quality, not the model.
- MEO/GEO labels are epoch-only (no Δv), so the Δv estimate is most meaningful on the Δv-labelled
  LEO core.

## Provenance

- **Dataset version (lockstep):** v{dataset_version}
- **Forecaster:** {bundle.checkpoint_id} @ {bundle.revision}

{_provenance_table(metadata)}

## License

Detector artifacts (thresholds, fine-tune): **MIT**. The forecaster checkpoint: **Apache-2.0** (its
own terms). The dataset and authored artifacts are **CC-BY-4.0**; the raw Space-Track element
history
is never redistributed. See the
[repository](https://github.com/astro-tools/maneuver-detect) for the full source terms, and
[`CITATION.cff`](https://github.com/astro-tools/maneuver-detect/blob/main/CITATION.cff) to cite.
"""
