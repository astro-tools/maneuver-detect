"""Publish a trained checkpoint bundle and its generated model card to the Hugging Face Hub.

A checkpoint is GPU-trained offline (D12), so it is published from the training environment with
this module — ``maneuver-detect models publish <name> <bundle.pt>`` — not by CI, which never has the
weights. The **model card is generated from the bundle's own provenance**
(:attr:`~maneuver_detect.models.checkpoint.ModelBundle.metadata`), so the documented training data
and metrics can never drift from the weights they describe (D8). Publishing **refuses a checkpoint
with no recorded held-out test metrics** (which would render a blank-metrics card) unless it is
explicitly allowed, so the published card always documents measured performance. The Hub repo id,
the revision tag, and the auth path come from :mod:`maneuver_detect.hub`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from maneuver_detect import hub
from maneuver_detect.datasets.catalogue import DATASET_VERSION

if TYPE_CHECKING:
    from maneuver_detect.calibration import BundledCalibration
    from maneuver_detect.models.checkpoint import ModelBundle

__all__ = ["build_model_card", "publish_checkpoint"]


def publish_checkpoint(
    name: str,
    bundle_path: str | Path,
    *,
    token: str | None = None,
    version: str = DATASET_VERSION,
    allow_unscored: bool = False,
) -> str:
    """Upload the checkpoint bundle ``bundle_path`` and its generated model card to the Hub.

    ``name`` is the registered detector name (``"bilstm-base"`` / ``"transformer-base"``); it
    selects the target Hub model repo (:data:`maneuver_detect.hub.MODELS`). The bundle is loaded
    first (validating it and supplying the card's provenance), then the ``.pt`` and a ``README.md``
    model card are uploaded and the lockstep ``v{version}`` tag is moved onto them. ``token`` is the
    HF write token (falls back to ``$HF_TOKEN`` / a prior login). Returns the model repo id.

    Refuses a checkpoint with no recorded held-out test metrics — whose card would document no
    measured performance — unless ``allow_unscored`` is set; score it first (the credentialed
    ``score_checkpoint`` driver back-fills the metrics into the bundle) so the published card
    carries real numbers. Raises :class:`~maneuver_detect.hub.HubError` for an unknown ``name`` or
    an unscored bundle published without ``allow_unscored``.
    """
    from maneuver_detect.models.checkpoint import load_bundle

    if name not in hub.MODELS:
        raise hub.HubError(f"unknown model {name!r}; publishable models: {sorted(hub.MODELS)}")
    model = hub.MODELS[name]

    bundle = load_bundle(bundle_path)
    if not allow_unscored and not _has_test_metrics(bundle):
        raise hub.HubError(
            f"checkpoint {bundle_path} has no recorded held-out test metrics, so its model card "
            "would document no measured performance; score it first (score_checkpoint), or pass "
            "allow_unscored=True (--allow-unscored) to publish it anyway"
        )
    card = build_model_card(bundle, name, version=version)

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


def _has_test_metrics(bundle: ModelBundle) -> bool:
    """Whether the bundle carries a held-out test report with at least one per-class metric."""
    report = bundle.metadata.get("test_report")
    return (
        isinstance(report, dict)
        and isinstance(report.get("per_class"), dict)
        and bool(report["per_class"])
    )


def _format_value(value: Any) -> str:
    """Render a provenance value for a markdown table cell (compact floats, plain otherwise)."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _provenance_table(metadata: dict[str, Any]) -> str:
    """A markdown table of the bundle's scalar provenance, or a note when nothing was recorded."""
    if not metadata:
        return "_No provenance was recorded in the checkpoint._"
    rows = "\n".join(f"| `{key}` | {_format_value(metadata[key])} |" for key in sorted(metadata))
    return f"| field | value |\n|---|---|\n{rows}"


# Orbit classes ordered by altitude for the per-class table, with unknowns sorted last by name.
_CLASS_ORDER = {"LEO": 0, "MEO": 1, "GEO": 2, "IGSO": 3, "HEO": 4}


def _fmt_metric(value: Any) -> str:
    """A recall/precision cell — two decimals, or an em dash when the metric is undefined."""
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


def _type_accuracy(confusion: Any) -> str:
    """Diagonal / total of a ``{true: {pred: count}}`` confusion matrix, or ``—`` when empty."""
    if not isinstance(confusion, dict):
        return "—"
    correct = sum(int(row.get(true, 0)) for true, row in confusion.items() if isinstance(row, dict))
    total = sum(int(n) for row in confusion.values() if isinstance(row, dict) for n in row.values())
    return f"{correct / total:.2f}" if total else "—"


def _test_report_table(report: dict[str, Any]) -> str:
    """Render the held-out test ``ScoreReport`` (as stored in metadata) as a per-class table.

    Returns an empty string when the report carries no per-class metrics, so an evaluated and an
    un-evaluated checkpoint both produce a valid card.
    """
    per_class = report.get("per_class")
    if not isinstance(per_class, dict) or not per_class:
        return ""
    operating_point = report.get("operating_point")
    ci_level = report.get("ci_level")
    op = f"{operating_point:g}" if isinstance(operating_point, (int, float)) else "the operating"
    ci = f" ({ci_level:.0%} CI)" if isinstance(ci_level, (int, float)) else ""
    rows = [
        "| Class | Recall | Precision | Operating pt | Above-floor labels | Type acc |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(per_class, key=lambda k: (_CLASS_ORDER.get(k, 99), k)):
        metrics = per_class[key]
        rows.append(
            f"| {key} | {_fmt_metric(metrics.get('recall'))} "
            f"| {_fmt_metric(metrics.get('precision'))} "
            f"| {_fmt_metric(metrics.get('operating_point_confidence'))} "
            f"| {metrics.get('n_labels_above_floor', '—')} "
            f"| {_type_accuracy(metrics.get('confusion'))} |"
        )
    table = "\n".join(rows)
    return (
        f"Held-out **test split** — recall/precision at {op} false alarm(s)/satellite-year over "
        f"the above-floor population{ci}. Operating pt is the per-class confidence cut admitted "
        f"within that false-alarm budget (in the detector's calibrated confidence units). Type acc "
        f"is the share of above-floor true positives whose maneuver type is correct.\n\n{table}\n\n"
    )


def _calibration_block(calibration: BundledCalibration | None) -> str:
    """Render the bundle's baked-in confidence calibration (or empty when none was fit).

    Shared by both model cards: documents that the detector emits calibrated confidence (the fitted
    temperature and conformal coverage) and the per-orbit-class expected calibration error of that
    calibrated confidence, pointing at the benchmark docs for the reliability diagrams themselves.
    """
    if calibration is None:
        return ""
    coverage = 1.0 - calibration.conformal_alpha
    rows = ["| Class | ECE |", "|---|---|"]
    for key in sorted(calibration.ece, key=lambda k: (_CLASS_ORDER.get(k, 99), k)):
        rows.append(f"| {key} | {calibration.ece[key]:.3f} |")
    table = "\n".join(rows)
    return (
        "This detector emits **calibrated** confidence: the raw score is mapped through a "
        f"temperature (**T = {calibration.temperature:.3f}**) fit on the val split only, so a "
        "confidence of *p* means about a fraction *p* of detections at that confidence are real. A "
        f"split-conformal predictor (marginal coverage **{coverage:.0%}**) accompanies it for "
        "prediction sets. Per-orbit-class expected calibration error (ECE) of the calibrated "
        f"confidence:\n\n{table}\n\nThe per-class reliability diagrams and the calibrated per-class "
        "operating points are published in the benchmark documentation.\n\n"
    )


def _class_threshold_line(class_thresholds: dict[str, float]) -> str:
    """A one-line card summary of the per-class detection gates, or empty when none were tuned."""
    if not class_thresholds:
        return ""
    ordered = sorted(class_thresholds, key=lambda k: (_CLASS_ORDER.get(k, 99), k))
    gates = ", ".join(f"{key} {class_thresholds[key]:.3f}" for key in ordered)
    return (
        f"\n- **Per-class detection thresholds:** {gates} "
        "(a class without its own gate uses the default above)."
    )


def build_model_card(bundle: ModelBundle, name: str, *, version: str = DATASET_VERSION) -> str:
    """Generate the Hugging Face model card (``README.md``) for ``bundle`` from its provenance.

    The card carries HF frontmatter (license, library, tags, the linked dataset) and documents how
    to load the checkpoint, the architecture-agnostic inference, the training data and splits, the
    measured metrics, the intended use and its limitations, and the full provenance — all read off
    the bundle so the card cannot drift from the weights.
    """
    architecture = str(bundle.network_config.get("network", "sequence"))
    n_params = sum(int(tensor.numel()) for tensor in bundle.state_dict.values())
    metadata = dict(bundle.metadata)
    # The held-out test report renders as its own per-class table, so pull it out of the scalar
    # provenance table (a nested dict would render as one unreadable cell).
    test_report = metadata.pop("test_report", None)
    eval_block = _test_report_table(test_report) if isinstance(test_report, dict) else ""
    calibration_block = _calibration_block(bundle.calibration)
    class_threshold_line = _class_threshold_line(bundle.class_thresholds)
    dataset_version = str(metadata.get("dataset_version", version))
    recall = metadata.get("best_val_recall")
    recall_line = (
        f"Best validation-split above-floor recall during training: **{recall:.3f}**.\n\n"
        if isinstance(recall, float)
        else ""
    )

    return f"""---
license: mit
library_name: maneuver-detect
tags:
- orbital-mechanics
- maneuver-detection
- space-situational-awareness
- time-series
- satellite
datasets:
- {hub.DATASET_REPO}
metrics:
- recall
- precision
---

# {name}

A {architecture} maneuver detector for [`maneuver-detect`](https://github.com/astro-tools/maneuver-detect).
It localises maneuvers in a satellite's mean-element TLE history; the same vis-viva / Gauss physics
inversion the classical detector uses then recovers the Δv magnitude and maneuver type for each
detection (the model localises, the physics inverts).

## How to use

The checkpoint is fetched from this repo automatically on first use — CPU-only, cached on disk, no
weights at install time:

```python
from maneuver_detect import detect, datasets

history = datasets.tle_history(norad_id=25544, start="2024-01-01")
maneuvers = detect(history, model="{name}")
# DataFrame columns: epoch, confidence, type, delta_v_estimate, plus provenance
```

## Model description

- **Architecture:** {architecture} sequence network, {n_params:,} parameters.
- **Input:** the frozen irregular-sampling encoding (time-encoded element deltas, no interpolation);
  inputs are standardised with the **train-split** per-class statistics frozen into this checkpoint,
  so inference reproduces training-time standardisation.
- **Window / stride:** {bundle.window} / {bundle.stride}.
- **Default detection threshold:** {bundle.threshold:.3f}.{class_threshold_line}
- **Inference:** CPU-only; the bundle ships the network weights, the normaliser, and these
  parameters together, so loading it never needs the training stack.

## Training data

Trained on the `maneuver-detect` labelled dataset **v{dataset_version}**
([`{hub.DATASET_REPO}`](https://huggingface.co/datasets/{hub.DATASET_REPO})), versioned in lockstep
with this checkpoint. The dataset is distributed recipe-first (operator labels + a pinned
reconstruction recipe + a content-hash manifest; the raw Space-Track element history is never
redistributed) and partitioned by the frozen, leak-free temporal-holdout splits — novel satellites
scored in novel eras.

## Evaluation

{eval_block}{calibration_block}{recall_line}The benchmark scores precision/recall at a fixed
false-alarm rate per orbit class over the above-floor population, with per-class type confusion, via
the deterministic scorer.
Performance is sharply data-quality-stratified: well-tracked modern satellites reach
literature-level recall, while noisy historical series are bounded by the TLE detectability floor.

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
- **Parameters:** {n_params:,}

{_provenance_table(metadata)}

## License

Model weights: **MIT**. The dataset and authored artifacts are **CC-BY-4.0**; the raw Space-Track
element history is never redistributed. See the
[repository](https://github.com/astro-tools/maneuver-detect) for the full source terms, and
[`CITATION.cff`](https://github.com/astro-tools/maneuver-detect/blob/main/CITATION.cff) to cite.
"""
