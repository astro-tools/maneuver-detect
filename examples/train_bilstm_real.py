"""Train the BiLSTM baseline on the real, credentialed dataset and score it.

This is the offline run that produces the publishable checkpoint and its model-card numbers, end to
end and reproducibly:

1. reconstruct the committed recipe from Space-Track (credentialed; the elements are never
   written to the repo, per D2);
2. slice the labelled objects by the frozen leak-free temporal split;
3. train the BiLSTM through the shared Lightning harness on a single GPU (``accelerator="auto"``,
   within the V7 budget), selecting the checkpoint on the val-split recall under the chosen
   class-balance objective;
4. tune a **per-orbit-class** detection threshold on the **val** split through the benchmark and
   freeze the gates — plus the scalar fallback — with the weights and the train-split normaliser
   into the checkpoint bundle;
5. score the held-out **test** split through the benchmark — era-scoped labels, in-era detections,
   per-type floor — and print the recall / precision per class for the model card.

Unlike ``train_bilstm.py`` (a synthetic, offline demo), this needs Space-Track credentials and a
GPU. Set ``SPACETRACK_USERNAME`` / ``SPACETRACK_PASSWORD`` and run it from a CUDA box:

    export CUBLAS_WORKSPACE_CONFIG=:4096:8       # for deterministic CUDA matmul
    python examples/train_bilstm_real.py

Without credentials it prints how to set them and exits 0 (it never blocks on a prompt). The number
of epochs is the ``MANEUVER_DETECT_TRAIN_EPOCHS`` environment variable (default 150). The
class-balance of the checkpoint-selection and threshold-tuning objective is
``MANEUVER_DETECT_SELECTION_OBJECTIVE`` (``macro`` default — weight every orbit class equally,
lifting GEO without trading LEO/MEO away; ``pooled`` to optimise the pooled above-floor recall
instead). The output stays ASCII (the delta-v column prints as ``delta_v_estimate``).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pandas as pd

from maneuver_detect.benchmark import SplitName, TemporalSplit
from maneuver_detect.datasets.catalogue import DATASET_VERSION
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.schema import ManeuverType

_ROOT = Path(__file__).resolve().parents[1]
# Track the bundled catalogue version, so the committed labels/splits and the reconstruction recipe
# stay in lockstep (a catalogue bump repoints both without editing this driver).
_DATA = _ROOT / "dataset" / f"v{'.'.join(DATASET_VERSION.split('.')[:2])}"
_DEFAULT_EPOCHS = 150


def _load_labels_by_norad(path: Path) -> dict[int, list[ManeuverLabel]]:
    """Parse the dataset's ``labels.json`` into ManeuverLabels keyed by NORAD id."""
    by_norad: dict[int, list[ManeuverLabel]] = defaultdict(list)
    for record in json.loads(path.read_text()):
        norad_id = record["norad_id"]
        if norad_id is None:
            continue
        maneuver_type = record["maneuver_type"]
        by_norad[norad_id].append(
            ManeuverLabel(
                norad_id=norad_id,
                epoch=pd.Timestamp(record["epoch"]).to_pydatetime(),
                window_start=pd.Timestamp(record["window_start"]).to_pydatetime(),
                window_end=pd.Timestamp(record["window_end"]).to_pydatetime(),
                source=record["source"],
                source_ref=record["source_ref"],
                orbit_class=OrbitClass(record["orbit_class"]),
                maneuver_type=ManeuverType(maneuver_type) if maneuver_type else None,
                delta_v=None if record.get("delta_v") is None else float(record["delta_v"]),
            )
        )
    return by_norad


def _has_credentials() -> bool:
    return bool(os.environ.get("SPACETRACK_USERNAME") and os.environ.get("SPACETRACK_PASSWORD"))


def _selection_objective() -> str:
    """The class-balance objective from ``MANEUVER_DETECT_SELECTION_OBJECTIVE`` (default macro)."""
    value = os.environ.get("MANEUVER_DETECT_SELECTION_OBJECTIVE", "macro")
    if value not in ("pooled", "macro"):
        raise SystemExit(
            f"MANEUVER_DETECT_SELECTION_OBJECTIVE must be 'pooled' or 'macro', got {value!r}"
        )
    return value


def _guide_without_credentials() -> int:
    print("This run reconstructs the real dataset from Space-Track and needs credentials.")
    print("Set the two environment variables and re-run on a GPU box:")
    print("  export SPACETRACK_USERNAME='you@example.com'")
    print("  export SPACETRACK_PASSWORD='your-space-track-password'")
    print("The fetched elements are never written to the repo (reconstruct locally, per D2).")
    return 0


def main() -> int:
    if not _has_credentials():
        return _guide_without_credentials()

    # Imported here so the credentials-absent path above does not pull the modelling / data stack.
    from maneuver_detect.data.spacetrack import SpacetrackFetcher
    from maneuver_detect.datasets import recipe, reconstruct
    from maneuver_detect.detectors.bilstm import BiLstmDetector
    from maneuver_detect.models.checkpoint import save_bundle
    from maneuver_detect.models.datamodule import objects_from_labelled_dataset
    from maneuver_detect.models.evaluate import (
        fit_calibration_on_val,
        score_on_temporal_split,
        tune_thresholds_per_class_on_val,
    )
    from maneuver_detect.models.train import ValBenchmark, train_bilstm

    epochs = int(os.environ.get("MANEUVER_DETECT_TRAIN_EPOCHS", str(_DEFAULT_EPOCHS)))
    objective = _selection_objective()
    labels_by_norad = _load_labels_by_norad(_DATA / "labels.json")
    split = TemporalSplit.from_json((_DATA / "splits.json").read_text())

    print("Reconstructing the dataset from Space-Track (credentialed, rate-limited)...")
    dataset = reconstruct(recipe(), SpacetrackFetcher(), labels_by_norad)
    sliced = objects_from_labelled_dataset(dataset, split)
    series_by_norad = {obj.norad_id: obj.series for obj in dataset.objects}
    print(
        f"objects  train={len(sliced['train'])}  "
        f"val={len(sliced['val'])}  test={len(sliced['test'])}"
    )

    print(
        f"Training the BiLSTM for up to {epochs} epochs on a single GPU (objective={objective})..."
    )
    started = time.time()
    bundle = train_bilstm(
        sliced["train"],
        sliced["val"],
        max_epochs=epochs,
        seed=0,
        accelerator="auto",
        deterministic="warn",  # cuDNN LSTM has no deterministic backward; stay seed-level on GPU
        progress=True,  # a multi-minute interactive run, so show the bar + per-step loss
        # Select the checkpoint on the val-split benchmark recall (the metric we publish), not the
        # BCE val_loss surrogate (which bottoms out fast and undertrains the GEO signal).
        val_benchmark=ValBenchmark(
            series_by_norad=series_by_norad,
            labels=dataset.labels,
            split=split,
            objective=objective,
        ),
        metadata={"dataset_version": DATASET_VERSION},
    )
    gpu_hours = (time.time() - started) / 3600.0

    # Tune a per-orbit-class decision threshold on the val split (the weights are fixed; the
    # threshold is a selection), and freeze the per-class gates plus a scalar fallback into the
    # bundle before scoring test. GEO can take a lower gate than LEO/MEO.
    tuning = tune_thresholds_per_class_on_val(
        lambda t: BiLstmDetector(bundle, threshold=t),
        series_by_norad,
        dataset.labels,
        split,
        objective=objective,
    )
    bundle = replace(bundle, threshold=tuning.fallback, class_thresholds=tuning.thresholds)
    gates = ", ".join(f"{cls} {gate:g}" for cls, gate in sorted(tuning.thresholds.items()))
    print(
        f"tuned thresholds -> fallback {tuning.fallback:g} (val recall {tuning.recall:.2f})  "
        f"per-class: {gates or '(none)'}"
    )

    # Fit the confidence calibration on the val split (val only — no test-label leakage) and bake it
    # into the bundle, so the published detector emits calibrated confidence and the test report's
    # per-class operating point is read in calibrated units.
    calibration = fit_calibration_on_val(
        BiLstmDetector(bundle), series_by_norad, dataset.labels, split
    )
    bundle = replace(bundle, calibration=calibration)
    ece = ", ".join(f"{cls} {calibration.ece[cls]:.3f}" for cls in sorted(calibration.ece))
    print(f"calibrated confidence -> temperature {calibration.temperature:.3f}  per-class ECE: {ece}")

    # Score the held-out test split through the benchmark (the model-card / leaderboard numbers) and
    # record the full report into the checkpoint, so the generated model card carries the per-class
    # test metrics straight from the weights.
    report = score_on_temporal_split(
        BiLstmDetector(bundle), series_by_norad, dataset.labels, split, partition=SplitName.TEST
    )
    bundle = replace(
        bundle, metadata={**bundle.metadata, "test_report": json.loads(report.to_json())}
    )

    out = _ROOT / "bilstm-base.pt"
    save_bundle(bundle, out)
    print(f"checkpoint -> {out}  (trained in {gpu_hours:.2f} GPU-hours)")
    print("\nHeld-out test split (recall @ operating point, above-floor population):")
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
