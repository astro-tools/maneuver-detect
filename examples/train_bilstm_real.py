"""Train the BiLSTM baseline on the real, credentialed v0.2 dataset and score it.

This is the offline run that produces the publishable checkpoint and its model-card numbers, end to
end and reproducibly:

1. reconstruct the committed v0.2 recipe from Space-Track (credentialed; the elements are never
   written to the repo, per D2);
2. slice the labelled objects by the frozen leak-free temporal split;
3. train the BiLSTM through the shared Lightning harness on a single GPU (``accelerator="auto"``,
   within the V7 budget);
4. save the checkpoint bundle (weights + the train-split normaliser + windowing/threshold);
5. score the held-out **test** split through the benchmark — era-scoped labels, in-era detections,
   per-type floor — and print the recall / precision per class for the model card.

Unlike ``train_bilstm.py`` (a synthetic, offline demo), this needs Space-Track credentials and a
GPU. Set ``SPACETRACK_USERNAME`` / ``SPACETRACK_PASSWORD`` and run it from a CUDA box:

    export CUBLAS_WORKSPACE_CONFIG=:4096:8       # for deterministic CUDA matmul
    python examples/train_bilstm_real.py

Without credentials it prints how to set them and exits 0 (it never blocks on a prompt). The number
of epochs is the ``MANEUVER_DETECT_TRAIN_EPOCHS`` environment variable (default 150). The output
stays ASCII (the delta-v column prints as ``delta_v_estimate``).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from maneuver_detect.benchmark import SplitName, TemporalSplit
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.schema import ManeuverType

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "dataset" / "v0.2"
_DEFAULT_EPOCHS = 150


def _load_labels_by_norad(path: Path) -> dict[int, list[ManeuverLabel]]:
    """Parse ``dataset/v0.2/labels.json`` into ManeuverLabels keyed by NORAD id."""
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


def _guide_without_credentials() -> int:
    print("This run reconstructs the real v0.2 dataset from Space-Track and needs credentials.")
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
    from maneuver_detect.models.evaluate import score_on_temporal_split
    from maneuver_detect.models.train import train_bilstm

    epochs = int(os.environ.get("MANEUVER_DETECT_TRAIN_EPOCHS", str(_DEFAULT_EPOCHS)))
    labels_by_norad = _load_labels_by_norad(_DATA / "labels.json")
    split = TemporalSplit.from_json((_DATA / "splits.json").read_text())

    print("Reconstructing the v0.2 dataset from Space-Track (credentialed, rate-limited)...")
    dataset = reconstruct(recipe(), SpacetrackFetcher(), labels_by_norad)
    sliced = objects_from_labelled_dataset(dataset, split)
    print(
        f"objects  train={len(sliced['train'])}  "
        f"val={len(sliced['val'])}  test={len(sliced['test'])}"
    )

    print(f"Training the BiLSTM for up to {epochs} epochs on a single GPU...")
    started = time.time()
    bundle = train_bilstm(
        sliced["train"],
        sliced["val"],
        max_epochs=epochs,
        seed=0,
        accelerator="auto",
        deterministic="warn",  # cuDNN LSTM has no deterministic backward; stay seed-level on GPU
        metadata={"dataset_version": "0.2.0"},
    )
    gpu_hours = (time.time() - started) / 3600.0

    out = _ROOT / "bilstm-base.pt"
    save_bundle(bundle, out)
    print(f"checkpoint -> {out}  (trained in {gpu_hours:.2f} GPU-hours)")

    # Score the held-out test split through the benchmark (the model-card / leaderboard numbers).
    series_by_norad = {obj.norad_id: obj.series for obj in dataset.objects}
    report = score_on_temporal_split(
        BiLstmDetector(bundle), series_by_norad, dataset.labels, split, partition=SplitName.TEST
    )
    print("\nHeld-out test split (recall @ operating point, above-floor population):")
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
