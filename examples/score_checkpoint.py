"""Re-score an existing checkpoint on the held-out test split and write its per-class metrics in.

This adds the model-card metrics to a checkpoint that was trained before they were recorded, without
a GPU retrain. It reconstructs the committed v0.2 dataset from Space-Track (credentialed; the
elements are never written to the repo, per D2), scores the bundle on the leak-free **test** split
through the benchmark, records the full report into the bundle as ``metadata["test_report"]``, and
re-saves it in place. The generated model card then carries the per-class test recall/precision.

CPU only, no GPU. Needs Space-Track credentials. Set SPACETRACK_USERNAME / SPACETRACK_PASSWORD:

    python examples/score_checkpoint.py bilstm-base ./bilstm-base.pt
    python examples/score_checkpoint.py transformer-base ./transformer-base.pt

Without credentials it prints how to set them and exits 0 (it never blocks on a prompt). The output
stays ASCII (the delta-v column prints as ``delta_v_estimate``).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pandas as pd

from maneuver_detect.benchmark import SplitName, TemporalSplit
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.schema import ManeuverType

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "dataset" / "v0.2"


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
    print(
        "This re-score reconstructs the real v0.2 dataset from Space-Track and needs credentials."
    )
    print("Set the two environment variables and re-run (CPU only, no GPU):")
    print("  export SPACETRACK_USERNAME='you@example.com'")
    print("  export SPACETRACK_PASSWORD='your-space-track-password'")
    print("The fetched elements are never written to the repo (reconstruct locally, per D2).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score a checkpoint on the test split and write its per-class metrics in."
    )
    parser.add_argument(
        "name",
        choices=("bilstm-base", "transformer-base"),
        help="registered detector name of the checkpoint",
    )
    parser.add_argument("bundle", help="path to the checkpoint bundle (.pt) to re-score in place")
    args = parser.parse_args()

    if not _has_credentials():
        return _guide_without_credentials()

    # Imported here so the credentials-absent path above does not pull the modelling / data stack.
    from maneuver_detect.data.spacetrack import SpacetrackFetcher
    from maneuver_detect.datasets import recipe, reconstruct
    from maneuver_detect.detectors.bilstm import BiLstmDetector
    from maneuver_detect.detectors.transformer import TransformerDetector
    from maneuver_detect.models.checkpoint import load_bundle, save_bundle
    from maneuver_detect.models.evaluate import score_on_temporal_split

    detectors = {"bilstm-base": BiLstmDetector, "transformer-base": TransformerDetector}

    labels_by_norad = _load_labels_by_norad(_DATA / "labels.json")
    split = TemporalSplit.from_json((_DATA / "splits.json").read_text())

    print("Reconstructing the v0.2 dataset from Space-Track (credentialed, rate-limited)...")
    dataset = reconstruct(recipe(), SpacetrackFetcher(), labels_by_norad)
    series_by_norad = {obj.norad_id: obj.series for obj in dataset.objects}

    bundle_path = Path(args.bundle)
    bundle = load_bundle(bundle_path)
    detector = detectors[args.name](bundle)
    report = score_on_temporal_split(
        detector, series_by_norad, dataset.labels, split, partition=SplitName.TEST
    )
    bundle = replace(
        bundle, metadata={**bundle.metadata, "test_report": json.loads(report.to_json())}
    )
    save_bundle(bundle, bundle_path)
    print(f"re-scored {args.name} -> {bundle_path}")
    print("\nHeld-out test split (recall @ operating point, above-floor population):")
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
