"""Build the leaderboard's private bundle — the held-out scoring fixture plus seed predictions.

The hosted Space scores submissions against the frozen test split with the shipped deterministic
scorer. To do that without a Space-Track reconstruction at request time it needs the *derived*
scoring inputs — the per-object floor that flags each label above- or below-detectability, the
inter-elset gap each label's matching window spans, and the era-only observation span. This script
reconstructs the committed v0.2 dataset from Space-Track (credentialed; the elements are never
written to the repo, per D2), builds those inputs through the same ``models.evaluate`` path the
local scorer uses, and writes:

    <out>/fixture.json            the held-out ScoringFixture (labels + exposure + timing floor)
    <out>/seeds/<name>.json       each baseline's test-split predictions, re-scored as a seed entry

The fixture's matching windows are real elset epochs (derived Space-Track data, D2), so this bundle
is **private deploy-time data**: upload ``<out>/`` to a private Hugging Face Dataset the Space
reads, do not commit it. The held-out labels it encodes are already public (the v0.2 answer key is
committed — the D12 amendment), so this leaks nothing new; the bundle is private only to honour D2.

CPU only, no GPU. Needs Space-Track credentials. Set SPACETRACK_USERNAME / SPACETRACK_PASSWORD:

    python leaderboard/build_fixture.py --out leaderboard-bundle

Without credentials it prints how to set them and exits 0 (it never blocks on a prompt). Output
stays ASCII.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from maneuver_detect.benchmark import predictions_to_json
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.leaderboard import ScoringFixture, fixture_to_json
from maneuver_detect.schema import ManeuverType

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "dataset" / "v0.2"

# The published D11 timing-only "cheating floor": the rank-AUC a Δt-only model reaches, the score a
# submission must beat to be doing more than reading gap lengths. A benchmark constant (V5/D11),
# shown with every score so a result is read in context — never derived from a submission.
_TIMING_FLOOR = {"LEO": 0.62, "GEO": 0.68}


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
    print("Building the leaderboard bundle reconstructs the real v0.2 dataset from Space-Track.")
    print("Set the two environment variables and re-run (CPU only, no GPU):")
    print("  export SPACETRACK_USERNAME='you@example.com'")
    print("  export SPACETRACK_PASSWORD='your-space-track-password'")
    print("The fetched elements are never written to the repo (reconstruct locally, per D2).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the leaderboard's private fixture + seed-prediction bundle."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "leaderboard-bundle",
        help="output bundle directory (upload to a private HF Dataset; do not commit)",
    )
    parser.add_argument(
        "--bilstm",
        type=Path,
        default=_ROOT / "bilstm-base.pt",
        help="BiLSTM checkpoint to seed the board with (skipped if absent)",
    )
    parser.add_argument(
        "--transformer",
        type=Path,
        default=_ROOT / "transformer-base.pt",
        help="transformer checkpoint to seed the board with (skipped if absent)",
    )
    args = parser.parse_args()

    if not _has_credentials():
        return _guide_without_credentials()

    # Imported here so the credentials-absent path above does not pull the modelling / data stack.
    from maneuver_detect.benchmark import SplitName, TemporalSplit
    from maneuver_detect.data.spacetrack import SpacetrackFetcher
    from maneuver_detect.datasets import recipe, reconstruct
    from maneuver_detect.detectors.bilstm import BiLstmDetector
    from maneuver_detect.detectors.classical import ClassicalDetector
    from maneuver_detect.detectors.transformer import TransformerDetector
    from maneuver_detect.models.checkpoint import load_bundle
    from maneuver_detect.models.evaluate import (
        detections_for_partition,
        scoring_inputs_for_partition,
    )

    labels_by_norad = _load_labels_by_norad(_DATA / "labels.json")
    splits_text = (_DATA / "splits.json").read_text()
    dataset_version = str(json.loads(splits_text)["dataset_version"])
    split = TemporalSplit.from_json(splits_text)

    print("Reconstructing the v0.2 dataset from Space-Track (credentialed, rate-limited)...")
    dataset = reconstruct(recipe(), SpacetrackFetcher(), labels_by_norad)
    series_by_norad = {obj.norad_id: obj.series for obj in dataset.objects}

    labels, exposure = scoring_inputs_for_partition(
        series_by_norad, dataset.labels, split, partition=SplitName.TEST
    )
    fixture = ScoringFixture(
        dataset_version=dataset_version,
        labels=tuple(labels),
        exposure=tuple(exposure),
        timing_floor=_TIMING_FLOOR,
    )

    out: Path = args.out
    (out / "seeds").mkdir(parents=True, exist_ok=True)
    (out / "fixture.json").write_text(fixture_to_json(fixture))
    print(f"wrote {out / 'fixture.json'} ({len(labels)} labels, {len(exposure)} objects)")

    seeds: list[tuple[str, object]] = [("classical", ClassicalDetector())]
    if args.bilstm.exists():
        seeds.append(("bilstm-base", BiLstmDetector(load_bundle(args.bilstm))))
    else:
        print(f"skipping bilstm seed: {args.bilstm} not found")
    if args.transformer.exists():
        seeds.append(("transformer-base", TransformerDetector(load_bundle(args.transformer))))
    else:
        print(f"skipping transformer seed: {args.transformer} not found")

    for name, detector in seeds:
        detections = detections_for_partition(
            detector,  # type: ignore[arg-type]
            series_by_norad,
            split,
            partition=SplitName.TEST,
        )
        (out / "seeds" / f"{name}.json").write_text(predictions_to_json(detections))
        print(f"wrote {out / 'seeds' / f'{name}.json'} ({len(detections)} detections)")

    print(f"\nBundle ready at {out}. Upload it to the Space's private HF Dataset (do not commit).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
