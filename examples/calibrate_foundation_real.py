"""Calibrate + score a foundation forecast-residual bundle on the real, credentialed dataset.

This produces the publishable v0.3 foundation bundle and its held-out per-class numbers. It
reconstructs the committed dataset from Space-Track (credentialed; the elements are never written to
the repo, per D2), builds the forecaster for the chosen backend (zero-shot by default — CPU-only;
``--finetune`` adds a light Chronos fine-tune on a GPU), calibrates the residual-z operating point
on the val split, scores the result on the leak-free **test** split, records the per-class
recall/precision into the bundle, and writes it out. The saved bundle is then ready for
``maneuver-detect models publish <backend>-residual <out.pt>`` (its generated model card carries
those numbers) and to seed the leaderboard via ``leaderboard/build_fixture.py``.

It is a thin wrapper over the ``maneuver-detect models calibrate-foundation`` subcommand, which does
the work. Zero-shot needs only a Space-Track account — no GPU, no Hugging Face token:

    python examples/calibrate_foundation_real.py chronos ./chronos-residual.pt
    python examples/calibrate_foundation_real.py timesfm ./timesfm-residual.pt
    python examples/calibrate_foundation_real.py chronos ./chronos-residual.pt --finetune   # GPU

Without credentials it prints how to set them and exits 0 (it never blocks on a prompt). The output
stays ASCII (the delta-v column prints as ``delta_v_estimate``).
"""

from __future__ import annotations

import argparse

from maneuver_detect import cli


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate + score a foundation bundle on the real dataset (credentialed)."
    )
    parser.add_argument("backend", choices=("chronos", "timesfm"), help="the forecaster backend")
    parser.add_argument("out", help="path to write the calibrated, scored bundle (.pt)")
    parser.add_argument(
        "--revision", default="main", help="the pinned, Apache-2.0-confirmed checkpoint revision"
    )
    parser.add_argument(
        "--finetune", action="store_true", help="add a light Chronos fine-tune first (GPU)"
    )
    args = parser.parse_args()

    argv = ["models", "calibrate-foundation", args.backend, args.out, "--revision", args.revision]
    if args.finetune:
        argv.append("--finetune")
    return cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
