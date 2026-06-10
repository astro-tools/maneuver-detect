"""Command-line interface — the ``maneuver-detect detect`` one-shot detection subcommand.

``detect`` mirrors the public :func:`maneuver_detect.detect` API on either a NORAD catalogue id
(its TLE history fetched live) or a local TLE file, and prints the canonical maneuver DataFrame.
It is a thin shell over the API, so the CLI produces the same result as the equivalent API call.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from maneuver_detect import __version__

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run the requested command, returning a process exit code.

    The console-script wrapper passes the return value to :func:`sys.exit`, so ``0`` means
    success and a non-zero value a hard failure. Argument or usage errors exit ``2`` via
    :mod:`argparse`.
    """
    args = _build_parser().parse_args(argv)
    return _run_detect(target=args.target, model=args.model)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maneuver-detect",
        description="Detect orbital maneuvers from a satellite's public TLE history.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    detect_parser = subcommands.add_parser(
        "detect",
        help="detect maneuvers for a NORAD id or a local TLE file",
        description=(
            "Assemble the mean-element history for <target> (a NORAD catalogue id fetched "
            "live, or a path to a local TLE file), run the detector, and print the detected "
            "maneuvers."
        ),
    )
    detect_parser.add_argument("target", help="a NORAD catalogue id or a path to a TLE file")
    detect_parser.add_argument(
        "--model",
        default="classical",
        help="detector to run (default: the classical reference detector)",
    )
    return parser


def _run_detect(target: str, model: str) -> int:
    raise NotImplementedError("The detector layer is not implemented yet.")
