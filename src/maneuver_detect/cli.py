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
    if args.command == "dataset":
        return _run_dataset_build(
            out_dir=args.out,
            nanu_start_year=args.nanu_start_year,
            nanu_end_year=args.nanu_end_year,
        )
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

    dataset_parser = subcommands.add_parser(
        "dataset",
        help="build the reconstructable dataset artifacts",
        description="Reconstruct the v0.1 labelled dataset from the pinned recipe.",
    )
    dataset_actions = dataset_parser.add_subparsers(
        dest="dataset_command", required=True, metavar="<action>"
    )
    build_parser = dataset_actions.add_parser(
        "build",
        help="reconstruct the dataset and write recipe.json, labels.json, and manifest.json",
        description=(
            "Fetch each catalogue object's series from Space-Track (credentials via the "
            "SPACETRACK_USERNAME / SPACETRACK_PASSWORD environment variables) and the open "
            "maneuver-label files, reconstruct the labelled dataset, and write the recipe, labels, "
            "and content-hash manifest. The raw series is never written."
        ),
    )
    build_parser.add_argument(
        "--out",
        required=True,
        help="output directory for recipe.json, labels.json, and manifest.json",
    )
    build_parser.add_argument(
        "--nanu-start-year",
        type=int,
        default=2016,
        help="first year of the CelesTrak NANU archive to crawl for GPS labels (default: 2016)",
    )
    build_parser.add_argument(
        "--nanu-end-year",
        type=int,
        default=None,
        help="last NANU archive year to crawl (default: the current year)",
    )
    return parser


def _run_detect(target: str, model: str) -> int:
    raise NotImplementedError("The detector layer is not implemented yet.")


def _run_dataset_build(out_dir: str, nanu_start_year: int, nanu_end_year: int | None) -> int:
    """Reconstruct the v0.1 dataset and write the recipe / labels / manifest artifacts."""
    import logging
    import sys
    from datetime import datetime, timezone

    import httpx

    from maneuver_detect.data.ratelimit import RateLimiter
    from maneuver_detect.data.spacetrack import SpacetrackFetcher
    from maneuver_detect.datasets.build import build_dataset, fetch_labels
    from maneuver_detect.datasets.catalogue import v01_recipe
    from maneuver_detect.labels.record import OrbitClass

    # A build is a long run (a NANU-archive crawl plus a per-object Space-Track fetch), so surface
    # the per-year / per-object progress logs live on stderr (the handler flushes each record).
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    progress = logging.getLogger("maneuver_detect")
    progress.addHandler(handler)
    progress.setLevel(logging.INFO)

    end_year = nanu_end_year if nanu_end_year is not None else datetime.now(tz=timezone.utc).year
    recipe = v01_recipe()
    headers = {"User-Agent": f"maneuver-detect/{__version__}"}
    progress.info(
        "fetching labels (NANU archive %d-%d), then %d series...",
        nanu_start_year,
        end_year,
        len(recipe.entries),
    )
    with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
        labels = fetch_labels(
            recipe,
            client,
            nanu_start_year=nanu_start_year,
            nanu_end_year=end_year,
            rate_limiter=RateLimiter(1.0),
        )
        with SpacetrackFetcher() as fetcher:
            report = build_dataset(recipe, fetcher, labels, out_dir)

    counts = recipe.per_class_counts()
    print(
        f"reconstructed {report.n_objects} objects "
        f"(LEO {counts[OrbitClass.LEO]}, MEO {counts[OrbitClass.MEO]}, "
        f"GEO {counts[OrbitClass.GEO]})"
    )
    for orbit_class in OrbitClass:
        cov = report.coverage.per_class[orbit_class]
        print(
            f"  {orbit_class.value}: {cov.n_events} labelled events "
            f"({cov.n_with_delta_v} with dv, {cov.n_with_norad} catalogue-resolved)"
        )
    for name in ("recipe", "labels", "manifest"):
        print(f"wrote {name}: {report.paths[name]}")
    return 0
