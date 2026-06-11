"""Command-line interface — the ``maneuver-detect detect`` one-shot detection subcommand.

``detect`` mirrors the public :func:`maneuver_detect.detect` API on either a NORAD catalogue id
(its TLE history fetched live) or a local TLE file, and prints the canonical maneuver DataFrame.
It is a thin shell over the API, so the CLI produces the same result as the equivalent API call:
both inputs converge on the same mean-element history the detector consumes, and the rendered
columns are the canonical schema verbatim. Output stays ASCII (the Δv column prints as
``delta_v_estimate``) so it is safe on a cp1252 Windows console.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

from maneuver_detect import __version__
from maneuver_detect.data import DEFAULT_SOURCE, FETCHERS

if TYPE_CHECKING:
    import pandas as pd

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
    return _run_detect(
        target=args.target,
        model=args.model,
        source=args.source,
        start=args.start,
        end=args.end,
        output_format=args.format,
        output=args.output,
    )


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
            "Assemble the mean-element history for <target>, run the detector, and print the "
            "detected maneuvers as the canonical schema. <target> is read as a local TLE file when "
            "it names an existing file, otherwise as a NORAD catalogue id whose history is fetched "
            "live. The output matches the equivalent detect() API call."
        ),
    )
    detect_parser.add_argument("target", help="a NORAD catalogue id, or a path to a local TLE file")
    detect_parser.add_argument(
        "--model",
        default="classical",
        help="detector to run (default: the classical reference detector)",
    )
    detect_parser.add_argument(
        "--source",
        choices=sorted(FETCHERS),
        default=DEFAULT_SOURCE,
        help=(
            "catalogue source for a NORAD-id fetch: 'spacetrack' is the credentialled history "
            "archive, 'celestrak' the no-auth current elset. Ignored for a TLE file "
            "(default: %(default)s)"
        ),
    )
    detect_parser.add_argument(
        "--start", help="ISO-8601 lower epoch bound, e.g. 2024-01-01 (default: full history)"
    )
    detect_parser.add_argument(
        "--end", help="ISO-8601 upper epoch bound, e.g. 2024-06-30 (default: full history)"
    )
    detect_parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="output format (default: %(default)s)",
    )
    detect_parser.add_argument(
        "--output",
        "-o",
        help="write the result to this file instead of stdout",
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


def _run_detect(
    target: str,
    model: str,
    source: str,
    start: str | None,
    end: str | None,
    output_format: str,
    output: str | None,
) -> int:
    """Run one-shot detection on ``target`` and render the canonical maneuver frame.

    Resolves ``target`` to a mean-element history (a local TLE file, else a live NORAD-id fetch),
    runs the named detector, and writes the result in ``output_format`` to ``output`` (or stdout).
    The library's own failures — a missing Space-Track credential, an unreachable source, a
    malformed TLE, an unknown model, or a bad date / target — are reported as a one-line message on
    stderr and a non-zero exit, not a traceback.
    """
    import sys

    from maneuver_detect import detect
    from maneuver_detect.errors import ManeuverDetectError

    try:
        history = _load_history(target, source=source, start=start, end=end)
        result = detect(history, model=model)
    except (ManeuverDetectError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rendered = _render(result, output_format)
    if output is None:
        print(rendered)
    else:
        from pathlib import Path

        Path(output).write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {len(result)} maneuver(s) to {output}", file=sys.stderr)
    return 0


def _load_history(target: str, *, source: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Resolve ``target`` to a cleaned mean-element history DataFrame.

    ``target`` is read as a local TLE file when it names an existing file (windowed to
    ``[start, end]`` after parsing), otherwise as a NORAD catalogue id whose history is fetched live
    from ``source``. A target that is neither an existing file nor an all-digit id raises
    :class:`ValueError`.
    """
    import os

    from maneuver_detect import datasets
    from maneuver_detect.data import build_series, read_tle_file
    from maneuver_detect.data.base import in_range, parse_bound

    if os.path.isfile(target):
        elsets = read_tle_file(target)
        lo, hi = parse_bound(start), parse_bound(end)
        if lo is not None or hi is not None:
            elsets = [elset for elset in elsets if in_range(elset.epoch, lo, hi)]
        return build_series(elsets)
    if target.isdigit():
        return datasets.tle_history(int(target), start=start, end=end, source=source)
    raise ValueError(f"target {target!r} is neither an existing file nor a NORAD catalogue id")


def _render(result: pd.DataFrame, output_format: str) -> str:
    """Render the canonical maneuver frame as ASCII text in the requested format.

    ``csv`` and ``json`` (ISO epochs) are the machine-readable forms; ``table`` is the human form,
    a column-aligned dump or a short notice when nothing was detected. All three stay ASCII — the
    canonical column is ``delta_v_estimate``, never a literal ``Δ`` — so the output is safe to print
    on a cp1252 Windows console.
    """
    if output_format == "csv":
        # Pin the line terminator to "\n": to_csv defaults to os.linesep ("\r\n" on Windows), which
        # combined with a text-mode --output write would double the newlines. A fixed "\n" keeps the
        # output deterministic across platforms.
        return result.to_csv(index=False, lineterminator="\n").rstrip("\n")
    if output_format == "json":
        return str(result.to_json(orient="records", date_format="iso"))
    if result.empty:
        return "No maneuvers detected."
    return result.to_string(index=False)


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
