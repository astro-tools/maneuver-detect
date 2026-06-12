"""Detect maneuvers for one satellite by NORAD catalogue id.

The headline workflow: fetch a satellite's mean-element TLE history from Space-Track and run the
classical detector over it. This needs a free Space-Track account, with the credentials in the
environment (``SPACETRACK_USERNAME`` / ``SPACETRACK_PASSWORD``); without them the script explains
how to set them and exits cleanly, so it is safe to run anywhere.

    python examples/detect_norad.py            # default: the ISS (NORAD 25544)
    python examples/detect_norad.py 39634      # Sentinel-1A

The output stays ASCII so it prints cleanly on any console.
"""

from __future__ import annotations

import os
import sys

from maneuver_detect import datasets, detect
from maneuver_detect.errors import ManeuverDetectError

_DEFAULT_NORAD_ID = 25544  # the ISS


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    norad_id = int(args[0]) if args else _DEFAULT_NORAD_ID

    if not (os.environ.get("SPACETRACK_USERNAME") and os.environ.get("SPACETRACK_PASSWORD")):
        print(
            "Set SPACETRACK_USERNAME and SPACETRACK_PASSWORD to fetch TLE history from\n"
            "Space-Track (a free account is at https://www.space-track.org/), then re-run\n"
            "this example."
        )
        return 0

    print(f"Fetching TLE history for NORAD {norad_id} from Space-Track...")
    try:
        history = datasets.tle_history(norad_id=norad_id, start="2023-01-01")
        maneuvers = detect(history, model="classical")
    except ManeuverDetectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{len(history)} elsets, {len(maneuvers)} maneuver(s) detected.")
    if maneuvers.empty:
        print("No maneuvers detected.")
    else:
        print(maneuvers.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
