"""Build the distributable v0.1 artifacts — recipe, labels, and the content-hash manifest.

The logic behind ``maneuver-detect dataset build``: fetch each catalogue object's series from the
credentialled archive, fetch the open maneuver-label files, reconstruct the labelled dataset, and
write the three committable artifacts. The raw Space-Track series is **never written** — only its
SHA-256 digest (the manifest), the openly-licensed labels, and the recipe parameters, per the
recipe-first model (D2).

:func:`build_dataset` is the pure orchestration (a series ``Fetcher`` plus already-parsed labels in,
the artifacts out) and is the unit-tested core. :func:`fetch_labels` is the networked label-ingest
leg the CLI wires up — it downloads the DORIS ``man.txt`` files and the NANU notices over HTTP and
parses them with the label layer.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from maneuver_detect.data.base import Fetcher
from maneuver_detect.data.ratelimit import RateLimiter
from maneuver_detect.datasets.catalogue import gps_svn_to_norad
from maneuver_detect.datasets.recipe import Recipe
from maneuver_detect.datasets.reconstruct import LabelledDataset, reconstruct
from maneuver_detect.labels.doris import parse_doris
from maneuver_detect.labels.gps_nanu import parse_nanus
from maneuver_detect.labels.labeller import CoverageReport
from maneuver_detect.labels.record import SOURCE_DORIS_IDS, ManeuverLabel, OrbitClass
from maneuver_detect.schema import ManeuverType

__all__ = [
    "BuildReport",
    "build_dataset",
    "fetch_labels",
    "fetch_nanu_labels",
    "labels_from_json",
    "labels_to_json",
    "write_artifacts",
]

_IDS_MAN_URL = "https://ids-doris.org/documents/BC/satellites/{ref}man.txt"
# The GPS NANU archive — one file per notice under a per-year index, the source of the FCSTDV
# (forecast delta-V = maneuver) history. The current_nanu.nnu file holds only the latest notice.
_NANU_ARCHIVE_INDEX = "https://celestrak.org/GPS/NANU/{year}/"
_NANU_ARCHIVE_FILE = "https://celestrak.org/GPS/NANU/{year}/{name}"


def labels_to_json(labels: Sequence[ManeuverLabel]) -> str:
    """Serialise maneuver labels to canonical JSON (the committable ``labels.json`` artifact)."""
    payload = [
        {
            "norad_id": label.norad_id,
            "epoch": label.epoch.isoformat(),
            "window_start": label.window_start.isoformat(),
            "window_end": label.window_end.isoformat(),
            "source": label.source,
            "source_ref": label.source_ref,
            "orbit_class": label.orbit_class.value,
            "maneuver_type": None if label.maneuver_type is None else label.maneuver_type.value,
            "delta_v": label.delta_v,
        }
        for label in sorted(
            labels,
            key=lambda m: (m.source, -1 if m.norad_id is None else m.norad_id, m.epoch.isoformat()),
        )
    ]
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def labels_from_json(text: str) -> list[ManeuverLabel]:
    """Parse maneuver labels from :func:`labels_to_json` output."""
    data = json.loads(text)
    labels: list[ManeuverLabel] = []
    for item in data:
        maneuver_type = item["maneuver_type"]
        delta_v = item["delta_v"]
        labels.append(
            ManeuverLabel(
                norad_id=None if item["norad_id"] is None else int(item["norad_id"]),
                epoch=datetime.fromisoformat(str(item["epoch"])),
                window_start=datetime.fromisoformat(str(item["window_start"])),
                window_end=datetime.fromisoformat(str(item["window_end"])),
                source=str(item["source"]),
                source_ref=str(item["source_ref"]),
                orbit_class=OrbitClass(str(item["orbit_class"])),
                maneuver_type=None if maneuver_type is None else ManeuverType(str(maneuver_type)),
                delta_v=None if delta_v is None else float(delta_v),
            )
        )
    return labels


def write_artifacts(
    dataset: LabelledDataset, recipe: Recipe, out_dir: str | Path
) -> dict[str, Path]:
    """Write the three artifact JSON files into ``out_dir``; return their paths keyed by name."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "recipe": out / "recipe.json",
        "labels": out / "labels.json",
        "manifest": out / "manifest.json",
    }
    paths["recipe"].write_text(recipe.to_json(), encoding="utf-8")
    paths["labels"].write_text(labels_to_json(dataset.labels), encoding="utf-8")
    paths["manifest"].write_text(dataset.manifest().to_json(), encoding="utf-8")
    return paths


@dataclass(frozen=True)
class BuildReport:
    """The outcome of a build: where the artifacts landed and the dataset's coverage.

    Attributes:
        paths: The written artifact paths, keyed ``"recipe"`` / ``"labels"`` / ``"manifest"``.
        n_objects: Number of reconstructed objects.
        coverage: The per-class label-coverage report.
    """

    paths: dict[str, Path]
    n_objects: int
    coverage: CoverageReport


def build_dataset(
    recipe: Recipe,
    series_fetcher: Fetcher,
    labels_by_norad: Mapping[int, Sequence[ManeuverLabel]],
    out_dir: str | Path,
) -> BuildReport:
    """Reconstruct ``recipe`` with ``series_fetcher`` and labels, then write the artifacts."""
    dataset = reconstruct(recipe, series_fetcher, labels_by_norad)
    paths = write_artifacts(dataset, recipe, out_dir)
    return BuildReport(paths=paths, n_objects=len(dataset.objects), coverage=dataset.coverage())


def fetch_nanu_labels(
    client: httpx.Client,
    *,
    start_year: int,
    end_year: int,
    svn_to_norad: Mapping[str, int],
    rate_limiter: RateLimiter | None = None,
) -> list[ManeuverLabel]:
    """Crawl the CelesTrak NANU archive over ``[start_year, end_year]`` for FCSTDV maneuver labels.

    Each year's index lists one file per notice; every file is fetched and parsed, keeping only the
    FCSTDV (maneuver) notices that resolve to a NORAD id. ``rate_limiter`` paces the per-file
    fetches (the archive holds tens of files per year) — pass one for a polite crawl. A missing
    year index (404) is skipped.
    """
    labels: list[ManeuverLabel] = []
    for year in range(start_year, end_year + 1):
        index = client.get(_NANU_ARCHIVE_INDEX.format(year=year))
        if index.status_code == 404:
            continue
        index.raise_for_status()
        for name in sorted(set(re.findall(r"nanu\.\d{7}\.txt", index.text))):
            if rate_limiter is not None:
                rate_limiter.acquire()
            response = client.get(_NANU_ARCHIVE_FILE.format(year=year, name=name))
            if response.status_code == 404:
                continue
            response.raise_for_status()
            labels.extend(
                label
                for label in parse_nanus(response.text, svn_to_norad=svn_to_norad)
                if label.norad_id is not None
            )
    return labels


def fetch_labels(
    recipe: Recipe,
    client: httpx.Client,
    *,
    nanu_start_year: int,
    nanu_end_year: int,
    rate_limiter: RateLimiter | None = None,
) -> dict[int, list[ManeuverLabel]]:
    """Download and parse the open maneuver-label files for ``recipe``, keyed by NORAD id.

    DORIS/IDS ``man.txt`` files are fetched one per LEO entry (a renamed/missing file is skipped
    with a warning); the GPS NANU FCSTDV notices come from the CelesTrak archive over
    ``[nanu_start_year, nanu_end_year]``, parsed with the full constellation crosswalk. Labels that
    do not resolve to a NORAD id are dropped (they cannot attach to a series).
    """
    by_norad: dict[int, list[ManeuverLabel]] = {}
    seen_refs: set[str] = set()
    for entry in recipe.entries:
        if entry.label_source != SOURCE_DORIS_IDS or entry.label_ref in seen_refs:
            continue
        seen_refs.add(entry.label_ref)
        response = client.get(_IDS_MAN_URL.format(ref=entry.label_ref))
        if response.status_code == 404:
            # The IDS server occasionally renames a man.txt file; skip a missing one with a loud
            # warning rather than aborting the whole build (that object just gets no labels).
            print(
                f"warning: DORIS file '{entry.label_ref}man.txt' not found (404); skipping",
                file=sys.stderr,
            )
            continue
        response.raise_for_status()
        for label in parse_doris(response.text):
            if label.norad_id is not None:
                by_norad.setdefault(label.norad_id, []).append(label)

    for label in fetch_nanu_labels(
        client,
        start_year=nanu_start_year,
        end_year=nanu_end_year,
        svn_to_norad=gps_svn_to_norad(),
        rate_limiter=rate_limiter,
    ):
        if label.norad_id is not None:
            by_norad.setdefault(label.norad_id, []).append(label)
    return by_norad
