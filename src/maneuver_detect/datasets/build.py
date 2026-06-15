"""Build the distributable dataset artifacts — recipe, labels, and the content-hash manifest.

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
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from maneuver_detect.data.base import Fetcher
from maneuver_detect.data.cache import Cache, default_cache
from maneuver_detect.data.ratelimit import RateLimiter
from maneuver_detect.datasets.catalogue import (
    galileo_gsat_to_norad,
    goes_name_to_norad,
    gps_svn_to_norad,
)
from maneuver_detect.datasets.recipe import Recipe
from maneuver_detect.datasets.reconstruct import LabelledDataset, reconstruct
from maneuver_detect.labels.doris import parse_doris
from maneuver_detect.labels.galileo_nagu import parse_nagus
from maneuver_detect.labels.gps_nanu import parse_nanus
from maneuver_detect.labels.labeller import CoverageReport
from maneuver_detect.labels.noaa_goes import parse_navsum
from maneuver_detect.labels.qzss_ohi import parse_qzss_ohi
from maneuver_detect.labels.record import (
    SOURCE_DORIS_IDS,
    SOURCE_GALILEO_NAGU,
    SOURCE_NOAA_GOES,
    SOURCE_QZSS_OHI,
    ManeuverLabel,
    OrbitClass,
)
from maneuver_detect.schema import ManeuverType

__all__ = [
    "BuildReport",
    "build_dataset",
    "fetch_galileo_nagu_labels",
    "fetch_labels",
    "fetch_nanu_labels",
    "fetch_noaa_goes_labels",
    "fetch_qzss_ohi_labels",
    "labels_from_json",
    "labels_to_json",
    "write_artifacts",
]

_logger = logging.getLogger(__name__)

_IDS_MAN_URL = "https://ids-doris.org/documents/BC/satellites/{ref}man.txt"
# The GPS NANU archive — one file per notice under a per-year index, the source of the FCSTDV
# (forecast delta-V = maneuver) history. The current_nanu.nnu file holds only the latest notice.
_NANU_ARCHIVE_INDEX = "https://celestrak.org/GPS/NANU/{year}/"
_NANU_ARCHIVE_FILE = "https://celestrak.org/GPS/NANU/{year}/{name}"
# The Galileo NAGU archive — the GSC publishes one .txt per notice at a stable URL keyed by the
# notice number (``<year><seq>``, sequential per year). There is no machine listing, so the crawl
# probes ``seq = 1, 2, ...`` per year until a run of consecutive misses ends the year.
_GSC_NAGU_FILE = (
    "https://www.gsc-europa.eu/sites/default/files/"
    "NOTICE_ADVISORY_TO_GALILEO_USERS_NAGU_{year}{seq:03d}.txt"
)
# Galileo NAGU numbers are sequential per year, so this many consecutive 404s ends a year's crawl.
_NAGU_MISS_RUN = 25
# The QZSS Operational History Information files — one per satellite at a stable URL keyed by the
# OHI file stem (e.g. ``ohi-qzs2.txt``), carrying the executed-maneuver Δv log.
_QZSS_OHI_URL = "https://qzss.go.jp/en/technical/qzssinfo/khp0mf0000000wuf-att/ohi-{ref}.txt"
# The NOAA OSPO navigation summary — a live-state file naming each GOES bird's last maneuver. Its
# maneuver *history* is recovered from the Internet Archive: the CDX API lists every archived
# snapshot (``collapse=digest`` keeps only content-distinct ones), each fetched verbatim (``id_``).
_NOAA_NAVSUM_URL = "https://www.ospo.noaa.gov/resources/cemscs/navsum.txt"
_NOAA_CDX_URL = (
    "https://web.archive.org/cdx/search/cdx"
    "?url=ospo.noaa.gov/resources/cemscs/navsum.txt&output=json&collapse=digest&fl=timestamp"
)
_NOAA_SNAPSHOT_URL = "https://web.archive.org/web/{timestamp}id_/" + _NOAA_NAVSUM_URL

# On-disk cache sources (subdirectories) for the label-file fetches. The crawls are otherwise a full
# re-download of immutable archives on every run — exactly the "repeated downloads" the providers
# rate-limit — so every fetched body is cached and re-served. CelesTrak explicitly asks for this:
# "individual NANUs are only updated once, so there is no need to download any NANU more than once".
_CACHE_SOURCE_NANU = "labels-nanu"
_CACHE_SOURCE_NAGU = "labels-nagu"
_CACHE_SOURCE_QZSS = "labels-qzss-ohi"
_CACHE_SOURCE_DORIS = "labels-doris"
_CACHE_SOURCE_NOAA = "labels-noaa-goes"

# TTLs. Notice files (a NANU/NAGU notice, an Internet-Archive snapshot) are immutable once issued,
# so cached effectively forever; archive *indexes* and the append-only OHI / man.txt / live files
# gain entries over time, so they carry a short TTL that still spares same-day re-runs.
_TTL_IMMUTABLE_S = 365.0 * 24 * 3600
_TTL_APPENDED_S = 24.0 * 3600
_TTL_INDEX_S = 6.0 * 3600


def _cached_text(
    client: httpx.Client,
    cache: Cache,
    *,
    source: str,
    url: str,
    ttl_s: float,
    rate_limiter: RateLimiter | None = None,
    tolerant: bool = False,
) -> str | None:
    """Return the text body of ``url``, served from / written to ``cache``.

    A fresh cache hit returns immediately with **no** network request (and so consumes no
    ``rate_limiter`` slot) — the whole point, so a re-run does not re-download an immutable archive.
    On a miss the URL is fetched (paced by ``rate_limiter``); a ``200`` body is cached and returned,
    a ``404`` returns ``None``, and any other status raises (``tolerant=True`` returns ``None``
    instead, for best-effort sources where one bad snapshot should not abort the crawl).
    """
    hit = cache.get(source, url, ttl_s=ttl_s)
    if hit is not None:
        return str(hit.value)
    if rate_limiter is not None:
        rate_limiter.acquire()
    response = client.get(url)
    if response.status_code == 200:
        cache.put(source, url, response.text)
        return response.text
    if response.status_code == 404 or tolerant:
        return None
    response.raise_for_status()
    return None  # unreachable: raise_for_status raised


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
    cache: Cache | None = None,
) -> list[ManeuverLabel]:
    """Crawl the CelesTrak NANU archive over ``[start_year, end_year]`` for FCSTDV maneuver labels.

    Each year's index lists one file per notice; every file is fetched and parsed, keeping only the
    FCSTDV (maneuver) notices that resolve to a NORAD id. Fetches are served from ``cache`` (the
    notices are immutable, so a re-run re-downloads nothing — the CelesTrak-requested behaviour);
    ``rate_limiter`` paces the per-file fetches that actually hit the network. A missing year index
    (404) is skipped. A past year's index is cached as immutable; the current (``end_year``) index
    carries a short TTL so newly issued notices are still picked up.
    """
    cache = cache or default_cache()
    labels: list[ManeuverLabel] = []
    for year in range(start_year, end_year + 1):
        index_ttl = _TTL_INDEX_S if year >= end_year else _TTL_IMMUTABLE_S
        index_text = _cached_text(
            client,
            cache,
            source=_CACHE_SOURCE_NANU,
            url=_NANU_ARCHIVE_INDEX.format(year=year),
            ttl_s=index_ttl,
        )
        if index_text is None:  # a missing year index (404)
            continue
        names = sorted(set(re.findall(r"nanu\.\d{7}\.txt", index_text)))
        before = len(labels)
        for name in names:
            text = _cached_text(
                client,
                cache,
                source=_CACHE_SOURCE_NANU,
                url=_NANU_ARCHIVE_FILE.format(year=year, name=name),
                ttl_s=_TTL_IMMUTABLE_S,
                rate_limiter=rate_limiter,
            )
            if text is None:
                continue
            labels.extend(
                label
                for label in parse_nanus(text, svn_to_norad=svn_to_norad)
                if label.norad_id is not None
            )
        _logger.info(
            "NANU archive %d: %d notices, %d FCSTDV labels", year, len(names), len(labels) - before
        )
    return labels


def fetch_galileo_nagu_labels(
    client: httpx.Client,
    *,
    start_year: int,
    end_year: int,
    gsat_to_norad: Mapping[str, int],
    rate_limiter: RateLimiter | None = None,
    cache: Cache | None = None,
) -> list[ManeuverLabel]:
    """Crawl the GSC Galileo NAGU archive over ``[start_year, end_year]`` for PLN_MANV labels.

    The GSC exposes no machine listing, so each year is probed by sequential notice number
    (``seq = 1, 2, ...``) against the stable ``.txt`` URL; a run of :data:`_NAGU_MISS_RUN`
    consecutive 404s ends the year (NAGU numbers are sequential per year). Every fetched notice is
    parsed and only the ``PLN_MANV`` notices that resolve to a NORAD id are kept. Issued notices are
    immutable, so they are served from ``cache`` on a re-run (only the trailing 404 probes re-hit
    the network); ``rate_limiter`` paces the fetches that do.
    """
    cache = cache or default_cache()
    labels: list[ManeuverLabel] = []
    for year in range(start_year, end_year + 1):
        before = len(labels)
        seq, misses = 0, 0
        while misses < _NAGU_MISS_RUN:
            seq += 1
            text = _cached_text(
                client,
                cache,
                source=_CACHE_SOURCE_NAGU,
                url=_GSC_NAGU_FILE.format(year=year, seq=seq),
                ttl_s=_TTL_IMMUTABLE_S,
                rate_limiter=rate_limiter,
            )
            if text is None:
                misses += 1
                continue
            misses = 0
            labels.extend(
                label
                for label in parse_nagus(text, gsat_to_norad=gsat_to_norad)
                if label.norad_id is not None
            )
        _logger.info(
            "Galileo NAGU %d: %d PLN_MANV labels (to seq %d)", year, len(labels) - before, seq
        )
    return labels


def fetch_qzss_ohi_labels(
    recipe: Recipe,
    client: httpx.Client,
    *,
    rate_limiter: RateLimiter | None = None,
    cache: Cache | None = None,
) -> list[ManeuverLabel]:
    """Fetch and parse the QZSS OHI executed-maneuver logs for the recipe's QZSS entries.

    One OHI file is fetched per QZSS recipe entry (keyed by the entry's ``label_ref`` OHI stem) and
    parsed with the entry's pinned NORAD id and orbit class (IGSO for QZS-2/4/1R, GEO for QZS-3/6).
    A missing file (404 — e.g. a freshly launched satellite without a log yet) is skipped with a
    warning rather than aborting. OHI files gain rows over time, so they are cached with a short TTL
    (``cache``); ``rate_limiter`` paces the fetches that hit the network.
    """
    cache = cache or default_cache()
    labels: list[ManeuverLabel] = []
    for entry in recipe.entries:
        if entry.label_source != SOURCE_QZSS_OHI:
            continue
        text = _cached_text(
            client,
            cache,
            source=_CACHE_SOURCE_QZSS,
            url=_QZSS_OHI_URL.format(ref=entry.label_ref),
            ttl_s=_TTL_APPENDED_S,
            rate_limiter=rate_limiter,
        )
        if text is None:
            print(
                f"warning: QZSS OHI file 'ohi-{entry.label_ref}.txt' not found (404); skipping",
                file=sys.stderr,
            )
            continue
        events = parse_qzss_ohi(
            text,
            norad_id=entry.norad_id,
            orbit_class=entry.orbit_class,
            qzs_label=entry.object_name,
        )
        labels.extend(events)
        _logger.info("QZSS OHI %s: %d maneuver events", entry.label_ref, len(events))
    return labels


def fetch_noaa_goes_labels(
    client: httpx.Client,
    *,
    goes_name_to_norad: Mapping[str, int],
    rate_limiter: RateLimiter | None = None,
    cache: Cache | None = None,
) -> list[ManeuverLabel]:
    """Build the GOES maneuver history from the NOAA navsum file's Internet-Archive snapshots.

    ``navsum.txt`` is a live-state file naming only each bird's *latest* maneuver, so the history is
    recovered by replaying its archived snapshots: the Internet Archive CDX API lists every
    content-distinct snapshot, each is fetched verbatim and parsed, and the distinct
    ``(norad_id, maneuver-day)`` epochs are accumulated. The current live file is parsed last so the
    newest maneuver is captured even before the archive catches up. Snapshots are immutable so they
    are cached forever; the CDX listing and the live file carry a short TTL. A snapshot that fails
    to fetch is skipped (best-effort). ``rate_limiter`` paces the fetches that hit the network.
    """
    cache = cache or default_cache()
    timestamps: list[str] = []
    index_text = _cached_text(
        client,
        cache,
        source=_CACHE_SOURCE_NOAA,
        url=_NOAA_CDX_URL,
        ttl_s=_TTL_INDEX_S,
        tolerant=True,
    )
    if index_text and index_text.strip():
        rows = json.loads(index_text)
        timestamps = [row[0] for row in rows[1:]]  # row 0 is the ["timestamp"] header
    snapshots = [(_NOAA_SNAPSHOT_URL.format(timestamp=ts), _TTL_IMMUTABLE_S) for ts in timestamps]
    snapshots.append((_NOAA_NAVSUM_URL, _TTL_INDEX_S))  # the live file (short TTL)

    seen: set[tuple[int, str]] = set()
    labels: list[ManeuverLabel] = []
    for url, ttl_s in snapshots:
        text = _cached_text(
            client,
            cache,
            source=_CACHE_SOURCE_NOAA,
            url=url,
            ttl_s=ttl_s,
            rate_limiter=rate_limiter,
            tolerant=True,
        )
        if text is None:
            continue
        for label in parse_navsum(text, goes_name_to_norad=goes_name_to_norad):
            if label.norad_id is None:
                continue
            key = (label.norad_id, label.window_start.isoformat())
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
    _logger.info(
        "NOAA GOES: %d distinct maneuver epochs over %d snapshots", len(labels), len(snapshots)
    )
    return labels


def fetch_labels(
    recipe: Recipe,
    client: httpx.Client,
    *,
    nanu_start_year: int,
    nanu_end_year: int,
    rate_limiter: RateLimiter | None = None,
    cache: Cache | None = None,
) -> dict[int, list[ManeuverLabel]]:
    """Download and parse the open maneuver-label files for ``recipe``, keyed by NORAD id.

    DORIS/IDS ``man.txt`` files are fetched one per LEO entry (a renamed/missing file is skipped
    with a warning); the GPS NANU FCSTDV notices come from the CelesTrak archive and — when the
    recipe carries Galileo entries — the Galileo NAGU notices from the GSC archive, both
    over ``[nanu_start_year, nanu_end_year]`` with the full constellation crosswalks; QZSS OHI and
    NOAA GOES labels come from their entries' sources. Every fetched file is served from / written
    to ``cache`` (defaulting to the shared on-disk cache), so a re-run re-downloads only the mutable
    archive indexes and append-only files — the immutable notices are reused, which is what the
    providers (CelesTrak especially) ask for. Labels that do not resolve to a NORAD id are dropped.
    The self-labelled GEO/HEO sources carry no external file — those are derived at reconstruction.
    """
    cache = cache or default_cache()
    by_norad: dict[int, list[ManeuverLabel]] = {}
    seen_refs: set[str] = set()
    for entry in recipe.entries:
        if entry.label_source != SOURCE_DORIS_IDS or entry.label_ref in seen_refs:
            continue
        seen_refs.add(entry.label_ref)
        # The IDS server occasionally renames a man.txt file; a missing one (404 -> None) is skipped
        # with a loud warning rather than aborting the build (that object just gets no labels).
        text = _cached_text(
            client,
            cache,
            source=_CACHE_SOURCE_DORIS,
            url=_IDS_MAN_URL.format(ref=entry.label_ref),
            ttl_s=_TTL_APPENDED_S,
        )
        if text is None:
            print(
                f"warning: DORIS file '{entry.label_ref}man.txt' not found (404); skipping",
                file=sys.stderr,
            )
            continue
        kept = 0
        for label in parse_doris(text):
            if label.norad_id is not None:
                by_norad.setdefault(label.norad_id, []).append(label)
                kept += 1
        _logger.info("DORIS %sman.txt: %d labels", entry.label_ref, kept)

    for label in fetch_nanu_labels(
        client,
        start_year=nanu_start_year,
        end_year=nanu_end_year,
        svn_to_norad=gps_svn_to_norad(),
        rate_limiter=rate_limiter,
        cache=cache,
    ):
        if label.norad_id is not None:
            by_norad.setdefault(label.norad_id, []).append(label)

    if any(entry.label_source == SOURCE_GALILEO_NAGU for entry in recipe.entries):
        for label in fetch_galileo_nagu_labels(
            client,
            start_year=nanu_start_year,
            end_year=nanu_end_year,
            gsat_to_norad=galileo_gsat_to_norad(),
            rate_limiter=rate_limiter,
            cache=cache,
        ):
            if label.norad_id is not None:
                by_norad.setdefault(label.norad_id, []).append(label)

    if any(entry.label_source == SOURCE_QZSS_OHI for entry in recipe.entries):
        for label in fetch_qzss_ohi_labels(recipe, client, rate_limiter=rate_limiter, cache=cache):
            if label.norad_id is not None:
                by_norad.setdefault(label.norad_id, []).append(label)

    if any(entry.label_source == SOURCE_NOAA_GOES for entry in recipe.entries):
        for label in fetch_noaa_goes_labels(
            client, goes_name_to_norad=goes_name_to_norad(), rate_limiter=rate_limiter, cache=cache
        ):
            if label.norad_id is not None:
                by_norad.setdefault(label.norad_id, []).append(label)
    return by_norad
