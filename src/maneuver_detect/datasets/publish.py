"""Publish the recipe-first labelled dataset and its dataset card to the Hugging Face Hub.

The distributable dataset is small, committed, and deterministic (the recipe, labels, manifest, and
splits — never the raw element series, D2), so it is published by CI on each release tag with this
module — ``maneuver-detect dataset publish <dir>``. It uploads the committed artifacts verbatim plus
a generated dataset card, and moves the lockstep ``v{version}`` tag onto them. The Hub repo id, the
revision tag, and the auth path come from :mod:`maneuver_detect.hub`.
"""

from __future__ import annotations

from pathlib import Path

from maneuver_detect import hub
from maneuver_detect.datasets.catalogue import DATASET_VERSION

__all__ = ["ARTIFACTS", "build_dataset_card", "publish_dataset"]

#: The distributable dataset artifacts, uploaded verbatim when present. ``splits.json`` is published
#: once a version's benchmark split is frozen; the others are always present in a built dataset dir.
ARTIFACTS = ("recipe.json", "labels.json", "manifest.json", "splits.json")


def publish_dataset(
    dataset_dir: str | Path,
    *,
    version: str = DATASET_VERSION,
    repo_id: str = hub.DATASET_REPO,
    token: str | None = None,
) -> str:
    """Upload the dataset artifacts in ``dataset_dir`` and a generated dataset card to the Hub.

    Uploads each of :data:`ARTIFACTS` present in ``dataset_dir`` verbatim plus a generated
    ``README.md`` dataset card, then moves the lockstep ``v{version}`` tag onto them. ``token`` is
    the HF write token (falls back to ``$HF_TOKEN`` / a prior login). Returns the dataset repo id.
    Raises :class:`~maneuver_detect.hub.HubError` if ``dataset_dir`` holds none of the artifacts.
    """
    directory = Path(dataset_dir)
    present = [name for name in ARTIFACTS if (directory / name).is_file()]
    if not present:
        raise hub.HubError(f"no dataset artifacts ({', '.join(ARTIFACTS)}) found in {directory}")
    card = build_dataset_card(version)

    api = hub.hf_api(token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Dataset card v{version}",
    )
    for name in present:
        api.upload_file(
            path_or_fileobj=str(directory / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Publish dataset v{version}",
        )
    hub.move_tag(api, repo_id, repo_type="dataset", version=version)
    return repo_id


def build_dataset_card(version: str = DATASET_VERSION) -> str:
    """Generate the Hugging Face dataset card (``README.md``) for the published dataset."""
    minor = ".".join(version.split(".")[:2])  # 0.2.0 -> 0.2 (the dataset/ subdirectory name)
    return f"""---
license: cc-by-4.0
pretty_name: maneuver-detect labelled maneuver dataset
tags:
- orbital-mechanics
- maneuver-detection
- space-situational-awareness
- satellite
- tle
---

# maneuver-detect — labelled maneuver dataset (v{version})

The labelled dataset behind [`maneuver-detect`](https://github.com/astro-tools/maneuver-detect): a
curated set of satellites whose maneuvers are known from public operator announcements, paired with
the mean-element TLE history those maneuvers show up in. Versioned in lockstep with the library and
its model checkpoints.

## Recipe-first distribution

The raw multi-year TLE history comes from Space-Track, whose terms do **not** permit redistributing
the data or analysis derived from it — so the raw series is **never** shipped here. Instead the
dataset is published as a pinned reconstruction recipe you rebuild locally from your own Space-Track
account, verified byte-for-byte by a content-hash manifest:

| File | What it is |
|---|---|
| `recipe.json` | Pinned catalogue: each object's NORAD id, class, label source, fetch window. |
| `labels.json` | Parsed operator maneuver labels (epoch / window / type / Δv if given). |
| `manifest.json` | One SHA-256 per reconstructed series — the byte-for-byte integrity check. |
| `splits.json` | The frozen, leak-free temporal-holdout train / val / test partition. |

Reconstruct and verify with the package (Space-Track credentials in the environment):

```bash
pip install maneuver-detect
export SPACETRACK_USERNAME='you@example.com' SPACETRACK_PASSWORD='...'
maneuver-detect dataset build --out dataset/v{minor}
```

## Class scope

- **LEO** — DORIS/IDS altimetry satellites (the Δv-labelled core) and SPOT imaging satellites.
- **MEO** — GPS (FCSTDV NANUs) and Galileo (PLN_MANV NAGUs); epoch-only.
- **GEO** — geostationary satellites: the GOES weather satellites carry operator-announced labels
  from the NOAA OSPO navigation summary, the QZS-3/6 satellites carry QZSS OHI operator Δv, and the
  Meteosat/Himawari satellites are self-labelled by longitude-drift inspection (best-effort).
- **IGSO** — the inclined/eccentric-geosynchronous QZSS satellites (QZS-2/4/1R), new in v0.3, with
  executed-Δv labels from the Cabinet Office of Japan's Operational History Information (OHI) files.
- **HEO** — the high-eccentricity regime is a reserved class with no objects yet: no ingestible
  maneuver source exists and self-labelling on deep-space TLEs is perturbation-dominated, so it is
  deferred to a future source.

## Licence

The authored artifacts (recipe, labels, splits, manifest) are **CC-BY-4.0**. The label sources pass
through under their own terms: DORIS/IDS open data; GPS NANUs and NOAA GOES navigation summaries are
US-Government public domain; Galileo NAGUs are reused with attribution (© EU); QZSS labels reused
under the Quasi-Zenith Satellite System website terms (CC-BY-4.0, "Source: Quasi-Zenith Satellite
System website"); the self-labelled GEO/HEO epochs are authored. The **raw Space-Track element
history is not redistributed** under any licence — it is re-fetched locally from each user's own
account. See the [repository](https://github.com/astro-tools/maneuver-detect) for full source terms.
"""
