"""Hugging Face Hub integration — load checkpoints and the dataset on demand, and publish them.

The package ships no model weights and no raw dataset. A learned detector pulls its trained
checkpoint from the Hub the first time it runs (:func:`checkpoint_path`); the dataset accessor pulls
the published recipe / labels / manifest / splits the first time they are asked for
(:func:`fetch_dataset`, :func:`dataset_path`). Both are CPU-only, cached on disk by
``huggingface_hub`` (under ``HF_HOME`` / the XDG cache — nothing is fetched at install time), and
pinned to a release revision so a given library version loads the dataset and checkpoints it was
released in **lockstep** with (D8).

The same module backs publishing: :func:`hf_api` and :func:`move_tag` are the shared, authenticated
substrate the checkpoint publisher (:mod:`maneuver_detect.models.publish`) and the dataset publisher
(:mod:`maneuver_detect.datasets.publish`) build on, so the Hub repo ids, the revision, and the auth
path are defined once here. ``huggingface_hub`` is imported lazily inside the functions, so
importing the package (or running the classical detector) never pays for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from maneuver_detect.datasets.catalogue import DATASET_VERSION
from maneuver_detect.errors import ManeuverDetectError

if TYPE_CHECKING:
    from huggingface_hub import HfApi

__all__ = [
    "DATASET_REPO",
    "MODELS",
    "HubError",
    "HubModel",
    "checkpoint_path",
    "dataset_path",
    "fetch_dataset",
    "hf_api",
    "hub_revision",
    "move_tag",
]

#: The dataset Hub repo (``repo_type="dataset"``) — the recipe, labels, manifest, splits, and the
#: dataset card. The raw element series is never uploaded (D2); only the reconstruction recipe is.
DATASET_REPO = "astro-tools/maneuver-detect"


@dataclass(frozen=True)
class HubModel:
    """A Hub-published checkpoint: the model repo it lives in and the bundle filename inside it."""

    repo_id: str
    filename: str


#: Detector name → its Hub model repo + bundle filename. ``detect(history, model=<name>)`` with no
#: local checkpoint (no explicit bundle, no ``$…_CHECKPOINT`` env var) pulls the bundle from here.
#: The ``*-residual`` entries are the v0.3 foundation baselines — their bundle is a
#: :class:`~maneuver_detect.models.foundation.FoundationBundle` (a pinned forecaster id + calibrated
#: per-class thresholds), not the torch-network
#: :class:`~maneuver_detect.models.checkpoint.ModelBundle`.
MODELS: dict[str, HubModel] = {
    "bilstm-base": HubModel("astro-tools/maneuver-detect-bilstm-base", "bilstm-base.pt"),
    "transformer-base": HubModel(
        "astro-tools/maneuver-detect-transformer-base", "transformer-base.pt"
    ),
    "chronos-residual": HubModel(
        "astro-tools/maneuver-detect-chronos-residual", "chronos-residual.pt"
    ),
    "timesfm-residual": HubModel(
        "astro-tools/maneuver-detect-timesfm-residual", "timesfm-residual.pt"
    ),
}

#: Environment override for the Hub revision artifacts load from. Point it at ``"main"`` to validate
#: against the latest un-tagged push before a release is cut; unset, it pins the lockstep release
#: tag.
_REVISION_ENV = "MANEUVER_DETECT_HUB_REVISION"


class HubError(ManeuverDetectError):
    """A Hugging Face Hub download or publish failed — network, auth, or a missing repo/revision."""


def hub_revision() -> str:
    """The Hub revision artifacts load from: ``$MANEUVER_DETECT_HUB_REVISION`` or the release tag.

    Unset, the default pins both the dataset and the checkpoints to ``v{DATASET_VERSION}`` so a
    given library version loads the dataset and checkpoints it was released in lockstep with (D8).
    Set the env var to ``"main"`` (or any branch / tag / commit) to load from there instead — useful
    for validating a published artifact before its release tag exists.
    """
    override = os.environ.get(_REVISION_ENV)
    return override if override else f"v{DATASET_VERSION}"


def checkpoint_path(name: str, *, revision: str | None = None) -> Path:
    """Download the Hub checkpoint bundle for detector ``name`` and return its local cached path.

    ``revision`` defaults to :func:`hub_revision`. The bundle is cached on disk by
    ``huggingface_hub`` and loaded CPU-only by the caller; no weights are fetched at install time.
    Raises :class:`HubError` if ``name`` is not a Hub-published model, or the download fails (off
    the network, unauthenticated for a gated repo, or the repo / revision does not exist).
    """
    try:
        model = MODELS[name]
    except KeyError:
        raise HubError(
            f"no Hub-published checkpoint for model {name!r}; published models: {sorted(MODELS)}"
        ) from None
    rev = revision if revision is not None else hub_revision()
    return _download_file(model.repo_id, model.filename, revision=rev, repo_type="model")


def dataset_path(filename: str, *, revision: str | None = None) -> Path:
    """Download one published dataset artifact (e.g. ``"recipe.json"``) and return its cached path.

    ``revision`` defaults to :func:`hub_revision`. Raises :class:`HubError` if the download fails.
    """
    rev = revision if revision is not None else hub_revision()
    return _download_file(DATASET_REPO, filename, revision=rev, repo_type="dataset")


def fetch_dataset(*, revision: str | None = None) -> Path:
    """Download the whole published dataset and return the local snapshot directory.

    Snapshots the dataset repo (recipe / labels / manifest / splits + the dataset card) at
    ``revision`` (default :func:`hub_revision`), cached on disk. Raises :class:`HubError` on
    failure.
    """
    from huggingface_hub import snapshot_download

    rev = revision if revision is not None else hub_revision()
    try:
        local = snapshot_download(DATASET_REPO, repo_type="dataset", revision=rev)
    except Exception as exc:
        # Any hub/transport failure is surfaced as one typed library error, not a raw stack.
        raise HubError(
            f"could not fetch the dataset from the Hub dataset repo {DATASET_REPO!r}@{rev} "
            f"({type(exc).__name__}: {exc}); check your network or set "
            "$MANEUVER_DETECT_HUB_REVISION to an available revision"
        ) from exc
    return Path(local)


def _download_file(repo_id: str, filename: str, *, revision: str, repo_type: str) -> Path:
    """Download one file from a Hub repo, wrapping any failure in :class:`HubError`."""
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, repo_type=repo_type
        )
    except Exception as exc:
        # Any hub/transport failure is surfaced as one typed library error, not a raw stack.
        hint = (
            "point the matching $MANEUVER_DETECT_*_CHECKPOINT env var at a local bundle"
            if repo_type == "model"
            else "set $MANEUVER_DETECT_HUB_REVISION to an available revision"
        )
        raise HubError(
            f"could not fetch {filename!r} from the Hub {repo_type} repo {repo_id!r}@{revision} "
            f"({type(exc).__name__}: {exc}); check your network, or {hint}"
        ) from exc
    return Path(local)


def hf_api(token: str | None = None) -> HfApi:
    """An :class:`huggingface_hub.HfApi` for publishing.

    ``token`` is the write token; when ``None``, ``huggingface_hub`` falls back to ``$HF_TOKEN`` or
    a prior ``huggingface-cli login`` — which is how CI (with an ``HF_TOKEN`` secret) authenticates.
    """
    from huggingface_hub import HfApi

    return HfApi(token=token)


def move_tag(api: HfApi, repo_id: str, *, repo_type: str, version: str) -> str:
    """Point the lockstep release tag ``v{version}`` at the repo's current head, and return it.

    Idempotent: an existing tag is deleted and recreated so re-publishing the same version moves the
    tag onto the freshly uploaded content rather than leaving it on the old commit.
    """
    from contextlib import suppress

    from huggingface_hub.errors import RevisionNotFoundError

    tag = f"v{version}"
    with suppress(RevisionNotFoundError):  # first publish — there is no tag to move yet
        api.delete_tag(repo_id, tag=tag, repo_type=repo_type)
    api.create_tag(repo_id, tag=tag, repo_type=repo_type)
    return tag
