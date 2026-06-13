"""Tests for the Hugging Face Hub load path — repo-id / revision resolution and the CPU load smoke.

The Hub is mocked throughout (``hf_hub_download`` is monkeypatched), so these run with no network,
no GPU, and no large download — the load-from-Hub path the package exposes is exercised against a
tiny locally-trained bundle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import maneuver_detect
from _synthetic import Burn, object_series, synthetic_series
from maneuver_detect import hub
from maneuver_detect.datasets.catalogue import DATASET_VERSION
from maneuver_detect.detectors.bilstm import CHECKPOINT_ENV
from maneuver_detect.models.bilstm import BiLstmConfig
from maneuver_detect.models.checkpoint import save_bundle
from maneuver_detect.models.train import train_bilstm
from maneuver_detect.schema import COLUMNS

_CONFIG = BiLstmConfig(hidden_size=8, num_layers=1, dropout=0.0)


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny CPU-trained checkpoint saved to disk, standing in for the Hub-hosted bundle."""
    objects = [
        object_series(norad_id=1, seed=1, burns=(Burn(40, "in_track_ms", 4.0),), n=90),
        object_series(norad_id=2, seed=2, burns=(Burn(55, "cross_track_ms", 4.0),), n=90),
    ]
    bundle = train_bilstm(
        objects,
        config=_CONFIG,
        max_epochs=1,
        seed=0,
        window=32,
        stride=16,
        batch_size=8,
        accelerator="cpu",
    )
    path = tmp_path_factory.mktemp("hub") / "bilstm-base.pt"
    save_bundle(bundle, path)
    return path


def test_hub_revision_defaults_to_release_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANEUVER_DETECT_HUB_REVISION", raising=False)
    assert hub.hub_revision() == f"v{DATASET_VERSION}"


def test_hub_revision_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANEUVER_DETECT_HUB_REVISION", "main")
    assert hub.hub_revision() == "main"


def test_checkpoint_path_unknown_model_raises() -> None:
    with pytest.raises(hub.HubError, match="no Hub-published checkpoint"):
        hub.checkpoint_path("nope")


def test_checkpoint_path_resolves_repo_filename_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path / "x.pt")

    monkeypatch.delenv("MANEUVER_DETECT_HUB_REVISION", raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    out = hub.checkpoint_path("bilstm-base")
    assert out == tmp_path / "x.pt"
    assert captured == {
        "repo_id": "astro-tools/maneuver-detect-bilstm-base",
        "filename": "bilstm-base.pt",
        "revision": f"v{DATASET_VERSION}",
        "repo_type": "model",
    }


def test_checkpoint_path_honours_revision_override(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return "/tmp/x.pt"

    monkeypatch.setenv("MANEUVER_DETECT_HUB_REVISION", "main")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    hub.checkpoint_path("transformer-base")
    assert captured["revision"] == "main"
    assert captured["repo_id"] == "astro-tools/maneuver-detect-transformer-base"


def test_checkpoint_path_wraps_download_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    with pytest.raises(hub.HubError, match="could not fetch"):
        hub.checkpoint_path("bilstm-base")


def test_detect_loads_checkpoint_from_hub_on_cpu(
    bundle_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The DoD CPU smoke: detect(model=...) with no local checkpoint pulls the named bundle from the
    # Hub (mocked) and runs on CPU — no GPU, no network, no large download in the default test run.
    captured: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(bundle_path)

    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    monkeypatch.delenv("MANEUVER_DETECT_HUB_REVISION", raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    frame = synthetic_series(norad_id=1, seed=5, burns=(Burn(45, "in_track_ms", 4.0),), n=90)
    out = maneuver_detect.detect(frame, model="bilstm-base")

    assert captured["repo_id"] == "astro-tools/maneuver-detect-bilstm-base"
    assert list(out.columns) == list(COLUMNS)


def test_dataset_path_uses_dataset_repo_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path / "recipe.json")

    monkeypatch.delenv("MANEUVER_DETECT_HUB_REVISION", raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    hub.dataset_path("recipe.json")
    assert captured["repo_id"] == hub.DATASET_REPO
    assert captured["repo_type"] == "dataset"
    assert captured["filename"] == "recipe.json"


def test_fetch_dataset_snapshots_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(tmp_path))
    assert hub.fetch_dataset() == tmp_path


def test_fetch_dataset_wraps_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> str:
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    with pytest.raises(hub.HubError, match="could not fetch the dataset"):
        hub.fetch_dataset()


def test_hf_api_returns_an_api() -> None:
    from huggingface_hub import HfApi

    assert isinstance(hub.hf_api(), HfApi)


def test_datasets_load_recipe_round_trips_from_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from maneuver_detect import datasets
    from maneuver_detect.datasets import recipe as build_recipe

    path = tmp_path / "recipe.json"
    path.write_text(build_recipe().to_json(), encoding="utf-8")
    monkeypatch.setattr("maneuver_detect.hub.dataset_path", lambda filename, *, revision=None: path)

    out = datasets.load_recipe()
    assert out.dataset_version == build_recipe().dataset_version
    assert out.norad_ids() == build_recipe().norad_ids()


def test_datasets_load_manifest_and_labels_from_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from maneuver_detect import datasets
    from maneuver_detect.datasets import Manifest, SeriesDigest
    from maneuver_detect.datasets.build import labels_to_json

    manifest = Manifest(
        dataset_version="0.2.0",
        digests=(SeriesDigest(norad_id=1, n_elsets=2, sha256="ab"),),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(labels_to_json([]), encoding="utf-8")

    def fake_dataset_path(filename: str, *, revision: object = None) -> Path:
        return manifest_path if filename == "manifest.json" else labels_path

    monkeypatch.setattr("maneuver_detect.hub.dataset_path", fake_dataset_path)

    assert datasets.load_manifest().dataset_version == "0.2.0"
    assert datasets.load_labels() == []


def test_datasets_fetch_dataset_delegates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from maneuver_detect import datasets

    monkeypatch.setattr("maneuver_detect.hub.fetch_dataset", lambda *, revision=None: tmp_path)
    assert datasets.fetch_dataset() == tmp_path
