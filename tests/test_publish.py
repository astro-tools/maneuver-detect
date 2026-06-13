"""Tests for Hugging Face Hub publishing — checkpoint + dataset upload, tagging, and card output.

The Hub API is faked (a recording :class:`HfApi` stand-in), so these run offline: they assert
that the publishers create the repo, upload the bundle / artifacts and the generated card, and move
the lockstep version tag, and that the cards carry the expected frontmatter and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from _synthetic import Burn, object_series
from maneuver_detect import hub
from maneuver_detect.datasets.catalogue import DATASET_VERSION
from maneuver_detect.datasets.publish import build_dataset_card, publish_dataset
from maneuver_detect.models.bilstm import BiLstmConfig
from maneuver_detect.models.checkpoint import load_bundle, save_bundle
from maneuver_detect.models.publish import build_model_card, publish_checkpoint
from maneuver_detect.models.train import train_bilstm

_CONFIG = BiLstmConfig(hidden_size=8, num_layers=1, dropout=0.0)


@dataclass
class _Upload:
    repo_id: str
    repo_type: str
    path_in_repo: str
    content: Any


class _FakeApi:
    """A recording stand-in for ``huggingface_hub.HfApi`` — the methods the publishers call."""

    def __init__(self, *, raise_on_delete: bool = False) -> None:
        self.raise_on_delete = raise_on_delete
        self.created: list[tuple[str, str]] = []
        self.uploads: list[_Upload] = []
        self.deleted: list[tuple[str, str, str]] = []
        self.tagged: list[tuple[str, str, str]] = []

    def create_repo(
        self, repo_id: str, *, repo_type: str, exist_ok: bool = False, **kw: Any
    ) -> None:
        self.created.append((repo_id, repo_type))

    def upload_file(
        self,
        *,
        path_or_fileobj: Any,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str | None = None,
        **kw: Any,
    ) -> None:
        self.uploads.append(_Upload(repo_id, repo_type, path_in_repo, path_or_fileobj))

    def delete_tag(self, repo_id: str, *, tag: str, repo_type: str) -> None:
        if self.raise_on_delete:
            from huggingface_hub.errors import RevisionNotFoundError

            raise RevisionNotFoundError("no such tag")
        self.deleted.append((repo_id, repo_type, tag))

    def create_tag(self, repo_id: str, *, tag: str, repo_type: str, **kw: Any) -> None:
        self.tagged.append((repo_id, repo_type, tag))

    def uploaded_names(self) -> set[str]:
        return {upload.path_in_repo for upload in self.uploads}


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
    path = tmp_path_factory.mktemp("publish") / "bilstm-base.pt"
    save_bundle(bundle, path)
    return path


def _install_fake(monkeypatch: pytest.MonkeyPatch, api: _FakeApi) -> None:
    monkeypatch.setattr(hub, "hf_api", lambda token=None: api)


def _write_dataset(directory: Path, names: tuple[str, ...]) -> None:
    for name in names:
        (directory / name).write_text("{}\n", encoding="utf-8")


def test_publish_checkpoint_uploads_bundle_card_and_tag(
    bundle_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeApi()
    _install_fake(monkeypatch, api)

    repo_id = publish_checkpoint("bilstm-base", bundle_path, token="x")

    assert repo_id == "astro-tools/maneuver-detect-bilstm-base"
    assert api.created == [("astro-tools/maneuver-detect-bilstm-base", "model")]
    assert api.uploaded_names() == {"bilstm-base.pt", "README.md"}
    by_name = {upload.path_in_repo: upload for upload in api.uploads}
    assert by_name["bilstm-base.pt"].content == str(bundle_path)  # the .pt is uploaded by path
    assert b"license: mit" in by_name["README.md"].content  # the card is uploaded as bytes
    assert api.tagged == [
        ("astro-tools/maneuver-detect-bilstm-base", "model", f"v{DATASET_VERSION}")
    ]


def test_publish_checkpoint_unknown_model_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeApi()
    _install_fake(monkeypatch, api)
    with pytest.raises(hub.HubError, match="unknown model"):
        publish_checkpoint("nope", tmp_path / "x.pt", token="x")


def test_publish_first_time_tolerates_missing_tag(
    bundle_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a first publish there is no tag to delete; move_tag must tolerate that and still create it.
    api = _FakeApi(raise_on_delete=True)
    _install_fake(monkeypatch, api)
    publish_checkpoint("bilstm-base", bundle_path, token="x")
    assert api.tagged == [
        ("astro-tools/maneuver-detect-bilstm-base", "model", f"v{DATASET_VERSION}")
    ]


def test_build_model_card_renders_frontmatter_and_provenance(bundle_path: Path) -> None:
    bundle = load_bundle(bundle_path)
    card = build_model_card(bundle, "bilstm-base")
    assert card.startswith("---\n")
    assert "license: mit" in card
    assert f"- {hub.DATASET_REPO}" in card  # datasets: linkage in the frontmatter
    assert "# bilstm-base" in card
    assert "Intended use" in card
    assert "| field | value |" in card  # the provenance table
    assert "`seed`" in card  # a recorded provenance field


def test_publish_dataset_uploads_card_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeApi()
    _install_fake(monkeypatch, api)
    _write_dataset(tmp_path, ("recipe.json", "labels.json", "manifest.json", "splits.json"))

    repo_id = publish_dataset(tmp_path, token="x")

    assert repo_id == hub.DATASET_REPO
    assert api.created == [(hub.DATASET_REPO, "dataset")]
    assert api.uploaded_names() == {
        "README.md",
        "recipe.json",
        "labels.json",
        "manifest.json",
        "splits.json",
    }
    assert api.tagged == [(hub.DATASET_REPO, "dataset", f"v{DATASET_VERSION}")]


def test_publish_dataset_skips_absent_optional_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeApi()
    _install_fake(monkeypatch, api)
    _write_dataset(tmp_path, ("recipe.json", "labels.json", "manifest.json"))  # no splits.json

    publish_dataset(tmp_path, token="x")

    assert "splits.json" not in api.uploaded_names()
    assert {"recipe.json", "labels.json", "manifest.json", "README.md"} <= api.uploaded_names()


def test_publish_dataset_empty_dir_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = _FakeApi()
    _install_fake(monkeypatch, api)
    with pytest.raises(hub.HubError, match="no dataset artifacts"):
        publish_dataset(tmp_path, token="x")


def test_build_dataset_card_has_frontmatter_and_minor_dir() -> None:
    card = build_dataset_card("0.2.0")
    assert "license: cc-by-4.0" in card
    assert "dataset/v0.2" in card
    assert "recipe-first" in card.lower()


def test_cli_dataset_publish_defaults_dir_and_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from maneuver_detect import cli

    calls: dict[str, Any] = {}

    def fake_publish(dataset_dir: str, *, version: str, repo_id: str, token: str | None) -> str:
        calls.update(dataset_dir=dataset_dir, version=version, repo_id=repo_id, token=token)
        return repo_id

    monkeypatch.setattr("maneuver_detect.datasets.publish.publish_dataset", fake_publish)

    rc = cli.main(["dataset", "publish", "--token", "tok"])

    assert rc == 0
    minor = ".".join(DATASET_VERSION.split(".")[:2])
    assert calls == {
        "dataset_dir": f"dataset/v{minor}",
        "version": DATASET_VERSION,
        "repo_id": hub.DATASET_REPO,
        "token": "tok",
    }
    assert "published dataset" in capsys.readouterr().out


def test_cli_models_publish_dispatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from maneuver_detect import cli

    calls: dict[str, Any] = {}

    def fake_publish(name: str, bundle: str, *, token: str | None, version: str) -> str:
        calls.update(name=name, bundle=bundle, token=token, version=version)
        return f"astro-tools/maneuver-detect-{name}"

    monkeypatch.setattr("maneuver_detect.models.publish.publish_checkpoint", fake_publish)

    rc = cli.main(["models", "publish", "bilstm-base", str(tmp_path / "b.pt")])

    assert rc == 0
    assert calls["name"] == "bilstm-base"
    assert calls["version"] == DATASET_VERSION
    assert "published bilstm-base" in capsys.readouterr().out
