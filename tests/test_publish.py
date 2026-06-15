"""Tests for Hugging Face Hub publishing — checkpoint + dataset upload, tagging, and card output.

The Hub API is faked (a recording :class:`HfApi` stand-in), so these run offline: they assert
that the publishers create the repo, upload the bundle / artifacts and the generated card, and move
the lockstep version tag, and that the cards carry the expected frontmatter and provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from _synthetic import Burn, object_series
from maneuver_detect import hub
from maneuver_detect.benchmark.scoring import score
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

    # The synthetic fixture is unscored, so allow it through — the scoring gate is its own test.
    repo_id = publish_checkpoint("bilstm-base", bundle_path, token="x", allow_unscored=True)

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
    publish_checkpoint("bilstm-base", bundle_path, token="x", allow_unscored=True)
    assert api.tagged == [
        ("astro-tools/maneuver-detect-bilstm-base", "model", f"v{DATASET_VERSION}")
    ]


def test_publish_checkpoint_refuses_an_unscored_bundle(
    bundle_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A checkpoint with no recorded held-out test metrics would render a blank-metrics card; refuse
    # it by default (D8 — the card must document measured performance) and upload nothing.
    api = _FakeApi()
    _install_fake(monkeypatch, api)
    with pytest.raises(hub.HubError, match="no recorded held-out test metrics"):
        publish_checkpoint("bilstm-base", bundle_path, token="x")
    assert api.uploads == []  # nothing was uploaded before the refusal


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


def test_build_model_card_renders_test_metrics_table(bundle_path: Path) -> None:
    test_report = {
        "operating_point": 1.0,
        "ci_level": 0.95,
        "per_class": {
            "GEO": {
                "recall": 0.10,
                "precision": 0.89,
                "operating_point_confidence": 0.66,
                "n_labels_above_floor": 120,
                "confusion": {},
            },
            "LEO": {
                "recall": 0.35,
                "precision": 0.86,
                "operating_point_confidence": 0.74,
                "n_labels_above_floor": 23,
                "confusion": {
                    "in_track": {"in_track": 8, "cross_track": 1, "radial": 0},
                    "cross_track": {"in_track": 0, "cross_track": 2, "radial": 0},
                    "radial": {"in_track": 0, "cross_track": 0, "radial": 0},
                },
            },
        },
    }
    bundle = replace(load_bundle(bundle_path), metadata={"seed": 0, "test_report": test_report})
    card = build_model_card(bundle, "bilstm-base")

    assert "| Class | Recall | Precision | Operating pt | Above-floor labels | Type acc |" in card
    assert "| LEO | 0.35 | 0.86 | 0.74 | 23 | 0.91 |" in card  # type acc = 10/11
    assert "| GEO | 0.10 | 0.89 | 0.66 | 120 | — |" in card  # empty confusion -> no type acc
    assert "false alarm(s)/satellite-year" in card
    assert "(95% CI)" in card
    assert card.index("| LEO |") < card.index("| GEO |")  # altitude order, not alphabetical
    # The nested report is rendered as the table, not dumped into the scalar provenance table.
    assert "`test_report`" not in card


def test_build_model_card_renders_per_class_thresholds(bundle_path: Path) -> None:
    base = load_bundle(bundle_path)
    # An untuned bundle (no per-class gates) carries no per-class threshold line.
    assert "Per-class detection thresholds" not in build_model_card(base, "bilstm-base")
    tuned = replace(base, class_thresholds={"GEO": 0.3, "LEO": 0.6})
    card = build_model_card(tuned, "bilstm-base")
    assert "Per-class detection thresholds" in card
    assert "GEO 0.300" in card
    assert card.index("LEO 0.600") < card.index("GEO 0.300")  # altitude order, not alphabetical


def test_build_model_card_renders_calibration_when_present(bundle_path: Path) -> None:
    from maneuver_detect.calibration import BundledCalibration, ReliabilityBin, ReliabilityCurve

    base = load_bundle(bundle_path)
    # An uncalibrated bundle carries no calibration section.
    assert "emits **calibrated** confidence" not in build_model_card(base, "bilstm-base")

    calibration = BundledCalibration(
        temperature=1.8,
        conformal_q=0.5,
        conformal_alpha=0.1,
        reliability={"LEO": ReliabilityCurve(bins=(ReliabilityBin(0.0, 0.5, 4, 0.3, 0.25),))},
        ece={"LEO": 0.12, "GEO": 0.04},
    )
    card = build_model_card(replace(base, calibration=calibration), "bilstm-base")
    assert "emits **calibrated** confidence" in card
    assert "T = 1.800" in card
    assert "90%" in card  # conformal marginal coverage 1 - alpha
    assert "| LEO | 0.120 |" in card
    assert card.index("| LEO | 0.120 |") < card.index("| GEO | 0.040 |")  # altitude order


def test_build_model_card_test_table_matches_real_report_schema(bundle_path: Path) -> None:
    # Guards against drift between ScoreReport.to_json() and the card renderer: a real (empty)
    # report round-trips through metadata and renders an all-undefined table without error.
    report = score([], [], [])
    bundle = replace(
        load_bundle(bundle_path),
        metadata={"test_report": json.loads(report.to_json())},
    )
    card = build_model_card(bundle, "bilstm-base")
    assert "| Class | Recall | Precision | Operating pt | Above-floor labels | Type acc |" in card
    # no labels -> undefined recall/precision/operating-point/type acc
    assert "| LEO | — | — | — | 0 | — |" in card


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


def test_build_dataset_card_documents_v03_classes_and_source_terms() -> None:
    card = build_dataset_card("0.3.0")
    assert "dataset/v0.3" in card
    # The five-class scope, including the v0.3 additions.
    assert "**IGSO**" in card and "**HEO**" in card
    # The new source attributions stack under CC-BY-4.0.
    assert "NOAA" in card  # GOES navsum, US-Government public domain
    assert "Quasi-Zenith Satellite System website" in card  # QZSS reuse attribution
    assert "© EU" in card  # Galileo NAGU attribution carried forward


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

    def fake_publish(
        name: str, bundle: str, *, token: str | None, version: str, allow_unscored: bool
    ) -> str:
        calls.update(
            name=name, bundle=bundle, token=token, version=version, allow_unscored=allow_unscored
        )
        return f"astro-tools/maneuver-detect-{name}"

    monkeypatch.setattr("maneuver_detect.models.publish.publish_checkpoint", fake_publish)

    rc = cli.main(["models", "publish", "bilstm-base", str(tmp_path / "b.pt"), "--allow-unscored"])

    assert rc == 0
    assert calls["name"] == "bilstm-base"
    assert calls["version"] == DATASET_VERSION
    assert calls["allow_unscored"] is True
    assert "published bilstm-base" in capsys.readouterr().out
