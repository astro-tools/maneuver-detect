"""Tests for foundation-bundle publishing — card generation, upload/tag, scoring gate, CLI routing.

The Hub API is faked (a recording stand-in), so these run offline: they assert the publisher creates
the repo, uploads the bundle and the generated card, moves the lockstep tag, refuses an unscored
bundle, and that the CLI routes a ``*-residual`` name to the foundation publisher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from maneuver_detect import hub
from maneuver_detect.datasets.catalogue import DATASET_VERSION
from maneuver_detect.models.foundation import FoundationBundle, save_foundation_bundle
from maneuver_detect.models.publish_foundation import build_foundation_card, publish_foundation

_TEST_REPORT = {
    "operating_point": 1.0,
    "ci_level": 0.95,
    "per_class": {
        "LEO": {"recall": 0.35, "precision": 0.86, "n_labels_above_floor": 23, "confusion": {}},
        "GEO": {"recall": 0.10, "precision": 0.89, "n_labels_above_floor": 120, "confusion": {}},
    },
}


@dataclass
class _Upload:
    repo_id: str
    repo_type: str
    path_in_repo: str
    content: Any


class _FakeApi:
    """A recording stand-in for ``huggingface_hub.HfApi`` — the methods the publisher calls."""

    def __init__(self) -> None:
        self.created: list[tuple[str, str]] = []
        self.uploads: list[_Upload] = []
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
        from huggingface_hub.errors import RevisionNotFoundError

        raise RevisionNotFoundError("no such tag")  # first publish — nothing to delete

    def create_tag(self, repo_id: str, *, tag: str, repo_type: str, **kw: Any) -> None:
        self.tagged.append((repo_id, repo_type, tag))

    def uploaded_names(self) -> set[str]:
        return {upload.path_in_repo for upload in self.uploads}


def _scored_bundle() -> FoundationBundle:
    return FoundationBundle(
        backend="chronos",
        checkpoint_id="amazon/chronos-bolt-small",
        revision="rev-9",
        context_length=64,
        class_thresholds={"LEO": 4.5, "MEO": 4.0, "GEO": 3.5},
        metadata={"dataset_version": "0.2.0", "mode": "zero-shot", "test_report": _TEST_REPORT},
    )


def test_build_foundation_card_renders_recipe_metrics_and_thresholds() -> None:
    card = build_foundation_card(_scored_bundle(), "chronos-residual")
    assert card.startswith("---\n")
    assert "license: mit" in card
    assert "- foundation-model" in card  # the foundation tag in the frontmatter
    assert f"- {hub.DATASET_REPO}" in card
    assert "# chronos-residual" in card
    assert "forecast-residual" in card
    assert "amazon/chronos-bolt-small" in card  # the pinned forecaster checkpoint
    assert "Apache-2.0" in card  # the forecaster licence
    assert "| Class | Residual-z threshold |" in card  # the calibrated operating point
    assert "| LEO | 4.50 |" in card
    # The held-out metrics render as the per-class table (shared with the torch card renderer).
    assert "| Class | Recall | Precision | Above-floor labels | Type acc |" in card
    assert "| LEO | 0.35 | 0.86 | 23 | — |" in card
    assert "`test_report`" not in card  # the nested report is the table, not a scalar cell


def test_publish_foundation_uploads_bundle_card_and_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeApi()
    monkeypatch.setattr(hub, "hf_api", lambda token=None: api)
    path = tmp_path / "chronos-residual.pt"
    save_foundation_bundle(_scored_bundle(), path)

    repo_id = publish_foundation("chronos-residual", path, token="x")

    assert repo_id == "astro-tools/maneuver-detect-chronos-residual"
    assert api.created == [("astro-tools/maneuver-detect-chronos-residual", "model")]
    assert api.uploaded_names() == {"chronos-residual.pt", "README.md"}
    by_name = {upload.path_in_repo: upload for upload in api.uploads}
    assert by_name["chronos-residual.pt"].content == str(path)
    assert b"license: mit" in by_name["README.md"].content
    assert api.tagged == [
        ("astro-tools/maneuver-detect-chronos-residual", "model", f"v{DATASET_VERSION}")
    ]


def test_publish_foundation_refuses_an_unscored_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeApi()
    monkeypatch.setattr(hub, "hf_api", lambda token=None: api)
    bundle = FoundationBundle(
        backend="timesfm",
        checkpoint_id="google/timesfm-2.5-200m-pytorch",
        revision="main",
        context_length=64,
        class_thresholds={"LEO": 4.0},
        metadata={"mode": "zero-shot"},  # no test_report
    )
    path = tmp_path / "timesfm-residual.pt"
    save_foundation_bundle(bundle, path)

    with pytest.raises(hub.HubError, match="no recorded held-out test metrics"):
        publish_foundation("timesfm-residual", path, token="x")
    assert api.uploads == []  # nothing uploaded before the refusal


def test_publish_foundation_unknown_model_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hub, "hf_api", lambda token=None: _FakeApi())
    with pytest.raises(hub.HubError, match="unknown model"):
        publish_foundation("nope", tmp_path / "x.pt", token="x")


def test_cli_models_publish_routes_residual_to_foundation_publisher(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from maneuver_detect import cli

    calls: dict[str, Any] = {}

    def fake_publish(
        name: str, bundle: str, *, token: str | None, version: str, allow_unscored: bool
    ) -> str:
        calls.update(name=name, bundle=bundle, allow_unscored=allow_unscored)
        return f"astro-tools/maneuver-detect-{name}"

    monkeypatch.setattr(
        "maneuver_detect.models.publish_foundation.publish_foundation", fake_publish
    )

    rc = cli.main(
        ["models", "publish", "chronos-residual", str(tmp_path / "c.pt"), "--allow-unscored"]
    )

    assert rc == 0
    assert calls["name"] == "chronos-residual"
    assert calls["allow_unscored"] is True
    assert "published chronos-residual" in capsys.readouterr().out


def test_cli_calibrate_foundation_guides_without_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from maneuver_detect import cli

    monkeypatch.delenv("SPACETRACK_USERNAME", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    # No credentials: it explains how to set them and exits 0 rather than reconstructing or hanging.
    rc = cli.main(["models", "calibrate-foundation", "chronos", "chronos-residual.pt"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SPACETRACK_USERNAME" in out
    assert "SPACETRACK_PASSWORD" in out
