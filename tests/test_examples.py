"""Smoke tests for the runnable examples in ``examples/``.

The documented examples must keep running against the shipped package. They are exercised as
subprocesses (not imported) so they stay out of the type-checked module graph and are tested through
their real ``python examples/...`` entry point. ``reproduce_baseline.py`` runs fully offline;
``detect_norad.py`` is tested on its credentials-absent path, the one branch that runs without a
Space-Track account or the network.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(
    script: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_EXAMPLES / script), *args],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )


def test_reproduce_baseline_runs_and_scores() -> None:
    result = _run("reproduce_baseline.py")
    assert result.returncode == 0, result.stderr
    # Scores a synthetic population through the benchmark; prints the headline metrics.
    assert "recall" in result.stdout
    assert "precision" in result.stdout
    assert "above-floor maneuver labels" in result.stdout


def test_train_bilstm_example_trains_and_scores() -> None:
    result = _run("train_bilstm.py")
    assert result.returncode == 0, result.stderr
    # Trains the BiLSTM on a synthetic population and scores it through the benchmark.
    assert "recall" in result.stdout
    assert "precision" in result.stdout
    assert "detections" in result.stdout


def test_detect_norad_guides_without_credentials() -> None:
    env = os.environ.copy()
    env.pop("SPACETRACK_USERNAME", None)
    env.pop("SPACETRACK_PASSWORD", None)
    result = _run("detect_norad.py", env=env)
    assert result.returncode == 0, result.stderr
    # No credentials, no network: it explains how to set them rather than failing.
    assert "SPACETRACK_USERNAME" in result.stdout
    assert "SPACETRACK_PASSWORD" in result.stdout


def test_train_bilstm_real_guides_without_credentials() -> None:
    env = os.environ.copy()
    env.pop("SPACETRACK_USERNAME", None)
    env.pop("SPACETRACK_PASSWORD", None)
    result = _run("train_bilstm_real.py", env=env)
    assert result.returncode == 0, result.stderr
    # The credentialed run needs Space-Track; with none it guides rather than failing or hanging.
    assert "SPACETRACK_USERNAME" in result.stdout
    assert "SPACETRACK_PASSWORD" in result.stdout


def test_train_transformer_example_trains_and_scores() -> None:
    result = _run("train_transformer.py")
    assert result.returncode == 0, result.stderr
    # Trains the transformer on a synthetic population and scores it through the benchmark.
    assert "recall" in result.stdout
    assert "precision" in result.stdout
    assert "detections" in result.stdout


def test_train_transformer_real_guides_without_credentials() -> None:
    env = os.environ.copy()
    env.pop("SPACETRACK_USERNAME", None)
    env.pop("SPACETRACK_PASSWORD", None)
    result = _run("train_transformer_real.py", env=env)
    assert result.returncode == 0, result.stderr
    # The credentialed run needs Space-Track; with none it guides rather than failing or hanging.
    assert "SPACETRACK_USERNAME" in result.stdout
    assert "SPACETRACK_PASSWORD" in result.stdout


def test_score_checkpoint_guides_without_credentials() -> None:
    env = os.environ.copy()
    env.pop("SPACETRACK_USERNAME", None)
    env.pop("SPACETRACK_PASSWORD", None)
    # Valid args so argparse passes; the re-score then needs Space-Track and guides without it.
    result = _run("score_checkpoint.py", "bilstm-base", "bilstm-base.pt", env=env)
    assert result.returncode == 0, result.stderr
    assert "SPACETRACK_USERNAME" in result.stdout
    assert "SPACETRACK_PASSWORD" in result.stdout


def test_foundation_residual_example_runs_offline() -> None:
    # Demonstrates the forecast-residual recipe with the dependency-free stand-in forecaster, so it
    # runs without the [foundation] extra, credentials, or the network.
    result = _run("foundation_residual.py")
    assert result.returncode == 0, result.stderr
    assert "forecast-residual" in result.stdout
    assert "detected" in result.stdout
    result.stdout.encode("ascii")  # the run-script stdout must stay ASCII on a cp1252 console


def test_example_stdout_stays_ascii() -> None:
    # The example output is smoke-tested on a cp1252 Windows console, so it must stay ASCII
    # (the delta-v column prints as ``delta_v_estimate``, never a literal Greek delta).
    result = _run("reproduce_baseline.py")
    try:
        result.stdout.encode("ascii")
    except UnicodeEncodeError as exc:  # pragma: no cover - the assertion message carries the detail
        pytest.fail(f"example stdout is not ASCII: {exc}")
