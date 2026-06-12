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


def _run(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_EXAMPLES / script)],
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


def test_detect_norad_guides_without_credentials() -> None:
    env = os.environ.copy()
    env.pop("SPACETRACK_USERNAME", None)
    env.pop("SPACETRACK_PASSWORD", None)
    result = _run("detect_norad.py", env=env)
    assert result.returncode == 0, result.stderr
    # No credentials, no network: it explains how to set them rather than failing.
    assert "SPACETRACK_USERNAME" in result.stdout
    assert "SPACETRACK_PASSWORD" in result.stdout


def test_example_stdout_stays_ascii() -> None:
    # The example output is smoke-tested on a cp1252 Windows console, so it must stay ASCII
    # (the delta-v column prints as ``delta_v_estimate``, never a literal Greek delta).
    result = _run("reproduce_baseline.py")
    try:
        result.stdout.encode("ascii")
    except UnicodeEncodeError as exc:  # pragma: no cover - the assertion message carries the detail
        pytest.fail(f"example stdout is not ASCII: {exc}")
