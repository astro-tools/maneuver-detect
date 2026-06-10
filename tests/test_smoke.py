"""Import-surface smoke tests — runnable without data or model weights."""

from __future__ import annotations

import pytest

import maneuver_detect
from maneuver_detect import cli


@pytest.mark.smoke
def test_version_is_a_nonempty_string() -> None:
    assert isinstance(maneuver_detect.__version__, str)
    assert maneuver_detect.__version__


@pytest.mark.smoke
def test_public_surface_is_exposed() -> None:
    assert callable(maneuver_detect.detect)
    assert hasattr(maneuver_detect, "datasets")
    assert callable(maneuver_detect.datasets.tle_history)


@pytest.mark.smoke
def test_cli_version_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
