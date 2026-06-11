"""Tests for ``maneuver_detect.cli`` argument parsing (no network is exercised)."""

from __future__ import annotations

import pytest

from maneuver_detect.cli import _build_parser


def test_detect_parses() -> None:
    args = _build_parser().parse_args(["detect", "25544"])
    assert args.command == "detect"
    assert args.target == "25544"
    assert args.model == "classical"


def test_dataset_build_parses() -> None:
    args = _build_parser().parse_args(["dataset", "build", "--out", "dist"])
    assert args.command == "dataset"
    assert args.dataset_command == "build"
    assert args.out == "dist"


def test_dataset_requires_an_action() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["dataset"])


def test_dataset_build_requires_out() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["dataset", "build"])
