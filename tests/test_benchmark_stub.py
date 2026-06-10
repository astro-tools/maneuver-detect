"""Benchmark-layer stub.

The scorer-determinism and dataset-reconstruction checks land here as the dataset and benchmark
layers are built; for now this fixes the ``benchmark`` marker that the dedicated CI job runs.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.benchmark
def test_benchmark_package_imports() -> None:
    module = importlib.import_module("maneuver_detect.benchmark")
    assert module.__name__ == "maneuver_detect.benchmark"
