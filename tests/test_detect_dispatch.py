"""Tests for detector registration and ``detect()`` dispatch."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest

import maneuver_detect
from maneuver_detect import Detector
from maneuver_detect.detectors import (
    _REGISTRY,
    available_models,
    get_detector,
    register_detector,
)
from maneuver_detect.schema import Maneuver, ManeuverType, from_frame, to_frame


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Snapshot the global detector registry and restore it after the test."""
    snapshot = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


def test_detector_is_abstract() -> None:
    with pytest.raises(TypeError):
        Detector()  # type: ignore[abstract]


def test_register_and_dispatch(clean_registry: None) -> None:
    sentinel = Maneuver(
        epoch=pd.Timestamp("2024-01-02T00:00:00", tz="UTC"),
        confidence=1.0,
        type=ManeuverType.IN_TRACK,
        delta_v_estimate=0.1,
        norad_id=99999,
        elset_epoch_before=pd.Timestamp("2024-01-01T00:00:00", tz="UTC"),
        elset_epoch_after=pd.Timestamp("2024-01-03T00:00:00", tz="UTC"),
    )

    @register_detector
    class _DummyDetector(Detector):
        name = "dummy-test"

        def detect(self, history: pd.DataFrame) -> pd.DataFrame:
            return to_frame([sentinel])

    assert "dummy-test" in available_models()
    assert isinstance(get_detector("dummy-test"), _DummyDetector)
    result = maneuver_detect.detect(pd.DataFrame(), model="dummy-test")
    assert from_frame(result) == [sentinel]


def test_register_rejects_name_clash(clean_registry: None) -> None:
    @register_detector
    class _First(Detector):
        name = "clash"

        def detect(self, history: pd.DataFrame) -> pd.DataFrame:
            return to_frame([])

    with pytest.raises(ValueError, match="already registered"):

        @register_detector
        class _Second(Detector):
            name = "clash"

            def detect(self, history: pd.DataFrame) -> pd.DataFrame:
                return to_frame([])


def test_unknown_model_raises_listing_available() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        maneuver_detect.detect(pd.DataFrame(), model="does-not-exist")
