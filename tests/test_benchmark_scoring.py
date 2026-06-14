"""Tests for ``maneuver_detect.benchmark.scoring`` — the deterministic scorer and its file I/O.

The load-bearing guarantee (D8) is that a committed predictions file reproduces committed scores
identically across runs and platforms. The committed ``tests/data/benchmark`` fixtures pin that: the
scorer's report must equal ``scores.json`` byte-for-byte, scoring twice must agree, and the
predictions file must round-trip through the schema unchanged. The toy scenario exercises every path
the metric has — a true positive, a false alarm, a miss, a below-floor recovery, an epoch-only
match, a type confusion, and a class with no labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from maneuver_detect.benchmark.matching import ScoredLabel
from maneuver_detect.benchmark.metrics import ObjectExposure
from maneuver_detect.benchmark.scoring import predictions_to_json, read_predictions, score
from maneuver_detect.labels.labeller import LabelledInterval
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import Maneuver, ManeuverType, to_frame

pytestmark = pytest.mark.benchmark

_DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "benchmark"
_PREDICTIONS = _DATA_DIR / "predictions.json"
_SCORES = _DATA_DIR / "scores.json"


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _interval(
    norad_id: int,
    epoch: str,
    gap_start: str,
    gap_end: str,
    tol_start: str,
    tol_end: str,
    *,
    orbit_class: OrbitClass,
    maneuver_type: ManeuverType | None,
    delta_v: float | None,
) -> LabelledInterval:
    return LabelledInterval(
        norad_id=norad_id,
        epoch=_ts(epoch),
        elset_epoch_before=_ts(gap_start),
        elset_epoch_after=_ts(gap_end),
        tol_start=_ts(tol_start),
        tol_end=_ts(tol_end),
        maneuver_type=maneuver_type,
        delta_v=delta_v,
        source="TEST",
        source_ref=f"{norad_id}:{epoch}",
        orbit_class=orbit_class,
    )


def _scenario_labels() -> list[ScoredLabel]:
    """The held-out labels of the toy scenario (LEO with a below-floor miss; MEO epoch-only)."""
    return [
        # LEO 11111 — three above-floor maneuvers (one undetected) and one below-floor maneuver.
        ScoredLabel(
            _interval(
                11111,
                "2024-01-03T12:00:00",
                "2024-01-03",
                "2024-01-04",
                "2024-01-02",
                "2024-01-05",
                orbit_class=OrbitClass.LEO,
                maneuver_type=ManeuverType.IN_TRACK,
                delta_v=0.5,
            )
        ),
        ScoredLabel(
            _interval(
                11111,
                "2024-01-06T12:00:00",
                "2024-01-06",
                "2024-01-07",
                "2024-01-05",
                "2024-01-08",
                orbit_class=OrbitClass.LEO,
                maneuver_type=ManeuverType.CROSS_TRACK,
                delta_v=0.3,
            )
        ),
        ScoredLabel(
            _interval(
                11111,
                "2024-01-09T12:00:00",
                "2024-01-09",
                "2024-01-10",
                "2024-01-08",
                "2024-01-11",
                orbit_class=OrbitClass.LEO,
                maneuver_type=ManeuverType.IN_TRACK,
                delta_v=0.001,
            ),
            above_floor=False,
        ),
        ScoredLabel(
            _interval(
                11111,
                "2024-01-13T12:00:00",
                "2024-01-13",
                "2024-01-14",
                "2024-01-12",
                "2024-01-15",
                orbit_class=OrbitClass.LEO,
                maneuver_type=ManeuverType.RADIAL,
                delta_v=0.4,
            )
        ),
        # MEO 22222 — one epoch-only maneuver (no announced type).
        ScoredLabel(
            _interval(
                22222,
                "2024-01-04T12:00:00",
                "2024-01-04",
                "2024-01-05",
                "2024-01-03",
                "2024-01-06",
                orbit_class=OrbitClass.MEO,
                maneuver_type=None,
                delta_v=None,
            )
        ),
    ]


def _scenario_exposure() -> list[ObjectExposure]:
    return [
        ObjectExposure(11111, OrbitClass.LEO, 2.0),
        ObjectExposure(22222, OrbitClass.MEO, 1.0),
        ObjectExposure(33333, OrbitClass.GEO, 0.5),  # observed, never labelled — only a false alarm
    ]


def _scenario_detections() -> list[Maneuver]:
    """The predictions of the frozen toy scenario — kept in lockstep with ``predictions.json``."""

    def man(
        norad_id: int,
        epoch: str,
        confidence: float,
        mtype: ManeuverType,
        delta_v: float | None,
        before: str,
        after: str,
    ) -> Maneuver:
        return Maneuver(
            epoch=_ts(epoch),
            confidence=confidence,
            type=mtype,
            delta_v_estimate=delta_v,
            norad_id=norad_id,
            elset_epoch_before=_ts(before),
            elset_epoch_after=_ts(after),
        )

    return [
        man(
            11111,
            "2024-01-03T12:00:00",
            0.9,
            ManeuverType.IN_TRACK,
            0.48,
            "2024-01-03",
            "2024-01-04",
        ),
        man(
            11111,
            "2024-01-21T12:00:00",
            0.7,
            ManeuverType.IN_TRACK,
            1.0,
            "2024-01-21",
            "2024-01-22",
        ),
        man(
            11111,
            "2024-01-06T12:00:00",
            0.6,
            ManeuverType.IN_TRACK,
            0.2,
            "2024-01-06",
            "2024-01-07",
        ),
        man(
            11111,
            "2024-01-09T12:00:00",
            0.5,
            ManeuverType.IN_TRACK,
            0.05,
            "2024-01-09",
            "2024-01-10",
        ),
        man(
            22222,
            "2024-01-04T12:00:00",
            0.6,
            ManeuverType.CROSS_TRACK,
            None,
            "2024-01-04",
            "2024-01-05",
        ),
        man(
            33333,
            "2024-01-02T12:00:00",
            0.55,
            ManeuverType.RADIAL,
            None,
            "2024-01-02",
            "2024-01-03",
        ),
    ]


# --- the frozen-artifact reproducibility guarantee (D8) -------------------------------------------


def test_scorer_reproduces_the_committed_scores() -> None:
    predictions = read_predictions(_PREDICTIONS.read_text(encoding="utf-8"))
    report = score(predictions, _scenario_labels(), _scenario_exposure())
    assert report.to_json() == _SCORES.read_text(encoding="utf-8")


def test_scoring_is_byte_stable_across_runs() -> None:
    predictions = read_predictions(_PREDICTIONS.read_text(encoding="utf-8"))
    first = score(predictions, _scenario_labels(), _scenario_exposure())
    second = score(predictions, _scenario_labels(), _scenario_exposure())
    assert first.to_json() == second.to_json()


def test_predictions_file_round_trips_and_matches_the_scenario() -> None:
    text = _PREDICTIONS.read_text(encoding="utf-8")
    # The committed file is exactly the canonical serialisation of the in-code scenario predictions.
    assert predictions_to_json(_scenario_detections()) == text
    # And it round-trips through the schema unchanged.
    assert predictions_to_json(read_predictions(text)) == text


def test_dataframe_and_record_inputs_score_identically() -> None:
    detections = _scenario_detections()
    from_records = score(detections, _scenario_labels(), _scenario_exposure())
    from_frame = score(to_frame(detections), _scenario_labels(), _scenario_exposure())
    assert from_frame.to_json() == from_records.to_json()


# --- the scored numbers, spot-checked against the hand-counted scenario ------------------------


def test_headline_numbers_match_the_hand_count() -> None:
    report = score(_scenario_detections(), _scenario_labels(), _scenario_exposure())

    leo = report.per_class[OrbitClass.LEO]
    # 3 above-floor labels, 2 recovered at 1 FA/sat-year (the third is the undetected one); one FP
    # is affordable within the 2-sat-year budget, so precision is 2/3.
    assert leo.recall == pytest.approx(2 / 3)
    assert leo.precision == pytest.approx(2 / 3)
    assert leo.full_population_recall == pytest.approx(
        0.75
    )  # +1 below-floor recovery over 4 labels
    assert leo.confusion.counts[ManeuverType.CROSS_TRACK][ManeuverType.IN_TRACK] == 1

    meo = report.per_class[OrbitClass.MEO]
    assert meo.recall == pytest.approx(1.0)
    assert meo.confusion.total() == 0  # the only match is epoch-only, so it is untyped

    geo = report.per_class[OrbitClass.GEO]
    assert geo.recall is None  # no GEO labels — recall is undefined, not zero

    assert report.headline()[OrbitClass.LEO] == pytest.approx(2 / 3)


# --- the human-readable summary and the predictions-file validation contract ----------------------


def test_summary_renders_every_class_with_na_for_undefined_recall() -> None:
    report = score(_scenario_detections(), _scenario_labels(), _scenario_exposure())
    text = report.summary()
    assert "FA/sat-year" in text  # the header carries the operating point
    for orbit_class in (OrbitClass.LEO, OrbitClass.MEO, OrbitClass.GEO):
        assert orbit_class.value in text
    assert "recall=0.667" in text  # LEO recall 2/3 to three decimals
    assert "recall=n/a" in text  # GEO recall is undefined (no labels) — exercises _fmt(None)


def test_read_predictions_rejects_a_record_missing_a_canonical_field() -> None:
    # A leaderboard submission missing a required column is a hard error, not a silent drop.
    bad = '[{"epoch": "2024-01-01T00:00:00+00:00", "confidence": 0.9}]'
    with pytest.raises(ValueError, match="missing canonical fields"):
        read_predictions(bad)


def test_read_predictions_enforces_a_fixed_schema_both_ways() -> None:
    # The schema is fixed in both directions (the D12 fixed-schema integrity surface): a record
    # carrying any field beyond the canonical columns is rejected, so a submission cannot smuggle a
    # query or any other payload past the reader even when every required field is present.
    smuggled = json.loads(predictions_to_json(_scenario_detections()[:1]))
    smuggled[0]["give_me"] = "the labels"
    with pytest.raises(ValueError, match="unknown fields"):
        read_predictions(json.dumps(smuggled))

    # A non-array payload, or a record that is not a JSON object, is rejected the same way rather
    # than iterated over and mis-parsed.
    with pytest.raises(ValueError, match="must be a JSON array"):
        read_predictions('{"give_me": "the labels"}')
    with pytest.raises(ValueError, match="must be a JSON object"):
        read_predictions('["SELECT * FROM labels"]')


# --- per-class confidence intervals -------------------------------------------------------------


def test_report_carries_per_class_confidence_intervals_and_level() -> None:
    report = score(_scenario_detections(), _scenario_labels(), _scenario_exposure())
    assert report.ci_level == pytest.approx(0.95)

    leo = report.per_class[OrbitClass.LEO]
    assert leo.ci_level == pytest.approx(0.95)
    assert leo.recall is not None
    assert leo.recall_ci is not None
    low, high = leo.recall_ci
    assert low < leo.recall < high  # the headline estimate sits inside its interval

    # MEO is a perfect 1-of-1: an honest interval whose upper bound is exactly 1.0, lower below it.
    meo = report.per_class[OrbitClass.MEO]
    assert meo.recall == pytest.approx(1.0)
    assert meo.recall_ci is not None
    assert meo.recall_ci[1] == pytest.approx(1.0)
    assert meo.recall_ci[0] < 1.0

    # GEO has no above-floor labels — its interval is undefined, not a spurious zero-width one.
    assert report.per_class[OrbitClass.GEO].recall_ci is None


def test_summary_shows_the_interval_and_its_level() -> None:
    text = score(_scenario_detections(), _scenario_labels(), _scenario_exposure()).summary()
    assert "95% CI" in text  # the header names the confidence level
    assert "recall=0.667 [0.208, 0.939]" in text  # LEO carries its bracketed interval
    assert "recall=n/a precision=n/a" in text  # an undefined estimate renders without a bracket


def test_ci_level_is_configurable_and_widens_the_interval() -> None:
    args = (_scenario_detections(), _scenario_labels(), _scenario_exposure())
    narrow = score(*args, ci_level=0.90).per_class[OrbitClass.LEO].recall_ci
    wide = score(*args, ci_level=0.99).per_class[OrbitClass.LEO].recall_ci
    assert narrow is not None
    assert wide is not None
    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_invalid_ci_level_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="ci_level"):
        score(_scenario_detections(), _scenario_labels(), _scenario_exposure(), ci_level=bad)
