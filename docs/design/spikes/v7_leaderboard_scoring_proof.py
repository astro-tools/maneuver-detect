"""V7 proof — the leaderboard's server-side scoring leg, run against hidden labels.

A documented dry-run of the one operation the public leaderboard performs on every submission:
take a user's predictions file, score it against **held-out labels the host never reveals**, and
return a rank. It reuses the *real* deterministic scorer
(:mod:`maneuver_detect.benchmark.scoring`) — the same code path the Hugging Face Space would call —
so the integrity properties shown here are properties of the shipped scorer, not a mock.

It demonstrates the five claims D12 rests on:

  1. **Hidden labels never leave the host.** The only thing returned is an aggregate score; the
     held-out label epochs do not appear anywhere in the payload.
  2. **The response is aggregate-only.** Per-class recall at the operating point plus the published
     D11 timing-only "cheating floor" — never the per-label match table the scorer computes
     internally (which would hand an attacker the answer key one row at a time).
  3. **The submission channel can't carry a query.** The fixed-schema parser
     (:func:`~maneuver_detect.benchmark.scoring.read_predictions`) accepts only canonical maneuver
     records and rejects anything else, so a submission cannot smuggle a label-exfiltration request.
  4. **Scoring is byte-deterministic (D8).** The same submission scores identically across runs, so
     the board is reproducible and a re-submission can't probe numerical noise.
  5. **The probing attack is bounded.** A single-detection oracle attack — submit one detection per
     candidate gap and watch whether recall ticks up — does recover public-split labels, but needs
     one submission per candidate gap: at ``R`` scored submissions/user/day, exfiltrating ``G``
     candidate gaps takes ``ceil(G / R)`` submission-days, is detectable as anomalous submission
     volume, and only ever touches the *public* split — the private split (scored once at release)
     is never exposed.

stdlib + the installed package only; no network, no GPU, deterministic across runs (no RNG, fixed
synthetic data — catalogue ids 9000x are fictional, so the proof ships no redistributed TLEs, the
V1/D2 practice).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pandas as pd

from maneuver_detect.benchmark.matching import ScoredLabel
from maneuver_detect.benchmark.metrics import ObjectExposure
from maneuver_detect.benchmark.scoring import (
    ScoreReport,
    predictions_to_json,
    read_predictions,
    score,
)
from maneuver_detect.labels.labeller import LabelledInterval
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.schema import Maneuver, ManeuverType

# The published D11 "cheating floor": the AUC a Δt-only ("timing alone") model reaches, the score
# any submission must beat to be doing more than reading gap lengths. A benchmark constant, computed
# server-side once and published with the protocol — NOT derived from a submission. Shown alongside
# every score so the board contextualises a result without exposing a single label.
PUBLISHED_TIMING_FLOOR = {"LEO": 0.62, "GEO": 0.68}  # timing-only rank-AUC (V5/D11)

DAY = pd.Timedelta(days=1)
SUBMISSIONS_PER_USER_PER_DAY = 5  # the D12 cadence


# --------------------------------------------------------------------------------------------------
# The host's secret: the held-out benchmark. In production this is a Space secret / private HF
# Dataset; here it is built inline from fictional elements so the artifact is self-contained.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HiddenBenchmark:
    """The server-side held-out set — labels + exposure that never appear in any returned payload."""

    labels: list[ScoredLabel]
    exposure: list[ObjectExposure]
    # The per-object daily gap grid an attacker could probe (the public attack surface).
    candidate_gaps: dict[int, list[pd.Timestamp]]


def _label(
    norad_id: int,
    day: int,
    orbit_class: OrbitClass,
    maneuver_type: ManeuverType | None,
    delta_v: float | None,
    *,
    above_floor: bool = True,
) -> ScoredLabel:
    """A held-out label on the gap [day, day+1), with the D4 ±1-adjacent-gap matching window."""
    base = pd.Timestamp("2026-03-01T00:00:00", tz="UTC")
    gap_start = base + day * DAY
    gap_end = gap_start + DAY
    return ScoredLabel(
        LabelledInterval(
            norad_id=norad_id,
            epoch=gap_start + pd.Timedelta(hours=12),
            elset_epoch_before=gap_start,
            elset_epoch_after=gap_end,
            tol_start=gap_start - DAY,  # ±1 adjacent gap (D4)
            tol_end=gap_end + DAY,
            maneuver_type=maneuver_type,
            delta_v=delta_v,
            source="SYNTH",
            source_ref=f"{norad_id}:{day}",
            orbit_class=orbit_class,
        ),
        above_floor=above_floor,
    )


def build_hidden_benchmark() -> HiddenBenchmark:
    """A two-object held-out set: a LEO and a GEO object, each observed daily for 30 days.

    Labels are spaced ≥3 gaps apart so the D4 ±1 tolerance gives each its own oracle cluster.
    """
    base = pd.Timestamp("2026-03-01T00:00:00", tz="UTC")
    span_days = 30
    candidate_gaps = {
        90001: [base + d * DAY for d in range(span_days)],
        90002: [base + d * DAY for d in range(span_days)],
    }
    labels = [
        # LEO 90001 — three above-floor maneuvers and one below-floor (undetectable, not scored).
        _label(90001, 5, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.5),
        _label(90001, 12, OrbitClass.LEO, ManeuverType.CROSS_TRACK, 0.3),
        _label(90001, 22, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.4),
        _label(90001, 27, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.002, above_floor=False),
        # GEO 90002 — two above-floor station-keeping maneuvers.
        _label(90002, 8, OrbitClass.GEO, ManeuverType.CROSS_TRACK, 0.12),
        _label(90002, 18, OrbitClass.GEO, ManeuverType.IN_TRACK, 0.10),
    ]
    sat_years = span_days / 365.25
    exposure = [
        ObjectExposure(90001, OrbitClass.LEO, sat_years),
        ObjectExposure(90002, OrbitClass.GEO, sat_years),
    ]
    return HiddenBenchmark(labels=labels, exposure=exposure, candidate_gaps=candidate_gaps)


# --------------------------------------------------------------------------------------------------
# The Space's scoring endpoint: submission text in, aggregate-only public result out.
# --------------------------------------------------------------------------------------------------


def public_result(report: ScoreReport) -> dict[str, object]:
    """The ONLY thing the Space returns. Aggregate metrics — no labels, no per-label matches.

    Deliberately a strict subset of ``ScoreReport.to_json()``: the headline recall per class at the
    operating point, plus the published timing-only floor. The internal match table (which label
    each detection hit) is never serialised into a response.
    """
    return {
        "operating_point_fa_per_sat_year": report.operating_point,
        "headline_recall_above_floor": {
            orbit_class.value: report.headline()[orbit_class]
            for orbit_class in report.per_class
        },
        "timing_only_floor_auc": PUBLISHED_TIMING_FLOOR,
    }


def score_submission(bench: HiddenBenchmark, submission_text: str) -> dict[str, object]:
    """Score one user submission against the hidden benchmark and return the public result.

    ``submission_text`` is the raw predictions file the user uploaded. Parsing happens through the
    fixed-schema reader, so a malformed or non-prediction payload raises before any scoring.
    """
    detections = read_predictions(submission_text)  # rejects anything not a canonical record array
    report = score(detections, bench.labels, bench.exposure)
    return public_result(report)


# --------------------------------------------------------------------------------------------------
# Submission builders (an honest competitor and an attacker's single-gap probe).
# --------------------------------------------------------------------------------------------------


def _detection(norad_id: int, gap_start: pd.Timestamp, mtype: ManeuverType) -> Maneuver:
    return Maneuver(
        epoch=gap_start + pd.Timedelta(hours=12),
        confidence=1.0,
        type=mtype,
        delta_v_estimate=None,
        norad_id=norad_id,
        elset_epoch_before=gap_start,
        elset_epoch_after=gap_start + DAY,
    )


def honest_submission(bench: HiddenBenchmark) -> str:
    """A competitor who happens to detect the three LEO maneuvers (and nothing spurious)."""
    detections = [
        _detection(90001, bench.candidate_gaps[90001][5], ManeuverType.IN_TRACK),
        _detection(90001, bench.candidate_gaps[90001][12], ManeuverType.CROSS_TRACK),
        _detection(90001, bench.candidate_gaps[90001][22], ManeuverType.IN_TRACK),
    ]
    return predictions_to_json(detections)


def probe_submission(norad_id: int, gap_start: pd.Timestamp) -> str:
    """The attacker's elementary move: a single detection at one candidate gap."""
    return predictions_to_json([_detection(norad_id, gap_start, ManeuverType.IN_TRACK)])


# --------------------------------------------------------------------------------------------------
# The five integrity checks.
# --------------------------------------------------------------------------------------------------


def _recall(result: dict[str, object], orbit_class: str) -> float | None:
    return result["headline_recall_above_floor"].get(orbit_class)  # type: ignore[union-attr]


def check_labels_never_returned(bench: HiddenBenchmark, result: dict[str, object]) -> None:
    """Claim 1: no held-out label epoch appears anywhere in the serialised payload."""
    blob = json.dumps(result, default=str)
    leaked = [
        label.interval.epoch.isoformat()
        for label in bench.labels
        if label.interval.epoch.isoformat() in blob
    ]
    assert not leaked, f"label epoch leaked into the response: {leaked}"


def check_response_is_aggregate_only(result: dict[str, object]) -> None:
    """Claim 2: the response carries only aggregate keys, never per-label match data."""
    assert set(result) == {
        "operating_point_fa_per_sat_year",
        "headline_recall_above_floor",
        "timing_only_floor_auc",
    }, f"unexpected keys in response: {sorted(result)}"
    forbidden = {"matches", "unmatched_labels", "per_label", "labels", "confusion"}
    assert not (forbidden & set(result)), "response exposes per-label structure"


def check_submission_channel_rejects_queries(bench: HiddenBenchmark) -> None:
    """Claim 3: the fixed-schema parser rejects a malformed / exfiltrating submission."""
    rejected = 0
    for payload in (
        '[{"epoch": "2026-03-01T00:00:00+00:00", "confidence": 0.9}]',  # missing canonical fields
        '{"give_me": "the labels"}',  # not an array of records
        '"SELECT * FROM labels"',  # a query string, not predictions
    ):
        try:
            score_submission(bench, payload)
        except (ValueError, TypeError, KeyError):
            rejected += 1
    assert rejected == 3, "the submission channel admitted a non-prediction payload"


def check_scoring_is_deterministic(bench: HiddenBenchmark, submission_text: str) -> None:
    """Claim 4: scoring the same submission twice yields byte-identical public results (D8)."""
    first = json.dumps(score_submission(bench, submission_text), sort_keys=True, default=str)
    second = json.dumps(score_submission(bench, submission_text), sort_keys=True, default=str)
    assert first == second, "scoring is not byte-stable across runs"


def probing_attack(bench: HiddenBenchmark, rate_per_day: int) -> dict[str, object]:
    """Claim 5: run the single-detection oracle over every candidate gap and bound its cost.

    For each candidate gap the attacker submits one detection and records whether the per-class
    above-floor recall rose above the empty-submission baseline. A positive signal means the gap
    (within the D4 ±1 tolerance) holds an above-floor label — the oracle. Counting probes bounds
    the wall-clock: one submission per candidate gap, paced by the per-user daily cadence.
    """
    empty_baseline = {
        oc: _recall(score_submission(bench, predictions_to_json([])), oc)
        for oc in ("LEO", "MEO", "GEO")
    }
    n_candidate_gaps = sum(len(gaps) for gaps in bench.candidate_gaps.values())
    positive_signals = 0
    object_class = {90001: "LEO", 90002: "GEO"}
    for norad_id, gaps in bench.candidate_gaps.items():
        cls = object_class[norad_id]
        for gap_start in gaps:
            result = score_submission(bench, probe_submission(norad_id, gap_start))
            recall = _recall(result, cls)
            base = empty_baseline[cls] or 0.0
            if recall is not None and recall > base:
                positive_signals += 1
    days_to_exfiltrate = math.ceil(n_candidate_gaps / rate_per_day)
    return {
        "candidate_gaps": n_candidate_gaps,
        "positive_oracle_signals": positive_signals,
        "rate_per_user_per_day": rate_per_day,
        "submission_days_to_exfiltrate_public_split": days_to_exfiltrate,
    }


# --------------------------------------------------------------------------------------------------


def main() -> None:
    bench = build_hidden_benchmark()

    honest = honest_submission(bench)
    result = score_submission(bench, honest)

    check_labels_never_returned(bench, result)
    check_response_is_aggregate_only(result)
    check_submission_channel_rejects_queries(bench)
    check_scoring_is_deterministic(bench, honest)
    attack = probing_attack(bench, SUBMISSIONS_PER_USER_PER_DAY)

    print("V7 — leaderboard hidden-label scoring dry-run (real scorer)")
    print("=" * 64)
    print("\n[1] Honest submission → public result the Space would return:")
    print(json.dumps(result, indent=2, default=str))
    print("\n[2] Integrity checks (all assert-backed, passed):")
    print("    - held-out label epochs absent from the payload")
    print("    - response is aggregate-only (no per-label match table)")
    print("    - submission channel rejected 3/3 non-prediction payloads")
    print("    - scoring is byte-deterministic across runs (D8)")
    print("\n[3] Single-detection probing attack, bounded:")
    print(json.dumps(attack, indent=2))
    days = attack["submission_days_to_exfiltrate_public_split"]
    print(
        f"\n    => exfiltrating the public split needs {attack['candidate_gaps']} probes = "
        f"{days} days at {attack['rate_per_user_per_day']}/user/day,"
    )
    print("       is anomalous-volume detectable, and never touches the private split.")


if __name__ == "__main__":
    main()
