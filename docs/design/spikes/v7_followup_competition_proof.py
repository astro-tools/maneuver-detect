"""V7 follow-up proof — the hidden-label competition firewall on a forward holdout.

The v0.2 leaderboard ships in *reproducibility* mode: the answer key is committed
(``dataset/v0.2/labels.json`` + ``splits.json``), so the hidden-label firewall V7 designed is
unbuildable on it (the **D12 amendment**). A true competition needs a test set whose labels live in
no public file — a **never-committed forward holdout**: maneuvers with an epoch *after* the public
dataset's freeze, reconstructed from the same operator feeds (D2) and kept only as private
deploy-time data. This dry-run shows that on such a holdout the firewall the open dataset could not
support is restored.

It reuses the *real* server-side scoring service (:class:`maneuver_detect.leaderboard.service`'s
``LeaderboardService``) and the shipped deterministic scorer — the same code the competition Space
would run — so the properties shown here are properties of the shipped code, not a mock.

It demonstrates the claims D16 rests on:

  1. **The forward holdout is disjoint from the committed public set, by a clean epoch cut.** Every
     committed-public label has epoch ``<=`` the freeze; every holdout label has epoch ``>`` it. The
     partition is a single timestamp comparison — auditable, and structurally impossible to leak a
     holdout label into ``labels.json`` / ``splits.json`` (the property the v0.2 open dataset could
     not have).
  2. **Hidden labels never leave the host.** The only thing returned is an aggregate score; the
     held-out (public-subset *and* private-subset) label epochs appear nowhere in the payload.
  3. **The response is aggregate-only** — per-class above-floor recall at the operating point plus
     the published D11 timing-only floor, never the per-label match table.
  4. **The submission channel can't carry a query** — the fixed-schema reader rejects anything that
     is not a canonical predictions array.
  5. **Scoring is byte-deterministic (D8)** — the same submission scores identically across runs.
  6. **The rate limit binds** — the real service enforces five scored submissions per user per UTC
     day (now an *integrity* bound, not a courtesy guard: probing genuinely leaks hidden labels).
  7. **The public/private firewall holds.** The live board scores a *public* subset of the holdout;
     a single-detection probing oracle recovers only that subset and never a *private*-subset label,
     and a competitor who memorised the public subset still scores ~zero on the held-back private
     subset that decides the final ranking. On the open v0.2 answer key this firewall was
     unbuildable; on a never-committed forward holdout it holds.

stdlib + the installed package only; no network, no GPU, deterministic across runs (no RNG, fixed
synthetic data — catalogue ids 9000x are fictional, so the proof ships no redistributed elements,
the V1/D2 practice).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from maneuver_detect.benchmark.matching import ScoredLabel
from maneuver_detect.benchmark.metrics import ObjectExposure
from maneuver_detect.benchmark.scoring import (
    ScoreReport,
    predictions_to_json,
    score,
)
from maneuver_detect.labels.labeller import LabelledInterval
from maneuver_detect.labels.record import OrbitClass
from maneuver_detect.leaderboard.fixture import ScoringFixture
from maneuver_detect.leaderboard.service import (
    DEFAULT_RATE_LIMIT_PER_DAY,
    LeaderboardService,
)
from maneuver_detect.schema import Maneuver, ManeuverType

# The public dataset's freeze: the cut that defines the forward holdout. A committed-public label
# sits on or before it; a holdout label sits strictly after it. Synthetic (a v0.3-style freeze).
FREEZE = pd.Timestamp("2026-09-01T00:00:00", tz="UTC")
DAY = pd.Timedelta(days=1)

# The forward window the holdout is drawn from — days 5..45 after the freeze, observed daily.
FORWARD_START_DAY = 5
FORWARD_END_DAY = 45
FORWARD_SPAN_DAYS = FORWARD_END_DAY - FORWARD_START_DAY + 1

# The published D11 "cheating floor": the rank-AUC a Δt-only model reaches, the score a submission
# must beat to be doing more than reading gap lengths (V5/D11). A benchmark constant shown with
# every score — never derived from a submission.
PUBLISHED_TIMING_FLOOR = {"LEO": 0.62, "GEO": 0.68}

# The three holdout objects: a dense LEO class plus the two classes the v0.3 dataset growth (D15)
# made operator-real (GEO via NOAA GOES, IGSO via QZSS OHI) — a GEO/IGSO holdout is thin, not empty.
OBJECT_CLASS = {90001: "LEO", 90002: "GEO", 90003: "IGSO"}

# A fixed UTC clock so the per-UTC-day rate-limit demonstration is deterministic.
_FIXED_NOW = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)


def _label(
    norad_id: int,
    day_offset: int,
    orbit_class: OrbitClass,
    maneuver_type: ManeuverType | None,
    delta_v: float | None,
    *,
    above_floor: bool = True,
) -> ScoredLabel:
    """A label on the gap ``[freeze + day_offset, +1 day)``, with the D4 ±1-adjacent-gap window.

    ``day_offset`` is relative to the freeze: negative places the label in the committed public
    history (epoch ``<=`` freeze), positive in the never-committed forward holdout (epoch ``>``
    freeze).
    """
    gap_start = FREEZE + day_offset * DAY
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
            source_ref=f"{norad_id}:{day_offset:+d}",
            orbit_class=orbit_class,
        ),
        above_floor=above_floor,
    )


@dataclass(frozen=True)
class Holdout:
    """The synthetic forward holdout — committed-public history plus the never-committed holdout.

    Attributes:
        committed_public: Labels with epoch ``<=`` freeze — the public answer key (``labels.json``).
        holdout_public: Holdout labels the *live* board scores against (the public subset).
        holdout_private: Holdout labels held back, scored once at the release reveal — they decide
            the final ranking and are never exposed to the live board.
        exposure: The scored population over the forward window (the sat-year denominator).
        candidate_gaps: The per-object daily gap grid an attacker could probe.
    """

    committed_public: list[ScoredLabel]
    holdout_public: list[ScoredLabel]
    holdout_private: list[ScoredLabel]
    exposure: list[ObjectExposure]
    candidate_gaps: dict[int, list[pd.Timestamp]]


def build_holdout() -> Holdout:
    """A three-object forward holdout: committed history pre-freeze, a public + private holdout after.

    Public- and private-subset labels are spaced ≥3 gaps apart so the D4 ±1 tolerance gives each its
    own oracle window, and the below-floor LEO label sits clear of both (undetectable, not scored).
    """
    committed_public = [
        # The public answer key — same objects, but every epoch is on/before the freeze.
        _label(90001, -30, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.5),
        _label(90001, -18, OrbitClass.LEO, ManeuverType.CROSS_TRACK, 0.3),
        _label(90002, -25, OrbitClass.GEO, ManeuverType.IN_TRACK, 0.12),
        _label(90003, -12, OrbitClass.IGSO, ManeuverType.CROSS_TRACK, 0.20),
    ]
    holdout_public = [
        # LEO — three above-floor maneuvers and one below-floor (undetectable, not scored).
        _label(90001, 8, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.5),
        _label(90001, 16, OrbitClass.LEO, ManeuverType.CROSS_TRACK, 0.3),
        _label(90001, 24, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.4),
        _label(90001, 28, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.002, above_floor=False),
        # GEO + IGSO — one operator-announced maneuver each (thin, but real — D15).
        _label(90002, 12, OrbitClass.GEO, ManeuverType.IN_TRACK, 0.12),
        _label(90003, 20, OrbitClass.IGSO, ManeuverType.CROSS_TRACK, 0.18),
    ]
    holdout_private = [
        # The held-back subset — the same forward window, disjoint gaps; decides the final ranking.
        _label(90001, 34, OrbitClass.LEO, ManeuverType.IN_TRACK, 0.45),
        _label(90001, 40, OrbitClass.LEO, ManeuverType.CROSS_TRACK, 0.35),
        _label(90002, 30, OrbitClass.GEO, ManeuverType.IN_TRACK, 0.11),
        _label(90003, 38, OrbitClass.IGSO, ManeuverType.CROSS_TRACK, 0.16),
    ]
    sat_years = FORWARD_SPAN_DAYS / 365.25
    exposure = [
        ObjectExposure(90001, OrbitClass.LEO, sat_years),
        ObjectExposure(90002, OrbitClass.GEO, sat_years),
        ObjectExposure(90003, OrbitClass.IGSO, sat_years),
    ]
    candidate_gaps = {
        norad_id: [FREEZE + d * DAY for d in range(FORWARD_START_DAY, FORWARD_END_DAY + 1)]
        for norad_id in OBJECT_CLASS
    }
    return Holdout(
        committed_public=committed_public,
        holdout_public=holdout_public,
        holdout_private=holdout_private,
        exposure=exposure,
        candidate_gaps=candidate_gaps,
    )


def _fixture(labels: list[ScoredLabel], exposure: list[ObjectExposure]) -> ScoringFixture:
    """Wrap a label subset + exposure as the private scoring fixture the Space loads."""
    return ScoringFixture(
        dataset_version="0.3.0-competition",
        labels=tuple(labels),
        exposure=tuple(exposure),
        timing_floor=PUBLISHED_TIMING_FLOOR,
    )


def _score_public(service: LeaderboardService, submission_text: str) -> dict[str, object]:
    """Score a submission through the real service's aggregate-only public path (no rate limit).

    Uses the shipped ``score`` + the service's ``public_result`` — the exact scoring leg ``submit``
    runs, minus the per-user counter — so the integrity properties are the service's own. (The rate
    limit is exercised separately, on ``submit``, in :func:`check_rate_limit_binds`.)
    """
    from maneuver_detect.benchmark.scoring import read_predictions

    detections = read_predictions(submission_text)  # rejects any non-prediction payload
    report = score(detections, list(service.fixture.labels), list(service.fixture.exposure))
    return service.public_result(report)


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


def _recall(result: dict[str, object], orbit_class: str) -> float | None:
    return result["headline_recall_above_floor"].get(orbit_class)  # type: ignore[union-attr]


def honest_submission(holdout: Holdout) -> str:
    """A competitor who detects the three above-floor LEO public-subset maneuvers (nothing else)."""
    gaps = holdout.candidate_gaps[90001]

    def gap(day: int) -> pd.Timestamp:
        return gaps[day - FORWARD_START_DAY]

    detections = [
        _detection(90001, gap(8), ManeuverType.IN_TRACK),
        _detection(90001, gap(16), ManeuverType.CROSS_TRACK),
        _detection(90001, gap(24), ManeuverType.IN_TRACK),
    ]
    return predictions_to_json(detections)


def probe_submission(norad_id: int, gap_start: pd.Timestamp) -> str:
    """The attacker's elementary move: a single detection at one candidate gap."""
    return predictions_to_json([_detection(norad_id, gap_start, ManeuverType.IN_TRACK)])


# --------------------------------------------------------------------------------------------------
# Integrity checks.
# --------------------------------------------------------------------------------------------------


def check_disjoint_partition(holdout: Holdout) -> None:
    """Claim 1: the holdout is a clean epoch cut, disjoint from the committed public set."""
    for label in holdout.committed_public:
        assert label.interval.epoch <= FREEZE, "a committed-public label sits after the freeze"
    for label in (*holdout.holdout_public, *holdout.holdout_private):
        assert label.interval.epoch > FREEZE, "a holdout label sits on/before the freeze"
    committed_epochs = {label.interval.epoch for label in holdout.committed_public}
    holdout_epochs = {
        label.interval.epoch for label in (*holdout.holdout_public, *holdout.holdout_private)
    }
    assert not (committed_epochs & holdout_epochs), "a holdout epoch appears in the committed set"


def check_labels_never_returned(holdout: Holdout, result: dict[str, object]) -> None:
    """Claim 2: no held-out label epoch (public or private) appears in the serialised payload."""
    blob = json.dumps(result, default=str)
    secret = (*holdout.holdout_public, *holdout.holdout_private)
    leaked = [
        label.interval.epoch.isoformat()
        for label in secret
        if label.interval.epoch.isoformat() in blob
    ]
    assert not leaked, f"a held-out label epoch leaked into the response: {leaked}"


def check_response_is_aggregate_only(result: dict[str, object]) -> None:
    """Claim 3: the response carries only aggregate keys, never per-label match data."""
    assert set(result) == {
        "operating_point_fa_per_sat_year",
        "headline_recall_above_floor",
        "timing_only_floor_auc",
    }, f"unexpected keys in response: {sorted(result)}"
    forbidden = {"matches", "unmatched_labels", "per_label", "labels", "confusion"}
    assert not (forbidden & set(result)), "response exposes per-label structure"


def check_submission_channel_rejects_queries(service: LeaderboardService) -> None:
    """Claim 4: the fixed-schema parser rejects a malformed / exfiltrating submission."""
    rejected = 0
    for payload in (
        '[{"epoch": "2026-09-10T00:00:00+00:00", "confidence": 0.9}]',  # missing canonical fields
        '{"give_me": "the labels"}',  # not an array of records
        '"SELECT * FROM labels"',  # a query string, not predictions
    ):
        try:
            _score_public(service, payload)
        except (ValueError, TypeError, KeyError):
            rejected += 1
    assert rejected == 3, "the submission channel admitted a non-prediction payload"


def check_scoring_is_deterministic(service: LeaderboardService, submission_text: str) -> None:
    """Claim 5: scoring the same submission twice yields byte-identical public results (D8)."""
    first = json.dumps(_score_public(service, submission_text), sort_keys=True, default=str)
    second = json.dumps(_score_public(service, submission_text), sort_keys=True, default=str)
    assert first == second, "scoring is not byte-stable across runs"


def check_rate_limit_binds(holdout: Holdout, submission_text: str) -> int:
    """Claim 6: the real service enforces the per-user-per-UTC-day scored-submission cadence."""
    from maneuver_detect.leaderboard.service import RateLimitError

    service = LeaderboardService(
        _fixture(holdout.holdout_public, holdout.exposure), clock=lambda: _FIXED_NOW
    )
    for _ in range(DEFAULT_RATE_LIMIT_PER_DAY):
        service.submit("prober", submission_text)
    try:
        service.submit("prober", submission_text)
    except RateLimitError:
        return DEFAULT_RATE_LIMIT_PER_DAY
    raise AssertionError("the rate limit did not bind on the (limit + 1)th submission")


def _label_window(label: ScoredLabel) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    return (label.interval.norad_id, label.interval.tol_start, label.interval.tol_end)


def _recovered(
    positives: list[tuple[int, pd.Timestamp]], labels: list[ScoredLabel]
) -> int:
    """How many above-floor ``labels`` have a probe positive inside their D4 tolerance window."""
    count = 0
    for label in labels:
        if not label.above_floor:
            continue
        norad_id, start, end = _label_window(label)
        if any(pid == norad_id and start <= gap <= end for pid, gap in positives):
            count += 1
    return count


def probing_firewall(holdout: Holdout) -> dict[str, object]:
    """Claim 7: probe the live (public-subset) board and show it never recovers a private label.

    For each candidate gap the attacker submits one detection and records whether the per-class
    above-floor recall rose above the empty-submission baseline — the single-detection oracle. The
    live board scores only the *public* subset, so a positive can only fall in a public-subset
    label's window; the private subset is in a different fixture the live board never touches.
    """
    public_board = LeaderboardService(_fixture(holdout.holdout_public, holdout.exposure))
    empty = {oc: _recall(_score_public(public_board, predictions_to_json([])), oc) for oc in OBJECT_CLASS.values()}

    positives: list[tuple[int, pd.Timestamp]] = []
    n_candidate_gaps = 0
    for norad_id, gaps in holdout.candidate_gaps.items():
        cls = OBJECT_CLASS[norad_id]
        for gap_start in gaps:
            n_candidate_gaps += 1
            result = _score_public(public_board, probe_submission(norad_id, gap_start))
            recall = _recall(result, cls)
            baseline = empty[cls] or 0.0
            if recall is not None and recall > baseline:
                positives.append((norad_id, gap_start))

    public_above_floor = [label for label in holdout.holdout_public if label.above_floor]
    return {
        "candidate_gaps": n_candidate_gaps,
        "public_subset_above_floor_labels": len(public_above_floor),
        "public_labels_recovered_by_probing": _recovered(positives, holdout.holdout_public),
        "private_subset_above_floor_labels": len(holdout.holdout_private),
        "private_labels_recovered_by_probing": _recovered(positives, holdout.holdout_private),
        "rate_per_user_per_day": DEFAULT_RATE_LIMIT_PER_DAY,
        "submission_days_to_exfiltrate_public_subset": math.ceil(
            n_candidate_gaps / DEFAULT_RATE_LIMIT_PER_DAY
        ),
    }


def overfit_does_not_transfer(holdout: Holdout) -> dict[str, object]:
    """A prober who memorised the public subset still scores ~zero on the held-back private subset.

    The submission predicts exactly the public-subset above-floor labels (everything probing can
    teach). It tops the live (public) board, but the final ranking is recomputed on the private
    subset — which it never saw — so its private recall is ~zero. This is the Kaggle firewall: the
    public board can be overfit, the private board decides the prize.
    """
    overfit = predictions_to_json(
        [
            _detection(
                label.interval.norad_id, label.interval.elset_epoch_before, ManeuverType.IN_TRACK
            )
            for label in holdout.holdout_public
            if label.above_floor
        ]
    )
    public_board = LeaderboardService(_fixture(holdout.holdout_public, holdout.exposure))
    private_board = LeaderboardService(_fixture(holdout.holdout_private, holdout.exposure))

    def pooled(result: dict[str, object]) -> float:
        per_class = result["headline_recall_above_floor"]
        vals = [v for v in per_class.values() if v is not None]  # type: ignore[union-attr]
        return max(vals) if vals else 0.0

    return {
        "overfit_headline_recall_public_subset": pooled(_score_public(public_board, overfit)),
        "overfit_headline_recall_private_subset": pooled(_score_public(private_board, overfit)),
    }


def main() -> None:
    holdout = build_holdout()
    public_board = LeaderboardService(_fixture(holdout.holdout_public, holdout.exposure))

    honest = honest_submission(holdout)
    result = _score_public(public_board, honest)

    check_disjoint_partition(holdout)
    check_labels_never_returned(holdout, result)
    check_response_is_aggregate_only(result)
    check_submission_channel_rejects_queries(public_board)
    check_scoring_is_deterministic(public_board, honest)
    rate_limit = check_rate_limit_binds(holdout, honest)
    firewall = probing_firewall(holdout)
    transfer = overfit_does_not_transfer(holdout)

    print("V7 follow-up — hidden-label competition firewall on a forward holdout (real service)")
    print("=" * 84)
    print("\n[1] Forward-holdout partition (the clean timestamp cut):")
    print(
        json.dumps(
            {
                "public_freeze": FREEZE.isoformat(),
                "committed_public_labels": len(holdout.committed_public),
                "forward_holdout_labels": len(holdout.holdout_public) + len(holdout.holdout_private),
                "holdout_public_subset": len(holdout.holdout_public),
                "holdout_private_subset": len(holdout.holdout_private),
            },
            indent=2,
        )
    )
    print("\n[2] Honest submission → public result the competition Space would return:")
    print(json.dumps(result, indent=2, default=str))
    print("\n[3] Integrity checks (all assert-backed, passed):")
    print("    - forward holdout disjoint from the committed public set (clean epoch cut)")
    print("    - held-out label epochs absent from the payload")
    print("    - response is aggregate-only (no per-label match table)")
    print("    - submission channel rejected 3/3 non-prediction payloads")
    print("    - scoring is byte-deterministic across runs (D8)")
    print(f"    - rate limit binds at {rate_limit} scored submissions / user / UTC day")
    print("\n[4] Public/private firewall under the single-detection probing oracle:")
    print(json.dumps(firewall, indent=2))
    print("\n[5] A prober who memorised the public subset cannot win the private ranking:")
    print(json.dumps(transfer, indent=2))
    print(
        f"\n    => probing recovers only the public subset "
        f"({firewall['public_labels_recovered_by_probing']}/"
        f"{firewall['public_subset_above_floor_labels']} of its above-floor labels) and never a"
    )
    print(
        f"       private-subset label ({firewall['private_labels_recovered_by_probing']} "
        f"recovered); the held-back private subset, revealed once at the release,"
    )
    print("       decides the final ranking. On the open v0.2 answer key this firewall was")
    print("       unbuildable; on a never-committed forward holdout it holds.")


if __name__ == "__main__":
    main()
