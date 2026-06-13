"""The leaderboard's server-side scoring service — a submission in, an aggregate-only result out.

Wraps the shipped deterministic scorer (:mod:`maneuver_detect.benchmark.scoring`) with the v0.2
leaderboard policy (the D12 amendment). A fixed-schema submission is scored against the held-out
:class:`~maneuver_detect.leaderboard.fixture.ScoringFixture` and only an **aggregate** result is
returned — the headline above-floor recall per class plus the published timing-only floor, never the
per-label match table the scorer computes internally. So a submission can carry nothing but
predictions (the fixed-schema :func:`~maneuver_detect.benchmark.scoring.read_predictions` rejects
anything else) and a response leaks no label.

The board ranks entries by headline (above-floor) recall, deterministically and reproducibly because
the scorer is byte-stable (D8). A per-user-per-UTC-day rate limit bounds probing volume — on the
v0.2 board this is a **courtesy / abuse guard, not an integrity guarantee**: the answer key is
public (the D12 amendment), so the rate limit slows abusive volume rather than guarding labels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from maneuver_detect.benchmark import ScoreReport, read_predictions, score
from maneuver_detect.leaderboard.fixture import ScoringFixture

__all__ = [
    "DEFAULT_RATE_LIMIT_PER_DAY",
    "BoardEntry",
    "InvalidSubmissionError",
    "LeaderboardService",
    "RateLimitError",
    "SubmissionResult",
]

#: The v0.2 submission cadence — scored submissions per user per UTC day (the D12 amendment retains
#: this as a courtesy / abuse guard now that the answer key is public).
DEFAULT_RATE_LIMIT_PER_DAY = 5


class InvalidSubmissionError(ValueError):
    """A submission that is not a canonical predictions file (rejected before it is scored)."""


class RateLimitError(RuntimeError):
    """A user exceeded the per-UTC-day scored-submission cadence."""


@dataclass(frozen=True)
class BoardEntry:
    """One scored entry on the leaderboard.

    Attributes:
        name: The submitter's display name (a seed baseline's name, or the submitting user's id).
        submitted_at: The UTC time the entry was scored.
        headline_recall: Above-floor recall pooled across classes (label-count weighted) — the rank
            key; ``0.0`` when the submission recovers nothing scorable.
        per_class_recall: Per-class above-floor recall at the operating point (``None`` for a class
            with no above-floor labels).
        is_seed: Whether the entry is one of the baselines the board launches with.
    """

    name: str
    submitted_at: datetime
    headline_recall: float
    per_class_recall: dict[str, float | None]
    is_seed: bool = False


@dataclass(frozen=True)
class SubmissionResult:
    """The outcome of scoring one submission — what the Space returns to the submitter.

    Attributes:
        result: The aggregate-only public result (the only thing derived from the hidden labels).
        entry: The board entry the submission produced.
        remaining_today: The submitter's remaining scored submissions for the current UTC day.
    """

    result: dict[str, object]
    entry: BoardEntry
    remaining_today: int


@dataclass
class LeaderboardService:
    """Scores submissions against a held-out fixture and keeps a ranked, rate-limited board.

    The service is the whole server-side leg of the leaderboard: it never exposes the fixture, only
    aggregate scores and the ranked board. ``clock`` is injected so the per-UTC-day rate limit is
    testable; it defaults to the real UTC wall clock.
    """

    fixture: ScoringFixture
    rate_limit_per_day: int = DEFAULT_RATE_LIMIT_PER_DAY
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    _board: list[BoardEntry] = field(default_factory=list, init=False, repr=False)
    _counts: dict[tuple[str, date], int] = field(default_factory=dict, init=False, repr=False)

    def public_result(self, report: ScoreReport) -> dict[str, object]:
        """The only payload derived from the hidden labels — aggregate metrics, never a match table.

        A strict subset of :meth:`ScoreReport.to_json`: headline above-floor recall per class at the
        operating point, plus the published timing-only floor. The per-label match table the scorer
        builds internally is never serialised, so a response hands an attacker no row of the key.
        """
        return {
            "operating_point_fa_per_sat_year": report.operating_point,
            "headline_recall_above_floor": {
                orbit_class.value: report.headline()[orbit_class]
                for orbit_class in report.per_class
            },
            "timing_only_floor_auc": dict(self.fixture.timing_floor),
        }

    def add_seed(self, name: str, predictions_text: str) -> BoardEntry:
        """Score a baseline's predictions and add it to the board as a seed entry (no rate limit).

        Seeds are the baselines the board launches with; they are re-scored through the same scorer
        as every submission, so a seed's number on the board is reproducible, not asserted.
        """
        report = self._score(predictions_text)
        entry = self._entry(name, report, is_seed=True)
        self._board.append(entry)
        return entry

    def submit(self, user_id: str, predictions_text: str) -> SubmissionResult:
        """Score a user's submission, append it to the board, and return the aggregate result.

        Raises :class:`RateLimitError` if the user is over the per-UTC-day cadence (checked before
        scoring, so an over-limit user spends no compute) and :class:`InvalidSubmissionError` if the
        payload is not a canonical predictions file (which does not consume the user's quota).
        """
        today = self.clock().astimezone(timezone.utc).date()
        key = (user_id, today)
        if self._counts.get(key, 0) >= self.rate_limit_per_day:
            raise RateLimitError(
                f"{user_id} has used all {self.rate_limit_per_day} scored submissions for {today}"
            )
        report = self._score(predictions_text)
        self._counts[key] = self._counts.get(key, 0) + 1
        entry = self._entry(user_id, report, is_seed=False)
        self._board.append(entry)
        return SubmissionResult(
            result=self.public_result(report),
            entry=entry,
            remaining_today=self.rate_limit_per_day - self._counts[key],
        )

    def board(self) -> list[BoardEntry]:
        """The board ranked by headline recall (ties broken by submission time, then name)."""
        return sorted(
            self._board,
            key=lambda entry: (-entry.headline_recall, entry.submitted_at, entry.name),
        )

    def _score(self, predictions_text: str) -> ScoreReport:
        try:
            detections = read_predictions(predictions_text)
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidSubmissionError(str(exc)) from exc
        return score(detections, list(self.fixture.labels), list(self.fixture.exposure))

    def _entry(self, name: str, report: ScoreReport, *, is_seed: bool) -> BoardEntry:
        return BoardEntry(
            name=name,
            submitted_at=self.clock().astimezone(timezone.utc),
            headline_recall=_pooled_above_floor_recall(report),
            per_class_recall={
                orbit_class.value: report.per_class[orbit_class].recall
                for orbit_class in report.per_class
            },
            is_seed=is_seed,
        )


def _pooled_above_floor_recall(report: ScoreReport) -> float:
    """Above-floor recall pooled across classes, weighted by each class's above-floor label count.

    The single scalar the board ranks by — the overall above-floor recall. Classes with no
    above-floor labels (or an undefined recall) are skipped; an empty population scores ``0.0``.
    """
    hit = 0.0
    total = 0
    for metrics in report.per_class.values():
        if metrics.recall is not None and metrics.n_labels_above_floor > 0:
            hit += metrics.recall * metrics.n_labels_above_floor
            total += metrics.n_labels_above_floor
    return hit / total if total else 0.0
