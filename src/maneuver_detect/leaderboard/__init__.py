"""The public leaderboard's server-side logic — the held-out fixture and the scoring service.

The hosted Hugging Face Space is a thin Gradio front end over this package: it loads a
:class:`~maneuver_detect.leaderboard.fixture.ScoringFixture` (the frozen test-split labels and
exposure, supplied as private deploy-time data per D2) and drives a
:class:`~maneuver_detect.leaderboard.service.LeaderboardService`, which scores each submission with
the shipped deterministic scorer and returns an aggregate-only result over a ranked, rate-limited
board.

Per the D12 amendment the v0.2 board is a **reproducibility / convenience** board on the public test
split — the answer key is already published, so the rate limit and aggregate-only response are
courtesy / abuse guards, not a hidden-label firewall. The Gradio app, the offline fixture builder,
and the deploy steps live under the repository's top-level ``leaderboard/`` directory.
"""

from __future__ import annotations

from maneuver_detect.leaderboard.fixture import (
    ScoringFixture,
    fixture_to_json,
    load_fixture,
)
from maneuver_detect.leaderboard.service import (
    DEFAULT_RATE_LIMIT_PER_DAY,
    BoardEntry,
    InvalidSubmissionError,
    LeaderboardService,
    RateLimitError,
    SubmissionResult,
)

__all__ = [
    "DEFAULT_RATE_LIMIT_PER_DAY",
    "BoardEntry",
    "InvalidSubmissionError",
    "LeaderboardService",
    "RateLimitError",
    "ScoringFixture",
    "SubmissionResult",
    "fixture_to_json",
    "load_fixture",
]
