"""The maneuver-detect leaderboard — a Gradio Space over the shipped deterministic scorer.

A thin front end over :mod:`maneuver_detect.leaderboard`: it loads the private scoring bundle (the
held-out fixture + seed predictions, built by ``build_fixture.py`` and supplied as a private HF
Dataset), seeds the board with the baselines, and scores each uploaded ``predictions.json`` against
the frozen test split. Only an aggregate result is returned, the board ranks by headline recall, and
a per-user-per-UTC-day rate limit bounds submission volume (a courtesy guard — the answer key is
public, the D12 amendment).

Runs on the free Hugging Face CPU tier: scoring is pure element-arithmetic, no GPU. Configure with:

    LEADERBOARD_BUNDLE_DIR    local path to a bundle directory (fixture.json + seeds/), or
    LEADERBOARD_BUNDLE_REPO   a private HF Dataset id to download the bundle from (with HF_TOKEN)
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from maneuver_detect.leaderboard import (
    InvalidSubmissionError,
    LeaderboardService,
    RateLimitError,
    load_fixture,
)

_INTRO = """# maneuver-detect leaderboard

Score a maneuver-detection method on the **frozen v0.2 test split**, with the same deterministic
scorer and matching rule the package ships. Upload a `predictions.json` — a JSON array of canonical
maneuver records — and get the per-class **above-floor recall** at the operating point, ranked
against the classical, BiLSTM, and transformer baselines.

The test labels are part of the public CC-BY-4.0 dataset, so this board is for **reproducible,
comparable** scoring on identical splits — not a hidden-label competition. Submissions are
rate-limited as a courtesy. The published timing-only "cheating floor" is shown with every score:
beat it to be doing more than reading gap lengths.
"""

_BOARD_HEADERS = ["#", "entry", "headline recall", "LEO", "MEO", "GEO", "seed"]


def _bundle_dir() -> Path:
    """Resolve the scoring bundle — a local dir, or a private HF Dataset downloaded at startup."""
    local = os.environ.get("LEADERBOARD_BUNDLE_DIR")
    if local:
        return Path(local)
    repo = os.environ.get("LEADERBOARD_BUNDLE_REPO")
    if not repo:
        raise RuntimeError("set LEADERBOARD_BUNDLE_DIR or LEADERBOARD_BUNDLE_REPO")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=repo, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    )


def _build_service() -> LeaderboardService:
    """Load the fixture and seed the board with every baseline prediction file in the bundle."""
    bundle = _bundle_dir()
    service = LeaderboardService(load_fixture((bundle / "fixture.json").read_text()))
    seeds_dir = bundle / "seeds"
    for path in sorted(seeds_dir.glob("*.json")) if seeds_dir.is_dir() else []:
        service.add_seed(path.stem, path.read_text())
    return service


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _board_rows(service: LeaderboardService) -> list[list[str]]:
    rows: list[list[str]] = []
    for rank, entry in enumerate(service.board(), start=1):
        per_class = entry.per_class_recall
        rows.append(
            [
                str(rank),
                entry.name,
                f"{entry.headline_recall:.3f}",
                _fmt(per_class.get("LEO")),
                _fmt(per_class.get("MEO")),
                _fmt(per_class.get("GEO")),
                "yes" if entry.is_seed else "",
            ]
        )
    return rows


def build_app() -> gr.Blocks:
    service = _build_service()

    def submit(
        user_id: str, file_path: str | None
    ) -> tuple[dict[str, object], list[list[str]], str]:
        if not user_id.strip():
            return {}, _board_rows(service), "Enter a name / Hugging Face user id first."
        if not file_path:
            return {}, _board_rows(service), "Upload a predictions.json first."
        try:
            outcome = service.submit(user_id.strip(), Path(file_path).read_text())
        except InvalidSubmissionError as exc:
            return {}, _board_rows(service), f"Not a valid predictions file: {exc}"
        except RateLimitError as exc:
            return {}, _board_rows(service), f"Rate limit: {exc}"
        status = f"Scored. {outcome.remaining_today} scored submissions left today."
        return outcome.result, _board_rows(service), status

    with gr.Blocks(title="maneuver-detect leaderboard") as app:
        gr.Markdown(_INTRO)
        with gr.Row():
            user_box = gr.Textbox(label="name / Hugging Face user id", scale=2)
            file_box = gr.File(label="predictions.json", file_types=[".json"], type="filepath")
        submit_button = gr.Button("Score submission", variant="primary")
        status_box = gr.Markdown()
        result_box = gr.JSON(label="your aggregate result")
        board_box = gr.Dataframe(
            headers=_BOARD_HEADERS,
            value=_board_rows(service),
            label="leaderboard",
            interactive=False,
        )
        submit_button.click(
            submit, inputs=[user_box, file_box], outputs=[result_box, board_box, status_box]
        )
    return app


if __name__ == "__main__":
    build_app().launch()
