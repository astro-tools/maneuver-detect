# Contributing to maneuver-detect

Thanks for your interest. This page is the one place to learn the workflow.

## Getting set up

```bash
git clone https://github.com/astro-tools/maneuver-detect.git
cd maneuver-detect
uv sync --all-groups
```

This installs the package, its runtime dependencies, and the dev and docs groups. The optional
time-series foundation-model stack lives behind the `[foundation]` extra and is not installed by
default; add `--extra foundation` if you are working on that baseline.

## Branches and PRs

- One issue per branch. Branch names use a short prefix for type:
  - `feat/<slug>` — new capability, tied to a `type:feature` issue.
  - `fix/<slug>` — bug fix, tied to a `type:bug` issue.
  - `chore/<slug>` — infra / tooling / hygiene.
  - `docs/<slug>` — docs-only change.
- Open a PR against `main`. Put `Closes #<N>` in the PR description so the issue
  auto-closes on merge and the project board advances the card to Done.
- Squash-merge is the only merge method. The PR title becomes the squash commit
  subject — write it as a complete imperative sentence.

## Local checks before pushing

```bash
uv run pytest               # tests
uv run ruff check           # lint
uv run ruff format --check  # formatting — CI fails if the tree is not formatted
uv run mypy                 # types (strict)
```

CI re-runs all four on Ubuntu and Windows, across Python 3.10, 3.11, and 3.12. Note that the
lint job runs `ruff format --check`, not just `ruff check` — run the formatter locally before
pushing.

## Commit messages

Keep them short and imperative. One subject line, optional body.

Do not include AI or tool attribution trailers in commits, PR titles, PR descriptions,
or comments — see the repo-level convention.

## Scope discipline

maneuver-detect is a dataset, a benchmark, and a set of detectors for orbital maneuvers in public
TLE history: a TLE history in, a maneuver DataFrame out. It is deliberately not a maneuver
*predictor*, a real-time / streaming pipeline, a new propagator or force model, or a general
time-series-anomaly framework, and it uses only publicly redistributable data. New maneuver-label
sources and new detector baselines are the most welcome contributions. Before opening a feature
issue, check the charter and the frozen design decisions (`docs/design/`) to make sure the work
belongs here.

## Questions

Open a [discussion](https://github.com/orgs/astro-tools/discussions) rather than an
issue for open-ended questions, usage help, or brainstorming. The astro-tools org runs
a single shared discussions space — there is no per-repo discussions board.
