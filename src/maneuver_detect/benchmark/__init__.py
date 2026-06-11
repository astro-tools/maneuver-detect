"""The frozen benchmark — splits, matching rule, metrics, and the scorer.

Leak-free splits by satellite and time window (seeded and byte-stable), the detection-matching
rule, the metric (precision and recall at a fixed false-alarm rate per satellite class, with
per-class type confusion), and the deterministic scorer the leaderboard runs. Frozen by release.
"""

from __future__ import annotations

from maneuver_detect.benchmark.matching import (
    DetectionMatch,
    Matching,
    ScoredLabel,
    match_detections,
)
from maneuver_detect.benchmark.metrics import (
    DEFAULT_OPERATING_POINT,
    DEFAULT_SWEEP,
    ClassMetrics,
    Confusion,
    ObjectExposure,
    PRPoint,
    class_metrics,
)
from maneuver_detect.benchmark.scoring import (
    ScoreReport,
    predictions_to_json,
    read_predictions,
    score,
)
from maneuver_detect.benchmark.splits import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    Split,
    SplitCounts,
    SplitName,
    make_splits,
    split_counts,
)

__all__ = [
    "DEFAULT_OPERATING_POINT",
    "DEFAULT_RATIOS",
    "DEFAULT_SEED",
    "DEFAULT_SWEEP",
    "ClassMetrics",
    "Confusion",
    "DetectionMatch",
    "Matching",
    "ObjectExposure",
    "PRPoint",
    "ScoreReport",
    "ScoredLabel",
    "Split",
    "SplitCounts",
    "SplitName",
    "class_metrics",
    "make_splits",
    "match_detections",
    "predictions_to_json",
    "read_predictions",
    "score",
    "split_counts",
]
