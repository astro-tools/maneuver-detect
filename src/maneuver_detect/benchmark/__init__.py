"""The frozen benchmark — splits, matching rule, metrics, and the scorer.

Leak-free splits by satellite and time window (seeded and byte-stable), the detection-matching
rule, the metric (precision and recall at a fixed false-alarm rate per satellite class, with
per-class type confusion), and the deterministic scorer the leaderboard runs. Frozen by release.
"""

from __future__ import annotations
