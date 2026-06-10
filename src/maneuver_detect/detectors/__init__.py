"""Maneuver detectors — one module per detector, behind a common interface.

Every detector consumes a per-object mean-element series and returns the canonical maneuver
schema. The classical reference detector (Holt-Winters smoothing + rule-based jump detection +
the Δv inversion) is the baseline every learned model must beat; the learned baselines arrive on
top of the same interface.
"""

from __future__ import annotations
