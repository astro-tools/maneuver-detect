"""Operator-announcement ingest and the epoch-to-elset-gap labeller.

One module per maneuver-label source, each normalising a heterogeneous operator log to a common
``(object, epoch, [type], [Δv])`` record, plus the labeller that maps a maneuver epoch onto the
inter-elset gap that first reflects it under the frozen matching tolerance.
"""

from __future__ import annotations
