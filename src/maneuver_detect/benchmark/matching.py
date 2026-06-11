"""The detection-matching rule — assigning detections to labelled maneuvers under the D4 tolerance.

A maneuver is observable only as a discontinuity between consecutive elsets, so a label is an
inter-elset gap, not an instant (D4). The labeller
(:func:`~maneuver_detect.labels.labeller.label_series`) has already mapped each label onto its
bracketing gap and precomputed the matching window — that gap plus one adjacent gap on each side
(≈ ±2 days), as the ``tol_start`` / ``tol_end`` of a
:class:`~maneuver_detect.labels.labeller.LabelledInterval`. This module applies the rule: a
detection **matches** a label when it is the same object and its detection epoch falls within that
window, and detections are assigned to labels **one-to-one**.

The label unit the benchmark scores against is the :class:`ScoredLabel` — a labelled interval tagged
with whether it sits **above the per-object detectability floor** (D4). The floor itself is not
computed here (it is a per-object, TLE-quality-dependent calibration); this layer only carries the
flag through so the metric layer can score the above-floor population as the headline and the full
population as a secondary lower bound.

Assignment is greedy by descending confidence (the standard detection protocol): the most confident
detection claims its nearest in-window label first, so the result is a single pass the metric layer
can threshold at any confidence without re-matching. Ties are broken deterministically, so the whole
matching is byte-stable across runs and platforms.
"""

from __future__ import annotations

from dataclasses import dataclass

from maneuver_detect.labels.labeller import LabelledInterval
from maneuver_detect.schema import Maneuver

__all__ = [
    "DetectionMatch",
    "Matching",
    "ScoredLabel",
    "match_detections",
]


@dataclass(frozen=True)
class ScoredLabel:
    """A held-out label as the benchmark scores it — an interval plus its above-floor status.

    Attributes:
        interval: The label mapped onto its bracketing inter-elset gap, carrying the D4 matching
            window (:class:`~maneuver_detect.labels.labeller.LabelledInterval`).
        above_floor: Whether the maneuver is above the per-object detectability floor (D4). The
            headline metric scores the above-floor population; below-floor labels are physically
            undetectable from TLEs and are *ignored* rather than counted as misses. Defaults to
            ``True`` — the floor calibration is supplied upstream, not computed here.
    """

    interval: LabelledInterval
    above_floor: bool = True


@dataclass(frozen=True)
class DetectionMatch:
    """One detection and the label it was assigned, or ``None`` when it matched nothing.

    Attributes:
        detection: The predicted maneuver.
        label: The :class:`ScoredLabel` it was assigned under the D4 rule, or ``None`` — an
            unmatched detection, which the metric layer counts as a false positive.
    """

    detection: Maneuver
    label: ScoredLabel | None


@dataclass(frozen=True)
class Matching:
    """The one-to-one assignment of detections to labels under the detection-matching rule.

    Attributes:
        matches: One :class:`DetectionMatch` per detection, in the descending-confidence order the
            greedy assignment ran (the order the metric layer thresholds along).
        unmatched_labels: The matchable labels no detection claimed — above-floor ones are the false
            negatives. Labels with no ``norad_id`` are dropped (they attach to no scored object).
    """

    matches: tuple[DetectionMatch, ...]
    unmatched_labels: tuple[ScoredLabel, ...]


def _in_window(detection: Maneuver, label: ScoredLabel) -> bool:
    """Whether ``detection`` falls in ``label``'s D4 window (same object, within tolerance).

    Inclusive on both ends: a detection epoch exactly on ``tol_start`` or ``tol_end`` matches.
    """
    if detection.norad_id != label.interval.norad_id:
        return False
    return label.interval.tol_start.value <= detection.epoch.value <= label.interval.tol_end.value


def match_detections(
    detections: list[Maneuver] | tuple[Maneuver, ...],
    labels: list[ScoredLabel] | tuple[ScoredLabel, ...],
) -> Matching:
    """Assign ``detections`` to ``labels`` one-to-one under the D4 detection-matching rule.

    Detections are processed in descending confidence order (ties broken by epoch then NORAD id, so
    the pass is deterministic). Each detection claims the nearest still-unclaimed label of the same
    object whose ``[tol_start, tol_end]`` window contains its epoch (nearest by ``|Δepoch|``, ties
    broken toward the earlier label epoch); a detection with no such label is unmatched. The result
    is threshold-independent: dropping the lowest-confidence detections never changes the matches of
    the ones that remain, so the metric layer can sweep a confidence threshold over a single pass.

    Labels whose ``norad_id`` is ``None`` cannot attach to a scored object and are ignored entirely.
    """
    label_list = list(labels)
    matchable = [i for i, label in enumerate(label_list) if label.interval.norad_id is not None]

    order = sorted(
        range(len(detections)),
        key=lambda j: (
            -detections[j].confidence,
            detections[j].epoch.value,
            detections[j].norad_id,
        ),
    )

    claimed: set[int] = set()
    assigned: dict[int, int] = {}  # detection index -> label index
    for j in order:
        detection = detections[j]
        candidates = [
            i for i in matchable if i not in claimed and _in_window(detection, label_list[i])
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda i: (
                abs(detection.epoch.value - label_list[i].interval.epoch.value),
                label_list[i].interval.epoch.value,
                i,
            ),
        )
        claimed.add(best)
        assigned[j] = best

    matches = tuple(
        DetectionMatch(
            detection=detections[j],
            label=label_list[assigned[j]] if j in assigned else None,
        )
        for j in order
    )
    unmatched_labels = tuple(label_list[i] for i in matchable if i not in claimed)
    return Matching(matches=matches, unmatched_labels=unmatched_labels)
