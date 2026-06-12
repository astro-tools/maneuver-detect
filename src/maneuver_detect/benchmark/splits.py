"""Leak-free train / val / test splits — by satellite and time window, seeded and byte-stable.

The benchmark holds out whole satellites *and* keeps overlapping maneuver windows together, so a
model is never trained on the same object — or the same instant of activity — it is later scored on
(D7). Both guarantees fall out of one construction: build the **satellite-overlap graph** (an edge
between two objects whenever any of their announced maneuver windows overlap), take its connected
components, and assign each *whole* component to one split. Because the graph node *is* the
satellite, no satellite can straddle a split; because overlapping windows share a component, no
overlapping window can either.

Overlap is computed on the announced ``[window_start, window_end]`` windows **verbatim**, not the
±2-day detection-matching envelope (D4). That envelope is a *per-object* tolerance — the scorer only
ever matches a detection against the labels of the same object — so it never reaches across
satellites, and padding the windows out to it collapses the dense modern catalogue into a single
component (everything in one split). The literal leak vector D7 names is an overlapping *window* —
that is what this guards.

Components are packed into splits largest-first toward target label-count ratios (default
70 / 15 / 15), each going to whichever split is furthest below its target. ``make_splits(...,
stratified=True)`` instead aims those ratios **within each orbit class**, sending each component to
the split whose per-class balance it least disturbs. Either way packing is deterministic; ``seed``
only orders equal-size components, so a frozen split reproduces byte-for-byte (D8): once a version's
split is committed, :func:`make_splits` on that version's labels yields it exactly.

Both modes pack *whole* components, so neither can rebalance a catalogue that collapses into
one overlap-component (e.g. dense, same-era GEO station-keeping whose windows all overlap) —
that takes a different split *construction*, not a different packing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from maneuver_detect.datasets.catalogue import DATASET_VERSION
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass

__all__ = [
    "DEFAULT_RATIOS",
    "DEFAULT_SEED",
    "Split",
    "SplitCounts",
    "SplitName",
    "make_splits",
    "split_counts",
]


class SplitName(str, Enum):
    """The three benchmark partitions, in canonical order."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


#: The pinned seed the frozen v0.1 split is generated under (D8). Only orders equal-size components.
DEFAULT_SEED = 0

#: Target label-count fractions for ``(train, val, test)``. Approximate — a whole component is the
#: smallest packable unit, so realised sizes are the nearest the component structure allows.
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class _Component:
    """One connected component of the satellite-overlap graph — packed as an indivisible unit.

    Attributes:
        norad_ids: The objects in the component (sorted), all bound to the same split.
        n_events: The number of maneuver labels the component carries (its packing weight).
        events_by_class: The label count split by :class:`OrbitClass` (sums to ``n_events``, keyed
            in canonical class order) — the per-class weight the stratified packer balances.
    """

    norad_ids: tuple[int, ...]
    n_events: int
    events_by_class: dict[OrbitClass, int]


@dataclass(frozen=True)
class Split:
    """A frozen leak-free partition of the labelled objects into train / val / test.

    Attributes:
        dataset_version: The dataset version the split was computed for (lockstep with manifest).
        seed: The seed the split was generated under (orders equal-size components).
        ratios: The target ``(train, val, test)`` label-count fractions the packing aimed at.
        train: NORAD ids assigned to the training split.
        val: NORAD ids assigned to the validation split.
        test: NORAD ids assigned to the test split.
    """

    dataset_version: str
    seed: int
    ratios: tuple[float, float, float]
    train: frozenset[int]
    val: frozenset[int]
    test: frozenset[int]

    def members(self, split: SplitName) -> frozenset[int]:
        """The NORAD ids assigned to ``split``."""
        return {SplitName.TRAIN: self.train, SplitName.VAL: self.val, SplitName.TEST: self.test}[
            split
        ]

    def by_norad(self) -> dict[int, SplitName]:
        """Map each assigned NORAD id to its split."""
        return {norad_id: name for name in SplitName for norad_id in self.members(name)}

    def name_of(self, norad_id: int | None) -> SplitName | None:
        """The split ``norad_id`` belongs to, or ``None`` if it is unset or in no split."""
        if norad_id is None:
            return None
        for name in SplitName:
            if norad_id in self.members(name):
                return name
        return None

    def assign(self, labels: Sequence[ManeuverLabel]) -> dict[SplitName, list[ManeuverLabel]]:
        """Group ``labels`` by the split of their object (every split key present, possibly empty).

        Labels whose ``norad_id`` is ``None`` or falls in no split are dropped — they cannot attach
        to an object the benchmark holds out.
        """
        grouped: dict[SplitName, list[ManeuverLabel]] = {name: [] for name in SplitName}
        membership = self.by_norad()
        for label in labels:
            if label.norad_id is None:
                continue
            name = membership.get(label.norad_id)
            if name is not None:
                grouped[name].append(label)
        return grouped

    def to_json(self) -> str:
        """Serialise to canonical, NORAD-sorted JSON (a stable, committable artifact)."""
        payload = {
            "dataset_version": self.dataset_version,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "splits": {name.value: sorted(self.members(name)) for name in SplitName},
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Split:
        """Parse a split from :meth:`to_json` output."""
        data = json.loads(text)
        splits = data["splits"]
        ratios = tuple(float(value) for value in data["ratios"])
        if len(ratios) != 3:
            raise ValueError(f"ratios must have three entries, got {len(ratios)}")
        return cls(
            dataset_version=str(data["dataset_version"]),
            seed=int(data["seed"]),
            ratios=(ratios[0], ratios[1], ratios[2]),
            train=frozenset(int(n) for n in splits[SplitName.TRAIN.value]),
            val=frozenset(int(n) for n in splits[SplitName.VAL.value]),
            test=frozenset(int(n) for n in splits[SplitName.TEST.value]),
        )


def _overlap_components(labels: Sequence[ManeuverLabel]) -> list[_Component]:
    """Connected components of the satellite-overlap graph, by a sweep over the windows.

    Each distinct ``norad_id`` is one node; two nodes are unioned when any of their windows overlap
    (closed intervals — touching endpoints count). A satellite's own windows need no edge: they are
    already the same node, so a satellite is always wholly within one component.
    """
    events: dict[int, int] = {}
    class_events: dict[int, dict[OrbitClass, int]] = {}
    for label in labels:
        if label.norad_id is None:
            continue
        events[label.norad_id] = events.get(label.norad_id, 0) + 1
        per_object = class_events.setdefault(label.norad_id, {})
        per_object[label.orbit_class] = per_object.get(label.orbit_class, 0) + 1

    parent: dict[int, int] = {norad_id: norad_id for norad_id in events}

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # union toward the smaller id (deterministic root)

    # Sweep the windows in start order; each unions with every still-open window of another object.
    windows: list[tuple[datetime, datetime, int]] = sorted(
        (label.window_start, label.window_end, label.norad_id)
        for label in labels
        if label.norad_id is not None
    )
    active: list[tuple[datetime, int]] = []  # (window_end, norad_id) of still-open windows
    for start, end, norad_id in windows:
        active = [(open_end, open_id) for open_end, open_id in active if open_end >= start]
        for _open_end, open_id in active:
            if open_id != norad_id:
                union(norad_id, open_id)
        active.append((end, norad_id))

    members: dict[int, list[int]] = {}
    for norad_id in events:
        members.setdefault(find(norad_id), []).append(norad_id)

    components: list[_Component] = []
    for group in members.values():
        combined: dict[OrbitClass, int] = {}
        for norad_id in group:
            for orbit_class, count in class_events[norad_id].items():
                combined[orbit_class] = combined.get(orbit_class, 0) + count
        components.append(
            _Component(
                norad_ids=tuple(sorted(group)),
                n_events=sum(events[norad_id] for norad_id in group),
                events_by_class={oc: combined[oc] for oc in OrbitClass if oc in combined},
            )
        )
    return components


def _tiebreak(seed: int, norad_id: int) -> str:
    """A platform-stable ordering key for equal-size components (Python's ``hash`` is salted)."""
    return hashlib.sha256(f"{seed}:{norad_id}".encode()).hexdigest()


def _pack(
    components: Iterable[_Component], ratios: tuple[float, float, float], seed: int
) -> dict[SplitName, set[int]]:
    """Greedily assign whole components to splits toward the target label-count ``ratios``.

    Components are ordered largest-first (ties broken by the seeded key), and each is placed in the
    split currently furthest below its target event share — first split wins an exact tie, so the
    result is deterministic and byte-stable for a given seed.
    """
    ordered = sorted(components, key=lambda c: (-c.n_events, _tiebreak(seed, c.norad_ids[0])))
    total_events = sum(c.n_events for c in ordered)
    targets = dict(zip(SplitName, (ratio * total_events for ratio in ratios), strict=True))

    assigned: dict[SplitName, set[int]] = {name: set() for name in SplitName}
    placed: dict[SplitName, int] = {name: 0 for name in SplitName}
    for component in ordered:
        target = max(SplitName, key=lambda name: targets[name] - placed[name])
        # max() returns the first maximal element, so an exact tie favours TRAIN > VAL > TEST.
        assigned[target].update(component.norad_ids)
        placed[target] += component.n_events
    return assigned


def _imbalance_increase(
    placed: Mapping[OrbitClass, int],
    target: Mapping[OrbitClass, float],
    events_by_class: Mapping[OrbitClass, int],
    class_totals: Mapping[OrbitClass, int],
) -> float:
    """The rise in summed squared *relative* per-class deviation from placing a component here.

    Only the classes the component carries can change, so the sum runs over just those. Each class's
    squared deviation is divided by that class's squared total event count, making every class's
    contribution scale-free — a small class (few labels) is balanced as hard as a large one rather
    than swamped by it.
    """
    increase = 0.0
    for orbit_class, n_events in events_by_class.items():
        norm = class_totals[orbit_class]  # > 0: the class occurs in at least this component
        before = placed[orbit_class] - target[orbit_class]
        after = before + n_events
        increase += (after * after - before * before) / (norm * norm)
    return increase


def _pack_stratified(
    components: Iterable[_Component], ratios: tuple[float, float, float], seed: int
) -> dict[SplitName, set[int]]:
    """Greedily assign whole components toward the target ``ratios`` *within each orbit class*.

    Like :func:`_pack`, components are ordered largest-first (the seeded key breaks ties) and placed
    one at a time; but each goes to the split whose per-class balance it least worsens — the one
    minimising :func:`_imbalance_increase` against per-class targets ``ratio * class_total``. The
    arithmetic is integer-derived and summed in a fixed (canonical class) order, so the choice is
    deterministic and byte-stable for a given seed (D8); an exact tie favours TRAIN > VAL > TEST,
    mirroring :func:`_pack`.

    Packing whole components cannot subdivide one, so a catalogue that collapses into a single
    overlap-component stays in one split under this mode too — rebalancing that needs a different
    split *construction*, not a different packing.
    """
    ordered = sorted(components, key=lambda c: (-c.n_events, _tiebreak(seed, c.norad_ids[0])))
    class_totals: dict[OrbitClass, int] = {}
    for component in ordered:
        for orbit_class, n_events in component.events_by_class.items():
            class_totals[orbit_class] = class_totals.get(orbit_class, 0) + n_events
    targets: dict[SplitName, dict[OrbitClass, float]] = {
        name: {orbit_class: ratio * total for orbit_class, total in class_totals.items()}
        for name, ratio in zip(SplitName, ratios, strict=True)
    }

    assigned: dict[SplitName, set[int]] = {name: set() for name in SplitName}
    placed: dict[SplitName, dict[OrbitClass, int]] = {
        name: dict.fromkeys(class_totals, 0) for name in SplitName
    }
    for component in ordered:
        target = SplitName.TRAIN
        best = float("inf")
        for name in SplitName:  # canonical order, so an exact tie favours TRAIN > VAL > TEST
            increase = _imbalance_increase(
                placed[name], targets[name], component.events_by_class, class_totals
            )
            if increase < best:
                best, target = increase, name
        assigned[target].update(component.norad_ids)
        for orbit_class, n_events in component.events_by_class.items():
            placed[target][orbit_class] += n_events
    return assigned


def make_splits(
    labels: Sequence[ManeuverLabel],
    *,
    dataset_version: str = DATASET_VERSION,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    stratified: bool = False,
) -> Split:
    """Partition the objects in ``labels`` into a leak-free train / val / test :class:`Split`.

    Objects whose maneuver windows overlap are kept together (so no overlapping window crosses a
    split), and each object lands wholly in one split (so no satellite crosses). ``ratios`` are the
    target ``(train, val, test)`` label-count fractions; ``seed`` orders equal-size components for a
    reproducible, byte-stable split (D8). Labels with no ``norad_id`` are ignored.

    By default the packer balances the *total* label count across splits. Pass ``stratified=True``
    to aim the ``ratios`` **within each orbit class** instead, so per-class val/test shares are
    targeted rather than incidental. Both modes hold the leak-free guarantees and are byte-stable
    per seed.
    """
    if len(ratios) != 3:
        raise ValueError(f"ratios must have three entries, got {len(ratios)}")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios!r}")
    if sum(ratios) <= 0:
        raise ValueError("ratios must sum to a positive value")

    components = _overlap_components(labels)
    assigned = (
        _pack_stratified(components, ratios, seed)
        if stratified
        else _pack(components, ratios, seed)
    )
    return Split(
        dataset_version=dataset_version,
        seed=seed,
        ratios=(ratios[0], ratios[1], ratios[2]),
        train=frozenset(assigned[SplitName.TRAIN]),
        val=frozenset(assigned[SplitName.VAL]),
        test=frozenset(assigned[SplitName.TEST]),
    )


@dataclass(frozen=True)
class _ClassCount:
    """Object and maneuver-event counts for one orbit class within one split."""

    n_objects: int
    n_events: int


@dataclass(frozen=True)
class SplitCounts:
    """Per-split, per-class object and maneuver-event counts (D7's reported figures).

    Attributes:
        per_split: ``{split: {orbit_class: counts}}`` with every split and class present (zero
            counts included), so the report shape is stable regardless of the input distribution.
    """

    per_split: dict[SplitName, dict[OrbitClass, _ClassCount]]

    def n_objects(self, split: SplitName) -> int:
        """Total objects in ``split`` across all classes."""
        return sum(count.n_objects for count in self.per_split[split].values())

    def n_events(self, split: SplitName) -> int:
        """Total maneuver events in ``split`` across all classes."""
        return sum(count.n_events for count in self.per_split[split].values())

    def summary(self) -> str:
        """A human-readable per-split, per-class count summary."""
        lines: list[str] = []
        for split in SplitName:
            lines.append(
                f"{split.value}: {self.n_objects(split)} objects, {self.n_events(split)} events"
            )
            for orbit_class in OrbitClass:
                count = self.per_split[split][orbit_class]
                lines.append(
                    f"  {orbit_class.value}: {count.n_objects} objects, {count.n_events} events"
                )
        return "\n".join(lines)


def split_counts(split: Split, labels: Sequence[ManeuverLabel]) -> SplitCounts:
    """Count objects and maneuver events per split and orbit class for ``labels`` under ``split``.

    An object is counted in a class once (its orbit class); events are the per-object label counts.
    Every split and :class:`OrbitClass` appears in the report even at zero count.
    """
    objects: dict[SplitName, dict[OrbitClass, set[int]]] = {
        name: {orbit_class: set() for orbit_class in OrbitClass} for name in SplitName
    }
    events: dict[SplitName, dict[OrbitClass, int]] = {
        name: dict.fromkeys(OrbitClass, 0) for name in SplitName
    }
    membership = split.by_norad()
    for label in labels:
        if label.norad_id is None:
            continue
        name = membership.get(label.norad_id)
        if name is None:
            continue
        objects[name][label.orbit_class].add(label.norad_id)
        events[name][label.orbit_class] += 1

    per_split = {
        name: {
            orbit_class: _ClassCount(
                n_objects=len(objects[name][orbit_class]),
                n_events=events[name][orbit_class],
            )
            for orbit_class in OrbitClass
        }
        for name in SplitName
    }
    return SplitCounts(per_split=per_split)
