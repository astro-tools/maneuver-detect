"""Tests for ``maneuver_detect.benchmark.splits`` — the leak-free, byte-stable benchmark splits.

The load-bearing tests run on the committed ``dataset/v0.2`` labels: that the partition leaks no
satellite and no overlapping maneuver window across train / val / test, and that ``make_splits`` is
byte-stable. The synthetic cases pin the overlap-component and packing behaviour directly —
including the class-stratified packer, exercised on multi-class synthetic data where each class
has components to distribute. (The frozen-split byte-reproduction and real-data target-ratio checks
return with the temporal-holdout split that re-freezes a balanced ``splits.json`` — the
window-overlap split degenerates on the dense GEO labels, fusing most of the catalogue into one
component, so no balanced frozen split exists yet to pin.)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from maneuver_detect.benchmark.splits import (
    DEFAULT_RATIOS,
    Split,
    SplitName,
    make_splits,
    split_counts,
)
from maneuver_detect.datasets.build import labels_from_json
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass

pytestmark = pytest.mark.benchmark

_UTC = timezone.utc
_T0 = datetime(2020, 1, 1, tzinfo=_UTC)
_LABELS_PATH = Path(__file__).resolve().parents[1] / "dataset" / "v0.2" / "labels.json"


def _label(
    norad_id: int | None,
    start: datetime,
    end: datetime,
    *,
    orbit_class: OrbitClass = OrbitClass.LEO,
) -> ManeuverLabel:
    """A minimal maneuver label with an explicit window — the only fields the split reads."""
    return ManeuverLabel(
        norad_id=norad_id,
        epoch=start,
        window_start=start,
        window_end=end,
        source="TEST",
        source_ref="ref",
        orbit_class=orbit_class,
    )


def _committed_labels() -> list[ManeuverLabel]:
    return labels_from_json(_LABELS_PATH.read_text(encoding="utf-8"))


# --- byte-stability + the definition-of-done leak-free invariants, on the real dataset ---


def test_make_splits_is_byte_stable() -> None:
    labels = _committed_labels()
    assert make_splits(labels).to_json() == make_splits(labels).to_json()


# --- the definition-of-done invariants, on the real dataset ---------------------------------------


def test_no_satellite_crosses_splits() -> None:
    split = make_splits(_committed_labels())
    train, val, test = split.train, split.val, split.test
    assert train & val == frozenset()
    assert train & test == frozenset()
    assert val & test == frozenset()


def test_no_overlapping_window_crosses_splits() -> None:
    labels = _committed_labels()
    membership = make_splits(labels).by_norad()
    # Sweep windows in start order; every pair of open, overlapping windows must share a split.
    windows = sorted(
        (label.window_start, label.window_end, membership[label.norad_id])
        for label in labels
        if label.norad_id is not None
    )
    open_windows: list[tuple[datetime, SplitName]] = []
    for start, end, name in windows:
        open_windows = [(o_end, o_name) for o_end, o_name in open_windows if o_end >= start]
        assert all(o_name == name for _o_end, o_name in open_windows)
        open_windows.append((end, name))


def test_split_counts_partition_the_dataset() -> None:
    labels = _committed_labels()
    split = make_splits(labels)
    counts = split_counts(split, labels)

    total_events = sum(counts.n_events(name) for name in SplitName)
    total_objects = sum(counts.n_objects(name) for name in SplitName)
    labelled = {label.norad_id for label in labels if label.norad_id is not None}
    assert total_events == len(labels)
    assert total_objects == len(labelled)
    # Every split and every class is present in the report (even at zero count).
    for name in SplitName:
        assert set(counts.per_split[name]) == set(OrbitClass)


def test_other_seed_is_still_leak_free() -> None:
    split = make_splits(_committed_labels(), seed=7)
    assert split.train & split.val == frozenset()
    assert split.train & split.test == frozenset()
    assert split.val & split.test == frozenset()
    # A different seed reorders equal-size components but partitions the same object set.
    base = make_splits(_committed_labels())
    assert split.train | split.val | split.test == base.train | base.val | base.test


# --- overlap-component and packing behaviour, on hand-built cases ---------------------------------


def test_overlapping_windows_share_a_split() -> None:
    # Two objects whose windows overlap form one component, so they cannot be split apart, however
    # many disjoint singletons surround them.
    labels = [
        _label(100, _T0, _T0 + timedelta(hours=6)),
        _label(200, _T0 + timedelta(hours=3), _T0 + timedelta(hours=9)),
    ]
    labels += [
        _label(300 + i, _T0 + timedelta(days=10 + i), _T0 + timedelta(days=10 + i, hours=1))
        for i in range(8)
    ]
    split = make_splits(labels)
    assert split.name_of(100) == split.name_of(200)


def test_touching_windows_count_as_overlap() -> None:
    # Closed-interval windows that share only an endpoint are treated as overlapping (conservative).
    labels = [
        _label(100, _T0, _T0 + timedelta(hours=1)),
        _label(200, _T0 + timedelta(hours=1), _T0 + timedelta(hours=2)),
    ]
    split = make_splits(labels)
    assert split.name_of(100) == split.name_of(200)


def test_same_object_windows_never_split_even_when_disjoint() -> None:
    # Two far-apart windows of the *same* object share the node, so the object stays whole.
    labels = [
        _label(100, _T0, _T0 + timedelta(hours=1)),
        _label(100, _T0 + timedelta(days=365), _T0 + timedelta(days=365, hours=1)),
    ]
    split = make_splits(labels)
    assert sum(100 in split.members(name) for name in SplitName) == 1


def test_disjoint_objects_distribute_across_splits() -> None:
    labels = [
        _label(100 + i, _T0 + timedelta(days=i), _T0 + timedelta(days=i, hours=1))
        for i in range(30)
    ]
    split = make_splits(labels)
    assert all(split.members(name) for name in SplitName)
    union = split.train | split.val | split.test
    assert union == {100 + i for i in range(30)}


def test_empty_labels_yield_empty_splits() -> None:
    split = make_splits([])
    assert split.train == frozenset()
    assert split.val == frozenset()
    assert split.test == frozenset()
    assert split == Split.from_json(split.to_json())


def test_labels_without_norad_are_dropped() -> None:
    labels = [
        _label(100, _T0, _T0 + timedelta(hours=1)),
        _label(None, _T0, _T0 + timedelta(hours=1)),
    ]
    split = make_splits(labels)
    assert split.name_of(100) is not None
    grouped = split.assign(labels)
    assert sum(len(group) for group in grouped.values()) == 1


def test_assign_groups_labels_by_split() -> None:
    labels = [
        _label(100 + i, _T0 + timedelta(days=i), _T0 + timedelta(days=i, hours=1))
        for i in range(30)
    ]
    split = make_splits(labels)
    grouped = split.assign(labels)
    assert set(grouped) == set(SplitName)
    for name, group in grouped.items():
        assert all(split.name_of(label.norad_id) == name for label in group)


# --- class-stratified packing, on multi-class synthetic data --------------------------------------


def _multiclass_singletons(count: int) -> list[ManeuverLabel]:
    """``count`` disjoint singletons, cycling through the orbit classes (windows never overlap)."""
    classes = list(OrbitClass)
    return [
        _label(
            100 + i,
            _T0 + timedelta(days=i),
            _T0 + timedelta(days=i, hours=1),
            orbit_class=classes[i % len(classes)],
        )
        for i in range(count)
    ]


def test_stratified_is_byte_stable() -> None:
    labels = _multiclass_singletons(30)
    assert (
        make_splits(labels, stratified=True).to_json()
        == make_splits(labels, stratified=True).to_json()
    )


def test_stratified_is_leak_free_multi_class() -> None:
    labels = _multiclass_singletons(30)
    split = make_splits(labels, stratified=True)
    assert split.train & split.val == frozenset()
    assert split.train & split.test == frozenset()
    assert split.val & split.test == frozenset()
    # No overlapping window crosses a split — the same sweep the real-data meta-test runs.
    membership = split.by_norad()
    windows = sorted(
        (label.window_start, label.window_end, membership[label.norad_id])
        for label in labels
        if label.norad_id is not None
    )
    open_windows: list[tuple[datetime, SplitName]] = []
    for start, end, name in windows:
        open_windows = [(o_end, o_name) for o_end, o_name in open_windows if o_end >= start]
        assert all(o_name == name for _o_end, o_name in open_windows)
        open_windows.append((end, name))


def test_stratified_targets_each_class_ratio() -> None:
    # 20 disjoint singleton objects per class — every component is one object, so the packer is free
    # to hit 70 / 15 / 15 within each class (14 / 3 / 3 of 20).
    labels: list[ManeuverLabel] = []
    norad = 1000
    day = 0
    for orbit_class in OrbitClass:
        for _ in range(20):
            start = _T0 + timedelta(days=day)
            labels.append(_label(norad, start, start + timedelta(hours=1), orbit_class=orbit_class))
            norad += 1
            day += 2

    counts = split_counts(make_splits(labels, stratified=True), labels)
    for orbit_class in OrbitClass:
        per_class = {name: counts.per_split[name][orbit_class].n_objects for name in SplitName}
        assert sum(per_class.values()) == 20
        assert all(n > 0 for n in per_class.values())  # every split carries some of the class
        assert per_class[SplitName.TRAIN] == max(per_class.values())  # train is the plurality
        for name, target in zip(SplitName, DEFAULT_RATIOS, strict=True):
            assert abs(per_class[name] / 20 - target) <= 0.1


def test_stratified_spreads_a_class_that_total_packing_starves() -> None:
    # 8 heavy GEO objects (4 events each) + 30 light LEO singletons. The total-count packer ranks
    # the heavy GEO components first and pours them into train; stratifying reaches val/test too.
    labels: list[ManeuverLabel] = []
    norad = 1000
    day = 0
    for _ in range(8):
        start = _T0 + timedelta(days=day)
        for hour in range(4):  # four disjoint windows of one object → one weight-4 component
            window = start + timedelta(hours=hour * 2)
            labels.append(
                _label(norad, window, window + timedelta(hours=1), orbit_class=OrbitClass.GEO)
            )
        norad += 1
        day += 5
    for _ in range(30):
        start = _T0 + timedelta(days=day)
        labels.append(_label(norad, start, start + timedelta(hours=1), orbit_class=OrbitClass.LEO))
        norad += 1
        day += 2

    total = split_counts(make_splits(labels), labels)
    stratified = split_counts(make_splits(labels, stratified=True), labels)
    geo = OrbitClass.GEO
    assert total.per_split[SplitName.VAL][geo].n_objects == 0  # total-count confines GEO to train
    assert total.per_split[SplitName.TEST][geo].n_objects == 0
    assert all(stratified.per_split[name][geo].n_objects > 0 for name in SplitName)  # all three


def test_stratified_keeps_multi_class_component_whole() -> None:
    # A LEO object whose window overlaps a GEO object's forms one component; stratified packing must
    # still place the whole component (both classes) in a single split.
    labels = [
        _label(100, _T0, _T0 + timedelta(hours=6), orbit_class=OrbitClass.LEO),
        _label(200, _T0 + timedelta(hours=3), _T0 + timedelta(hours=9), orbit_class=OrbitClass.GEO),
    ]
    labels += [
        _label(
            300 + i,
            _T0 + timedelta(days=10 + i),
            _T0 + timedelta(days=10 + i, hours=1),
            orbit_class=OrbitClass.MEO,
        )
        for i in range(8)
    ]
    split = make_splits(labels, stratified=True)
    assert split.name_of(100) == split.name_of(200)


def test_stratified_single_overlap_component_stays_whole() -> None:
    # When every window overlaps, the whole catalogue is one component (the dense-GEO degeneracy the
    # real v0.2 labels hit). No packing can subdivide a component, so all objects land in one split;
    # stratified mode keeps the leak-free guarantee rather than tearing the component apart.
    classes = list(OrbitClass)
    labels = [
        _label(100 + i, _T0, _T0 + timedelta(days=2), orbit_class=classes[i % len(classes)])
        for i in range(9)
    ]
    split = make_splits(labels, stratified=True)
    occupied = [split.members(name) for name in SplitName if split.members(name)]
    assert len(occupied) == 1
    assert occupied[0] == {100 + i for i in range(9)}


def test_stratified_other_seed_is_still_leak_free() -> None:
    labels = _multiclass_singletons(30)
    split = make_splits(labels, stratified=True, seed=7)
    assert split.train & split.val == frozenset()
    assert split.train & split.test == frozenset()
    assert split.val & split.test == frozenset()
    # A different seed reorders equal-size components but partitions the same object set.
    base = make_splits(labels, stratified=True)
    assert split.train | split.val | split.test == base.train | base.val | base.test


# --- input validation -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratios", [(0.5, 0.5), (0.7, 0.15, 0.1, 0.05), (-0.1, 0.5, 0.6), (0.0, 0.0, 0.0)]
)
def test_invalid_ratios_rejected(ratios: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        make_splits([], ratios=ratios)  # type: ignore[arg-type]


def test_from_json_rejects_wrong_length_ratios() -> None:
    bad = (
        '{"dataset_version": "0.1.0", "seed": 0, "ratios": [0.5, 0.5], '
        '"splits": {"train": [], "val": [], "test": []}}'
    )
    with pytest.raises(ValueError):
        Split.from_json(bad)


# --- name_of and the count summary (small, direct branches) ---


def test_name_of_handles_unset_and_unassigned_ids() -> None:
    split = Split(
        dataset_version="test",
        seed=0,
        ratios=DEFAULT_RATIOS,
        train=frozenset({1}),
        val=frozenset({2}),
        test=frozenset({3}),
    )
    assert split.name_of(1) is SplitName.TRAIN
    assert split.name_of(None) is None  # an unset id
    assert split.name_of(99999) is None  # an id in no split


def test_split_counts_drops_unattachable_labels_and_summarises() -> None:
    split = Split(
        dataset_version="test",
        seed=0,
        ratios=DEFAULT_RATIOS,
        train=frozenset({1}),
        val=frozenset(),
        test=frozenset(),
    )
    labels = [
        _label(1, _T0, _T0 + timedelta(days=1)),  # counted in train / LEO
        _label(None, _T0, _T0 + timedelta(days=1)),  # no norad id → dropped
        _label(42, _T0, _T0 + timedelta(days=1)),  # in no split → dropped
    ]
    counts = split_counts(split, labels)
    assert counts.n_objects(SplitName.TRAIN) == 1
    assert counts.n_events(SplitName.TRAIN) == 1
    assert counts.n_objects(SplitName.VAL) == 0
    text = counts.summary()
    # Every split and class appears with a stable shape.
    for split_name in (SplitName.TRAIN, SplitName.VAL, SplitName.TEST):
        assert f"{split_name.value}:" in text
    assert OrbitClass.LEO.value in text
