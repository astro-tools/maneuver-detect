"""Tests for ``maneuver_detect.benchmark.splits`` — the leak-free, byte-stable benchmark splits.

The split is *frozen by release* (D7), so the load-bearing tests run against the committed
``dataset/v0.1`` artifacts: that :func:`make_splits` reproduces the committed ``splits.json``
byte-for-byte, and that the partition leaks no satellite and no overlapping maneuver window across
train / val / test. The synthetic cases pin the overlap-component and packing behaviour directly.
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
_DATA_DIR = Path(__file__).resolve().parents[1] / "dataset" / "v0.1"


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
    return labels_from_json((_DATA_DIR / "labels.json").read_text(encoding="utf-8"))


# --- frozen-artifact reproducibility (the strongest byte-stability guarantee) ---


def test_make_splits_reproduces_frozen_artifact() -> None:
    committed = (_DATA_DIR / "splits.json").read_text(encoding="utf-8")
    assert make_splits(_committed_labels()).to_json() == committed


def test_frozen_artifact_round_trips() -> None:
    committed = (_DATA_DIR / "splits.json").read_text(encoding="utf-8")
    split = Split.from_json(committed)
    assert split == make_splits(_committed_labels())
    assert split.to_json() == committed


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
    # Every split and every class is present in the report, even GEO at zero (no GEO in v0.1).
    for name in SplitName:
        assert set(counts.per_split[name]) == set(OrbitClass)
    assert all(counts.per_split[name][OrbitClass.GEO].n_events == 0 for name in SplitName)


def test_target_ratios_are_approximately_met() -> None:
    labels = _committed_labels()
    counts = split_counts(make_splits(labels), labels)
    total = len(labels)
    for name, target in zip(SplitName, DEFAULT_RATIOS, strict=True):
        assert counts.n_events(name) / total == pytest.approx(target, abs=0.02)


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
