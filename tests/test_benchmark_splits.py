"""Tests for ``maneuver_detect.benchmark.splits`` — the leak-free, byte-stable benchmark splits.

Two constructions are covered. The **overlap-component** split (``make_splits``, default and
class-stratified) holds out whole satellites. The **temporal-holdout** split
(``make_temporal_split``) cuts the timeline into guard-separated eras with disjoint object sets, so
a dense class spreads across train / val / test where the overlap split fuses it into one component.
The load-bearing tests run on the committed ``dataset/v0.2`` labels — neither construction leaks a
satellite or an overlapping window across partitions, each is byte-stable, and the frozen
``splits.json`` (the temporal-holdout partition) reproduces byte-for-byte and is non-degenerate. The
synthetic cases pin the overlap-component, packing, and era-assignment behaviour directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from maneuver_detect.benchmark.splits import (
    DEFAULT_RATIOS,
    Split,
    SplitName,
    TemporalSplit,
    make_splits,
    make_temporal_split,
    split_counts,
)
from maneuver_detect.datasets.build import labels_from_json
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass

pytestmark = pytest.mark.benchmark

_UTC = timezone.utc
_T0 = datetime(2020, 1, 1, tzinfo=_UTC)
_DATA_DIR = Path(__file__).resolve().parents[1] / "dataset" / "v0.2"
_LABELS_PATH = _DATA_DIR / "labels.json"
_SPLITS_PATH = _DATA_DIR / "splits.json"

# The current (v0.3) frozen artifacts — the release this PR ships.
_V03_DIR = Path(__file__).resolve().parents[1] / "dataset" / "v0.3"
_V03_LABELS_PATH = _V03_DIR / "labels.json"
_V03_SPLITS_PATH = _V03_DIR / "splits.json"

#: The D4 ±1-adjacent-gap nominal matching tolerance; the temporal split's guard comfortably exceeds
#: it, so the envelope-overlap sweep is a faithful check of the matching-leak vector.
_MATCH_TOL = timedelta(days=2)


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


def _assert_no_cross_partition_envelope_overlap(
    grouped: dict[SplitName, list[ManeuverLabel]],
) -> None:
    """No label's ±tolerance match envelope overlaps a label assigned to a different partition.

    A sweep over every assigned label's ``[window_start - tol, window_end + tol]`` envelope: any two
    open, overlapping envelopes must belong to the same partition. This single check covers both the
    overlapping-window leak vector and the clean-temporal-boundary requirement at once.
    """
    spans = sorted(
        (label.window_start - _MATCH_TOL, label.window_end + _MATCH_TOL, name)
        for name, group in grouped.items()
        for label in group
    )
    open_spans: list[tuple[datetime, SplitName]] = []
    for start, end, name in spans:
        open_spans = [(o_end, o_name) for o_end, o_name in open_spans if o_end >= start]
        assert all(o_name == name for _o_end, o_name in open_spans)
        open_spans.append((end, name))


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


# --- temporal-holdout split: frozen v0.2 artifact -------------------------------------------------


def _committed_splits_text() -> str:
    return _SPLITS_PATH.read_text(encoding="utf-8")


def test_temporal_split_reproduces_frozen_artifact() -> None:
    # The committed splits.json is make_temporal_split on the committed labels, byte-for-byte (D8).
    # The frozen artifact under test is the v0.2 one, so the version is pinned to it rather than the
    # package's current DATASET_VERSION default.
    rebuilt = make_temporal_split(_committed_labels(), dataset_version="0.2.0")
    assert rebuilt.to_json() == _committed_splits_text()


def test_frozen_artifact_round_trips() -> None:
    committed = _committed_splits_text()
    assert TemporalSplit.from_json(committed).to_json() == committed


def test_frozen_split_is_non_degenerate() -> None:
    # Every class present in the dataset lands in every partition — the balance the overlap split
    # can't reach on dense GEO. The committed v0.2 labels cover LEO/MEO/GEO (IGSO/HEO arrive in the
    # v0.3 dataset), so the target is the classes actually in the data, not the whole enum.
    labels = _committed_labels()
    dataset_classes = {label.orbit_class for label in labels}
    grouped = make_temporal_split(labels).assign(labels)
    for name in SplitName:
        present = {label.orbit_class for label in grouped[name]}
        assert present == dataset_classes, f"{name.value} missing {dataset_classes - present}"


# --- temporal-holdout split: leak-free + byte-stable on the real dataset --------------------------


def test_temporal_split_is_leak_free_on_real_data() -> None:
    labels = _committed_labels()
    split = make_temporal_split(labels)
    # Satellite axis: object sets pairwise disjoint.
    assert split.train & split.val == frozenset()
    assert split.train & split.test == frozenset()
    assert split.val & split.test == frozenset()
    # Temporal axis: no match envelope crosses a partition (covers the window-overlap leak too).
    _assert_no_cross_partition_envelope_overlap(split.assign(labels))


def test_temporal_split_is_byte_stable_on_real_data() -> None:
    labels = _committed_labels()
    assert make_temporal_split(labels).to_json() == make_temporal_split(labels).to_json()


# --- v0.3 frozen artifacts: the release this PR ships ---------------------------------------------


def _v03_labels() -> list[ManeuverLabel]:
    return labels_from_json(_V03_LABELS_PATH.read_text(encoding="utf-8"))


def test_v03_temporal_split_reproduces_frozen_artifact() -> None:
    # The committed v0.3 splits.json is make_temporal_split on the committed v0.3 labels, byte-exact
    # (D8). The version stamp comes from the package default (0.3.0), matching the artifact.
    rebuilt = make_temporal_split(_v03_labels())
    assert rebuilt.to_json() == _V03_SPLITS_PATH.read_text(encoding="utf-8")


def test_v03_split_is_leak_free_and_non_degenerate() -> None:
    labels = _v03_labels()
    split = TemporalSplit.from_json(_V03_SPLITS_PATH.read_text(encoding="utf-8"))
    # Satellite axis: object sets pairwise disjoint.
    assert split.train & split.val == frozenset()
    assert split.train & split.test == frozenset()
    assert split.val & split.test == frozenset()
    # Temporal axis: no match envelope crosses a partition.
    grouped = split.assign(labels)
    _assert_no_cross_partition_envelope_overlap(grouped)
    # Every populated class (LEO/MEO/GEO/IGSO; HEO is reserved/empty) lands in every partition.
    dataset_classes = {label.orbit_class for label in labels}
    assert OrbitClass.IGSO in dataset_classes and OrbitClass.HEO not in dataset_classes
    for name in SplitName:
        present = {label.orbit_class for label in grouped[name]}
        assert present == dataset_classes, f"{name.value} missing {dataset_classes - present}"


def test_temporal_other_seed_is_still_leak_free() -> None:
    labels = _committed_labels()
    split = make_temporal_split(labels, seed=7)
    assert split.train & split.val == frozenset()
    assert split.train & split.test == frozenset()
    assert split.val & split.test == frozenset()
    _assert_no_cross_partition_envelope_overlap(split.assign(labels))


# --- temporal-holdout split: assignment / serialisation behaviour ---------------------------------


def test_temporal_assign_drops_guard_band_and_wrong_era_labels() -> None:
    labels = _committed_labels()
    split = make_temporal_split(labels)
    train_obj = min(split.train)
    # A train object's label in the guard band around cut1 — dropped (it straddles the boundary).
    guard_label = _label(train_obj, split.cut1, split.cut1 + timedelta(hours=1))
    # The same object's label in the test era (after cut2) — dropped, keeping its era novel.
    late = split.cut2 + split.guard + timedelta(days=10)
    wrong_era_label = _label(train_obj, late, late + timedelta(hours=1))
    grouped = split.assign([*labels, guard_label, wrong_era_label])
    for name in SplitName:
        assert guard_label not in grouped[name]
        assert wrong_era_label not in grouped[name]


def test_temporal_from_json_rejects_non_temporal() -> None:
    # An object-set Split serialises without a "kind" marker, so it can't parse as temporal.
    with pytest.raises(ValueError):
        TemporalSplit.from_json(make_splits(_committed_labels()).to_json())


def test_temporal_empty_labels_rejected() -> None:
    with pytest.raises(ValueError):
        make_temporal_split([])


def test_temporal_collapsing_quantiles_rejected() -> None:
    # When the label epochs are too clustered for the two quantiles to land on distinct epochs, the
    # era cuts collapse (cut1 >= cut2); reject rather than emit a degenerate single-era split.
    same = datetime(2024, 1, 1, tzinfo=_UTC)
    labels = [_label(n, same, same + timedelta(hours=1)) for n in range(1, 6)]
    with pytest.raises(ValueError, match="collapse"):
        make_temporal_split(labels)


def test_temporal_object_with_only_boundary_straddling_labels_is_dropped() -> None:
    # An object whose every label window straddles a cut (start era != end era) belongs wholly to no
    # era, so it is assigned to no partition rather than leaking across the boundary. The clean band
    # objects are unaffected — the drop is specific to the straddler.
    labels: list[ManeuverLabel] = []
    norad = 1000
    for band_start in (2000, 2010, 2020):
        for k in range(6):
            start = datetime(band_start + 1, 6, 1, tzinfo=_UTC) + timedelta(days=20 * k)
            labels.append(_label(norad, start, start + timedelta(hours=1)))
            norad += 1
    straddler = 9999
    labels.append(
        _label(
            straddler,
            datetime(2002, 1, 1, tzinfo=_UTC),
            datetime(2019, 1, 1, tzinfo=_UTC),  # spans era 0 into era 2 — straddles both cuts
        )
    )
    split = make_temporal_split(labels, quantiles=(0.34, 0.67))
    assert straddler not in (split.train | split.val | split.test)
    assert split.train and split.val and split.test  # clean band objects still populate every era


def test_committed_temporal_split_clears_the_guard_band_with_margin() -> None:
    # D7 leak-freeness rests on the guard band clearing the series-derived match envelope (the
    # labelled gap ±1 adjacent gap) at each era boundary. make_temporal_split sees only announced
    # windows, not the envelope, so pin the structural margin on the committed split: every pair of
    # cross-partition kept labels is separated in time by more than the full 2*guard band — the
    # slack the envelope lives in. A grown label crowding a cut trips this before it could leak.
    labels = _committed_labels()
    split = make_temporal_split(labels)
    grouped = split.assign(labels)
    by_partition = {
        name: [(label.window_start, label.window_end) for label in grouped[name]]
        for name in SplitName
    }

    names = list(SplitName)
    min_cross_gap = min(
        max(s2 - e1, s1 - e2, timedelta(0))
        for i, n1 in enumerate(names)
        for n2 in names[i + 1 :]
        for (s1, e1) in by_partition[n1]
        for (s2, e2) in by_partition[n2]
    )
    assert min_cross_gap > 2 * split.guard


def test_temporal_synthetic_multi_band_is_leak_free() -> None:
    # Three well-separated year-bands per class, cuts placed between bands, so every class reaches
    # every partition; robust to the exact cut placement.
    labels: list[ManeuverLabel] = []
    norad = 1000
    for orbit_class in OrbitClass:
        for band_start in (2000, 2010, 2020):
            for k in range(4):
                start = datetime(band_start + 1, 6, 1, tzinfo=_UTC) + timedelta(days=20 * k)
                labels.append(
                    _label(norad, start, start + timedelta(hours=1), orbit_class=orbit_class)
                )
                norad += 1
    split = make_temporal_split(labels, quantiles=(0.34, 0.67))
    assert split.train and split.val and split.test  # every partition populated
    assert split.train & split.val == frozenset()
    assert split.train & split.test == frozenset()
    assert split.val & split.test == frozenset()
    _assert_no_cross_partition_envelope_overlap(split.assign(labels))
    assert make_temporal_split(labels, quantiles=(0.34, 0.67)).to_json() == split.to_json()


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
