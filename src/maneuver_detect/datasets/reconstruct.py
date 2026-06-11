"""The reconstruction engine — turn a recipe into the labelled dataset and verify it byte-for-byte.

:func:`reconstruct` walks a :class:`~maneuver_detect.datasets.recipe.Recipe`, and for each object
composes the layers already built: fetch the series (via any
:class:`~maneuver_detect.data.base.Fetcher` — the credentialled Space-Track archive in production, a
fake in tests), clean and assemble it into the mean-element series (the data layer), attach that
object's maneuver labels onto the inter-elset gaps (the label layer), and digest the series for the
manifest. The result
is the in-memory v0.1 labelled dataset (local reconstruction; the Hub upload is a later milestone).

:func:`verify` reconstructs and compares the fresh per-series digests against a pinned
:class:`~maneuver_detect.datasets.manifest.Manifest` — the byte-for-byte integrity check that makes
the recipe-first distribution model trustworthy (D2/D8).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from maneuver_detect.data.base import Fetcher
from maneuver_detect.data.clean import clean_elsets
from maneuver_detect.data.history import assemble
from maneuver_detect.datasets.manifest import Manifest, SeriesDigest, series_sha256
from maneuver_detect.datasets.recipe import Recipe
from maneuver_detect.labels.labeller import (
    CoverageReport,
    LabelledInterval,
    label_coverage,
    label_series,
)
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass

__all__ = ["LabelledDataset", "ObjectDataset", "reconstruct", "verify"]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectDataset:
    """One object's reconstructed slice of the dataset.

    Attributes:
        norad_id: NORAD id of the object.
        orbit_class: The object's orbit class.
        object_name: The recipe's human-readable name.
        series: The cleaned mean-element series
            (:data:`~maneuver_detect.data.history.MEAN_ELEMENT_COLUMNS`).
        intervals: The labelled inter-elset intervals (maneuver labels mapped onto gaps).
        unmatched: Labels whose epoch fell outside the series' span.
        digest: The content digest of ``series`` for the manifest.
    """

    norad_id: int
    orbit_class: OrbitClass
    object_name: str
    series: pd.DataFrame
    intervals: list[LabelledInterval]
    unmatched: list[ManeuverLabel]
    digest: SeriesDigest


@dataclass(frozen=True)
class LabelledDataset:
    """The reconstructed v0.1 labelled dataset.

    Attributes:
        dataset_version: The dataset version reconstructed (carried into the manifest).
        objects: The per-object slices, in recipe order.
        labels: Every maneuver label fed into the reconstruction (for the coverage report).
    """

    dataset_version: str
    objects: list[ObjectDataset]
    labels: list[ManeuverLabel]

    def by_norad(self) -> dict[int, ObjectDataset]:
        """Index the object slices by NORAD id."""
        return {obj.norad_id: obj for obj in self.objects}

    def manifest(self) -> Manifest:
        """The per-series content-hash manifest of this reconstruction."""
        return Manifest(
            dataset_version=self.dataset_version,
            digests=tuple(obj.digest for obj in self.objects),
        )

    def coverage(self) -> CoverageReport:
        """The per-class label-coverage report over the dataset's labels."""
        return label_coverage(self.labels)


def reconstruct(
    recipe: Recipe,
    fetcher: Fetcher,
    labels_by_norad: Mapping[int, Sequence[ManeuverLabel]] | None = None,
) -> LabelledDataset:
    """Reconstruct the labelled dataset for ``recipe`` using ``fetcher`` and per-object labels.

    For each recipe entry: fetch the series in the entry's window, clean and assemble it, map the
    object's labels (from ``labels_by_norad``, keyed by NORAD id; empty when omitted) onto its gaps,
    and digest the cleaned series. The same recipe + same fetched input yields a byte-identical
    manifest (the property :func:`verify` checks).
    """
    labels_by_norad = labels_by_norad or {}
    objects: list[ObjectDataset] = []
    all_labels: list[ManeuverLabel] = []
    total = len(recipe.entries)
    for index, entry in enumerate(recipe.entries, start=1):
        result = fetcher.fetch(entry.norad_id, start=entry.start, end=entry.end)
        cleaned = clean_elsets(list(result.elsets))
        series = assemble(cleaned)
        obj_labels = list(labels_by_norad.get(entry.norad_id, []))
        all_labels.extend(obj_labels)
        labelling = label_series(series, obj_labels)
        _logger.info(
            "[%d/%d] series NORAD %s (%s): %d elsets, %d labels",
            index,
            total,
            entry.norad_id,
            entry.object_name,
            len(cleaned),
            len(obj_labels),
        )
        objects.append(
            ObjectDataset(
                norad_id=entry.norad_id,
                orbit_class=entry.orbit_class,
                object_name=entry.object_name,
                series=series,
                intervals=labelling.intervals,
                unmatched=labelling.unmatched,
                digest=SeriesDigest(
                    norad_id=entry.norad_id,
                    n_elsets=len(cleaned),
                    sha256=series_sha256(cleaned),
                ),
            )
        )
    return LabelledDataset(
        dataset_version=recipe.dataset_version, objects=objects, labels=all_labels
    )


def verify(
    recipe: Recipe,
    fetcher: Fetcher,
    manifest: Manifest,
    labels_by_norad: Mapping[int, Sequence[ManeuverLabel]] | None = None,
) -> list[str]:
    """Reconstruct ``recipe`` and check it against the pinned ``manifest``.

    Returns the list of mismatch descriptions from :meth:`Manifest.verify` — empty means the
    reconstruction reproduces the pinned dataset byte-for-byte.
    """
    dataset = reconstruct(recipe, fetcher, labels_by_norad)
    return manifest.verify(dataset.manifest())
