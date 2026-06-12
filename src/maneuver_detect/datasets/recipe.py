"""The pinned reconstruction recipe — the catalogue + per-object fetch parameters (D2).

A :class:`Recipe` is the committed specification a reconstruction re-derives the dataset from: the
exact set of objects, their orbit class and label source, and the per-object catalogue source and
epoch window to fetch. It carries no catalogue *data* — only the parameters needed to re-fetch it —
which is what keeps the recipe-first model compliant (the raw Space-Track series is never shipped;
each user reconstructs from their own account). The companion ``manifest`` module pins the content
hash of each reconstructed series.

The recipe itself lives in :mod:`~maneuver_detect.datasets.catalogue` (``recipe``); this module is
the schema and its canonical JSON serialisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from maneuver_detect.labels.record import OrbitClass

__all__ = ["Recipe", "RecipeEntry"]


@dataclass(frozen=True)
class RecipeEntry:
    """One object in the recipe — what to fetch and which labels attach to it.

    Attributes:
        norad_id: NORAD catalogue id of the object (the series fetch key).
        orbit_class: The object's orbit class.
        object_name: A human-readable name, for the recipe's readability (e.g. ``"Jason-2"``).
        catalogue_source: The series source to fetch from (``"spacetrack"`` for multi-year history).
        label_source: The maneuver-label source for this object
            (:data:`~maneuver_detect.labels.record.SOURCE_DORIS_IDS` /
            :data:`~maneuver_detect.labels.record.SOURCE_GPS_NANU`).
        label_ref: The source-native key the label fetch uses — the DORIS satellite code (e.g.
            ``"ja2"`` for ``ja2man.txt``) or the GPS ``"SVN62"``.
        start: ISO-8601 start of the epoch window — scopes both the series fetch and the object's
            maneuver labels (``None`` for the full history).
        end: ISO-8601 end of the epoch window, likewise scoping the series and the labels (``None``
            for open-ended).
    """

    norad_id: int
    orbit_class: OrbitClass
    object_name: str
    catalogue_source: str
    label_source: str
    label_ref: str
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class Recipe:
    """A pinned, versioned set of recipe entries — the reconstructable dataset specification.

    Attributes:
        dataset_version: The dataset version (aligned with a later Hub release; lockstep with the
            manifest computed from it).
        entries: The objects to reconstruct.
    """

    dataset_version: str
    entries: tuple[RecipeEntry, ...]

    def norad_ids(self) -> tuple[int, ...]:
        """The NORAD ids in the recipe, in entry order."""
        return tuple(entry.norad_id for entry in self.entries)

    def per_class_counts(self) -> dict[OrbitClass, int]:
        """Number of objects per orbit class (every class present, zero included)."""
        counts = dict.fromkeys(OrbitClass, 0)
        for entry in self.entries:
            counts[entry.orbit_class] += 1
        return counts

    def to_json(self) -> str:
        """Serialise to canonical, NORAD-sorted JSON (a stable, committable artifact)."""
        payload = {
            "dataset_version": self.dataset_version,
            "entries": [
                {
                    "norad_id": e.norad_id,
                    "orbit_class": e.orbit_class.value,
                    "object_name": e.object_name,
                    "catalogue_source": e.catalogue_source,
                    "label_source": e.label_source,
                    "label_ref": e.label_ref,
                    "start": e.start,
                    "end": e.end,
                }
                for e in sorted(self.entries, key=lambda e: e.norad_id)
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Recipe:
        """Parse a recipe from :meth:`to_json` output."""
        data = json.loads(text)
        entries = tuple(
            RecipeEntry(
                norad_id=int(item["norad_id"]),
                orbit_class=OrbitClass(str(item["orbit_class"])),
                object_name=str(item["object_name"]),
                catalogue_source=str(item["catalogue_source"]),
                label_source=str(item["label_source"]),
                label_ref=str(item["label_ref"]),
                start=None if item["start"] is None else str(item["start"]),
                end=None if item["end"] is None else str(item["end"]),
            )
            for item in data["entries"]
        )
        return cls(dataset_version=str(data["dataset_version"]), entries=entries)
