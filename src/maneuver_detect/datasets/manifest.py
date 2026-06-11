"""The per-series content-hash manifest — the integrity check recipe-first reconstruction relies on.

Because the dataset is distributed as *labels + a pinned recipe* rather than raw catalogue data
(D2), a reconstruction is only trustworthy if it is **byte-deterministic**: re-running the recipe on
the same pinned input must yield the identical series. The manifest pins a SHA-256 per object so a
reconstruction can be verified bit-for-bit against the one the baselines were built on (D8).

:func:`series_sha256` hashes the **verbatim** cleaned catalogue elements of one object's series —
the parse-faithful values only (no float arithmetic), so the digest is identical across platforms.
The derived ``semi_major_axis`` and ``dt_days`` are deterministic functions of those elements and
are deliberately excluded (a ``pow`` can differ by a unit in the last place across math libraries).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from maneuver_detect.data.elset import Elset

__all__ = ["Manifest", "SeriesDigest", "series_sha256"]


def _canonical_record(elset: Elset) -> dict[str, object]:
    """The verbatim catalogue fields that pin one elset, in a fixed order (nothing derived)."""
    return {
        "norad_id": elset.norad_id,
        "epoch": elset.epoch.isoformat(),
        "mean_motion": elset.mean_motion,
        "eccentricity": elset.eccentricity,
        "inclination": elset.inclination,
        "raan": elset.raan,
        "arg_perigee": elset.arg_perigee,
        "mean_anomaly": elset.mean_anomaly,
        "bstar": elset.bstar,
        "mean_motion_dot": elset.mean_motion_dot,
        "mean_motion_ddot": elset.mean_motion_ddot,
    }


def series_sha256(elsets: Sequence[Elset]) -> str:
    """SHA-256 over one object's epoch-ordered verbatim elements.

    The elsets are sorted by epoch first, so the digest is independent of fetch order (the cleaning
    layer already orders a series, but hashing must not depend on that). Floats are serialised via
    their round-trip ``repr`` (what :func:`json.dumps` emits), which is platform-independent for the
    correctly-rounded values a catalogue parse produces — so the digest is byte-stable everywhere.
    """
    ordered = sorted(elsets, key=lambda elset: elset.epoch)
    blob = json.dumps(
        [_canonical_record(elset) for elset in ordered], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class SeriesDigest:
    """The content digest of one object's reconstructed series.

    Attributes:
        norad_id: NORAD id of the object.
        n_elsets: Number of cleaned elsets in the series (a cheap check alongside the hash).
        sha256: The :func:`series_sha256` digest of the series.
    """

    norad_id: int
    n_elsets: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    """A pinned set of per-series digests for one dataset version.

    Attributes:
        dataset_version: The dataset version the digests were computed for.
        digests: One :class:`SeriesDigest` per object in the recipe.
    """

    dataset_version: str
    digests: tuple[SeriesDigest, ...]

    def __post_init__(self) -> None:
        # Canonicalise digest order (by NORAD id) so equality and serialisation don't depend on the
        # order objects were reconstructed in — the manifest is keyed by object, not ordered.
        object.__setattr__(
            self, "digests", tuple(sorted(self.digests, key=lambda digest: digest.norad_id))
        )

    def by_norad(self) -> dict[int, SeriesDigest]:
        """Index the digests by NORAD id."""
        return {digest.norad_id: digest for digest in self.digests}

    def to_json(self) -> str:
        """Serialise to canonical, NORAD-sorted JSON (a stable, committable artifact)."""
        payload = {
            "dataset_version": self.dataset_version,
            "digests": [
                {"norad_id": d.norad_id, "n_elsets": d.n_elsets, "sha256": d.sha256}
                for d in sorted(self.digests, key=lambda d: d.norad_id)
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Parse a manifest from :meth:`to_json` output."""
        data = json.loads(text)
        digests = tuple(
            SeriesDigest(
                norad_id=int(item["norad_id"]),
                n_elsets=int(item["n_elsets"]),
                sha256=str(item["sha256"]),
            )
            for item in data["digests"]
        )
        return cls(dataset_version=str(data["dataset_version"]), digests=digests)

    def verify(self, other: Manifest) -> list[str]:
        """Compare a freshly reconstructed manifest ``other`` against this pinned one.

        Returns a list of human-readable mismatch descriptions — an empty list means the
        reconstruction is byte-identical to the pinned dataset. Reports a version difference, any
        object missing or unexpected, and per-object hash / count mismatches.
        """
        mismatches: list[str] = []
        if self.dataset_version != other.dataset_version:
            mismatches.append(
                f"dataset_version: pinned {self.dataset_version!r} != got {other.dataset_version!r}"
            )
        pinned = self.by_norad()
        got = other.by_norad()
        for norad_id in sorted(set(pinned) | set(got)):
            want = pinned.get(norad_id)
            have = got.get(norad_id)
            if want is None:
                mismatches.append(f"NORAD {norad_id}: unexpected (not in the pinned manifest)")
            elif have is None:
                mismatches.append(f"NORAD {norad_id}: missing from the reconstruction")
            elif want.sha256 != have.sha256:
                mismatches.append(
                    f"NORAD {norad_id}: hash mismatch "
                    f"(pinned {want.sha256[:12]}… != got {have.sha256[:12]}…)"
                )
            elif want.n_elsets != have.n_elsets:
                mismatches.append(f"NORAD {norad_id}: n_elsets {want.n_elsets} != {have.n_elsets}")
        return mismatches
