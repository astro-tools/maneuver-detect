"""The typed exception hierarchy for maneuver-detect.

Every failure the library raises on its own behalf is one of these, so a caller can catch
:class:`ManeuverDetectError` to mean "something this library did, not a bug in the caller". The
data layer is the first consumer: :class:`DataSourceError` for a network / upstream-API failure
and :class:`MissingCredentialError` for a Space-Track query attempted without credentials. Later
layers extend the hierarchy under the same root.
"""

from __future__ import annotations

__all__ = [
    "DataSourceError",
    "ManeuverDetectError",
    "MissingCredentialError",
]


class ManeuverDetectError(Exception):
    """Root of the maneuver-detect exception hierarchy."""


class DataSourceError(ManeuverDetectError):
    """A network or upstream-API failure fetching from a catalogue source.

    Raised when a source (CelesTrak, Space-Track) is unreachable, returns an error status, or
    returns an unparseable payload, and no cached value is available to fall back on. The
    :attr:`source` attribute names the offending source so a caller can tell which leg failed.
    """

    def __init__(self, message: str, *, source: str) -> None:
        if not source:
            raise ValueError("DataSourceError requires a non-empty source name")
        super().__init__(message)
        self.source = source


class MissingCredentialError(ManeuverDetectError):
    """A credentialled source was queried without a complete credential.

    Raised by the Space-Track fetcher when the user has not supplied credentials (or supplied a
    partial set), instead of failing silently or leaking a confusing transport error. The
    :attr:`source` attribute names the source and :attr:`missing_fields` lists the credential
    fields that were absent or empty, so the remediation is precise.
    """

    def __init__(self, message: str, *, source: str, missing_fields: list[str]) -> None:
        super().__init__(message)
        self.source = source
        self.missing_fields = list(missing_fields)
