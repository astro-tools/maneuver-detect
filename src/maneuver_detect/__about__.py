"""The package version, in a standalone leaf module.

Kept separate from :mod:`maneuver_detect` so a module that only needs the version string (e.g. the
data layer's HTTP ``User-Agent``) can import it without triggering the package's top-level imports —
which would otherwise be a circular import while the package is still initialising.
"""

from __future__ import annotations

__version__ = "0.2.0"
