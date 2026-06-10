"""XDG-aware on-disk cache for catalogue responses.

One JSON file per ``(source, key)`` under a platformdirs-resolved cache root
(``~/.cache/maneuver-detect/`` on Linux, ``%LOCALAPPDATA%\\maneuver-detect\\Cache\\`` on Windows).
The fetchers cache the **raw OMM payload** each source returned, so a cache hit re-parses to the
same :class:`~maneuver_detect.data.elset.Elset` sequence without a network round-trip — which is
both the offline-resilience story and the rate-limit discipline each source asks for.

Caching is a *local* copy on the user's own disk: it is not redistribution, so it sits inside the
distribution model (D2) for Space-Track data the same way "save it locally" sits inside
Space-Track's own API rules.

**Atomic writes.** Every write goes to a tempfile in the destination directory, then
``os.replace`` swaps it in — atomic on POSIX (``rename``) and Windows (``MoveFileExW``). A reader
sees the old file or the new one, never a torn write, so parallel reconstruction runs sharing one
cache dir are safe without a lockfile.

**Disabled mode.** Setting ``MANEUVER_DETECT_CACHE_DIR=""`` (empty) disables the cache: ``get`` /
``get_stale`` always miss and ``put`` is a no-op — handy for a pristine reconstruction check.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import platformdirs

_logger = logging.getLogger(__name__)

_CACHE_DIR_ENV_VAR = "MANEUVER_DETECT_CACHE_DIR"
_APP_NAME = "maneuver-detect"

# Per-source default cache lifetimes (seconds).
#
# CelesTrak serves only the *current* GP for an object, which turns over a few times a day; a 6h
# TTL revalidates within one cadence (and the fetcher's If-Modified-Since revalidation makes an
# expired-but-unchanged entry free to refresh). Space-Track's gp_history is an *immutable* archive
# of past elsets — once fetched, a closed epoch range never changes — so a long 7-day TTL matches
# Space-Track's own guidance ("save it locally; do not query for the same data repeatedly").
DEFAULT_TTLS: Mapping[str, float] = {
    "celestrak": 6 * 60 * 60,
    "spacetrack": 7 * 24 * 60 * 60,
}


@dataclass(frozen=True)
class CacheHit:
    """A successful cache lookup: the stored value and when it was written."""

    value: Any  # JSON-serialisable (the raw OMM payload)
    fetched_at: datetime  # tz-aware UTC


def _resolve_cache_dir(explicit: Path | None) -> Path | None:
    """Resolve the cache directory: constructor arg → env var → platformdirs default.

    The env var set to an empty string disables the cache (returns ``None``).
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(_CACHE_DIR_ENV_VAR)
    if env is not None:
        if env == "":
            return None
        return Path(env)
    return Path(platformdirs.user_cache_dir(_APP_NAME))


def _hash_key(key: str) -> str:
    """Hash a key to a filesystem-safe, fixed-length filename."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class Cache:
    """On-disk cache for catalogue responses. See the module docstring for the safety story."""

    def __init__(self, directory: Path | None = None) -> None:
        self._dir = _resolve_cache_dir(directory)

    @property
    def enabled(self) -> bool:
        """``True`` unless the cache is disabled (empty-string env override)."""
        return self._dir is not None

    @property
    def directory(self) -> Path | None:
        """The resolved cache root, or ``None`` when caching is disabled."""
        return self._dir

    def get(self, source: str, key: str, *, ttl_s: float) -> CacheHit | None:
        """Return the cached value if present and younger than ``ttl_s``, else ``None``."""
        hit = self.get_stale(source, key)
        if hit is None:
            return None
        age = (datetime.now(tz=timezone.utc) - hit.fetched_at).total_seconds()
        if age > ttl_s:
            return None
        return hit

    def get_stale(self, source: str, key: str) -> CacheHit | None:
        """Return the cached value regardless of age, or ``None`` if missing / corrupt."""
        if self._dir is None:
            return None
        path = self._path(source, key)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            fetched_at = datetime.fromisoformat(payload["fetched_at"])
            value = payload["value"]
        except (OSError, ValueError, KeyError) as exc:
            # Corrupt or truncated entry — treat as a miss; the next put overwrites it cleanly.
            _logger.warning("ignoring corrupt cache entry at %s: %s", path, exc)
            return None
        return CacheHit(value=value, fetched_at=fetched_at)

    def put(self, source: str, key: str, value: Any) -> None:
        """Write ``value`` to the cache atomically, stamping ``fetched_at``. No-op when disabled.

        Re-``put``-ting an unchanged ``value`` is how the fetchers "touch" an entry after an
        If-Modified-Since revalidation: same payload, fresh timestamp.
        """
        if self._dir is None:
            return
        path = self._path(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            # Stored next to the value so the sha256-named files are not opaque to an operator
            # who greps the cache dir for a satellite they are debugging.
            "key": key,
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "value": value,
        }
        # The tempfile MUST live in the destination directory so the rename stays on one
        # filesystem (where it is atomic); a /tmp tempfile would degrade to a cross-device copy.
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise

    def _path(self, source: str, key: str) -> Path:
        if self._dir is None:
            raise RuntimeError("cache is disabled; check `enabled` before calling _path")
        return self._dir / source / f"{_hash_key(key)}.json"


_default_cache: Cache | None = None


def default_cache() -> Cache:
    """Return the lazy module-level cache singleton.

    Fetchers default to this so they share one cache per process. Tests should construct their own
    :class:`Cache` with a ``tmp_path`` directory and inject it, rather than touching the singleton.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache()
    return _default_cache
