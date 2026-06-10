"""Space-Track credential resolution from the environment.

Space-Track is a per-person, USSPACECOM-approved account: the library never ships or proxies
credentials, it reads the user's own from the environment and passes them straight through to the
login request. The convention is two environment variables:

- ``SPACETRACK_USERNAME``
- ``SPACETRACK_PASSWORD``

:func:`require_spacetrack_credential` returns both when present, or raises a typed
:class:`~maneuver_detect.errors.MissingCredentialError` naming the missing fields — never a silent
failure or a confusing transport error from an unauthenticated query. This module never logs or
echoes credential values.
"""

from __future__ import annotations

import os

from maneuver_detect.errors import MissingCredentialError

__all__ = ["SPACETRACK_ENV_VARS", "SPACETRACK_SOURCE", "require_spacetrack_credential"]

SPACETRACK_SOURCE = "spacetrack"

# field name -> environment variable. The dict the fetcher consumes is keyed by the field names.
SPACETRACK_ENV_VARS: dict[str, str] = {
    "username": "SPACETRACK_USERNAME",
    "password": "SPACETRACK_PASSWORD",
}


def require_spacetrack_credential() -> dict[str, str]:
    """Return ``{"username": ..., "password": ...}`` from the environment, or raise.

    Every required field must have a non-empty value; a partial credential is treated as absent so
    a half-configured environment fails loudly rather than producing a baffling login rejection.

    Raises:
        MissingCredentialError: When either field is unset or empty. Its ``missing_fields`` lists
            the field names not satisfied and its message names the environment variables to set.
    """
    credential: dict[str, str] = {}
    missing: list[str] = []
    for field, env_var in SPACETRACK_ENV_VARS.items():
        value = os.environ.get(env_var, "")
        if value:
            credential[field] = value
        else:
            missing.append(field)
    if missing:
        missing_vars = ", ".join(SPACETRACK_ENV_VARS[field] for field in missing)
        raise MissingCredentialError(
            f"Space-Track credentials are required; set {missing_vars} in the environment",
            source=SPACETRACK_SOURCE,
            missing_fields=missing,
        )
    return credential
