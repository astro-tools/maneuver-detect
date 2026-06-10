"""Tests for ``maneuver_detect.data.credentials`` — Space-Track env credential resolution."""

from __future__ import annotations

import pytest

from maneuver_detect.data.credentials import require_spacetrack_credential
from maneuver_detect.errors import MissingCredentialError


def test_returns_credential_when_both_env_vars_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPACETRACK_USERNAME", "alice@example.com")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "s3cret")
    assert require_spacetrack_credential() == {
        "username": "alice@example.com",
        "password": "s3cret",
    }


def test_missing_both_lists_both_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPACETRACK_USERNAME", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    with pytest.raises(MissingCredentialError) as excinfo:
        require_spacetrack_credential()
    err = excinfo.value
    assert err.source == "spacetrack"
    assert sorted(err.missing_fields) == ["password", "username"]
    # The message names the env vars to set so the remediation is actionable.
    assert "SPACETRACK_USERNAME" in str(err) and "SPACETRACK_PASSWORD" in str(err)


def test_partial_credential_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPACETRACK_USERNAME", "alice@example.com")
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    with pytest.raises(MissingCredentialError) as excinfo:
        require_spacetrack_credential()
    assert excinfo.value.missing_fields == ["password"]


def test_empty_value_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPACETRACK_USERNAME", "alice@example.com")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "")
    with pytest.raises(MissingCredentialError) as excinfo:
        require_spacetrack_credential()
    assert excinfo.value.missing_fields == ["password"]
