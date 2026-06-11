"""Tests for ``maneuver_detect.labels.doris`` — the IDS ``man.txt`` parser.

Fixtures are synthetic lines crafted to the published ``man.readme`` format (illustrative values,
not redistributed operator files); they exercise each parameter-type axis mapping, multi-burn
aggregation, the TOPEX window-only branch, the crosswalk, and the TAI→UTC conversion.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from maneuver_detect.labels.doris import parse_doris
from maneuver_detect.labels.record import SOURCE_DORIS_IDS, OrbitClass
from maneuver_detect.schema import ManeuverType

_UTC = timezone.utc


def _burn(
    year: int, doy: int, hour: int, minute: int, second: float, dv: tuple[float, float, float]
) -> str:
    # 5 date sub-fields + duration + 3 ΔV components + 3 acc + 3 delta-acc = 15 tokens.
    return (
        f"{year} {doy:03d} {hour:02d} {minute:02d} {second:06.3f} "
        f"5.0e+00 {dv[0]:.6e} {dv[1]:.6e} {dv[2]:.6e} 0 0 0 0 0 0"
    )


def _event(
    code: str,
    begin: tuple[int, int, int, int],
    end: tuple[int, int, int, int],
    *,
    param: str | None = None,
    spot_type: str | None = None,
    burns: tuple[str, ...] = (),
) -> str:
    head = (
        f"{code} {begin[0]} {begin[1]:03d} {begin[2]:02d} {begin[3]:02d} "
        f"{end[0]} {end[1]:03d} {end[2]:02d} {end[3]:02d}"
    )
    if param is None:
        return head
    parts = [head]
    if spot_type is not None:
        parts.append(spot_type)
    parts.extend([param, str(len(burns)), *burns])
    return " ".join(parts)


# A JASON (007: Q,S,W = radial, along, cross) event with a single radial burn.
_JASON_RADIAL = _event(
    "JASO1",
    (2024, 10, 12, 0),
    (2024, 10, 13, 0),
    param="007",
    burns=(_burn(2024, 10, 12, 30, 0.0, (2.5, 0.0, 0.0)),),
)
# A JASON event with two along-track burns that sum to 2.5 m/s.
_JASON_TWO_BURNS = _event(
    "JASO2",
    (2024, 11, 0, 0),
    (2024, 11, 2, 0),
    param="007",
    burns=(
        _burn(2024, 11, 0, 30, 0.0, (0.0, 1.0, 0.0)),
        _burn(2024, 11, 1, 30, 0.0, (0.0, 1.5, 0.0)),
    ),
)
# ENVISAT (006: radial, along, cross) — an along-track (in-track) burn.
_ENVISAT_INTRACK = _event(
    "ENVI1",
    (2024, 20, 0, 0),
    (2024, 20, 1, 0),
    param="006",
    burns=(_burn(2024, 20, 0, 30, 0.0, (0.0, 1.2, 0.0)),),
)
# CryoSat-2 (006) — a cross-track burn.
_CRYOSAT_CROSS = _event(
    "CRYO2",
    (2024, 25, 0, 0),
    (2024, 25, 1, 0),
    param="006",
    burns=(_burn(2024, 25, 0, 30, 0.0, (0.0, 0.0, 3.0)),),
)
# SPOT (005: T,R,L = cross, along, radial) with the SPOT-only MCC type token; an L (radial) burn.
_SPOT_RADIAL = _event(
    "SPOT4",
    (2024, 30, 0, 0),
    (2024, 30, 1, 0),
    param="005",
    spot_type="MCC",
    burns=(_burn(2024, 30, 0, 30, 0.0, (0.0, 0.0, 4.0)),),
)
# TOPEX — window only, no burn detail.
_TOPEX_WINDOW = _event("TOPEX", (2024, 40, 0, 0), (2024, 40, 2, 0))
# An un-crosswalked code (window only) ingests with norad_id=None.
_UNMAPPED = _event("SWOT1", (2024, 50, 0, 0), (2024, 50, 1, 0))

_FIXTURE = "\n".join(
    [
        "# a comment line, skipped",
        "",
        _JASON_RADIAL,
        _JASON_TWO_BURNS,
        _ENVISAT_INTRACK,
        _CRYOSAT_CROSS,
        _SPOT_RADIAL,
        _TOPEX_WINDOW,
        _UNMAPPED,
    ]
)


def _by_code() -> dict[str, object]:
    return {label.source_ref.split()[0]: label for label in parse_doris(_FIXTURE)}


def test_parses_all_events_skipping_comments() -> None:
    assert len(parse_doris(_FIXTURE)) == 7


def test_empty_text_is_empty_list() -> None:
    assert parse_doris("") == []


def test_crosswalk_resolves_known_codes() -> None:
    by_code = _by_code()
    assert by_code["JASO1"].norad_id == 26997  # type: ignore[attr-defined]
    assert by_code["ENVI1"].norad_id == 27386  # type: ignore[attr-defined]
    assert by_code["CRYO2"].norad_id == 36508  # type: ignore[attr-defined]
    assert by_code["SPOT4"].norad_id == 25260  # type: ignore[attr-defined]
    assert by_code["TOPEX"].norad_id == 22076  # type: ignore[attr-defined]


def test_unmapped_code_has_none_norad() -> None:
    assert _by_code()["SWOT1"].norad_id is None  # type: ignore[attr-defined]


def test_axis_mapping_to_maneuver_type() -> None:
    by_code = _by_code()
    assert by_code["JASO1"].maneuver_type is ManeuverType.RADIAL  # type: ignore[attr-defined]
    assert by_code["ENVI1"].maneuver_type is ManeuverType.IN_TRACK  # type: ignore[attr-defined]
    assert by_code["CRYO2"].maneuver_type is ManeuverType.CROSS_TRACK  # type: ignore[attr-defined]
    assert by_code["SPOT4"].maneuver_type is ManeuverType.RADIAL  # type: ignore[attr-defined]


def test_delta_v_magnitude_and_burn_aggregation() -> None:
    by_code = _by_code()
    assert by_code["JASO1"].delta_v == pytest.approx(2.5)  # type: ignore[attr-defined]
    assert by_code["CRYO2"].delta_v == pytest.approx(3.0)  # type: ignore[attr-defined]
    assert by_code["SPOT4"].delta_v == pytest.approx(4.0)  # type: ignore[attr-defined]
    # Two along-track burns of 1.0 + 1.5 sum to a 2.5 m/s event.
    assert by_code["JASO2"].delta_v == pytest.approx(2.5)  # type: ignore[attr-defined]
    assert by_code["JASO2"].maneuver_type is ManeuverType.IN_TRACK  # type: ignore[attr-defined]


def test_topex_is_epoch_only_leo() -> None:
    topex = _by_code()["TOPEX"]
    assert topex.delta_v is None  # type: ignore[attr-defined]
    assert topex.maneuver_type is None  # type: ignore[attr-defined]
    assert topex.orbit_class is OrbitClass.LEO  # type: ignore[attr-defined]
    assert topex.source == SOURCE_DORIS_IDS  # type: ignore[attr-defined]


def test_epochs_are_utc_and_tai_converted() -> None:
    # The JASON burn is at TAI 2024 DOY 010 12:30:00. UTC is ~37 s earlier than the TAI clock, so
    # reinterpreting that clock reading as UTC overshoots the real UTC epoch by the TAI-UTC offset.
    jason = _by_code()["JASO1"]
    assert jason.epoch.tzinfo is timezone.utc  # type: ignore[attr-defined]
    naive_clock_as_utc = datetime(2024, 1, 10, 12, 30, tzinfo=_UTC)
    offset = (naive_clock_as_utc - jason.epoch).total_seconds()  # type: ignore[attr-defined]
    assert 30.0 < offset < 45.0


def test_truncated_burn_block_raises() -> None:
    # A header declaring one burn (param 007, N=1) but carrying no burn tokens.
    truncated = _event("JASO1", (2024, 10, 12, 0), (2024, 10, 13, 0)) + " 007 1"
    with pytest.raises(ValueError, match="truncated burn block"):
        parse_doris(truncated)
