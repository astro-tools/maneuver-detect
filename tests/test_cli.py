"""Tests for ``maneuver_detect.cli`` — argument parsing and the ``detect`` subcommand.

The network is never exercised: the NORAD-id path is driven through a monkeypatched
``datasets.tle_history``, and the TLE-file path reads a small committed-in-test fixture.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maneuver_detect import cli, datasets, detect
from maneuver_detect.cli import _build_parser, _load_history, _render
from maneuver_detect.errors import MissingCredentialError
from maneuver_detect.schema import COLUMNS, Maneuver, ManeuverType, to_frame

# A valid ISS line pair as column-1-68 bodies; the day-of-year varies line 1 per epoch and the
# column-69 modulo-10 checksum is appended per line so the parser's checksum guard accepts the
# fixture. Four daily epochs — enough to assemble a series, too few for the detector to fire, so the
# TLE-file path exercises end-to-end without depending on a detection.
_TLE_LINE1_BODY = "1 25544U 98067A   240{day:02d}.50000000  .00016717  00000-0  10270-3 0  900"
_TLE_LINE2_BODY = "2 25544  51.6400 208.0000 0006703 130.0000 325.0000 15.5000000012345"


def _tle_line(body: str) -> str:
    """Append the TLE column-69 modulo-10 checksum to a 68-character line body."""
    total = sum((ord(c) - 48 if "0" <= c <= "9" else 1 if c == "-" else 0) for c in body)
    return f"{body}{total % 10}"


def _write_tle_file(tmp_path: Path, days: tuple[int, ...] = (1, 2, 3, 4)) -> Path:
    lines = ["ISS (ZARYA)"]
    for day in days:
        lines.append(_tle_line(_TLE_LINE1_BODY.format(day=day)))
        lines.append(_tle_line(_TLE_LINE2_BODY))
    path = tmp_path / "history.tle"
    path.write_text("\n".join(lines) + "\n")
    return path


def _history_with_maneuver(norad_id: int = 25544, n: int = 60, seed: int = 1) -> pd.DataFrame:
    """A deterministic daily mean-element series with one clear in-track jump at the midpoint."""
    rng = np.random.default_rng(seed)
    epochs = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    mu_km3_s2 = 398600.8
    a = 7000.0 + rng.normal(0.0, 0.005, n)
    a[n // 2 :] += 0.6  # a clean in-track jump, well above the metre-level TLE noise
    e = 0.001 + rng.normal(0.0, 1e-5, n)
    inc = 51.6 + rng.normal(0.0, 3e-4, n)
    raan = (0.05 * np.arange(n)) % 360.0
    mean_motion = np.sqrt(mu_km3_s2 / a**3) * 86400.0 / (2.0 * math.pi)
    return pd.DataFrame(
        {
            "epoch": pd.Series(epochs, dtype="datetime64[ns, UTC]"),
            "norad_id": norad_id,
            "mean_motion": mean_motion,
            "semi_major_axis": a,
            "eccentricity": e,
            "inclination": inc,
            "raan": raan,
            "arg_perigee": 0.0,
            "mean_anomaly": 0.0,
            "bstar": 0.0,
            "dt_days": np.concatenate(([np.nan], np.ones(n - 1))),
        }
    )


def test_detect_parses() -> None:
    args = _build_parser().parse_args(["detect", "25544"])
    assert args.command == "detect"
    assert args.target == "25544"
    assert args.model == "classical"
    # The new options carry sensible defaults.
    assert args.source == "spacetrack"
    assert args.start is None and args.end is None
    assert args.format == "table"
    assert args.output is None


def test_detect_parses_all_options() -> None:
    args = _build_parser().parse_args(
        [
            "detect",
            "history.tle",
            "--source",
            "celestrak",
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-30",
            "--format",
            "json",
            "-o",
            "out.json",
        ]
    )
    assert args.target == "history.tle"
    assert args.source == "celestrak"
    assert args.start == "2024-01-01" and args.end == "2024-06-30"
    assert args.format == "json"
    assert args.output == "out.json"


@pytest.mark.parametrize("flag_value", [["--source", "nope"], ["--format", "yaml"]])
def test_detect_rejects_invalid_choices(flag_value: list[str]) -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["detect", "25544", *flag_value])


def test_dataset_build_parses() -> None:
    args = _build_parser().parse_args(["dataset", "build", "--out", "dist"])
    assert args.command == "dataset"
    assert args.dataset_command == "build"
    assert args.out == "dist"
    assert args.nanu_start_year == 2016  # default recent-window start
    assert args.nanu_end_year is None  # defaults to the current year at run time


def test_dataset_build_nanu_year_flags() -> None:
    args = _build_parser().parse_args(
        [
            "dataset",
            "build",
            "--out",
            "dist",
            "--nanu-start-year",
            "2010",
            "--nanu-end-year",
            "2020",
        ]
    )
    assert args.nanu_start_year == 2010
    assert args.nanu_end_year == 2020


def test_dataset_requires_an_action() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["dataset"])


def test_dataset_build_requires_out() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["dataset", "build"])


# --- rendering -----------------------------------------------------------------------------


def _one_maneuver_frame() -> pd.DataFrame:
    return to_frame(
        [
            Maneuver(
                epoch=pd.Timestamp("2024-01-30T12:00:00", tz="UTC"),
                confidence=0.94,
                type=ManeuverType.IN_TRACK,
                delta_v_estimate=0.33,
                norad_id=25544,
                elset_epoch_before=pd.Timestamp("2024-01-30T00:00:00", tz="UTC"),
                elset_epoch_after=pd.Timestamp("2024-01-31T00:00:00", tz="UTC"),
            ),
            Maneuver(  # a below-floor maneuver with no reported dv -> NaN renders as ASCII
                epoch=pd.Timestamp("2024-02-10T00:00:00", tz="UTC"),
                confidence=0.5,
                type=ManeuverType.CROSS_TRACK,
                delta_v_estimate=None,
                norad_id=25544,
                elset_epoch_before=pd.Timestamp("2024-02-09T00:00:00", tz="UTC"),
                elset_epoch_after=pd.Timestamp("2024-02-11T00:00:00", tz="UTC"),
            ),
        ]
    )


def test_render_table_is_ascii_with_canonical_columns() -> None:
    text = _render(_one_maneuver_frame(), "table")
    assert text.isascii()  # cp1252-safe: the dv column is delta_v_estimate, never a literal Δ
    assert "delta_v_estimate" in text and "Δ" not in text
    assert "in-track" in text


def test_render_table_empty_says_so() -> None:
    from maneuver_detect.schema import empty_frame

    assert _render(empty_frame(), "table") == "No maneuvers detected."


def test_render_csv_round_trips_columns() -> None:
    text = _render(_one_maneuver_frame(), "csv")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == list(COLUMNS)
    assert len(rows) == 3  # header + two maneuvers
    assert text.isascii()


def test_render_json_is_a_record_list() -> None:
    records = json.loads(_render(_one_maneuver_frame(), "json"))
    assert isinstance(records, list) and len(records) == 2
    assert set(COLUMNS) <= set(records[0])


# --- history resolution (TLE file vs NORAD id) ---------------------------------------------


def test_load_history_reads_a_tle_file(tmp_path: Path) -> None:
    path = _write_tle_file(tmp_path)
    history = _load_history(str(path), source="spacetrack", start=None, end=None)
    assert len(history) == 4
    assert int(history["norad_id"].iloc[0]) == 25544


def test_load_history_windows_a_tle_file(tmp_path: Path) -> None:
    path = _write_tle_file(tmp_path, days=(1, 2, 3, 4))
    # Epochs sit at 12:00 UTC (day-of-year .5), so an inclusive [02 00:00, 04 00:00] window keeps
    # the day-2 and day-3 epochs and drops day-1 (before) and day-4 (after).
    history = _load_history(str(path), source="spacetrack", start="2024-01-02", end="2024-01-04")
    assert [ts.day for ts in history["epoch"]] == [2, 3]


def test_load_history_norad_id_uses_tle_history(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake(norad_id: int, *, start: str | None, end: str | None, source: str) -> pd.DataFrame:
        seen.update(norad_id=norad_id, start=start, end=end, source=source)
        return _history_with_maneuver()

    monkeypatch.setattr(datasets, "tle_history", fake)
    history = _load_history("25544", source="celestrak", start="2024-01-01", end=None)
    assert len(history) == 60
    assert seen == {"norad_id": 25544, "start": "2024-01-01", "end": None, "source": "celestrak"}


def test_load_history_bad_target_raises() -> None:
    with pytest.raises(ValueError, match="neither an existing file nor a NORAD"):
        _load_history("not-a-file-or-id", source="spacetrack", start=None, end=None)


# --- end-to-end through cli.main -----------------------------------------------------------


def test_detect_on_tle_file_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_tle_file(tmp_path)
    rc = cli.main(["detect", str(path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.isascii()
    # Four epochs is below the detector's window, so nothing is found — but it runs cleanly.
    assert captured.out.strip() == "No maneuvers detected."


def test_detect_on_norad_id_matches_the_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    history = _history_with_maneuver()

    def fake(norad_id: int, *, start: str | None, end: str | None, source: str) -> pd.DataFrame:
        return history

    monkeypatch.setattr(datasets, "tle_history", fake)

    rc = cli.main(["detect", "25544"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.isascii()
    assert "in-track" in captured.out  # the injected maneuver is detected and printed
    # "the same result as the equivalent API call": CLI output == the rendered detect() frame.
    expected = _render(detect(history, model="classical"), "table")
    assert captured.out.rstrip("\n") == expected


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_detect_machine_formats_parse(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fmt: str
) -> None:
    monkeypatch.setattr(datasets, "tle_history", lambda *a, **k: _history_with_maneuver())
    rc = cli.main(["detect", "25544", "--format", fmt])
    out = capsys.readouterr().out
    assert rc == 0
    if fmt == "csv":
        rows = list(csv.reader(io.StringIO(out)))
        assert rows[0] == list(COLUMNS) and len(rows) == 2
    else:
        assert len(json.loads(out)) == 1


def test_detect_output_writes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(datasets, "tle_history", lambda *a, **k: _history_with_maneuver())
    out_path = tmp_path / "result.csv"
    rc = cli.main(["detect", "25544", "--format", "csv", "--output", str(out_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""  # the result went to the file, not stdout
    assert "wrote 1 maneuver(s)" in captured.err
    rows = list(csv.reader(io.StringIO(out_path.read_text())))
    assert rows[0] == list(COLUMNS) and len(rows) == 2


# --- error handling: a one-line message and a non-zero exit, never a traceback -------------


def test_detect_missing_credentials_is_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def raise_missing(*args: object, **kwargs: object) -> pd.DataFrame:
        raise MissingCredentialError(
            "Space-Track credentials are required; set SPACETRACK_USERNAME, SPACETRACK_PASSWORD",
            source="spacetrack",
            missing_fields=["username", "password"],
        )

    monkeypatch.setattr(datasets, "tle_history", raise_missing)
    rc = cli.main(["detect", "25544"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "error:" in captured.err and "credential" in captured.err.lower()
    assert "Traceback" not in captured.err


def test_detect_unknown_model_is_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(datasets, "tle_history", lambda *a, **k: _history_with_maneuver())
    rc = cli.main(["detect", "25544", "--model", "no-such-model"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "unknown model" in captured.err
    assert "Traceback" not in captured.err


def test_detect_bad_target_is_clean(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["detect", "not-a-file-or-id"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "neither an existing file nor a NORAD" in captured.err


def test_dataset_build_command_dispatches_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The whole `dataset build` execution path is driven offline: the networked label fetch, the
    # Space-Track fetcher, and the reconstruction are replaced with fakes, so the CLI dispatch, the
    # progress logging, and the report rendering are exercised without a network or credentials.
    from maneuver_detect.datasets.build import BuildReport
    from maneuver_detect.labels.labeller import ClassCoverage, CoverageReport
    from maneuver_detect.labels.record import OrbitClass

    captured: dict[str, object] = {}

    def fake_fetch_labels(
        recipe: object,
        client: object,
        *,
        nanu_start_year: int,
        nanu_end_year: int,
        rate_limiter: object = None,
    ) -> dict[int, list[object]]:
        captured["years"] = (nanu_start_year, nanu_end_year)
        return {}

    class FakeFetcher:
        def __enter__(self) -> FakeFetcher:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_build_dataset(
        recipe: object, fetcher: object, labels: object, out_dir: str
    ) -> BuildReport:
        captured["out_dir"] = out_dir
        per_class = {
            orbit_class: ClassCoverage(
                orbit_class=orbit_class,
                n_events=0,
                n_with_delta_v=0,
                n_with_norad=0,
                sources=(),
            )
            for orbit_class in OrbitClass
        }
        return BuildReport(
            paths={name: tmp_path / f"{name}.json" for name in ("recipe", "labels", "manifest")},
            n_objects=2,
            coverage=CoverageReport(per_class=per_class, total=0),
        )

    monkeypatch.setattr("maneuver_detect.datasets.build.fetch_labels", fake_fetch_labels)
    monkeypatch.setattr("maneuver_detect.datasets.build.build_dataset", fake_build_dataset)
    monkeypatch.setattr("maneuver_detect.data.spacetrack.SpacetrackFetcher", FakeFetcher)

    rc = cli.main(
        [
            "dataset",
            "build",
            "--out",
            str(tmp_path),
            "--nanu-start-year",
            "2024",
            "--nanu-end-year",
            "2024",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "reconstructed 2 objects" in out
    for name in ("recipe", "labels", "manifest"):
        assert f"wrote {name}:" in out
    assert captured["years"] == (2024, 2024)
    assert captured["out_dir"] == str(tmp_path)
