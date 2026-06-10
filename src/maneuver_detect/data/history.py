"""Per-NORAD mean-element time-series assembly — the canonical detector input.

``assemble`` turns a cleaned :class:`~maneuver_detect.data.elset.Elset` sequence into the canonical
mean-element DataFrame: one row per epoch, the SGP4 mean elements in the TEME frame, the derived
semi-major axis, and the inter-elset spacing. ``build_series`` is the convenience that cleans
first (:func:`~maneuver_detect.data.clean.clean_elsets`) then assembles.

This frame is the contract the feature and detector layers consume; its columns are
:data:`MEAN_ELEMENT_COLUMNS`. The elements are carried through verbatim from the catalog (TEME, no
frame change); the only derived quantities are ``semi_major_axis`` (from the mean motion via
Kepler's third law) and ``dt_days`` (days since the previous elset, which makes the irregular
cadence and any re-acquisition gaps visible to downstream without this layer acting on them).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from maneuver_detect.data.clean import clean_elsets
from maneuver_detect.data.elset import Elset

__all__ = ["MEAN_ELEMENT_COLUMNS", "assemble", "build_series"]

#: The canonical mean-element series columns, in order. ``epoch`` is timezone-aware UTC; angles
#: (``inclination``, ``raan``, ``arg_perigee``, ``mean_anomaly``) are degrees; ``mean_motion`` is
#: revolutions per day; ``semi_major_axis`` is kilometres; ``bstar`` is inverse earth radii;
#: ``dt_days`` is days since the previous row (``NaN`` for the first). The six mean elements plus
#: ``bstar`` are the SGP4 TEME mean elements; ``semi_major_axis`` is derived from ``mean_motion``.
MEAN_ELEMENT_COLUMNS: tuple[str, ...] = (
    "epoch",
    "norad_id",
    "mean_motion",
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "raan",
    "arg_perigee",
    "mean_anomaly",
    "bstar",
    "dt_days",
)

_DATETIME_DTYPE = "datetime64[ns, UTC]"

# Earth gravitational parameter, WGS72 (km^3 / s^2) — the value SGP4 itself uses, so the derived
# semi-major axis is consistent with the propagator the mean elements were fit for.
_MU_WGS72_KM3_S2 = 398600.8


def _semi_major_axis_km(mean_motion_rev_per_day: float) -> float:
    """Kozai-mean semi-major axis (km) from mean motion via ``a = (mu / n^2)^(1/3)``."""
    n_rad_per_s = mean_motion_rev_per_day * 2.0 * math.pi / 86400.0
    return float((_MU_WGS72_KM3_S2 / (n_rad_per_s * n_rad_per_s)) ** (1.0 / 3.0))


def _dt_days(elsets: Sequence[Elset]) -> list[float]:
    """Days since the previous elset for each row; ``NaN`` for the first, ``[]`` for no rows."""
    gaps: list[float] = []
    previous: Elset | None = None
    for elset in elsets:
        if previous is None:
            gaps.append(math.nan)
        else:
            gaps.append((elset.epoch - previous.epoch).total_seconds() / 86400.0)
        previous = elset
    return gaps


def assemble(elsets: Sequence[Elset]) -> pd.DataFrame:
    """Assemble ``elsets`` into the canonical mean-element DataFrame (:data:`MEAN_ELEMENT_COLUMNS`).

    Expects cleaned input (use :func:`build_series` to clean first); the rows are sorted by epoch
    defensively. An empty sequence yields an empty frame that still carries the full schema and
    dtypes, so a series with no elsets has the same shape as one with many.
    """
    ordered = sorted(elsets, key=lambda elset: elset.epoch)
    data = {
        "epoch": pd.Series([pd.Timestamp(e.epoch) for e in ordered], dtype=_DATETIME_DTYPE),
        "norad_id": pd.Series([e.norad_id for e in ordered], dtype="int64"),
        "mean_motion": pd.Series([e.mean_motion for e in ordered], dtype="float64"),
        "semi_major_axis": pd.Series(
            [_semi_major_axis_km(e.mean_motion) for e in ordered], dtype="float64"
        ),
        "eccentricity": pd.Series([e.eccentricity for e in ordered], dtype="float64"),
        "inclination": pd.Series([e.inclination for e in ordered], dtype="float64"),
        "raan": pd.Series([e.raan for e in ordered], dtype="float64"),
        "arg_perigee": pd.Series([e.arg_perigee for e in ordered], dtype="float64"),
        "mean_anomaly": pd.Series([e.mean_anomaly for e in ordered], dtype="float64"),
        "bstar": pd.Series([e.bstar for e in ordered], dtype="float64"),
        "dt_days": pd.Series(_dt_days(ordered), dtype="float64"),
    }
    return pd.DataFrame(data, columns=list(MEAN_ELEMENT_COLUMNS))


def build_series(elsets: Sequence[Elset]) -> pd.DataFrame:
    """Clean ``elsets`` then assemble the canonical series — the fetch-to-frame convenience.

    Equivalent to ``assemble(clean_elsets(elsets))``: the one call a caller hands a fetcher's raw
    elsets to get the detector-ready mean-element DataFrame.
    """
    return assemble(clean_elsets(elsets))
