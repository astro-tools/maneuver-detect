"""Train the BiLSTM baseline end to end and score it, fully offline on synthetic data.

This walks the whole learned-baseline workflow on a *synthetic* labelled population so it runs with
no credentials, no network, and no GPU: build per-object mean-element series with injected burns,
split them into train / val / test, train the BiLSTM through the shared Lightning harness, run the
trained detector, and score its detections through the same benchmark scorer the published
leaderboard uses. It trains a small model for a few epochs on CPU — a demonstration of the pipeline,
not a tuned run.

The real published checkpoint comes from the identical call on the *reconstructed* dataset: replace
the synthetic population with ``maneuver_detect.datasets.reconstruct(...)`` over the committed
recipe (needs Space-Track credentials), slice it by the frozen split with
``maneuver_detect.models.datamodule.objects_from_labelled_dataset``, and run ``train_bilstm`` on a
single GPU (``accelerator="auto"``) within the pinned compute budget; the measured cost and the test
scores are then recorded on the model card.

Run it with no arguments:

    python examples/train_bilstm.py

The output stays ASCII (the delta-v column prints as ``delta_v_estimate``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maneuver_detect.benchmark import ObjectExposure, ScoredLabel, score
from maneuver_detect.detectors.bilstm import BiLstmDetector
from maneuver_detect.labels.labeller import label_series
from maneuver_detect.labels.record import ManeuverLabel, OrbitClass
from maneuver_detect.models.bilstm import BiLstmConfig
from maneuver_detect.models.datamodule import ObjectSeries
from maneuver_detect.models.train import train_bilstm
from maneuver_detect.physics import EARTH_MU_KM3_S2, Orbit, gauss_forward, j2_secular_rates
from maneuver_detect.schema import ManeuverType, from_frame

_DEG = math.pi / 180.0
_SECONDS_PER_DAY = 86400.0
_LEO = Orbit(semi_major_axis_km=6778.0, eccentricity=0.001, inclination_rad=66.0 * _DEG)
_NOISE = {
    "semi_major_axis": 0.006,
    "eccentricity": 1.0e-5,
    "inclination": 3.0e-4,
    "raan": 3.0e-4,
    "arg_perigee": 1.0e-2,
}


@dataclass(frozen=True)
class Burn:
    """One injected maneuver: the gap it brackets, its RSW component, and its magnitude."""

    gap_index: int
    component: str
    delta_v_ms: float
    true_anomaly_rad: float = 0.7

    @property
    def maneuver_type(self) -> ManeuverType:
        return {
            "in_track_ms": ManeuverType.IN_TRACK,
            "cross_track_ms": ManeuverType.CROSS_TRACK,
            "radial_ms": ManeuverType.RADIAL,
        }[self.component]


# One maneuver schedule per satellite; empty tuples are maneuver-free objects that contribute
# exposure (a false-alarm budget) without labels. Train / val / test are sliced from this order.
_POPULATION: tuple[tuple[Burn, ...], ...] = (
    (Burn(35, "in_track_ms", 3.0), Burn(80, "cross_track_ms", 4.0)),
    (Burn(45, "in_track_ms", 3.5), Burn(95, "in_track_ms", 3.0)),
    (Burn(40, "cross_track_ms", 4.0), Burn(90, "in_track_ms", 3.5)),
    (Burn(50, "in_track_ms", 3.0),),
    (Burn(60, "in_track_ms", 3.5),),
    (),
)
_SPLIT = {"train": (0, 1, 2), "val": (3,), "test": (4, 5)}


def _mean_motion_rev_per_day(a_km: float) -> float:
    return math.sqrt(EARTH_MU_KM3_S2 / a_km**3) * _SECONDS_PER_DAY / (2.0 * math.pi)


def _series(norad_id: int, seed: int, burns: tuple[Burn, ...], n: int = 130) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    epochs = [t0 + pd.Timedelta(days=float(i)) for i in range(n)]
    days = np.arange(n, dtype=float)

    raan_dot, argp_dot, _ = j2_secular_rates(_LEO)
    deg_per_day = _SECONDS_PER_DAY / _DEG

    a = _LEO.semi_major_axis_km - 2.0e-3 * days
    e = _LEO.eccentricity + 1.0e-4 * np.sin(0.30 * days)
    inc = _LEO.inclination_rad / _DEG + 5.0e-3 * np.sin(0.20 * days)
    raan = 30.0 + raan_dot * deg_per_day * days
    argp = 90.0 + argp_dot * deg_per_day * days

    a += rng.normal(0.0, _NOISE["semi_major_axis"], n)
    e += rng.normal(0.0, _NOISE["eccentricity"], n)
    inc += rng.normal(0.0, _NOISE["inclination"], n)
    raan += rng.normal(0.0, _NOISE["raan"], n)
    argp += rng.normal(0.0, _NOISE["arg_perigee"], n)

    for burn in burns:
        kwargs = {"radial_ms": 0.0, "in_track_ms": 0.0, "cross_track_ms": 0.0}
        kwargs[burn.component] = burn.delta_v_ms
        step = gauss_forward(orbit=_LEO, true_anomaly_rad=burn.true_anomaly_rad, **kwargs)
        idx = burn.gap_index
        a[idx:] += step.delta_a_km
        e[idx:] += step.delta_eccentricity
        inc[idx:] += step.delta_inclination_rad / _DEG
        raan[idx:] += step.delta_raan_rad / _DEG

    mean_motion = np.array([_mean_motion_rev_per_day(value) for value in a], dtype=float)
    dt_days = np.concatenate(([np.nan], np.diff(days)))
    return pd.DataFrame(
        {
            "epoch": pd.Series(epochs, dtype="datetime64[ns, UTC]"),
            "norad_id": norad_id,
            "mean_motion": mean_motion,
            "semi_major_axis": a,
            "eccentricity": e,
            "inclination": inc,
            "raan": raan,
            "arg_perigee": argp,
            "mean_anomaly": 0.0,
            "bstar": 0.0,
            "dt_days": dt_days,
        }
    )


def _object_series(norad_id: int, frame: pd.DataFrame, burns: tuple[Burn, ...]) -> ObjectSeries:
    epochs = list(frame["epoch"])
    return ObjectSeries(
        norad_id=norad_id,
        series=frame,
        maneuver_epochs=tuple(pd.Timestamp(epochs[burn.gap_index]) for burn in burns),
    )


def _scored_labels_and_exposure(
    frame: pd.DataFrame, burns: tuple[Burn, ...]
) -> tuple[list[ScoredLabel], ObjectExposure]:
    norad_id = int(frame["norad_id"].iloc[0])
    epochs = list(frame["epoch"])
    labels = [
        ManeuverLabel(
            norad_id=norad_id,
            epoch=(epochs[b.gap_index - 1] + (epochs[b.gap_index] - epochs[b.gap_index - 1]) / 2),
            window_start=epochs[b.gap_index - 1],
            window_end=epochs[b.gap_index],
            source="SYNTHETIC",
            source_ref=f"{norad_id}-{b.gap_index}",
            orbit_class=OrbitClass.LEO,
            maneuver_type=b.maneuver_type,
            delta_v=b.delta_v_ms,
        )
        for b in burns
    ]
    intervals = label_series(frame, labels).intervals
    scored = [ScoredLabel(interval=iv, above_floor=True) for iv in intervals]
    span_years = (epochs[-1] - epochs[0]).total_seconds() / (365.25 * _SECONDS_PER_DAY)
    exposure = ObjectExposure(
        norad_id=norad_id, orbit_class=OrbitClass.LEO, observation_years=span_years
    )
    return scored, exposure


def main() -> int:
    frames = {
        index: _series(30000 + index, seed=200 + index, burns=burns)
        for index, burns in enumerate(_POPULATION)
    }
    train_objects = [_object_series(30000 + i, frames[i], _POPULATION[i]) for i in _SPLIT["train"]]
    val_objects = [_object_series(30000 + i, frames[i], _POPULATION[i]) for i in _SPLIT["val"]]

    print("Training the BiLSTM baseline on a synthetic LEO population (CPU, a few epochs)...")
    bundle = train_bilstm(
        train_objects,
        val_objects,
        config=BiLstmConfig(hidden_size=32, num_layers=1, dropout=0.0),
        max_epochs=40,
        seed=0,
        accelerator="cpu",
        threshold=0.5,
    )
    detector = BiLstmDetector(bundle)

    detections = []
    scored_labels: list[ScoredLabel] = []
    exposure: list[ObjectExposure] = []
    for index in _SPLIT["test"]:
        frame = frames[index]
        detections.extend(from_frame(detector.detect(frame)))
        labels, exp = _scored_labels_and_exposure(frame, _POPULATION[index])
        scored_labels.extend(labels)
        exposure.append(exp)

    report = score(detections, scored_labels, exposure)
    metrics = report.per_class[OrbitClass.LEO]

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    print(f"Scored {len(exposure)} held-out LEO objects ({len(scored_labels)} above-floor labels).")
    print(f"  operating point : {metrics.operating_point:g} false-alarm(s)/satellite-year")
    print(f"  recall          : {fmt(metrics.recall)}")
    print(f"  precision       : {fmt(metrics.precision)}")
    print(f"  detections      : {len(detections)}")
    print("Synthetic demo; real numbers come from training on the reconstructed dataset on a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
