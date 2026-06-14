"""Demonstrate the foundation forecast-residual recipe through the detector, fully offline.

The v0.3 foundation baseline (``chronos-residual``) detects maneuvers by
*forecasting* a satellite's mean-element series with a pretrained model and flagging the gaps where
the realised series departs from the forecast beyond a per-class threshold; the same Gauss physics
inversion the classical detector uses then recovers the delta-v and type. The real forecasters live
behind the optional ``[foundation]`` extra and pull weights from the Hub, so they cannot run in a
self-contained script.

This example shows the *recipe* on a synthetic series with the dependency-free
``DriftContinuationForecaster`` standing in for the model slot — the same stand-in the V6 spike used
to prove the wiring. A real run swaps in Chronos and changes nothing downstream:

    pip install "maneuver-detect[foundation]"
    python -c "from maneuver_detect import detect, datasets; \\
        print(detect(datasets.tle_history(25544, start='2024-01-01'), model='chronos-residual'))"

Run it with no arguments; it needs no credentials, no network, and no extra:

    python examples/foundation_residual.py

The output stays ASCII (the delta-v column prints as ``delta_v_estimate``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maneuver_detect.detectors.foundation import (
    ChronosResidualDetector,
    DriftContinuationForecaster,
)

# A per-orbit-class residual-z operating point (standardized-residual units) — the calibrated
# per-class threshold a published bundle carries; here it is fixed so the recipe is self-contained.
_CLASS_THRESHOLDS = {"LEO": 6.0, "MEO": 6.0, "GEO": 6.0}


def synthetic_series() -> pd.DataFrame:
    """A quiet LEO mean-element series with a clean in-track and a cross-track maneuver injected."""
    rng = np.random.default_rng(60)
    n = 120
    epochs = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    days = np.arange(n, dtype=float)

    a = 6778.0 - 2.0e-3 * days + rng.normal(0.0, 0.006, n)  # km, slow drag decay + TLE noise
    inc = 66.0 + rng.normal(0.0, 3.0e-4, n)  # deg
    a[40:] += 0.9  # an in-track burn steps the semi-major axis on the gap into token 40
    inc[80:] += 0.02  # a cross-track burn steps the inclination on the gap into token 80

    return pd.DataFrame(
        {
            "epoch": epochs,
            "norad_id": 90001,
            "semi_major_axis": a,
            "eccentricity": np.full(n, 0.001),
            "inclination": inc,
            "raan": 30.0 + rng.normal(0.0, 3.0e-4, n),
            "arg_perigee": np.full(n, 90.0),
        }
    )


def main() -> None:
    history = synthetic_series()
    # The registered detector, with the dependency-free stand-in filling the forecaster slot a real
    # Chronos model occupies. Everything downstream — the threshold, NMS, the Gauss
    # inversion, the canonical schema — is exactly what the real detector runs.
    detector = ChronosResidualDetector(
        forecaster=DriftContinuationForecaster(), class_thresholds=_CLASS_THRESHOLDS
    )
    maneuvers = detector.detect(history)

    print("Foundation forecast-residual recipe (stand-in forecaster, synthetic LEO series)")
    print("=" * 78)
    print(f"injected: in-track burn ~day 40, cross-track burn ~day 80; {len(maneuvers)} detected\n")
    if maneuvers.empty:
        print("No maneuvers detected.")
    else:
        print(maneuvers.to_string(index=False))
    print(
        '\nInstall "maneuver-detect[foundation]" and pass model="chronos-residual"\n'
        "to run the same recipe with the real pretrained forecaster."
    )


if __name__ == "__main__":
    main()
