"""The leak check the issue DoD requires: no feature may be derived from the label.

The feature layer's boundary is structural — :func:`build_channels` takes only the element series,
so there is no label, target, or split it *could* read. These tests pin that boundary: the encoder's
signature carries no label argument, the encoding is byte-deterministic, and swapping the per-token
target handed to the windower changes only the target tensor, never the features or the validity.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from maneuver_detect.features import (
    ClassNormaliser,
    build_channels,
    encode_history,
    make_windows,
)


def _frame(*, n: int = 160, step_at: int = 80) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    epochs = [start + timedelta(days=i) for i in range(n)]
    a = 6778.0 - 0.002 * np.arange(n)
    a[step_at:] += 0.4  # a real maneuver step, so the would-be label is non-trivial
    return pd.DataFrame(
        {
            "epoch": pd.Series(epochs, dtype="datetime64[ns, UTC]"),
            "norad_id": np.full(n, 25544, dtype=int),
            "semi_major_axis": a,
            "eccentricity": np.full(n, 0.001),
            "inclination": np.full(n, 66.0),
            "raan": (20.0 + 5.0 * np.arange(n)) % 360.0,
            "arg_perigee": np.full(n, 90.0),
        }
    )


def test_build_channels_takes_no_label_argument() -> None:
    params = list(inspect.signature(build_channels).parameters)
    assert params == ["history"]
    forbidden = ("label", "target", "split", "maneuver", "truth")
    assert not any(any(word in name for word in forbidden) for name in params)


def test_encoding_is_byte_deterministic_end_to_end() -> None:
    frame = _frame()
    rc = build_channels(frame)
    norm = ClassNormaliser.fit(encode_history(frame))
    one = norm.transform(rc).features
    two = ClassNormaliser.fit(encode_history(frame)).transform(build_channels(frame)).features
    assert one.tobytes() == two.tobytes()


def test_normaliser_fits_on_train_objects_only() -> None:
    # The other half of the leak-free contract (D11.3): the standardiser's statistics come from the
    # train split only, so a held-out object is never scaled by its own distribution. The feature
    # layer leaves this to the caller; pin it here so a regression that pooled the held-out object
    # into the fit would be caught.
    train = build_channels(_frame())
    held_out_frame = _frame()
    held_out_frame["semi_major_axis"] = held_out_frame["semi_major_axis"] + 80.0  # a different LEO
    held_out = build_channels(held_out_frame)
    cls = train.orbit_class
    assert held_out.orbit_class is cls  # same class, distinct statistics

    train_only = ClassNormaliser.fit([train])
    pooled = ClassNormaliser.fit([train, held_out])

    # Adding the held-out object to the fit moves the statistics, so the train-only fit the harness
    # uses cannot have been informed by the held-out object's scale.
    assert not np.allclose(train_only.medians[cls], pooled.medians[cls])
    # The train-only statistics are exactly the train block's own medians — nothing else entered.
    assert np.allclose(train_only.medians[cls], np.median(train.element_block(), axis=0))


def test_features_do_not_depend_on_the_target() -> None:
    frame = _frame(step_at=80)
    matrix = ClassNormaliser.fit([build_channels(frame)]).transform(build_channels(frame))

    # Two completely different per-token targets — including the "true" maneuver label at gap 80.
    truth = np.zeros(matrix.n_tokens)
    truth[80] = 1.0
    decoy = np.ones(matrix.n_tokens)

    with_truth = make_windows(matrix, target=truth)
    with_decoy = make_windows(matrix, target=decoy)
    without = make_windows(matrix)

    assert with_truth.features.tobytes() == with_decoy.features.tobytes()
    assert with_truth.features.tobytes() == without.features.tobytes()
    assert np.array_equal(with_truth.validity, with_decoy.validity)
    # only the target tensor reflects the label
    assert with_truth.target is not None and with_decoy.target is not None
    assert with_truth.target.tobytes() != with_decoy.target.tobytes()
