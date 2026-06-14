"""The TimesFM forecaster backend — the drop-in second entry (needs the ``[foundation]`` extra).

Wraps a pretrained TimesFM model as the :class:`~maneuver_detect.detectors.foundation.Forecaster`
the forecast-residual detector consumes, on the same recipe as the Chronos backend (D14.4). TimesFM
leads with a point forecast; its predictive scale is taken from its continuous quantile head when
available (the spread of the returned quantiles) and otherwise estimated from the robust spread of
the model's own recent one-step residuals — so the residual standardisation has a scale either way.
The model is loaded from the Hub on CPU at the pinned revision and ``compile``d once; ``timesfm`` is
imported lazily, so this module is reached only with the extra installed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from maneuver_detect.detectors.foundation import Forecast, _rolling_contexts

__all__ = ["TimesFmForecaster"]

FloatArray = npt.NDArray[np.float64]

_SCALE_FLOOR = 1e-12


class TimesFmForecaster:
    """A pretrained TimesFM model as a one-step-ahead element-channel forecaster (CPU-only)."""

    def __init__(self, *, checkpoint_id: str, revision: str, context_length: int) -> None:
        import timesfm

        self._context_length = context_length
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint_id, revision=revision)
        model.compile(
            timesfm.ForecastConfig(
                max_context=context_length,
                max_horizon=1,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
            )
        )
        self._model = model

    def forecast(self, series: FloatArray) -> Forecast:
        n = series.shape[0]
        mean = np.full(n, np.nan, dtype=np.float64)
        scale = np.ones(n, dtype=np.float64)
        if n < 2:
            return Forecast(mean=mean, scale=scale)

        inputs = [
            np.asarray(context, dtype=np.float32)
            for context in _rolling_contexts(series, self._context_length)
        ]
        point, quantiles = self._model.forecast(horizon=1, inputs=inputs)
        predicted = np.asarray(point, dtype=np.float64)[:, 0]
        mean[1:] = predicted

        spread = self._predictive_spread(np.asarray(quantiles, dtype=np.float64), series, predicted)
        scale[1:] = np.maximum(np.abs(spread), _SCALE_FLOOR)
        return Forecast(mean=mean, scale=scale)

    def _predictive_spread(
        self, quantiles: FloatArray, series: FloatArray, predicted: FloatArray
    ) -> FloatArray:
        """Half the predictive interval width per forecast, or a robust residual-MAD fallback.

        Uses the quantile head's outermost spread (min/max across the quantile axis, robust to the
        quantile ordering) when it is non-degenerate; otherwise falls back to a single robust scale
        from the realised one-step residuals (median absolute deviation, ``·1.4826``), so a
        point-only forecast still standardises.
        """
        if quantiles.ndim == 3 and quantiles.shape[-1] >= 2:
            width = quantiles.max(axis=-1)[:, 0] - quantiles.min(axis=-1)[:, 0]
            if np.any(width > 0.0):
                return np.asarray(width / 2.0, dtype=np.float64)
        residual = series[1:] - predicted
        robust = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        return np.full(predicted.shape[0], max(robust, _SCALE_FLOOR), dtype=np.float64)
