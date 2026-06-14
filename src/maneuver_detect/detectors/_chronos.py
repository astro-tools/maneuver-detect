"""The Chronos forecaster backend — the lead foundation model (needs the ``[foundation]`` extra).

Wraps a pretrained Chronos pipeline as the :class:`~maneuver_detect.detectors.foundation.Forecaster`
the forecast-residual detector consumes: a rolling one-step-ahead forecast of a single element
channel, with the predictive scale read off Chronos's native probabilistic quantiles (the half-width
of the 80% predictive interval) — the property that made Chronos the D14.4 lead. The model is loaded
from the Hub at the pinned revision (Apache-2.0 confirmed at ingest, D14.2) onto a GPU when present
(else CPU) and run under ``no_grad``; an optional light fine-tune ``state_dict`` is loaded onto it.

:func:`finetune_chronos_model` is the offline light-fine-tune driver (D14.3 *optional polish*): a
short AdamW pass over the objects' element windows specialising the quiet-dynamics prior to the
satellite-element domain, returning the fine-tuned ``state_dict`` and the measured cost. Everything
here imports ``chronos`` / ``torch`` lazily; this module is reached only with the extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from maneuver_detect.detectors.foundation import Forecast, _rolling_contexts

if TYPE_CHECKING:
    import pandas as pd
    import torch

__all__ = ["ChronosForecaster", "finetune_chronos_model"]

FloatArray = npt.NDArray[np.float64]


def _device() -> str:
    """The compute device — a GPU when one is present, else CPU (a GPU is never required)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


#: Quantile levels requested per forecast; the residual's predictive scale is half the spread of the
#: outer two (the 80% predictive interval) — Chronos's probabilistic edge over a point forecaster.
_QUANTILE_LEVELS = [0.1, 0.5, 0.9]
#: Rolling one-step contexts forecast per batched ``predict_quantiles`` call (caps peak memory on a
#: multi-year daily series, which yields one context per token).
_FORECAST_BATCH = 256
_SCALE_FLOOR = 1e-12


class ChronosForecaster:
    """A pretrained Chronos pipeline as a one-step-ahead element-channel forecaster.

    Loads onto a GPU when one is present, else CPU (a GPU is never required); the predict outputs
    are pulled back to CPU as NumPy.
    """

    def __init__(
        self,
        *,
        checkpoint_id: str,
        revision: str,
        context_length: int,
        finetune_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        import torch
        from chronos import BaseChronosPipeline

        self._context_length = context_length
        pipeline = BaseChronosPipeline.from_pretrained(
            checkpoint_id, revision=revision, device_map=_device(), torch_dtype=torch.float32
        )
        if finetune_state is not None:
            pipeline.model.load_state_dict(finetune_state)
        pipeline.model.eval()
        self._pipeline = pipeline

    def forecast(self, series: FloatArray) -> Forecast:
        import torch

        n = series.shape[0]
        mean = np.full(n, np.nan, dtype=np.float64)
        scale = np.ones(n, dtype=np.float64)
        if n < 2:
            return Forecast(mean=mean, scale=scale)

        contexts = [
            torch.tensor(context, dtype=torch.float32)
            for context in _rolling_contexts(series, self._context_length)
        ]
        means: list[FloatArray] = []
        lows: list[FloatArray] = []
        highs: list[FloatArray] = []
        with torch.no_grad():
            for start in range(0, len(contexts), _FORECAST_BATCH):
                chunk = contexts[start : start + _FORECAST_BATCH]
                quantiles, point = self._pipeline.predict_quantiles(
                    chunk, prediction_length=1, quantile_levels=_QUANTILE_LEVELS
                )
                means.append(point[:, 0].to(torch.float64).cpu().numpy())
                lows.append(quantiles[:, 0, 0].to(torch.float64).cpu().numpy())
                highs.append(quantiles[:, 0, -1].to(torch.float64).cpu().numpy())

        mean[1:] = np.concatenate(means)
        spread = (np.concatenate(highs) - np.concatenate(lows)) / 2.0
        scale[1:] = np.maximum(np.abs(spread), _SCALE_FLOOR)
        return Forecast(mean=mean, scale=scale)


def _finetune_windows(
    series: list[pd.DataFrame], context_length: int, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(context, target)`` training windows from the objects' trigger-channel series.

    Pools the same two channels the detector forecasts — the semi-major axis and the inclination —
    from every object, sliding a ``context_length`` context against the following ``horizon``
    target. Chronos instance-normalises each context, so pooling the two very-different-scale
    channels is fine. Returns the stacked context / target tensors (empty when none is long enough).
    """
    import torch

    from maneuver_detect.detectors.learned import _AlignedElements
    from maneuver_detect.features.channels import build_channels

    contexts: list[FloatArray] = []
    targets: list[FloatArray] = []
    for frame in series:
        channels = build_channels(frame)
        if channels.n_tokens < context_length + horizon:
            continue
        elements = _AlignedElements.from_channels(channels)
        for values in (elements.a_km, elements.inc_rad):
            for start in range(values.shape[0] - context_length - horizon + 1):
                ctx = values[start : start + context_length]
                tgt = values[start + context_length : start + context_length + horizon]
                contexts.append(ctx.astype(np.float64))
                targets.append(tgt.astype(np.float64))
    if not contexts:
        return torch.empty(0), torch.empty(0)
    return (
        torch.tensor(np.stack(contexts), dtype=torch.float32),
        torch.tensor(np.stack(targets), dtype=torch.float32),
    )


def finetune_chronos_model(
    *,
    checkpoint_id: str,
    revision: str,
    context_length: int,
    series: list[pd.DataFrame],
    max_steps: int = 200,
    learning_rate: float = 1e-4,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Light fine-tune of the Chronos model on the element windows; return its state and the cost.

    A short AdamW pass over ``(context, target)`` windows from the objects' trigger-channel series,
    minimising the model's own forecast loss — the optional polish on the zero-shot baseline
    (D14.3), sized to the V7 single-GPU envelope. CPU-capable for a smoke run; a full fine-tune
    wants a GPU. Raises :class:`ValueError` when no series is long enough to form a window.
    """
    import torch
    from chronos import BaseChronosPipeline
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = _device()
    pipeline = BaseChronosPipeline.from_pretrained(
        checkpoint_id, revision=revision, device_map=device, torch_dtype=torch.float32
    )
    model = pipeline.model
    horizon = int(model.config.chronos_config["prediction_length"])
    contexts, targets = _finetune_windows(series, context_length, horizon)
    if contexts.shape[0] == 0:
        raise ValueError(
            f"no fine-tune windows: every series is shorter than context_length + horizon "
            f"({context_length} + {horizon})"
        )

    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(contexts, targets), batch_size=32, shuffle=True)
    step = 0
    last_loss = float("nan")
    while step < max_steps:
        for context_batch, target_batch in loader:
            optimiser.zero_grad()
            output = model(context=context_batch.to(device), target=target_batch.to(device))
            output.loss.backward()
            optimiser.step()
            last_loss = float(output.loss.detach())
            step += 1
            if step >= max_steps:
                break

    model.eval()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    cost = {"steps": step, "final_loss": last_loss, "n_windows": int(contexts.shape[0])}
    return state, cost
