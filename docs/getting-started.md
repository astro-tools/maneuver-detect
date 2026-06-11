# Getting started

## Installation

```bash
pip install maneuver-detect
```

The base install carries only permissive dependencies. It includes the PyTorch / Lightning
modelling stack and the Hugging Face Hub / `datasets` libraries; a GPU is needed only to *train* new
baselines, never to install the package or run the classical detector. The time-series
foundation-model baseline lives behind an optional extra:

```bash
pip install "maneuver-detect[foundation]"
```

maneuver-detect supports Python 3.10, 3.11, and 3.12.

## Detect maneuvers for a satellite

Hand `detect` a TLE history for a NORAD catalogue id and read back the detected maneuvers:

```python
from maneuver_detect import detect, datasets

history = datasets.tle_history(norad_id=25544, start="2024-01-01")
maneuvers = detect(history, model="classical")
# DataFrame columns: epoch, confidence, type, delta_v_estimate, plus provenance
```

Each row is one detected maneuver: the detection `epoch` (UTC), a calibrated `confidence` in
`[0, 1]`, the maneuver `type` (in-track / cross-track / radial), and a `delta_v_estimate` in m/s.

## From the command line

The CLI mirrors the API for a one-shot detection on a NORAD id (fetched live) or a local TLE
file:

```bash
maneuver-detect detect 25544
maneuver-detect detect path/to/history.tle
```
