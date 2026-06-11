# API reference

The public surface of maneuver-detect. Everything under `maneuver_detect` documented here is part of the
frozen library contract; modules not listed are internal and may change between releases.

## Top-level surface

::: maneuver_detect
    options:
      members:
        - detect
        - datasets
        - Detector
        - Maneuver
        - ManeuverType
        - available_models

## Datasets

::: maneuver_detect.datasets
    options:
      members:
        - tle_history

## Output schema

::: maneuver_detect.schema

## Δv inversion (physics)

::: maneuver_detect.physics

## Detectors

::: maneuver_detect.detectors

## Benchmark

::: maneuver_detect.benchmark
