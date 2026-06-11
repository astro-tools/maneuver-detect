# Output schema and Δv inversion

Every detector returns the same **canonical maneuver schema**, and the Δv estimate and maneuver type in it
come from a single physical **inversion** of the mean-element step across the gap. This page documents both —
the schema as the frozen library contract, and the inversion as the physics behind the `type` and
`delta_v_estimate` columns.

## The canonical schema

`detect` returns a pandas DataFrame with one row per detected maneuver and these columns, in order:

| Column | Type | Meaning |
|---|---|---|
| `epoch` | UTC datetime | Detection epoch — the maneuver epoch, inside the bracketing gap. |
| `confidence` | float, `[0, 1]` | Calibrated detection confidence. Monotone in the residual-jump significance, so a benchmark can threshold it to set an operating point. |
| `type` | string | `in-track`, `cross-track`, or `radial` — the dominant Δv component. |
| `delta_v_estimate` | float, m/s | Estimated `|Δv|`. `NaN` when the inversion falls below the per-type detectability floor (the magnitude is not trustworthy there). |
| `norad_id` | int | NORAD catalogue id of the object. |
| `elset_epoch_before` | UTC datetime | Epoch of the elset bounding the **start** of the inter-elset gap. |
| `elset_epoch_after` | UTC datetime | Epoch of the elset bounding the **end** of that gap. |

A detector that finds nothing returns an **empty frame with the same columns and dtypes**, so downstream code
never special-cases the no-detection result. The schema is the single source of truth, exposed as
`maneuver_detect.Maneuver` (one record) and `maneuver_detect.ManeuverType` (the type enum); the
`maneuver_detect.schema` module provides the lossless `to_frame` / `from_frame` round-trip the detectors and
the scorer share. The schema is **frozen by release**.

The provenance pair `(elset_epoch_before, elset_epoch_after)` is the gap the detection brackets — the same
inter-elset gap the benchmark's [matching rule](benchmark.md#detection-matching-rule) scores against.

## The Δv inversion

A detector sees the orbit only through its SGP4 **mean** elements: a step in semi-major axis, eccentricity,
inclination, and node across the gap that brackets a burn. The inversion (`maneuver_detect.physics`) reads a
Δv back out of that step using the impulsive form of the **Gauss variational equations** — the exact
first-order relation between an impulsive Δv, decomposed in the radial / in-track / cross-track (RSW) frame,
and the resulting element change:

- **in-track** Δv shows up as a step in semi-major axis (vis-viva) and eccentricity;
- **cross-track** Δv shows up as a step in inclination and node, in closed form;
- **radial** Δv shows up as an eccentricity-vector change beyond what the in-track burn explains — weakly
  observable, so it is reported low-confidence.

The maneuver **type** is the dominant component, and the magnitude of the combined impulse is the **Δv
estimate**. Two physical facts shape the result:

### Secular drift is detrended first

The natural J2 nodal regression moves the node by several degrees per day in LEO — far more than any
station-keeping burn — so a raw element difference reads as a huge spurious cross-track Δv. Before inverting,
each element step is measured with a **two-sided local-linear fit** that subtracts the local secular trend, so
steady drift leaves no standing residual and only the anomalous step remains.

### There is a per-class detectability floor

Below roughly cm/s in LEO and ~0.1 m/s in GEO the element step is buried in TLE noise and neither the Δv nor
the type is recoverable; above it the inversion is good to about ±25%. The `delta_v_estimate` is gated against
that floor — a below-floor inversion yields `NaN` rather than a noise figure — and the floor is **per type**,
because the node and inclination are good only to arc-seconds while the semi-major axis is good to metres, so
a cross-track burn must be larger than an in-track one to clear the noise.

### Signs and observability

The in-track component keeps its sign — positive raises the orbit, negative lowers it — because that is fixed
by the sign of the semi-major-axis step. The cross-track and radial components are reported as magnitudes:
their sign is not observable from a mean-element step without knowing where in the orbit the burn occurred.

The quantitative accuracy of the recovered Δv against *published* burn magnitudes is validated against the
DORIS/IDS Δv ground truth; the in-library contract is method correctness — the forward and inverse models
round-trip, and the type rule and magnitudes match the textbook impulsive-maneuver relations.

See the [API reference](api.md) for the full `maneuver_detect.schema` and `maneuver_detect.physics` surface.
