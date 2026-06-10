# The mean-element series

The canonical per-object input the feature and detector layers consume. The data layer fetches raw
elsets, **cleans** them, and **assembles** them into this series. It is the same DataFrame
`detect()` is handed and the benchmark scores against, so its schema is a fixed contract.

## Schema

One row per epoch, ordered by epoch ascending with strictly increasing epochs. Columns, in order:

| Column | Dtype | Unit | Meaning |
|---|---|---|---|
| `epoch` | `datetime64[ns, UTC]` | — | Element-set epoch (timezone-aware UTC). |
| `norad_id` | `int64` | — | NORAD catalogue id of the object (constant down the series). |
| `mean_motion` | `float64` | rev/day | SGP4 mean motion *n*. |
| `semi_major_axis` | `float64` | km | Derived Kozai-mean semi-major axis *a* (see below). |
| `eccentricity` | `float64` | — | Eccentricity *e*. |
| `inclination` | `float64` | deg | Inclination *i*. |
| `raan` | `float64` | deg | Right ascension of the ascending node Ω. |
| `arg_perigee` | `float64` | deg | Argument of perigee ω. |
| `mean_anomaly` | `float64` | deg | Mean anomaly *M*. |
| `bstar` | `float64` | 1/earth-radii | B\* drag term. |
| `dt_days` | `float64` | days | Days since the previous row (`NaN` for the first). |

The six mean elements plus `bstar` are the SGP4 mean elements in the **TEME** frame, carried
through from the catalog verbatim — no frame change is applied. `semi_major_axis` and `dt_days` are
the only derived columns.

### Derived `semi_major_axis`

From the mean motion via Kepler's third law, `a = (μ / n²)^(1/3)` with *n* in rad/s and
μ = 398600.8 km³/s² (the WGS72 value SGP4 uses), so the semi-major axis is consistent with the
propagator the elements were fit for. This is the Kozai-mean *a* — a direct conversion of the mean
motion, not a Brouwer-corrected osculating value.

### `dt_days` and gaps

`dt_days` exposes the irregular TLE cadence — typically around a day, but punctuated by
re-acquisition gaps (notably after a maneuver). Cleaning never splits, fills, or interpolates
across a gap: a maneuver is observable only as a discontinuity *between* two elsets, so the gap is
signal, and the matching benchmark labels a maneuver onto exactly such an inter-elset interval.
`dt_days` makes the spacing visible without the data layer acting on it.

## Cleaning rules

`clean_elsets` removes only **obvious catalog noise** — distinguishing a *bad elset* from a
*maneuver* is the detector's job, not this layer's.

**Validity.** An elset is dropped if it is non-physical on its face — a non-finite element,
eccentricity outside `[0, 1)`, non-positive mean motion, or inclination outside `[0, 180]` — or if
SGP4 cannot initialise it and propagate it at its own epoch (which catches a decayed orbit whose
perigee is inside the Earth). Anything that merely *looks* anomalous is kept; that is the
detector's call.

**Duplicate-epoch dedup.** At most one elset survives per epoch. The rule separates two cases:

- **Exact duplicates** — same epoch *and* identical elements (e.g. the same elset redistributed
  across CelesTrak and Space-Track) — collapse to one. Identity is decided by the elements, so this
  is robust even when `element_set_no` is a placeholder (CelesTrak frequently emits `999`).
- A genuine **same-epoch re-fit** — same epoch, *differing* elements (the catalog re-issued a
  revised fit) — keeps the highest `element_set_no` (the later revision), with a deterministic
  element-value fallback when `element_set_no` does not discriminate.

The dedup does not depend on input order, so a reconstructed series is byte-stable.
