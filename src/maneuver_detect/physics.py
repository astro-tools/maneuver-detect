"""The Δv inversion — turning a detected element jump into a maneuver type and a Δv estimate.

Uses vis-viva for the in-track component (from the change in semi-major axis) and the Gauss
variational equations for the radial / in-track / cross-track decomposition; secular drift
(notably J2 nodal regression) is removed before the inversion. The dominant component sets the
maneuver type and the magnitude gives the Δv estimate.
"""

from __future__ import annotations
