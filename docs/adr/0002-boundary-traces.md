# ADR-0002: Canonical Facets and Condensed Boundary Traces

## Status

Accepted.

## Context

Thermal and fluid conditions may cover partial, overlapping rectangles of an exterior
control-volume face. Keeping separate mesh faces and projected boundary geometry
duplicated topology, geometry, and coordinate interpretation. Temperature-dependent
thermal conditions must be evaluated at physical surface temperature.

## Decision

Each `ExteriorFacet` stores a rectangular `Facet`, the owner CellId and the
thermal boundary group. The geometry compiler partitions exterior faces at
thermal and fluid region cuts, including uncovered pieces. FacetIds are grouped
by owner and fluid tags are consumed during preprocessing. Later regions take
precedence within each physics. Uncovered pieces use the default thermal law.
Internal neighbors come from the Cartesian index.

For cell temperature `T`, trace `lambda`, normal conductivity `k`, center-to-wall
distance `d`, and facet area `A`, conductive flux is `F = k A (T - lambda) / d`.

Dirichlet prescribes `lambda` independently of temperature; Neumann satisfies
`F + A q(lambda) = 0` where positive `q` is inward. Convection equates `F` to
`A h(lambda) (lambda - T_inf(lambda))`. Boundary expressions receive the surface
trace as their temperature variable. `evaluate_boundary` takes the integration
patch, internal temperature, time and area-specific cell-to-wall resistance.

Each local trace is eliminated by static condensation. Trace-independent laws
use analytic formulas and temperature-dependent laws use a bracketed scalar solve.
Assembly, gradients and probes share the same law. The global state and matrix
remain cell-sized, without materialized ghost nodes.

## Rationale

- Canonical geometry and facet subdivision have one owner.
- Multiple physics independently bind conditions to the same FacetIds.
- No second boundary geometry, duplicated area, or mesh-dependent physical law.
- Local trace condensation leaves the global degree-of-freedom count unchanged.
