# ADR-0002: Canonical Facets and Condensed Boundary Traces

## Status

Accepted.

## Context

Thermal and fluid conditions may cover partial, overlapping rectangles of an exterior
control-volume face. Keeping separate mesh faces and projected boundary geometry
duplicated topology, geometry, and coordinate interpretation. Temperature-dependent
thermal conditions must be evaluated at physical surface temperature.

## Decision

`DiscreteMesh::facets` is the unique collection of rectangular flux integration
regions. `Facet` stores adjacent cells (one invalid for an exterior surface), normal
axis, plane coordinate and tangential bounds. It supplies area, center, owner,
outward sign and membership. `FacetId` identifies one canonical integration region.

The geometry compiler partitions exterior facets against the union of thermal and
fluid boundary region cuts **before** building cell-facet incidence. A partially
covered rectangle creates several Facets, including uncovered pieces. The compiler
then independently binds thermal conditions to `Model::facet_thermal_condition`
and fluid boundaries to those same FacetIds. Later regions take precedence within
each physics. The default thermal condition applies to uncovered exterior facets.
Internal facet geometry does not depend on exterior boundary assignments.

For cell temperature `T`, trace `lambda`, normal conductivity `k`, center-to-wall
distance `d`, and facet area `A`, conductive flux is `F = k A (T - lambda) / d`.

Dirichlet prescribes `lambda` independently of temperature; Neumann satisfies
`F + A q(lambda) = 0` where positive `q` is inward. Convection equates `F` to
`A h(lambda) (lambda - T_inf(lambda))`. Boundary expressions receive the surface
trace as their temperature variable. `evaluate_boundary` accepts an expression
context, integration area, and cell-to-wall conductance, independently of mesh
geometry or topology.

Each local trace is eliminated by static condensation. Trace-independent laws
use analytic formulas and temperature-dependent laws use a bracketed scalar solve.
Assembly, gradients and probes share the same law. The global state and matrix
remain cell-sized, without materialized ghost nodes.

## Rationale

- Canonical geometry and facet subdivision have one owner.
- Multiple physics independently bind conditions to the same FacetIds.
- No second boundary geometry, duplicated area, or mesh-dependent physical law.
- Local trace condensation leaves the global degree-of-freedom count unchanged.
