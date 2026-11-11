# ADR-0002: Canonical Facets and Condensed Boundary Traces

## Status

Accepted.

## Context

Thermal and fluid conditions may cover partial, overlapping rectangles of an exterior
control-volume face. Keeping separate mesh faces and projected boundary geometry
duplicated topology, geometry, and coordinate interpretation. Temperature-dependent
thermal conditions must be evaluated at physical surface temperature.

## Decision

`Model::boundary_facets` is the unique collection of external rectangular integration
regions. Each stored `Facet` has one valid CellId (minus or plus), a normal
axis, plane coordinate and tangential bounds; the missing side determines its
outward normal. `BoundaryFacetId` identifies one stored boundary piece.

The geometry compiler partitions these pieces against the union of thermal and
fluid boundary region cuts. A partially covered rectangle creates several pieces,
including uncovered regions. The compiler independently binds thermal conditions
through `Model::boundary_thermal_condition` and fluid conditions to those same
BoundaryFacetIds. Later regions take precedence within each physics. The default
thermal condition applies to uncovered pieces.

Internal contacts use CellId connections, implicit for ConformalMesh and explicit
for NonConformalMesh. A transient, oriented `Facet` with both sides populated
is derived during assembly. No internal Facet array or cell-facet incidence is retained.

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
- Multiple physics independently bind conditions to the same BoundaryFacetIds.
- No second boundary geometry, duplicated area, or mesh-dependent physical law.
- Local trace condensation leaves the global degree-of-freedom count unchanged.
