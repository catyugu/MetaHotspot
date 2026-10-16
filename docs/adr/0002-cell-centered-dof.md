# ADR-0002: Cell-Centered DOFs and Explicit Face Patches

## Status

Accepted.

## Context

Control-volume identity and face connectivity do not encode geometric coordinate indices. Boundary conditions act on surface integrals; representing a surface does not require a surface temperature unknown.

## Decision

Store one temperature unknown per control volume, directly indexed by CellId. `DiscreteMesh` stores axis-aligned cell bounds, unique oriented `FacePatch` records and cell-to-face CSR incidence. Face patches contain geometry and connectivity; materials and boundary assignments belong to the physical model.

`Model::boundary_terms` stores integration rectangles referencing original FaceIds and `boundary_table` indices. The table pairs each boundary type with compiled expression handles; Robin HTC and ambient temperature share one record. The compiler intersects physical boundary regions without modifying mesh topology, applies conditions in authoring order, and fills uncovered areas with the default condition. Heat sources use independent volume quadrature records. `Model` retains `OutputMesh` for sampling and output.

For normal conductivity `k`, owner-center distance `d` and patch area `A`:

- Dirichlet contributes `g = k A / d` to the diagonal and `g T_D` to the RHS.
- Neumann contributes `q A` to the RHS, with positive `q` adding heat.
- Robin contributes `g = A / (d/k + 1/h)` to the diagonal and `g T_inf` to the RHS; zero HTC contributes zero.

An internal solid face uses `g = A / (d_i/k_i + d_j/k_j)` and contributes `g [[1,-1],[-1,1]]`. Assembly selects solid conduction or fluid–solid exchange directly from the two physical types. Frozen fluid volume fluxes use the same unique oriented faces.

The current TPFA requires orthogonal axis-aligned boxes with diagonal conductivity; nonorthogonal center connections are rejected.

## Rationale

- A single internal-face contribution guarantees equal and opposite heat flow.
- Area projection preserves partial boundary coverage and repeated contributions to the same temperature DOF.
- Explicit incidence frees assembly and sampling from coordinate-index arithmetic.
- Keeping boundary integrals separate from unknown placement avoids unnecessary face DOFs.
