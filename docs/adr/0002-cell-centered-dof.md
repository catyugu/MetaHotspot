# ADR-0002: Cell-Centered DOFs with Boundary Integral BCs

## Status

Accepted

## Context

The original XML uses face-key strings (e.g., `Z|E|0|0,50,50,100`) to specify boundary conditions. The mesh uses cell-centered DOFs (temperature at cell centers). Boundary conditions include:

- **First-type (Dirichlet)**: Fixed temperature
- **Second-type (Neumann)**: Fixed heat flux
- **Third-type (Cauchy/Robin)**: Convection `h(T - T_∞)`

A concern was raised about future microfluid support (face velocity/pressure BCs), which might require actual face DOFs.

## Decision

Use **cell-centered DOFs only**. Boundary conditions are applied directly as boundary integrals over cell faces, without storing separate face DOFs.

- **Dirichlet BC**: Ghost cell method — one phantom cell value beyond the boundary. The cell equation at the boundary uses the ghost cell temperature `T_ghost = 2·T_dirichlet - T_boundary`.
- **Neumann BC (flux)**: The flux `q·n` directly enters the cell's RHS as `Σ q·A_face`.
- **Cauchy BC**: The convective term `h·A·(T_boundary - T_∞)` is linearized and contributes both to the RHS (`h·A·T_∞`) and to the diagonal of the Jacobian (`h·A·T_boundary` coefficient).

**Face BC data**: `preprocessor` pre-resolves all face-key strings into per-face BC type + parameter index arrays, indexed by face. Assembly hot loops do array lookups only, no string parsing.

## Rationale

- Flux-type BCs (Neumann, Cauchy) don't need face DOFs — they contribute as boundary integrals.
- Ghost cell method for Dirichlet is standard practice in finite volume methods.
- Dual DOF system (volume + face) would roughly double memory and complicate assembly.
- Future microfluid face fields (velocity, pressure) would be a Phase 2 extension, requiring a separate face-field storage — at that point, face DOFs would be added intentionally for that subsystem only.

## Notes

- Face BC arrays are stored SoA alongside cell arrays for cache efficiency during assembly.
- Each of 6 mesh faces (Z-, Z+, Y-, Y+, X-, X+) has its own BC type + param arrays.
