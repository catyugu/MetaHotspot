# ADR-0002: Cell-Centered DOFs with Boundary Integral BCs

## Status

Accepted. **Superseded in storage form by ADR-0005** — per-face SoA arrays were replaced by cell-level `CellBC` storage. The mathematical decision (cell-centered DOF, no face DOF, BCs applied as boundary integrals) is unchanged.

## Context

The XML uses face-key strings to specify BCs. The mesh uses cell-centered DOFs. BC types: Dirichlet / Neumann / Cauchy. A future microfluid phase might want face DOFs for velocity/pressure.

## Decision

**Cell-centered DOFs only.** BCs are applied as boundary integrals over cell faces — no face DOFs stored.

- **Dirichlet**: ghost cell method. The boundary cell's equation uses the ghost value `T_ghost = 2·T_dirichlet − T_boundary` (linear contribution to diag and RHS).
- **Neumann**: flux `q·n` enters the cell RHS directly as `Σ q·A_face`.
- **Cauchy**: linearized as `h·A·(T_boundary − T_∞)` → contributes to the diagonal (`h·A` coefficient) and to the RHS (`h·A·T_∞`).

## Rationale

- Flux-type BCs need no face DOF — they are integrals.
- Ghost cell for Dirichlet is standard FVM.
- A dual volume+face DOF system would roughly double memory.
- Microfluid face fields (Phase 2) would warrant their own face-field storage — face DOFs would be added intentionally for that subsystem only, not retrofitted here.

## Notes

- The preprocessor pre-resolves every face-key string into per-cell `CellBC { types[6], param_idxs[6] }` (see ADR-0005). Assembly hot loops do array lookups only — no string parsing.
- All geometry stays in SI meters.
