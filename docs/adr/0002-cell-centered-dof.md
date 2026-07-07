# ADR-0002: Cell-Centered DOFs with Cell-Level BC Storage

## Status

Accepted. Combines the original ADR-0002 (cell-centered DOF, no face DOF, BCs as boundary integrals) and the original ADR-0005 (per-cell `CellBC` storage), which previously cross-superseded each other and are merged into a single decision here.

## Context

The XML uses face-key strings to specify BCs. The mesh uses cell-centered DOFs. BC types: Dirichlet / Neumann / Cauchy. A future microfluid phase might want face DOFs for velocity/pressure.

Two storage concerns must be answered together:

1. **Mathematical placement of DOFs.** Flux-type BCs are integrals, not field values; a face DOF would not be used by them. Storing a face DOF alongside a cell DOF roughly doubles memory.
2. **Storage shape when block face projections overlap in the same layer.** A per-face SoA array holds only one BC per face, but two overlapping blocks in the same layer can disagree on what that face's BC should be.

## Decision

**Cell-centered DOFs only, stored at the cell level.** BCs are applied as boundary integrals over cell faces — no face DOFs stored, no per-face SoA arrays. Each cell holds its own 6-face BC descriptor, indexed into a shared parameter table:

```cpp
struct CellBC {
    std::array<BcType, FACE_COUNT> types;        // xm, xp, ym, yp, zm, zp
    std::array<uint16_t, FACE_COUNT> param_idxs; // indices into BCParamTable
};
```

Discrete treatments:

- **Dirichlet (FirstType)** — ghost cell method. The boundary cell's equation uses the ghost value `T_ghost = 2·T_dirichlet − T_boundary` (linear contribution to diag and RHS).
- **Neumann (SecondType)** — flux `q·n` enters the cell RHS directly as `Σ q·A_face`.
- **Cauchy (ThirdType)** — linearized as `h·A·(T_boundary − T_∞)`; contributes to the diagonal (`h·A` coefficient) and to the RHS (`h·A·T_∞`).

Anisotropic `k`: at assembly, pick `k_along(dir) ∈ {kx, ky, kz}` per face normal.

`other_bc` is applied during preprocessing to every face that was not explicitly specified — including faces of virtual neighbors — so the assembly hot loop sees a fully-populated `CellBC` per cell.

The fluid subsystem is independent of `CellBC`: thermal BCs and fluid BCs coexist on the same face. Fluid BCs live in `FluidCellBC` + `FluidBCParamTable` (see `internal_model.hpp`).

## Rationale

- **No projection ambiguity.** Each cell's face is independent; overlapping blocks each carry their own resolution.
- **Virtual-neighbor consistency.** When a cell's neighbor is virtual, the cell's face BC is already set to `other_bc` during preprocessing — same channel as overlap resolution.
- **Flexibility.** Different cells in the same layer can have different BC types on the same directional face.
- **Simpler preprocessing.** No global face arrays; assignment is at cell level.
- **Flux-type BCs need no face DOF.** They are integrals, and the ghost-cell pattern handles Dirichlet without face storage.
- **Memory.** A dual volume+face DOF system would roughly double memory.
- **Microfluid face fields (Phase 2).** Would warrant their own face-field storage; face DOFs would be added intentionally for that subsystem only, not retrofitted here.

## Data flow

```text
IOStructure
  └─> Preprocessor::load()
        ├─> preprocessor::resolve_layers()
        │     ├─> valid_mask, index_map            (full-grid tier)
        │     └─> material_id                      (compact tier; parallel to cell_bcs)
        ├─> preprocessor::resolve_face_keys()
        │     ├─> flatten (boundary, face_key) pairs
        │     └─> single grid traversal → CellBC per cell, per face, with `other_bc` fallback (handles overlap)
        └─> Compile expressions → BCParamTable + heat_source_table
```

## Notes

- `BCParamTable` remains: shared BC parameters live there, referenced by `param_idx`.
- `other_bc` is applied during preprocessing, not at assembly time. Virtual neighbors: the active cell's face touching a virtual cell gets `other_bc` set during `resolve_face_keys()`.
- The heat source is **not** part of `CellBC` — it is a deduplicated dictionary indexed by `uint16_t` (see ADR-0004 §Heat source dictionary).
- The preprocessor pre-resolves every face-key string into per-cell `CellBC { types[6], param_idxs[6] }`. Assembly hot loops do array lookups only — no string parsing.
- All geometry stays in SI meters.
