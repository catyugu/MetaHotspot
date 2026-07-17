# ADR-0002: Cell-Centered DOFs with Cell-Level BC Storage

## Status

Accepted. Combines the original ADR-0002 (cell-centered DOF, no face DOF, BCs as boundary integrals) and the original ADR-0005 (per-cell `CellBC` storage), which previously cross-superseded each other and are merged into a single decision here.

## Context

The XML uses face-key strings to specify BCs. The mesh uses cell-centered DOFs. BC types: Dirichlet / Neumann / Cauchy. A future microfluid phase might want face DOFs for velocity/pressure.

Two storage concerns must be answered together:

1. **Mathematical placement of DOFs.** Flux-type BCs are integrals, not field values; a face DOF would not be used by them. Storing a face DOF alongside a cell DOF roughly doubles memory.
2. **Storage shape when block face projections overlap in the same layer.** A per-face SoA array holds only one BC per face, but two overlapping blocks in the same layer can disagree on what that face's BC should be.

## Decision

**Cell-centered DOFs only, stored at the cell level.** BCs are applied as boundary integrals over cell faces — no face DOFs stored, no per-face SoA arrays. Each face of every cell carries its own BC descriptor, stored as a flat array on `Model`, indexed into a shared parameter table:

```cpp
struct FaceBC {
    BcType type = BcType::None;
    uint16_t param_idx = 0;      // → BCParamTable
};

// Model::face_bcs is std::vector<FaceBC> with size N_active * 6.
// face_bcs[c * 6 + dir] gives the BC for cell c's face dir.
```

Discrete treatments:

- **Dirichlet (FirstType)** — ghost cell method. The boundary cell's equation uses the ghost value `T_ghost = 2·T_dirichlet − T_boundary` (linear contribution to diag and RHS).
- **Neumann (SecondType)** — flux `q·n` enters the cell RHS directly as `Σ q·A_face`.
- **Cauchy (ThirdType)** — linearized as `h·A·(T_boundary − T_∞)`; contributes to the diagonal (`h·A` coefficient) and to the RHS (`h·A·T_∞`).

Anisotropic `k`: at assembly, pick `k_along(dir) ∈ {kx, ky, kz}` per face normal.

`other_bc` is applied during preprocessing to every face that was not explicitly specified — including faces of virtual neighbors — so the assembly hot loop sees a fully-populated per-face BC for every cell.

The fluid subsystem is independent of thermal BC storage: thermal faces use `FaceBC`; fluid boundary records are temporary preprocessing state and only assembly-ready values enter `FluidDomain`.

## Rationale

- **No projection ambiguity.** Each individual face BC is independent; overlapping blocks each carry their own resolution.
- **Virtual-neighbor consistency.** When a cell's neighbor is virtual, the cell's face BC is already set to `other_bc` during preprocessing — same channel as overlap resolution.
- **Flexibility.** Different cells in the same layer can have different BC types on the same directional face.
- **Simpler preprocessing.** No global face arrays; assignment is at cell level.
- **Flux-type BCs need no face DOF.** They are integrals, and the ghost-cell pattern handles Dirichlet without face storage.
- **Memory.** A dual volume+face DOF system would roughly double memory.
- **Microfluid face fields (Phase 2).** Would warrant their own face-field storage; face DOFs would be added intentionally for that subsystem only, not retrofitted here.

## Data flow

```text
ModelDefinition
  └─> build_model()
        ├─> preprocessor::assign_cell_layers()
        │     ├─> grid_to_cell                   (full-grid)
        │     ├─> cell_to_grid                   (compact inverse)
        │     └─> material_id + heat_source_idx (compact)
        ├─> preprocessor::parse_all_face_keys()
        │     └─> flatten (boundary, face_key) pairs
        ├─> preprocessor::resolve_boundary_patches()
        │     └─> face_bcs [N_active * 6] (handles overlap)
        └─> Compile expressions → BCParamTable + heat_source_table
```

## Notes

- `BCParamTable` remains: shared BC parameters live there, referenced by `param_idx`.
- `other_bc` is applied during preprocessing, not at assembly time. Virtual neighbors: the active cell's face touching a virtual cell gets `other_bc` set during `resolve_boundary_patches()`.
- The heat source is **not** part of `FaceBC` — it is stored in a Block-level expression table indexed per cell by `uint16_t`.
- The preprocessor pre-resolves every face-key string into `FaceBC` entries in `Model::face_bcs[c * 6 + dir]`. Assembly hot loops do array lookups only — no string parsing.
- All geometry stays in SI meters.
