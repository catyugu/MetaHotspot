# ADR-0002: Cell-Centered DOFs with Cell-Level BC Storage

## Status

Accepted.

## Context

The authoring model uses structured face regions to specify BCs; the XML adapter translates its legacy face-key encoding at the I/O boundary. The mesh uses cell-centered DOFs. BC types: Dirichlet / Neumann / Cauchy. A future microfluid phase might want face DOFs for velocity/pressure.

Two storage concerns must be answered together:

1. **Mathematical placement of DOFs.** Flux-type BCs are integrals, not field values; a face DOF would not be used by them. Storing a face DOF alongside a cell DOF roughly doubles memory.
2. **Storage shape when block face projections overlap in the same layer.** A per-face SoA array holds only one BC per face, but two overlapping blocks in the same layer can disagree on what that face's BC should be.

## Decision

**Cell-centered DOFs only, stored at the cell level.** BCs are applied as boundary integrals over cell faces — no face DOFs stored, no per-face SoA arrays. Each face of every cell carries its own BC descriptor, stored as a flat array on `Model`, indexed into a shared parameter table:

```cpp
struct FaceBC {
    BcType type = BcType::None;
    TableIndex param_idx = 0;    // → BCParamTable
};

// Model::face_bcs is std::vector<FaceBC> with size N_active * 6.
// face_bcs[c * 6 + dir] gives the BC for cell c's face dir.
```

Discrete treatments:

- **Dirichlet (FirstType)** — ghost cell method. The boundary cell's equation uses the ghost value `T_ghost = 2·T_dirichlet − T_boundary` (linear contribution to diag and RHS).
- **Neumann (SecondType)** — flux `q·n` enters the cell RHS directly as `Σ q·A_face`.
- **Cauchy (ThirdType)** — linearized as `h·A·(T_boundary − T_∞)`; contributes to the diagonal (`h·A` coefficient) and to the RHS (`h·A·T_∞`).

Anisotropic `k`: at assembly, pick `k_along(dir) ∈ {kx, ky, kz}` per face normal.

The default boundary is applied during preprocessing to every face that was not explicitly specified — including faces of virtual neighbors — so the assembly hot loop sees a fully-populated per-face BC for every cell. Explicit patches are applied in authoring order, so later patches override earlier ones.

The fluid subsystem is independent of thermal BC storage: thermal faces use `FaceBC`; fluid boundary records are temporary preprocessing state and only assembly-ready values enter `FluidDomain`.

## Rationale

- **No projection ambiguity.** Each individual face BC is independent; overlapping blocks each carry their own resolution.
- **Virtual-neighbor consistency.** When a cell's neighbor is virtual, the cell's face BC is already set to the default boundary during preprocessing — the same channel as overlap resolution.
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
        ├─> preprocessor::compile_boundary_patches()
        │     └─> flatten ordered (condition, FaceRegion) pairs
        ├─> preprocessor::resolve_boundary_patches()
        │     └─> face_bcs [N_active * 6] (handles overlap)
        └─> Compile expressions → BCParamTable + heat_source_table
```

## Notes

- `BCParamTable` remains: shared BC parameters live there, referenced by `param_idx`.
- The default boundary is applied during preprocessing, not at assembly time. For virtual neighbors, the active cell's touching face is populated during `resolve_boundary_patches()`.
- The heat source is **not** part of `FaceBC` — it is stored in a Block-level expression table indexed per cell by the 32-bit `TableIndex`.
- The preprocessor resolves every structured `FaceRegion` into `FaceBC` entries in `Model::face_bcs[c * 6 + dir]`. Assembly hot loops do array lookups only — no region matching or string parsing.
- All geometry stays in SI meters.
