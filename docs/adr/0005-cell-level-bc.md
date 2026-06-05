# ADR-0005: Cell-Level BC Storage

## Status

Accepted. Supersedes the per-face SoA arrays described in ADR-0002.

## Context

When block face projections overlap in the same layer, a per-face SoA array can only hold one BC per face — but two overlapping blocks in the same layer can disagree on what that face's BC should be.

## Decision

Store BCs **per cell per face**:

```cpp
struct CellBC {
    std::array<BcType, FACE_COUNT> types;        // xm, xp, ym, yp, zm, zp
    std::array<uint16_t, FACE_COUNT> param_idxs; // indices into BCParamTable
};
```

## Rationale

- **No projection ambiguity**: each cell's face is independent.
- **Virtual-neighbor consistency**: when a cell's neighbor is virtual, the cell's face BC is already set to `other_bc` during preprocessing — same channel as overlap resolution.
- **Flexibility**: different cells in the same layer can have different BC types on the same directional face.
- **Simpler preprocessing**: no global face arrays; assignment is at cell level.

## Data flow

```text
IOStructure
  └─> Preprocessor::load()
        ├─> preprocessor::resolve_layers()
        │     ├─> valid_mask, index_map
        │     └─> material_id, layer_id (full-grid)
        ├─> preprocessor::resolve_face_keys()
        │     ├─> flatten (boundary, face_key) pairs
        │     └─> single grid traversal → CellBC per cell, per face, with `other_bc` fallback (handles overlap)
        └─> Compile expressions → BCParamTable + heat_source_table
```

## Notes

- `BCParamTable` remains: shared BC parameters live there, referenced by `param_idx`.
- `other_bc` is applied during preprocessing, not at assembly time.
- Virtual neighbors: the active cell's face touching a virtual cell gets `other_bc` set during `resolve_face_keys()`.
- The heat source is **not** part of `CellBC` — it is a deduplicated dictionary indexed by `uint16_t` (see ADR-0004 §Heat source dictionary).
