# ADR-0005: Cell-Level BC Storage

## Status

Accepted

## Decision

Boundary conditions are stored at **cell level**, with each cell storing BC information for its 6 faces independently:

```cpp
struct CellBC {
    std::array<BcType, 6> types;        // xm, xp, ym, yp, zm, zp
    std::array<uint16_t, 6> param_idxs; // indices into BCParamTable
};

struct CellFields {
    // Compact size (N_active): active cells only
    std::vector<CellBC> cell_bcs;
    std::vector<CompiledExpression> heat_source;
};
```

`FaceBCFields` is removed.

## Rationale

- **No projection ambiguity**: Each cell's face has independent BC, regardless of overlapping projections
- **Consistent with virtual cell handling**: When a cell's neighbor is virtual, the cell's face BC is already set to `other_bc` during preprocessing
- **Flexibility**: Different cells in the same layer can have different BC types on the same directional face
- **Simpler preprocessing**: FaceKeyProcessor no longer needs to manage global face arrays; BC assignment happens at cell level

## Data Flow

```text
IOStructure
  └─> ModelBuilder::build()
        ├─> LayerProcessor::resolve()
        │     ├─> Generate valid_mask, index_map
        │     └─> Assign material_id, layer_id per cell
        ├─> FaceKeyProcessor::resolve()
        │     └─> Assign CellBC to each cell's faces
        │         (handles projection overlap by cell-to-face assignment)
        ├─> Apply other_bc to unspecified faces
        └─> Compile expressions → BCParamTable, heat_source
```

## Notes

- `BCParamTable` still exists: shared BC parameters are indexed by `param_idx`
- `other_bc` is applied during preprocessing, not at assembly time
- Virtual cell neighbors: the active cell's face touching a virtual cell gets `other_bc` set during LayerProcessor.resolve()
