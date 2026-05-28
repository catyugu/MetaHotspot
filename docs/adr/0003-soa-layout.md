# ADR-0003: SoA Data Layout Throughout

## Status

Accepted

## Context

CLAUDE.md mandates "面向数据设计" (data-oriented design) and "扁平化的 SoA 设计" (flat Structure-of-Arrays design). The simulation operates on a 3D structured grid with up to millions of cells.

## Decision

All mesh and field data uses **SoA (Structure of Arrays)** layout throughout the internal model:

```cpp
// Cell-centered fields (SoA)
temperature[N]:      float          // solution DOF vector
material_id[N]:      uint8_t        // which material per cell
layer_id[N]:         uint8_t        // which layer
bc_applied[N]:      uint8_t        // flags for BC application

// Per-face fields (6 faces, SoA)
face_zm_bc_type[N_xy]: uint8_t
face_zm_bc_param[N_xy]: uint16_t
// ... zp, ym, yp, xm, xp

// Mesh geometry
vertex_x[nx+1]:     float
vertex_y[ny+1]:     float
vertex_z[nz+1]:     float
```

Where `N = nx · ny · nz` (total cell count).

## Rationale

- SoA enables vectorized assembly: iterating over cells, reading all fields of the same type in sequence → optimal cache utilization and SIMD opportunity.
- AoS would scatter reads across struct members — bad for hot loops that only need one field at a time.
- CLAUDE.md explicitly requires SoA.
- SoA also simplifies parallelization with TBB: each thread operates on a contiguous range of cell indices, reading its private field segment.

## Notes

- IO model (XML serialization) uses AoS structs that mirror the XML schema — that's intentional, as it maps directly to XML elements.
- Conversion IO → Internal happens once at preprocessor stage, converting AoS IO structures to flat SoA internal arrays.
