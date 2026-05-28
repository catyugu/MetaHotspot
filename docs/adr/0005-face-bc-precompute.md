# ADR-0005: Precomputed Face BC Arrays

## Status

Accepted

## Context

Boundary conditions are specified in the XML using face-key strings like `Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100`. The format encodes: `Face|Direction|LayerIndex|X_min,Y_min,X_max,Y_max;...`. Assembling the linear system requires knowing, for each boundary face of each cell, what BC type applies and what parameters to use.

## Decision

`preprocessor` resolves all face-key strings into **precomputed face BC arrays**:

- Per face direction (Z-, Z+, Y-, Y+, X-, X+): `bc_type[ni·nj]` and `bc_param[ni·nj]` arrays
- `bc_type`: `None=0`, `Dirichlet=1`, `Neumann=2`, `Cauchy=3`
- `bc_param`: index into a `BCParameterTable` (e.g., for Cauchy: param stores `{h, T_inf}` packed)

During assembly, for each cell, the hot loop does:

```cpp
auto bc = face_bc_type[face_idx];        // O(1) array lookup
auto param = face_bc_param[face_idx];     // O(1) array lookup
apply_bc(bc, param, cell_temperature, A, b);
```

## Rationale

- Assembly hot loop must be as fast as possible — no string parsing, no geometric intersection tests.
- Pre-resolving once at preprocessing cost is trivial (done once, used N·6 times per assembly call).
- Face BC arrays are SoA alongside cell arrays — cache-friendly iteration.
- The preprocessing cost of parsing face keys is negligible compared to the N assembly iterations.

## Notes

- Face keys also carry a `LayerIndex` component (e.g., `Z|E|30|...`) — this is used during preprocessing to only apply the BC to cells whose Z range overlaps with that layer index.
- The `BCParameterTable` is a flat array of parameter structs: `{double temperature}` for Dirichlet, `{double flux}` for Neumann, `{double h, double T_inf}` for Cauchy.
