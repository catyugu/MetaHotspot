# ADR-0003: Contiguous Geometry and Attribute Storage

## Status

Accepted.

## Context

Large control-volume models require compact storage and predictable iteration without per-entity allocations.

## Decision

Store conformal geometry as three coordinate axes and derive internal contacts from indices. Store nonconformal geometry as contiguous cell boxes plus CellId connection pairs. Keep explicit boundary pieces and Cell physical attributes at model level. See `include/metahotspot/model/mesh.hpp` and `include/metahotspot/model/model.hpp`. Interior Facets are derived on demand; no internal Facet array, persistent reverse adjacency, or cached connection geometry is retained. `ModelDefinition` describes physical regions independently of mesh specifications; conversion happens once in the compiler.

## Rationale

- Implicit conformal geometry reduces storage; contiguous explicit connections and boundary loops avoid pointer chasing.
- Attribute arrays separate physical assignments from geometric and algebraic identities.
- AoS in IO is correct: it maps 1:1 to XML elements.
