# ADR-0003: Contiguous Geometry and Attribute Storage

## Status

Accepted.

## Context

Large control-volume models require compact storage and predictable iteration without per-entity allocations.

## Decision

Store cell geometry and face patches in contiguous vectors, adjacency in CSR arrays, and physical attributes in separate arrays and expression tables. See `include/metahotspot/core/discrete_mesh.hpp` and `include/metahotspot/core/model.hpp`. `ModelDefinition` keeps the input-oriented structure; conversion happens once in the compiler.

## Rationale

- Contiguous cell and face loops avoid pointer chasing.
- Attribute arrays separate physical assignments from geometric and algebraic identities.
- AoS in IO is correct: it maps 1:1 to XML elements.
