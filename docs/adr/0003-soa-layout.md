# ADR-0003: Contiguous Geometry and Attribute Storage

## Status

Accepted.

## Context

Large control-volume models require compact storage and predictable iteration without per-entity allocations.

## Decision

Store three vertex coordinate axes, a dense grid-to-cell map and a compact
cell-to-grid map. Derive cell geometry and adjacency from these arrays. Store
exterior facets in a contiguous vector grouped by owner, with a single offset
array. Keep cell attributes in separate arrays and expression tables.
`FomDefinition` holds the input structure and the compiler builds the runtime
model.

## Rationale

- Compact cell IDs index state and material arrays directly.
- Attribute arrays separate physical assignments from geometric and algebraic identities.
- AoS in IO is correct: it maps 1:1 to XML elements.
