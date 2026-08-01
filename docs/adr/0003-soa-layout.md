# ADR-0003: SoA Data Layout Throughout

## Status

Accepted.

## Context

CLAUDE.md mandates data-oriented design and flat SoA. The structured grid can hold millions of cells.

## Decision

All mesh and field data uses SoA throughout the internal model. See `src/common/model.hpp` for concrete types. `ModelDefinition` keeps the input-oriented structure; conversion happens once in the preprocessor.

## Rationale

- Hot assembly loops iterate cells, reading one field at a time — SoA gives cache-friendly contiguous reads and SIMD opportunity.
- TBB parallel_for partitions the contiguous cell range cleanly per worker.
- AoS in IO is correct: it maps 1:1 to XML elements.
