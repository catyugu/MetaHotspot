# Discrete model migration

Scope: consume an explicit mesh, separate cell identity from geometry, adjacency,
physical assignments and algebraic ordering. No quadtree generation, adaptive
refinement, backwards ABI compatibility or unverified nonorthogonal flux scheme.

## Acceptance contract

- Cells have physical bounds; faces are unique oriented rectangular patches.
- Topology is a cell/face incidence graph, not coordinate-index arithmetic.
- Boundary conditions are assigned to exterior patches, separately from geometry.
- Thermal unknowns are addressed through a bijective `DofMap`.
- Internal conduction is assembled once per face with equal/opposite flux.
- Solid/fluid interface physics is selected directly, not applied as a correction.
- Sampling and output consume geometry and incidence; CTM consumes native face
  geometry and explicit cell-to-state maps.
- Cartesian axes are an input/output adapter concern only.
- The current TPFA method supports orthogonal axis-aligned boxes with diagonal
  conductivity. Representability of a mesh is not evidence of numerical accuracy.

## Workflow

Each migration round adds or rewrites tests, observes the expected failure,
implements the change, runs unit and case regressions, then commits and pushes.
Intermediate commits may retain old consumers until the next round; the final
model and public interfaces do not retain the old structured-grid contracts.

## Validation record

Commands and outcomes are recorded below as each round completes.

### Baseline and round 1

- Baseline: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`,
  `cmake --build build --parallel 4`,
  `ctest --test-dir build --output-on-failure`: 90/90 passed.
- Baseline: `python run_cases.py`: all eight registered cases passed the existing
  thresholds. `PYTHONPATH=python OPENBLAS_NUM_THREADS=1 python -m pytest python/tests -q`:
  16 passed.
- Red: `g++ -std=c++20 -fsyntax-only -Isrc -I.cache/cpm/googletest/9eec/googletest/include tests/test_discrete_mesh.cpp`
  failed because `core/discrete_mesh.hpp` did not exist.
- Green: `cmake --build build --parallel 6`,
  `ctest --test-dir build --output-on-failure`: 92/92 passed, including closed
  oriented incidence, unique internal faces, invalid mesh and DOF permutations.
- Round 1 case regression: `python run_cases.py` passed all existing thresholds;
  Python regression: 16 passed with the baseline command.
