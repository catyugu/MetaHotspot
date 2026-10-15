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

### Round 2: conservative operators and boundary projection

- Red: `build/tests/mhs_tests --gtest_filter=DiscreteMesh.ThermalOperatorsRespectIndependentDofOrdering`
  failed with a nonuniform state and a permuted temperature DOF map.
- Red: `build/tests/mhs_tests --gtest_filter=DiscreteMesh.PartialBoundaryIntegratesActualOverlap`
  failed because a 0.01 m² patch missed the old face-center classifier.
- Green: `cmake --build build --parallel 6`, `ctest --test-dir build --output-on-failure`:
  94/94 passed. `python run_cases.py` passed all eight cases; Python regression:
  16 passed using the baseline command.
- Both microfluid models were assembled through Python before and after this
  round. At 22,400 and 9,432 states, relative Frobenius differences in K were
  1.56e-14 and 3.23e-15; C differed by 6.12e-15 and 5.82e-15. Maximum absolute RHS
  changes were 1.69e-14 and 4.18e-14. These are floating-point summation changes.
- The old structured data remains only for consumers awaiting round 3; thermal
  and fluid assembly now consume explicit face topology and temperature DOF maps.

### Round 3: consumers and public snapshot ABI

- Red: the new external-mesh test did not compile (missing overload and output
  connectivity). The Python mesh snapshot test failed on missing `face_count`.
  Two numerical CTM tests failed on repeated face contributions and permuted DOFs.
- Green: `ctest --test-dir build --output-on-failure`: 96/96 passed;
  `python run_cases.py`: all eight cases passed; Python regression: 19 passed.
- Removed runtime `MeshGeometry`, grid/cell maps, six-slot boundary arrays and
  `core/mesh.hpp`. Only `compiler/cartesian_mesh.cpp` converts supplied Cartesian
  axes into explicit geometry/connectivity. No old ABI adapter is retained.
- The C/Python snapshot exports cell bounds, oriented faces, patch bounds/areas,
  normal distances, region identities and cell-to-state maps. CTM boundary
  groups now reference projected native surfaces, including face identity.
- VTU and node interpolation consume an explicit output mesh. The optional
  Cartesian sampling shape is presentation metadata used only by XML output.
- At shared-face probe coordinates, location selects the lexicographically
  greatest cell lower corner; this is independent of storage and DOF order.


### Round 4: imported meshes, conservative source projections and final checks

- Red tests reproduced missing C/Python external-mesh and DOF-order entry points,
  acceptance of empty/duplicated closed shells and overlapping cells, and loss of
  subcell source volumes. Mixed-material control volumes were also accepted by
  the old center classifier.
- Added `VolumeAssignment` records carrying independent cell/region/source IDs,
  subvolume measures and quadrature centroids. Constant source densities conserve
  integrated power over covered regions. General spatial expressions use midpoint
  quadrature on each geometric piece. Unsupported domain/material cut cells are
  rejected; material tensors are not arbitrarily averaged.
- External import validates cell/face bounds, normal distances, per-side coverage,
  patch intersections, incidence and cell-volume overlap. A bounding-volume
  hierarchy checks overlaps without a global all-pairs search. TPFA rejects
  nonorthogonal cell-center connections. No quadtree generation or refinement is
  implemented.
- CTM now uses native face normals and owner distances for Robin condensation.
  Tests reproduced the previous implicit Z-normal assumption and a junction
  observation using algebraic indices as physical cell indices. Repeated patch
  contributions are accumulated in boundary and interface projections.
- The historical test treating a cell center on a block edge as evidence of full
  cell inclusion was replaced with explicit cut-cell rejection and an aligned
  input-mesh success case.

- Final verification: `cmake --build build --parallel 6` and
  `ctest --test-dir build --output-on-failure`: 100/100 passed.
  `PYTHONPATH=python OPENBLAS_NUM_THREADS=1 python -m pytest python/tests -q`:
  27 passed. `python run_cases.py`: all eight registered cases passed unchanged
  reference thresholds.
- Additional red/green checks reproduced silent truncation of fractional Python
  mesh/DOF IDs. Integer input validation now rejects fractional, nonfinite and
  out-of-range values before native conversion; signed -1 remains accepted for
  exterior neighbors. Focused Ruff checks and formatting checks passed for the
  updated Python input wrapper and mesh contract tests.

## Final interface conventions

- `ModelDefinition` still describes physical entities and expressions. Cartesian
  axes are one supplied mesh format, not a numerical-kernel topology.
- C++: `build_model(definition, discrete_mesh)` consumes a fully specified mesh
  in SI metres. C: `mhs_model_set_discrete_mesh`, `mhs_model_set_dof_order`.
- Python: `Model.set_discrete_mesh(cell_min=..., cell_max=..., face_owner=...,
  face_neighbor=..., face_axis=..., face_sign=..., face_min=..., face_max=...)`.
  Bounds have shape `(count, 3)`, IDs/axes/signs have shape `(face_count,)`;
  external neighbors are `np.iinfo(np.uintp).max` (signed -1 is accepted on input).
  The caller supplies complete faces and connectivity; no neighbors are inferred.
  Coordinates are SI regardless of the physical authoring model's length unit.
- `set_dof_order(cell_to_dof)` selects a bijective temperature-state ordering;
  selecting a new mesh or reading XML clears that ordering. Compiled handles
  remain immutable and reusable.
- `K`, `C`, `f`, `state`, and `state_history` use DOF order. Geometry and evaluated
  material arrays use CellId order. `Solution.temperature` and
  `temperature_history` return physical CellId order (views for identity ordering).
- `Compiled.cells` contains explicit cell bounds, face incidence/geometry and
  volume-projection arrays. Singular region/layer/block tags use their maximum
  integer sentinel when a cell has mixed region membership; the projection rows
  preserve all actual memberships. There is no scalar per-cell source index.
- `Compiled.source_operator(state, time)` returns a sparse matrix of integrated
  source powers, one column per physical source. `SourcePort(source_ids=...,
  power_W=...)` preserves separate sources sharing a temperature DOF; explicit
  physical `cells=` selections remain available. Exactly one selector is required.
- `BoundaryGroup(surface=geometry.surface(Face.ZP), h_range=...)` retains actual
  FaceIds and projected areas. Robin parameters still use the existing effective
  affine approximation; this migration does not claim exact physical-HTC affinity.
- `OutputMesh` is a presentation view with explicit vertices/connectivity.
  VTU supports imported box meshes. Cartesian XML output requires its separate
  sampling-grid shape; it is not an unstructured result format.

## Reproduce the final regression

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=OFF
cmake --build build --parallel 6
ctest --test-dir build --output-on-failure
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 python -m pytest python/tests -q
python run_cases.py
```

The Python suite includes imported, randomly reordered 22,400- and 9,432-state
microfluid meshes. It restores the original algebraic ordering with an explicit
DOF map, compares K/C/f and physical source columns, and solves both models.
The pressure preprocessing uses the existing 1e-8 relative iterative tolerance;
its permutation-dependent flux differences are checked at that tolerance rather
than incorrectly requiring bitwise identity of iterative hydraulic solutions.
The capacity and source projection comparisons retain 1e-13 checks.

Measured relative K differences after permutation were 5.02e-11 and 1.89e-15;
capacity differences were zero; maximum physical temperature differences were
6.14e-7 K and 4.86e-11 K. Both microfluid CLI cases also solved and wrote VTU/XML.
Against the pre-refactor operators without reordering, final K differences were
4.77e-15 and 9.52e-16; final temperatures differed from SciPy direct solutions of
those baseline operators by at most 1.57e-6 K and 4.57e-7 K.

All eight registered cases passed unchanged reference thresholds. Seven final
node fields matched the saved baseline at XML output precision; the remaining
nonlinear case differed by at most 2.41e-4 K. No reference thresholds were relaxed.
