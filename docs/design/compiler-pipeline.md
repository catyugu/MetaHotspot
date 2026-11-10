# Model compiler stages

`build_model(definition, mesh)` compiles continuous box geometry and physical
assignments against a separate Cartesian grid. The public C++, C and Python
input layouts and PTA XML formats are unchanged.

## Ownership and stage boundaries

| Stage | Implementation | Temporary output |
| --- | --- | --- |
| Validate physics and compile expressions | `model_physics.cpp` | Boundary region tags, default thermal condition ID, initial material viscosities |
| Compile geometry | `conformal_mesh.cpp`, `volume_rasterizer.cpp` | `DiscreteMesh` and one input-region ID per active cell |
| Bind cell physics and probes | `model_physics.cpp`, `model_compiler.cpp` | Material IDs, source quadrature and probe cell IDs stored in `Model` |
| Partition and tag boundaries | `geometry_compiler.cpp` | Canonical facets and per-physics tag arrays |
| Prepare frozen flow | `fluid_preprocessor.cpp` | Assembly-ready flow data stored in `Model` |

Geometry borrows spans of box operations and never reads material or source
IDs. Region IDs refer to the original volume order. The region map is released
before boundary partitioning. The expression compiler owns its symbol table;
compiled expressions retain their own function state. All scratch data is local
to one compilation.

With DEBUG logging enabled (`MHS_LOG_LEVEL=debug`), a compilation reports elapsed
milliseconds for expressions, cells, boundaries and flow. The cells measurement
includes geometry, physical binding and probe location. The flow measurement
includes pressure-system assembly and solution.

## Cartesian region rasterization

Box bounds are located against sorted cell-center coordinates. Reverse traversal
of volumes and their operations applies the highest-priority decision first.
An Add records the region ID; a Subtract temporarily excludes cells from that
volume. Exclusions are cleared before visiting the next volume, so a lower
volume can remain visible through a subtraction. Fully assigned grids stop
traversal early. Ownership storage is reused as the structured-to-active-cell
map when constructing facets.

The aligned path costs `O(N + sum of box footprints)` plus coordinate searches,
rather than testing every cell against every box. Broad, partially overlapping
boxes can still require repeated visits. The dense map uses one entity index
per candidate cell; active cells also have a temporary region map. Subtraction
uses a reusable list of excluded indices.

Only effective region boundaries must align with grid planes, within the
existing `geometry_eps` tolerance. An unaligned raw face triggers validation of
its intersected grid cells. Each such cell is partitioned by intersecting box
faces, and every open piece must have the same final region owner as the cell
center. Hidden and cancelled faces are accepted; effective cuts through a cell
are rejected. This exceptional path can be expensive for many overlapping,
unaligned boxes and does not change the input grid or create smaller cells.

Probe lookup searches the input axes and the lexicographically ordered active
cells. Points on shared planes consider every adjacent candidate and select the
last active containing cell, matching the previous closed-box scan. Outside
points retain the invalid cell ID.

## Shared exterior partition

Boundary regions are indexed by axis and sorted plane coordinate. Each exterior
facet searches its plane candidates once, collects geometric cuts, and tags the
resulting pieces for all requested groups. Tags are opaque IDs; the geometry
compiler has no thermal or hydraulic policy. Original region order determines
precedence even when nearby plane coordinates sort differently. Tolerance-only
rectangle matches remain eligible for tagging without adding cuts.

Thermal and hydraulic conditions share the same FacetIds and incidence. Interior
facets retain invalid boundary tags. Rectangular cut ordering is preserved, so
many intersecting cuts can still produce a Cartesian product of pieces. The
index reduces unrelated-plane searches; it is not a two-dimensional spatial
index for arbitrarily many coplanar rectangles.

## Local validation, 2026-11-11

Baseline: `dev` at `be0ba03cc80ffaac11778619cde3332620ed76bd`.
Local builds used GCC 14.2, third-party headers/libraries from the supplied Linux
SDK, AMGCL 1.5.0 and GoogleTest 1.17.0. The repository dependency declarations were
not changed. Cross-platform CI remains necessary.

- Release build with project warnings treated as errors: passed.
- Existing CTest suite: 80/80 passed, also under AddressSanitizer and UBSan.
  LeakSanitizer was disabled because the execution environment uses ptrace.
- Existing Python tests: 26/26 passed against the rebuilt C library.
- All 10 registered CLI cases passed their reference comparisons; generated XML
  and VTU files matched the baseline byte for byte.
- Temporary differential checks over 200 random aligned models matched cells,
  facets, incidence, materials, source quadrature, boundary tags and probe IDs.
- A fine-grid ownership oracle checked 500 random CSG models against a coarser
  grid: 18 uniform models were accepted and 482 effective cuts were rejected.
- Compiler translation unit with logging compiled out: passed with `-Werror`.
- Installation/export configuration generation: passed.

Two synthetic `64^3`-cell fixtures measured complete `build_model` calls in
separate processes. Times are medians of three runs; peak RSS is the median of
those runs. Model fingerprints matched in both fixtures.

| Fixture | Baseline time | New time | Baseline peak RSS | New peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Full base volume, 2,000 unit boxes, 200 probes | 720.703 ms | 78.233 ms | 121,420 KiB | 125,960 KiB |
| Full volume, 6,000 small boundary rectangles across grid planes | 399.047 ms | 99.146 ms | 166,640 KiB | 171,348 KiB |

These fixtures measure the affected search patterns, not a general solver
speedup. Temporary validation programs and generated result files are excluded
from the change; the existing test sources are unchanged.
