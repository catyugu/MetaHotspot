# AGENTS.md - MetaHotspot

## Project Overview

MetaHotspot is a multi-language thermal simulation platform for VLSI/heterogeneous integration. It converts HotSpot-format inputs into structured mesh + JSON configs, then solves steady-state and transient thermal problems using a Finite Volume Method (FVM) solver.

**Architecture**: Adapter (Python) -> Mesher (Gmsh) -> FVM Solver (Python/SciPy)
**Long-term goal**: Migrate the solver core to C++17/20 with Eigen/PARDISO.

### Subprojects

| Directory      | Language     | Purpose                                                         |
| -------------- | ------------ | --------------------------------------------------------------- |
| `metahotspot/` | Python 3.10+ | Core library: parser, converter, mesher, FVM solver, data model |
| `scripts/`     | Python 3.10+ | CLI entry points: adapter, solver, pipeline runner              |
| `Hotspot/`     | C (C99)      | Reference HotSpot thermal simulator                             |
| `examples/`    | -            | Converted HotSpot example data (example1-example4)              |
| `docs/`        | Markdown     | Project documentation (Chinese)                                 |
| `cpp/`         | C++20        | Future C++ solver (empty, planned)                              |

---

## Environment Setup

```bash
conda activate numerical
```

All Python dependencies (numpy, scipy, meshio, gmsh) and C/C++ tools (clang, clang-cl, cmake) are in this conda environment.

---

## Python Workflow (Full Pipeline)

**Convert HotSpot examples → mesh → run steady + transient:**

```bash
# One-shot: convert example1-example4 and mesh them
python scripts/adapter.py --batch-four

# Run steady→init→transient on one converted example
python scripts/run_example_pipeline.py examples/hotspot_converted/example1
```

**Single example (manual step-by-step):**

```bash
python scripts/adapter.py Hotspot/examples/example1 examples/output --mode both
python scripts/run_example_pipeline.py examples/output
```

---

## C (HotSpot Subproject)

See `Hotspot/AGENTS.md` for full details. Quick reference:

```bash
cd Hotspot/build
cmake -G Ninja ..
ninja

# Run
./hotspot.exe -f <floorplan.flp> -p <power.ptrace> -c <config.config>
```

---

## Code Style Guidelines

### Python (metahotspot/, scripts/)

**Naming Conventions:**
- Classes: `PascalCase` - e.g., `FVMSolver`, `GmshMesher`, `SimulationModelBuilder25D`
- Functions/methods: `snake_case` - e.g., `solve_steady()`, `parse_flp()`
- Private methods: `_leading_underscore` - e.g., `_prepare_mesh()`, `_extract_faces()`
- Constants: `UPPER_SNAKE_CASE` - e.g., `DEFAULT_CONFIG`, `STANDARD_MATERIALS`
- Variables: `snake_case` - e.g., `hex_data`, `boundary_faces`

**Type Annotations:** Required on all function signatures.
- Use `from typing import Dict, List, Tuple, Optional, Any`
- Use `@dataclass(slots=True)` for data structures

**Imports:** Standard library → third-party → local (absolute imports)

**Error Handling:**
- `ValueError` for invalid data; `FileNotFoundError` for missing files
- Log prefix: `[INFO]`, `[WARNING]`, `[RESULT]`
- Non-critical failures print warning, don't crash (e.g., pressure solve)
- No assertions in production code

**Data Model:** `model25d.py` is single source of truth for config/materials/stackup.
- Use `load_config()` + `load_stackup()` as entry points
- Config always merged with defaults via `merge_with_defaults()`

---

## Key Modules

| Module                  | Class/Functions                                                                                                          | Purpose                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| `model25d.py`           | `DEFAULT_CONFIG`, `STANDARD_MATERIALS`, `Unit2D`, `Layer25D`, `load_config()`, `merge_with_defaults()`, `load_stackup()` | Single source of truth for defaults                              |
| `assembler.py`          | `FVMAssembler`                                                                                                           | Face-to-cell extraction, conduction/advection matrix assembly    |
| `thermal_solver.py`     | `ThermalSolver`                                                                                                          | Steady-state (direct sparse solve) or transient (implicit Euler) |
| `hotspot_parser.py`     | `HotSpotParser` (static methods)                                                                                         | `.flp`, `.config`, `.materials`, `.lcf` parsing                  |
| `converter.py`          | `SimulationModelBuilder25D`, `convert_hotspot_to_metahotspot()`                                                          | HotSpot → MetaHotspot JSON                                       |
| `gmsh_mesher.py`        | `GmshMesher`                                                                                                             | Hexahedral mesh generation via gmsh API (**non-conformal mesh**) |
| `mesh_preprocessor.py`  | `MeshPreprocessor`                                                                                                       | .msh → `MeshTopology` + `PhysicalFields`                         |
| `fluid_preprocessor.py` | `FluidPreprocessor`                                                                                                      | Solve pressure and calculate hydro properties before thermal assembly         |
| `metahotspot_solver.py` | `MetaHotspotSolver`                                                                                                      | End-to-end: preprocess → fluid flow → assemble → solve → export VTU |

---

## File Formats

- **`solver_config.json`**: Solver config (JSON, 4-space indent)
- **`mesh.msh`**: Gmsh v4 mesh with physical groups
- **`result.vtu`** / **`transient_result.vtu`**: VTK Unstructured Grid with `Temperature_K`
- **`*_layout.json`**: Per-layer layout units (in `layouts/`)

---

## Architecture Principles

1. **Single source of truth**: `model25d.py` owns all defaults—never duplicate fallback logic
2. **No backward compatibility**: Update all call sites when refactoring
3. **Sequential completion**: Each phase complete before next starts
4. **Config-driven**: All params flow through `solver_config.json`