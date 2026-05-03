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
| `ARTSim/`      | Python       | Reference thermal simulator (submodule, Makefile-based)         |
| `examples/`    | -            | Converted HotSpot example data (example1-example4)              |
| `docs/`        | Markdown     | Project documentation (Chinese)                                 |
| `cpp/`         | -            | Future C++ solver (currently empty)                             |

---

## Build & Run Commands

### Environment Setup

```bash
conda activate numerical
```

* All Python dependencies (numpy, scipy, meshio, gmsh) must be available in this conda environment.
* All C/C++ tools: `clang`, `clang-cl`, `cmake` are also all in the `numerical` miniconda environment.

### MetaHotspot Pipeline (Python)

**Step 1 - Convert HotSpot examples to MetaHotspot format:**
```bash
python scripts/adapter.py Hotspot/examples examples/hotspot_converted --batch-four
```

**Step 2 - Run solver on a converted example:**
```bash
python scripts/solver.py examples/hotspot_converted/example1/solver_config_steady.json
```

**Step 3 - Run full steady->transient pipeline:**
```bash
python scripts/run_example_pipeline.py examples/hotspot_converted/example1
```

**Convert a single example (not batch):**
```bash
python scripts/adapter.py <input_dir> <output_dir> --mode steady
python scripts/adapter.py <input_dir> <output_dir> --mode transient
python scripts/adapter.py <input_dir> <output_dir> --mode both
```

---

## Code Style Guidelines

### Python (metahotspot/, scripts/)

**Naming Conventions:**
- Classes: `PascalCase` - e.g., `FVMSolver`, `GmshMesher`, `SimulationModelBuilder25D`
- Functions/methods: `snake_case` - e.g., `solve_steady_state()`, `parse_flp()`
- Private methods: `_leading_underscore` - e.g., `_prepare_mesh()`, `_extract_faces()`
- Constants: `UPPER_SNAKE_CASE` - e.g., `DEFAULT_CONFIG`, `STANDARD_MATERIALS`, `GEOMETRY_TOLERANCE`
- Variables: `snake_case` - e.g., `hex_data`, `boundary_faces`

**Type Annotations:**
- Always use type hints on function signatures
- Use `from typing import Dict, List, Tuple, Optional, Any` for complex types
- Use `dataclass` for data structures (`@dataclass` with `slots=True` when performance matters)
- Example: `def parse_flp(file_path: str) -> List[dict]:`

**Imports:**
- Standard library first, then third-party, then local
- Use absolute imports from the package: `from metahotspot.model25d import load_config`
- Third-party: `numpy as np`, `scipy.sparse as sp`, `meshio`, `gmsh`

**Error Handling:**
- Use `ValueError` for invalid data (e.g., `raise ValueError("No hexahedron cells found")`)
- Use `FileNotFoundError` for missing files
- Print warnings with `[WARNING]` prefix, info with `[INFO]`, results with `[RESULT]`
- Try/except around non-critical operations (e.g., pressure solve failure prints warning, doesn't crash)
- Assertions not used in production code - use explicit checks instead

**Comments:**
- Chinese comments are acceptable and common in this codebase
- Section dividers use `# ==========` patterns
- Docstrings on public classes and functions

**Data Model Pattern:**
- `model25d.py` is the single source of truth for config/materials/stackup
- Use `load_config()` and `load_stackup()` as unified entry points
- Config always merged with defaults via `merge_with_defaults()`
- Data classes: `Unit2D`, `Layer25D` in `model25d.py`; `Cell` in `fvm_solver.py`

**Solver Pattern:**
- FVMSolver builds sparse matrices (CSR format) and solves with `scipy.sparse.linalg`
- Builder pattern for model construction (chain calls: `.build_materials().build_chip_layers()...`)
- Solver outputs VTU files via `meshio`

---

## Key Module Reference

### metahotspot/model25d.py
- **`DEFAULT_CONFIG`**: Global default configuration dict (single source of truth)
- **`STANDARD_MATERIALS`**: Material library (silicon, copper, aluminum, tim, water, default_solid)
- **`Unit2D`**: 2D layout unit dataclass
- **`Layer25D`**: Stackup layer dataclass with list of `Unit2D`
- **`load_config(path)`**: Unified config loader + merger
- **`merge_with_defaults(raw)`**: Type-safe config merging with fallback handling
- **`load_stackup(config, base_dir)`**: Build layer list from config

### metahotspot/converter.py
- **`SimulationModelBuilder25D`**: Builder that converts HotSpot inputs to MetaHotspot JSON
- **`convert_hotspot_to_metahotspot()`**: Top-level conversion function
- **`convert_hotspot_with_modes()`**: Supports steady/transient/both modes

### metahotspot/gmsh_mesher.py
- **`GmshMesher`**: Generates hexahedral meshes via gmsh Python API
- Adaptive refinement near heat sources
- Morton-code spatial sorting for cache-friendly cell ordering
- **IMPORTANT**: The mesh generated is **non-conformal**.

### metahotspot/fvm_solver.py
- **`FVMSolver`**: Core FVM thermal solver
- Face-to-cell extraction for boundary/internal face classification
- Conduction matrix assembly (overlap-area based)
- Pressure-driven advection for fluid cells (hydrodynamic resistance model)
- London-Shah Nusselt number correlation for conjugate heat transfer
- Steady-state: direct sparse solve
- Transient: implicit Euler with LU factorization

### metahotspot/hotspot_parser.py
- **`HotSpotParser`**: Static methods to parse `.flp`, `.config`, `.materials`, `.lcf` files
- Skips comment lines (starting with `#`), strips whitespace

### scripts/adapter.py
- CLI for converting HotSpot examples + meshing in one step
- `--batch-four` mode for example1-example4

### scripts/solver.py
- Thin CLI wrapper: `FVMSolver(config_path).solve()`

### scripts/run_example_pipeline.py
- Full pipeline: steady solve -> copy result as init -> transient solve

---

## File Formats

- **`solver_config.json`**: Unified solver configuration (JSON, 4-space indent)
- **`mesh.msh`**: Gmsh v4 format mesh with physical groups
- **`result.vtu`** / **`transient_result.vtu`**: VTK Unstructured Grid with `Temperature_K` cell data
- **`*_layout.json`**: Per-layer layout unit definitions (in `layouts/` subdirectory)

## Architecture Principles

1. **Single source of truth**: `model25d.py` owns all config/material defaults. Never duplicate fallback logic.
2. **No backward compatibility**: When refactoring, update all call sites. Do not add compatibility shims.
3. **Sequential completion**: Each phase of work must be fully complete before starting the next.
4. **Config-driven**: All simulation parameters flow through `solver_config.json`. The solver reads only this + the mesh file.
