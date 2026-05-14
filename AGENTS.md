# AGENTS.md - MetaHotspot

## Project Overview

MetaHotspot is a multi-language thermal simulation platform for VLSI/heterogeneous integration. It converts HotSpot-format inputs into structured mesh + JSON configs, then solves steady-state and transient thermal problems using a Finite Volume Method (FVM) solver.

**Architecture**: Adapter (Python) → Mesher (native Python) → FVM Solver (Python/SciPy)
**Long-term goal**: Migrate the solver core to C++17/20 with Eigen/PARDISO.

### Subprojects

| Directory      | Language     | Purpose                                                         |
| -------------- | ------------ | --------------------------------------------------------------- |
| `metahotspot/` | Python 3.10+ | Core library: parser, converter, mesher, FVM solver, data model |
| `scripts/`     | Python 3.10+ | CLI entry points: adapter, solver, pipeline runner              |
| `Hotspot/`     | C (C99)      | Reference HotSpot thermal simulator (submodule)                 |
| `examples/`    | —            | Converted HotSpot example data (example1–example4)              |
| `docs/`        | Markdown     | Project documentation (Chinese)                                 |
| `cpp/`         | C++20        | Future C++ solver (planned, empty)                              |

---

## Environment Setup

```bash
conda activate numerical
```

All Python dependencies (numpy, scipy, meshio, gmsh, numba) and C/C++ tools (clang, clang-cl, cmake, ninja) are in this conda environment.

## Python Workflow

### Convert examples (one-time setup)

```bash
# From project root — uses default paths Hotspot/examples → examples/hotspot_converted/
python scripts/adapter.py --batch-four
```

Equivalent explicit form:

```bash
python scripts/adapter.py Hotspot/examples examples/hotspot_converted --batch-four
```

This produces `solver_config_steady.json`, `solver_config_transient.json` per example directory.

### Run steady → init → transient pipeline

```bash
python scripts/run_example_pipeline.py examples/hotspot_converted/example1
```

Each converted example also has a `run.py` shortcut:

```bash
python examples/hotspot_converted/example1/run.py
```

**Pipeline flow:**

1. Steady solve → writes `result.vtu`
2. Copies `result.vtu` → `init.vtu` (initial field for transient)
3. Transient solve → writes `transient_result.vtu`
4. Copies all outputs to `outputs/`

### Manual single-example workflow

```bash
# Convert and mesh
python scripts/adapter.py Hotspot/examples/example1 examples/output --mode both

# Run pipeline
python scripts/run_example_pipeline.py examples/output
```

---

## Validation Commands

From `docs/VALIDATION.md` — these are the canonical verification steps:

```bash
# Convert examples 1-4 (one-time setup)
python scripts/adapter.py Hotspot/examples examples/hotspot_converted --batch-four

# Run a single example (steady → init → transient pipeline)
python examples/hotspot_converted/example1/run.py
# Shortcut: python examples/hotspot_converted/example1/run.py  (calls scripts/run_example_pipeline.py internally)
```

Each `examples/hotspot_converted/exampleN/run.py` is a thin wrapper:

```python
subprocess.run([sys.executable, str(script), str(example_dir)], check=True)
# script = project_root / "scripts" / "run_example_pipeline.py"
```

Output goes to `examples/hotspot_converted/exampleN/outputs/` (steady_result.vtu, transient_result.vtu, init.vtu).

## C++ Solver (placeholder)

`cpp/` directory is currently empty — future C++20 solver target. No build infrastructure exists yet.

## HotSpot Subproject

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

| Element           | Convention            | Example                                |
| ----------------- | --------------------- | -------------------------------------- |
| Classes           | `PascalCase`          | `FVMSolver`, `GmshMesher`              |
| Functions/methods | `snake_case`          | `solve_steady()`, `parse_flp()`        |
| Private methods   | `_leading_underscore` | `_prepare_mesh()`                      |
| Constants         | `UPPER_SNAKE_CASE`    | `DEFAULT_CONFIG`, `STANDARD_MATERIALS` |
| Variables         | `snake_case`          | `hex_data`, `boundary_faces`           |

- **Type annotations** required on all function signatures
- Use `@dataclass(slots=True)` for data structures
- Imports: standard library → third-party → local (absolute imports)
- Log prefix: `[INFO]`, `[WARNING]`, `[RESULT]`
- No assertions or fallbacks in production code
- Non-critical failures print warning, don't crash (e.g., pressure solve)

### Data Model

`model25d.py` is the **single source of truth** for defaults—never duplicate fallback logic.

- Always use `load_config()` + `load_stackup()` as entry points
- Config always merged with defaults via `merge_with_defaults()`
- Property resolution priority: unit > unit material > layer material > default material

## Key Modules

| Module                     | Class/Functions                                                                                                             | Purpose                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `model25d.py`              | `DEFAULT_CONFIG`, `STANDARD_MATERIALS`, `UnitRegion`, `LayerRegion`, `merge_with_defaults()`, `parse_computational_model()` | Single source of truth for defaults                                 |
| `legacy/converter.py`      | `SimulationModelBuilder25D`, `convert_hotspot_with_modes()`                                                                 | HotSpot → MetaHotspot JSON + layout files                           |
| `legacy/hotspot_parser.py` | `HotSpotParser` (static methods)                                                                                            | `.flp`, `.config`, `.materials`, `.lcf` parsing                     |
| `assembler.py`             | `FVMAssembler`                                                                                                              | Face-to-cell extraction, conduction/advection matrix assembly       |
| `assembler_kernels.py`     | `overlap_area_kernel`, `find_adjacent_pairs_kernel`                                                                         | Numba JIT compute kernels                                           |
| `thermal_solver.py`        | `ThermalSolver`                                                                                                             | Steady-state (direct sparse) or transient (implicit Euler)          |
| `mesher.py`                | `Mesher`                                                                                                                    | Hexahedral mesh generation (native Python, not gmsh)                |
| `fluid_preprocessor.py`    | `FluidPreprocessor`                                                                                                         | Solve pressure, calculate hydro properties before thermal assembly  |
| `metahotspot_solver.py`    | `MetaHotspotSolver`                                                                                                         | End-to-end: preprocess → fluid flow → assemble → solve → export VTU |

---

## File Formats (output)

- **`solver_config_steady.json`** / **`solver_config_transient.json`**: Solver config (JSON, 4-space indent)
  - `simulation_type`: `"steady"` or `"transient"`
  - `mesh_file_path`: path to `mesh.msh` (relative to config dir)
  - `ptrace_file_path`: power trace file (`.ptrace`)
  - `init_temperature_file_path`: used by transient (set to `"init.vtu"` after steady run)
- **`result.vtu`** / **`transient_result.vtu`**: VTK Unstructured Grid with `Temperature_K` cell data
- **`init.vtu`**: Steady result reused as transient initial field
- **`layouts/*_layout.json`**: Per-layer unit definitions (name, lx, ly, dx, dy, k, cp, is_fluid...)

---

## Example Summary

| Example  | Description          | Key feature                                     |
| -------- | -------------------- | ----------------------------------------------- |
| example1 | 2D EV6 processor     | Simple steady + transient                       |
| example2 | 3D heterogeneous     | 4-layer stackup                                 |
| example3 | 3D cache stacking    | 6-layer stackup                                 |
| example4 | Microfluidic cooling | Water fluid + pressure BCs (mc_inlet/mc_outlet) |

## Architecture Principles

1. **Single source of truth**: `model25d.py` owns all defaults—never duplicate fallback logic
2. **No backward compatibility**: Update all call sites when refactoring
3. **Sequential completion**: Each phase complete before next starts
4. **Config-driven**: All params flow through `solver_config.json`
