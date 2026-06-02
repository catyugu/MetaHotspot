# MetaHotspot Context

Thermal simulation engine for electronic packaging. Models heat transfer in multi-layer electronic chip assemblies using finite volume method on structured grids.

## Core Concepts

### Study Types

- **Steady**: Static thermal equilibrium (`StudyType>Steady` in XML). Treated as transient at t=0 with no time advancement — one direct nonlinear solve.
- **Transient**: Time-dependent simulation with `TransientStudyDuration` and `TransientStudyTimeStep`. Expressions evaluated with `t` advancing from 0.

### Mesh

- **Structured grid**: Cell-centered DOFs, regular Cartesian mesh. Only supported mesh type. 2D (`Dimension2D`) is explicitly unsupported — the mesh always uses 3D vertex arrays.
- **Cell**: Fundamental volume element. Temperature DOF stored at cell center.
- **Face**: Cell surface. Boundary conditions applied here via boundary integrals.
- **Vertex coordinates**: Grid lines defining cell boundaries.

### Boundary Conditions (热边界条件)

- **First-type (Dirichlet)**: Fixed temperature `T = T₀`. Applied via ghost cell method.
- **Second-type (Neumann)**: Fixed heat flux `q = q₀·n`. Enters cell RHS directly.
- **Third-type (Cauchy/Robin)**: Convection `h(T - T_∞)`. Linearized into Jacobian + RHS contributions.
- **Other BC (`other_bc`)**: Default BC applied to faces not covered by any face key. Specified at IO level via `other_bc_type` + `other_bc_first/second/third`; preprocessor applies it to all `BcType::None` faces during BC array initialization.

### Material Properties

- **DaoreXishu (导热系数)**: Thermal conductivity `k` [W/(m·K)].
- **Midu (密度)**: Density `ρ` [kg/m³].
- **BiRerong (比热容)**: Specific heat `c` [J/(kg·K)].
- Properties can be constant or functions of `{x, y, z, T, t}`.

### Layers

- **Top layer**: Die-attach and chip materials (copper, etc.).
- **Middle layer**: Substrate (silicon).
- **Bottom layer**: PCB or heat spreader.
- Each layer has a `ThicknessExpression` and mesh size hints. The layer thickness is the **only** Z-axis dimension — blocks inherit the full Z extent of their parent layer and have no independent Z thickness or offset.
- **Block geometry**: Blocks define shape only in the XY plane via add/sub `Rect` operations. A block's Z range is always `[layer.z_start, layer.z_end]`.
- **Block heat source**: Each block has one `ti_reyuan_expr` (体热源, [W/m³]). Preprocessor expands this to a per-cell `heat_source` array indexed by `cell_idx`.

### Expressions

- **Geometry expressions**: `w_top/2`, `h_middle` — evaluated via `expr::eval_geometry()`. Context: none (variables pre-registered).
- **Field expressions**: Material properties, BC parameters. Context: `{x, y, z, T, t}`. Pre-compiled to `CompiledExpression`.
- **Expr registry**: Global, thread-safe. `Preprocessor::load()` calls `clear_registry()` then populates from `IOStructure` variables/functions.
- **Native functions**: C++ functions registered via `expr::register_native()`. Used for piecewise functions and other forms easier to express in code than strings.
- **Expr eval concurrency**: Cached ExprTK expressions share a singleton `ExprTKCompiled` per formula string. `eval()` uses a mutex to protect shared x_/y_/z_/T_/t_ members, which serializes evaluation. Constant expressions (`make_constant`) are lock-free.

### Face Keys

Boundary face specification format: `Face|Direction|CoordValue|X_min,Y_min,X_max,Y_max;...`
Example: `Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100`

- `Face`: Z/Y/X — axis perpendicular to the boundary plane
- `Direction`: E — boundary category (Electrical, currently the only category)
- `CoordValue`: spatial coordinate of the boundary plane (e.g., `0` = Z=0mm, `30` = Z=30mm), multiplied by si_scale internally. **Not a layer index** — boundary selection is purely coordinate-based, independent of layer ordering.
- The trailing coordinates describe one or more rectangular regions in the 2D projection of the boundary face

## Solver Pipeline

- **Preprocessor**: `Preprocessor::load(IOStructure)` → `unique_ptr<InternalModel>`. Stateless class; calls free functions in `mhs::preprocessor` namespace. Not a `ModelBuilder` class.
    - **IO model** (`io_model.hpp`): AoS structs mirroring XML schema. Uses `ThermalBCType` (FirstType, SecondType, ThirdType) matching XML element names. Length unit (`LengthUnit`: M, Mm, Um, Nm, Inch, Mil) converted to SI (meters) here.
    - **Internal model** (`internal_model.hpp`): All geometry in SI units (meters), no unit storage.
    - **Internal model** (`internal_model.hpp`): Flat SoA arrays. Uses `BcType` (None, FirstType, SecondType, ThirdType) — the `None` variant marks faces with no BC. Conversion happens once at preprocessing.
    - **IO function converters**: `ExpressionFunction`, `GaussFunction`, `SineFunction`, `PieceWiseFunction`, `DoubleExponentialFunction` are planned but not yet defined — even in headers. Current `IOStructure.functions` is a flat `unordered_map<string, FieldEvaluator>` using native function registration.

2. **Scheduler**: Outer loop — time stepping + nonlinear Anderson accelerated iteration (namespace `mhs`)
3. **Assembler**: Given model + current state → evaluates A(T)·T = b(T) as linear system
4. **Solver**: Eigen `SparseLU` or `BiCGSTAB` — virtual factory pattern (namespace `mhs`)
5. **Postprocessor**: Pure computation — cell-to-node interpolation, max/min temperature. No file I/O. (namespace `mhs`)
6. **io module**: `read_xml(xml_path)` reads XML; `write_vtu(path, model, node_temperature)` writes VTU; `write_xml(input_path, output_path, model, node_temperature)` copies and updates XML. Free functions, not class.

## GlobalState

Persistent state across simulation, stored in `model::GlobalState`:

- **Core fields**: `T` (current temperature), `T_prev` (previous time step), `residual`
- **Time stepping**: `current_time`, `time_step` (step counter), `dt` (current step size)

## Key Design Principles

1. **No raw strings in internal model** — all expressions compiled to `CompiledExpression`
2. **Expr registry is global** — `Preprocessor::load()` calls `clear_registry()` then populates; external code uses `parse()`/`eval()`
3. **Thread-safe expr module** — `parse()`/`register_*()` mutex-protected; `eval()` is lock-free — each `CompiledExpression` owns its own `ExprTKCompiled` instance, no shared mutable state. Constant expressions (`make_constant`) are also lock-free.
4. **Precomputed sparsity pattern** — assemble only fills values, does not rebuild structure
5. **Backward Euler** — transient time discretization (θ=1.0), the code uses `ρ*c*vol/dt * (T - T_prev)` mass term
6. **TBB parallel assembly** — `tbb::parallel_for` over cells. `CompiledExpression::eval()` is lock-free (per-instance ownership), so no serialization bottleneck.
7. **Single source of truth for internal types** — `types.hpp` defines all internal enums (`StudyType`, `BcType`) and core types (`FieldContext`, `FaceDir`, `FieldEvaluator`); `io_model.hpp` includes it instead of redeclaring

## Glossary

| Term              | Chinese  | Notes                                                                                                       |
| ----------------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| Structure         | 结构体   | Top-level XML element                                                                                       |
| Layer             | 层       | Stack of material blocks                                                                                    |
| Block             | 块       | Geometry defined by add/sub rects — contains material and heat source                                       |
| CellBoundaryGroup | 面边界组 | BC definition group, one per Block. Each group has 6 faces (xm/xp/ym/yp/zm/zp) with independent BC settings |
| Rect              | 矩形     | Add or subtract operation                                                                                   |
| Boundary          | 边界     | Face BC specification                                                                                       |
| Face key          | 面键     | String encoding boundary face                                                                               |
| Material          | 材料     | copper, silicon, TIM                                                                                        |
| Variable          | 变量     | Geometry parameter (w_top, etc.)                                                                            |
| Function          | 函数     | User-defined expression function                                                                            |
| DAORE XISHU       | 导热系数 | Thermal conductivity                                                                                        |
| MIDU              | 密度     | Density                                                                                                     |
| BI RERONG         | 比热容   | Specific heat                                                                                               |

## Virtual Cell & Mesh Mask

Structured grid creates `nx × ny × nz` cells, but not all cells are within valid geometry (electronic package has voids).

| Concept      | Description                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------ |
| valid_mask   | `std::vector<uint8_t>` (size = nx*ny*nz). `1` = active cell, `0` = virtual                                   |
| index_map    | `std::vector<size_t>` (size = nx*ny*nz). Maps old grid index → compact active index. SIZE_MAX = virtual cell |
| active_count | Number of valid cells (N_active). Matrix dimension = active_count                                            |

**CellFields layout:**

- **Full-grid size** (nx*ny*nz): `index_map`, `valid_mask`, `material_id`, `layer_id`
- **Compact size** (N_active): `cell_bcs`, `heat_source`

Keeping full-grid arrays for material/layer IDs simplifies debugging and maintains consistent array style across the model.

## Cell-Level BC

BC is stored at cell level (not face-array level) to handle overlapping projections between blocks in the same layer.

```cpp
struct CellBC {
    std::array<BcType, 6> types;           // xm, xp, ym, yp, zm, zp
    std::array<uint16_t, 6> param_idxs;   // indices into BCParamTable
};

struct CellFields {
    int cell_count = 0;  // = N_active

    // Full-grid size (nx*ny*nz): virtual + active
    std::vector<size_t> index_map;
    std::vector<uint8_t> valid_mask;
    std::vector<size_t> material_id;
    std::vector<size_t> layer_id;

    // Compact size (N_active): active cells only
    std::vector<CellBC> cell_bcs;
    std::vector<CompiledExpression> heat_source;
};
```

- Face projection overlap between blocks is resolved — each cell's face has independent BC
- `other_bc` is applied during preprocessing for faces not explicitly specified
- Virtual cell neighbors are also handled in preprocessing (neighboring active cells get other_bc on that face)

**FaceBCFields removed**: Replaced by cell-level `CellBC`. Each cell stores its 6 face BCs independently, eliminating face projection ambiguity.
