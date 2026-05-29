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
- Each layer has a `ThicknessExpression` and mesh size hints.
- **Block heat source**: Each block has one `ti_reyuan_expr` (体热源, [W/m³]). Preprocessor expands this to a per-cell `heat_source` array indexed by `cell_idx`.

### Expressions

- **Geometry expressions**: `w_top/2`, `h_middle` — evaluated via `expr::eval_geometry()`. Context: none (variables pre-registered).
- **Field expressions**: Material properties, BC parameters. Context: `{x, y, z, T, t}`. Pre-compiled to `FieldExpression`.
- **Expr registry**: Global, thread-safe. Populated by `ModelBuilder` from `IOStructure` variables/functions.
- **Native functions**: C++ functions registered via `expr::register_native()`. Used for piecewise functions and other forms easier to express in code than strings.

### Face Keys

Boundary face specification format: `Face|Direction|LayerIndex|X_min,Y_min,X_max,Y_max;...`
Example: `Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100`

## Solver Pipeline

1. **Preprocessor**: IO model → Internal SoA model (mesh, BC arrays, compiled expressions)
   - **IO model** (`io_model.hpp`): AoS structs mirroring XML schema. Uses `ThermalBCType` (FirstType, SecondType, ThirdType) matching XML element names. Length unit (`LengthUnit`: M, Mm, Um, Nm, Inch, Mil) converted to SI (meters) here.
   - **Internal model** (`internal_model.hpp`): All geometry in SI units (meters), no unit storage.
   - **Internal model** (`internal_model.hpp`): Flat SoA arrays. Uses `BcType` (None, FirstType, SecondType, ThirdType) — the `None` variant marks faces with no BC. Conversion happens once at preprocessing.
   - **IO function converters**: `ExpressionFunction`, `GaussFunction`, `SineFunction`, `PieceWiseFunction` 等需经由 `FunctionConverter` 转换为 `FieldEvaluator`，再包装为 `CompiledExpression`。
2. **Scheduler**: Outer loop — time stepping + nonlinear Newton iteration
3. **Assembler**: Given model + current state → evaluates A(T)·T = b(T) as linear system
4. **Solver**: Eigen `SparseLU` or `BiCGSTAB` — factory pattern
5. **Postprocessor**: Pure computation — cell-to-node interpolation, max/min temperature. No file I/O.
6. **io module**: `read_xml(xml_path)` reads XML; `write_vtu(path, model, node_temperature)` writes VTU; `write_xml(output_path, input_path, model, node_temperature)` copies and updates XML.

## GlobalState

Persistent state across simulation, stored in `model::GlobalState`:

- **Core fields**: `T` (current temperature), `T_prev` (previous time step), `residual`
- **Ring buffers**: `T_history` (past time steps), `nl_history` (non-linear snapshots), `dt_history`
- **Ring buffer capacity**: Configurable, default 5
- **Convergence status**: `Running`, `Converged`, `Diverged`

## Key Design Principles

1. **No raw strings in internal model** — all expressions compiled to `FieldExpression`
2. **Expr registry is internal** — `ModelBuilder` populates, external code uses clean API
3. **Thread-safe expr module** — `parse()`/`register_*()` mutex-protected, `eval()` lock-free
4. **Precomputed sparsity pattern** — assemble only fills values, does not rebuild structure
5. **Crank-Nicolson (θ=0.5)** — transient time discretization with lumped mass
6. **TBB parallel assembly** — `tbb::parallel_for` over cells
7. **Single source of truth for internal types** — `types.hpp` defines all internal enums (`StudyType`, `BcType`, `ConvergenceStatus`); `io_model.hpp` includes it instead of redeclaring

## Glossary

| Term        | Chinese  | Notes                             |
| ----------- | -------- | --------------------------------- |
| Structure   | 结构体   | Top-level XML element             |
| Layer       | 层       | Stack of material blocks          |
| Block       | 块       | Geometry defined by add/sub rects |
| Rect        | 矩形     | Add or subtract operation         |
| Boundary    | 边界     | Face BC specification             |
| Face key    | 面键     | String encoding boundary face     |
| Material    | 材料     | copper, silicon, TIM              |
| Variable    | 变量     | Geometry parameter (w_top, etc.)  |
| Function    | 函数     | User-defined expression function  |
| DAORE XISHU | 导热系数 | Thermal conductivity              |
| MIDU        | 密度     | Density                           |
| BI RERONG   | 比热容   | Specific heat                     |
