# MetaHotspot Context

Thermal simulation engine for electronic packaging. Models heat transfer in multi-layer electronic chip assemblies using finite volume method on structured grids.

## Core Concepts

### Study Types

- **Steady**: Static thermal equilibrium (`StudyType>Steady` in XML). Treated as transient at t=0 with no time advancement — one direct nonlinear solve.
- **Transient**: Time-dependent simulation with `TransientStudyDuration` and `TransientStudyTimeStep`. Expressions evaluated with `t` advancing from 0.

### Mesh

- **Structured grid**: Cell-centered DOFs, regular Cartesian mesh. Only supported mesh type.
- **Cell**: Fundamental volume element. Temperature DOF stored at cell center.
- **Face**: Cell surface. Boundary conditions applied here via boundary integrals.
- **Vertex coordinates**: Grid lines defining cell boundaries.

### Boundary Conditions (热边界条件)

- **First-type (Dirichlet)**: Fixed temperature `T = T₀`. Applied via ghost cell method.
- **Second-type (Neumann)**: Fixed heat flux `q = q₀·n`. Enters cell RHS directly.
- **Third-type (Cauchy/Robin)**: Convection `h(T - T_∞)`. Linearized into Jacobian + RHS contributions.

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

### Expressions

- **Geometry expressions**: `w_top/2`, `h_middle`, evaluated at preprocessing to concrete numbers. Context: none (only variables like `w_top`).
- **Field expressions**: Material properties, BC parameters. Context: `{x, y, z, T, t}`.
- User-defined functions (e.g., `test_gaussian`) registered in function pool.
- **Native functions**: C++ functions `double(const FieldContext&)` registered via `register_native()`. Used for piecewise functions and other forms that are easier to express in code than as strings.

### Face Keys

Boundary face specification format: `Face|Direction|LayerIndex|X_min,Y_min,X_max,Y_max;...`
Example: `Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100`

## Solver Pipeline

1. **Preprocessor**: IO model → Internal SoA model (mesh, BC arrays, compiled expressions)
2. **Scheduler**: Outer loop — time stepping + nonlinear Newton iteration
3. **Assembler**: Given model + current state → evaluates A(T)·T = b(T) as linear system
4. **Solver**: Eigen `SparseLU` or `BiCGSTAB` — factory pattern
5. **Postprocessor**: VTU (ParaView) + XML result output

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
