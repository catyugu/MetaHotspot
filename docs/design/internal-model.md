# 内部模型结构（SoA）

扁平 SoA，针对缓存局部性和向量化优化。`src/engine/runtime_model.hpp`。**所有几何 SI 米**。

## MeshGeometry

```cpp
struct MeshGeometry {
    mhs::Index nx = 0, ny = 0, nz = 0;

    std::vector<double> dx, dy, dz;                    // sizes nx, ny, nz
    std::vector<double> cx, cy, cz;                    // sizes nx, ny, nz (centers)
};
```

## CellFields（SoA）

```cpp
// Per-cell per-face BC (ADR-0002) — stored as flat array on Model
struct FaceBC {
    BcType type = BcType::None;  // None = internal face or adiabatic
    uint16_t param_idx = 0;      // → BCParamTable
};

struct CellFields {
    std::vector<Index> grid_to_cell;         // grid index → active cell; invalidIndex = virtual
    std::vector<Index> cell_to_grid;         // active cell → grid index
    std::vector<uint16_t> material_id;       // index into material_table
    std::vector<uint16_t> heat_source_idx;   // index into heat_source_table
};
```

### 虚拟单元

结构化网格 `nx × ny × nz` 包含大量无效单元（封装有空洞）。`grid_to_cell`（full-grid tier）的 `invalidIndex` 值即标记虚拟单元；`cell_to_grid` 是活跃单元到结构化网格的精确逆映射。矩阵维度 = `N_active`，由 `cells.material_id.size()` 唯一确定。基础装配直接遍历 `cell_to_grid`，不再扫描并跳过整个结构化网格。

`face_bcs` 存储在 `Model` 中，为 `[N_active * 6]` 扁平数组：`face_bcs[c * 6 + dir]` 给出单元 c 方向 dir 的 BC。无 `CellBC` 结构体。

### 热源索引表

每个 Block 的 `volumetric_heat_source` 编入 `Model::heat_source_table`：

- `heat_source_idx` 是 compact 字段（与 `material_id` 同索引空间），每个活跃单元引用其最终覆盖 Block 的表达式
- `model.heat_source_table[hs_idx].eval(ctx)` 求值

## MaterialProps

```cpp
struct MaterialProps {
    CompiledExpression kx, ky, kz;      // thermal conductivity [W/(m·K)]
    CompiledExpression rho;              // density [kg/m³]
    CompiledExpression c;                // specific heat [J/(kg·K)]
};
```

流体标记与初始粘度只参与 `fluid::build_domain()`，不进入运行时材料表。

## BCParamTable

```cpp
struct BCParamTable {
    std::vector<CompiledExpression> dirichlet_T;   // N_dirichlet
    std::vector<CompiledExpression> neumann_q;     // N_neumann
    std::vector<CompiledExpression> cauchy_h;      // N_cauchy
    std::vector<CompiledExpression> cauchy_T_inf;  // N_cauchy
};
```

## Model

```cpp
struct Model {
    MeshGeometry       mesh;
    CellFields         cells;

    // Face-level BC storage: flat array [N_active * 6].
    // face_bcs[c * 6 + dir] gives the BC for cell c's face `dir`.
    std::vector<FaceBC> face_bcs;
    BCParamTable       bc_params;

    std::vector<MaterialProps> material_table;

    std::vector<CompiledExpression> heat_source_table;  // 每个 Block 一项

    double initial_temperature = 300.0;
    StudyType study_type = StudyType::Steady;
    double transient_duration    = 0.0;
    double transient_time_step   = 1.0; // output interval
    std::vector<ProbePoint> observation_points;

    // Fluid-solid coupled heat-transfer subsystem
    FluidDomain fluid;
};
```

## FluidDomain

```cpp
struct FluidDomain {
    std::vector<Index> fluid_to_global;   // fluid index → active cell
    std::vector<Index> global_to_fluid;   // active cell → fluid index / invalidIndex
    std::vector<double> face_volume_flux; // [N_fluid * 6], frozen signed flux
    std::vector<double> interface_heat_transfer_factor; // Nu / D_h
    std::vector<double> boundary_outflux; // NaN = use pressure-derived net flux
    std::vector<double> boundary_temperature;
};
```

压力、粘度、水力导通系数、通道尺寸和流体 BC 值仅存在于
`fluid_preprocessor.cpp` 的局部 `FluidPreprocessWorkspace`。压力求解结束后，
只把热组装需要的冻结面流量和换热因子写入 `FluidDomain`。

## 设计要点

- **双向拓扑**：assembler 通过 `cell_to_grid` 只遍历活跃单元；邻居查询和 postprocessor 通过 `grid_to_cell` 识别虚拟位置
- **面级 BC**：`face_bcs` 为 `[N_active * 6]` 扁平数组，消除了 `CellBC` 结构体。`face_bcs[c*6 + dir]` 直接索引。虚拟邻居已在 `resolve_boundary_patches()` 阶段填好默认边界
- **默认边界在预处理阶段填充**，不在装配时；显式边界按顺序覆盖
- **Ring buffer (`SolutionHistory`)**：容量由构造函数显式指定；当前调度器使用容量 2。`accepted.current() == T` 在每步接受后成立。
- **各向异性 k**：`MaterialProps` 按 X / Y / Z 三轴分字段 `kx / ky / kz`，与装配时面法向 1:1 对应。面法向助手统一定义在 `src/engine/mesh.hpp`。
- **流体域**：`FluidDomain` 只包含热组装所需的冻结面流量、流固换热因子和边界出流/温度；水力预处理状态不进入 `Model`
