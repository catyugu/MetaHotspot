# 内部模型结构（SoA）

内部结构为扁平 SoA（Structure of Arrays）数组，针对缓存局部性和向量化优化。

---

## 3.1 网格几何

```cpp
namespace mhs::model {

struct MeshGeometry {
    int nx = 0, ny = 0, nz = 0;      // 每个方向的单元数
    int cell_count = 0;              // nx * ny * nz

    // 顶点坐标（每个轴比单元数多一个）
    std::vector<double> vertex_x;    // 大小 nx+1
    std::vector<double> vertex_y;    // 大小 ny+1
    std::vector<double> vertex_z;    // 大小 nz+1

    // 单元尺寸（用于热通量计算）
    std::vector<double> dx;          // 大小 nx，相邻 x 顶点间距
    std::vector<double> dy;          // 大小 ny
    std::vector<double> dz;          // 大小 nz

    // 单元中心坐标（用于 BC 表达式求值）
    std::vector<double> cx;          // 大小 nx
    std::vector<double> cy;          // 大小 ny
    std::vector<double> cz;          // 大小 nz
};

} // namespace mhs::model
```

---

## 3.2 单元场（SoA）

```cpp
namespace mhs::model {

// 材料属性槽 — 预处理阶段全部预编译为 CompiledExpression。
// 若为常数表达式（is_constant=true），直接用 constant_value，无求值开销。
struct MaterialProps {
    expr::CompiledExpression k;   // 导热系数 k(x,y,z,T,t)
    expr::CompiledExpression rho; // 密度 rho(x,y,z,T,t)
    expr::CompiledExpression c;   // 比热容 c(x,y,z,T,t)
};

struct CellFields {
    int cell_count = 0;

    std::vector<size_t> material_id;   // 大小 cell_count
    std::vector<size_t> layer_id;         // 大小 cell_count

    // 每个单元的体热源 Q(x,y,z,T,t) [W/m³]。
    // 由 Block.ti_reyuan_expr 预编译而来。
    // 常数热源也存为 CompiledExpression（通过 make_constant()）。
    std::vector<expr::CompiledExpression> heat_source;  // 大小 cell_count

    // BC 应用标志（位掩码，标记哪些面已施加 BC）
    std::vector<uint8_t> bc_flags;         // 大小 cell_count
};

} // namespace mhs::model
```

---

## 3.3 面 BC 数组（SoA）

```cpp
namespace mhs::model {

enum class BcType : uint8_t { None = 0, FirstType = 1, SecondType = 2, ThirdType = 3 };

// BC 参数表 — 预处理阶段全部预编译为 CompiledExpression。
// 每个条目是一个函数：eval(ctx) -> value。
// bc_type 决定使用哪个参数向量（如 FirstType 使用 dirichlet_T）。
struct BCParamTable {
    std::vector<expr::CompiledExpression> dirichlet_T;          // 大小 N_dirichlet
    std::vector<expr::CompiledExpression> neumann_q;           // 大小 N_neumann
    std::vector<expr::CompiledExpression> cauchy_h;            // 大小 N_cauchy
    std::vector<expr::CompiledExpression> cauchy_T_inf;         // 大小 N_cauchy
};

struct FaceBCFields {
    // 6 个面，每个面大小为 N_xy 或 N_xz 或 N_yz。
    // bc_type[i] 决定 BC 类型；bc_param_idx[i] 索引到 BCParamTable 的对应向量中。
    //
    // Z- 面：大小 nx * ny
    std::vector<BcType> bc_type_zm;
    std::vector<uint16_t> bc_param_idx_zm;   // 索引到 dirichlet_T / neumann_q / cauchy_*
    // Z+ 面：大小 nx * ny
    std::vector<BcType> bc_type_zp;
    std::vector<uint16_t> bc_param_idx_zp;
    // Y- 面：大小 nx * nz
    std::vector<BcType> bc_type_ym;
    std::vector<uint16_t> bc_param_idx_ym;
    // Y+ 面：大小 nx * nz
    std::vector<BcType> bc_type_yp;
    std::vector<uint16_t> bc_param_idx_yp;
    // X- 面：大小 ny * nz
    std::vector<BcType> bc_type_xm;
    std::vector<uint16_t> bc_param_idx_xm;
    // X+ 面：大小 ny * nz
    std::vector<BcType> bc_type_xp;
    std::vector<uint16_t> bc_param_idx_xp;
};

} // namespace mhs::model
```

---

## 3.4 全局状态缓冲

```cpp
namespace mhs::model {

struct GlobalState {
    int cell_count = 0;
    double current_time = 0.0;   // t=0（稳态）或当前时间（瞬态）
    int time_step = 0;

    // 主解向量
    std::vector<double> T;           // 温度，大小 cell_count

    // 瞬态：前一时间步温度（用于时间导数离散）
    std::vector<double> T_prev;       // 大小 cell_count

    // Newton 迭代：残差向量
    std::vector<double> residual;    // 大小 cell_count
};

} // namespace mhs::model
```

---

## 3.5 完整内部模型

```cpp
namespace mhs::model {

struct InternalModel {
    MeshGeometry mesh;

    CellFields cells;

    FaceBCFields face_bcs;
    BCParamTable bc_params;

    // 材料属性表（按 MaterialID 索引）
    // MaterialID enum 值 -> MaterialProps（每个属性均为预编译 CompiledExpression）
    std::array<MaterialProps, 256> material_table;

    // 仿真元数据
    double initial_temperature = 300.0;
    double ambient_temperature = 300.0;
    StudyType study_type = StudyType::Steady;
    double transient_duration = 0.0;
    double transient_time_step = 1.0;
};

} // namespace mhs::model
```

---

## 设计要点

### Assembler Interface

- **Input**: `InternalModel` + `GlobalState` + `t` + `dt`
- **Output**: `LinearSystem` (sparse A, RHS b, residual)
- **GlobalState contains**: T, T_prev, T_history ring buffer, nl_history ring buffer, dt_history ring buffer
- **Crank-Nicolson**: θ = 0.5, lumped mass matrix
- **Convergence status**: `GlobalState::status` (Running/Converged/Diverged/Stalled)

### GlobalState Ring Buffers

```cpp
struct GlobalState {
    int cell_count = 0;
    double current_time = 0.0;
    int time_step = 0;
    ConvergenceStatus status = ConvergenceStatus::Running;

    std::vector<double> T;               // current temperature
    std::vector<double> T_prev;          // previous time step
    std::vector<double> residual;        // current residual

    // Ring buffers (std::deque, configurable capacity, default 5)
    std::deque<std::vector<double>> T_history;     // past time steps
    std::deque<std::vector<double>> nl_history;    // non-linear iteration snapshots
    std::deque<double> dt_history;                  // past Δt values
};
```

### DOF & BC Application

- **Cell-centered DOF**: Temperature stored at cell center
- **Dirichlet BC**: Ghost cell method — boundary outside one ghost cell, `T_ghost = 2·T_dirichlet - T_boundary`
- **Neumann BC**: Heat flux enters cell RHS directly (area integral)
- **Cauchy BC**: Convection linearized — Jacobian diagonal adds `h·A`, RHS adds `h·A·T_∞`
- **Default BC**: Configured in XML, not interior default

### SoA 布局优势

- 按字段顺序连续读取，优化缓存命中
- SIMD 向量化容易（连续内存操作）
- TBB 并行化简单：每个线程操作一段连续的单元索引

### 预处理阶段已完成的工作

所有表达式编译为 `CompiledExpression`：

- 材料属性（`k`、`ρ`、`c`）→ `MaterialProps`
- BC 参数（`T_dirichlet`、`q_neumann`、`h_cauchy`、`T_inf_cauchy`）→ `BCParamTable`
- 热源（每个单元）→ `CellFields.heat_source`
- 用户自定义函数（exprtk + native）→ expr 模块函数池
