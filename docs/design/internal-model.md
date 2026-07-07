# 内部模型结构（SoA）

扁平 SoA，针对缓存局部性和向量化优化。`src/data/internal_model.hpp`。**所有几何 SI 米**。

## MeshGeometry

```cpp
struct MeshGeometry {
    int nx = 0, ny = 0, nz = 0;

    std::vector<double> dx, dy, dz;                    // sizes nx, ny, nz
    std::vector<double> cx, cy, cz;                    // sizes nx, ny, nz (centers)
};
```

## CellFields（SoA）

```cpp
// Per-cell per-face BC (ADR-0002)
struct CellBC {
    std::array<BcType, FACE_COUNT> types;          // xm, xp, ym, yp, zm, zp
    std::array<uint16_t, FACE_COUNT> param_idxs;   // indices into BCParamTable
};

struct CellFields {

    std::vector<uint32_t> index_map;             // old grid index → compact; invalidIndex = virtual
    std::vector<uint16_t> material_id;           // index into material_table
    std::vector<uint16_t> heat_source_idx;       // index into heat_source_table
    std::vector<CellBC>   cell_bcs;
};
```

### 虚拟单元

结构化网格 `nx × ny × nz` 包含大量无效单元（封装有空洞）。`index_map`（full-grid tier）的 `invalidIndex` 值即标记虚拟单元，矩阵维度 = `N_active`，由 `cell_bcs.size()` 唯一确定（`cell_bcs` 是 compact tier 字段）。

### 热源字典化

`Block.ti_reyuan_expr` 字符串去重后编入 `InternalModel::heat_source_table`：

- `heat_source_idx` 是 compact 字段（与 `material_id` / `cell_bcs` 同索引空间），未匹配到任何 block 的活跃单元填 `0`（`make_constant(0.0)`）
- 重复公式只编译一次，每单元 2 字节索引
- `model.heat_source_table[hs_idx].eval(ctx)` 求值

## BCParamTable

```cpp
struct BCParamTable {
    std::vector<CompiledExpression> dirichlet_T;   // N_dirichlet
    std::vector<CompiledExpression> neumann_q;     // N_neumann
    std::vector<CompiledExpression> cauchy_h;      // N_cauchy
    std::vector<CompiledExpression> cauchy_T_inf;  // N_cauchy
};
```

## AssembleContext

定义在 `src/assembler/assembler.hpp`，是 `Assembler::assemble` 接收的最小数据容器：

```cpp
struct AssembleContext {
    Eigen::Ref<const Eigen::VectorXd> T;
    double current_time = 0.0;
};
```

**Invariants:**

- `T.size() == N_active`（与 `cells.cell_bcs.size()` 一致）。

历史步缓存（`SolutionHistory`）、`dt`、`time_step` 由 `Scheduler::run()` 内部持有，**不**放在 `AssembleContext` 中，也**不**放入 `InternalModel`。

## InternalModel

```cpp
struct InternalModel {
    MeshGeometry       mesh;
    CellFields         cells;
    BCParamTable       bc_params;
    std::vector<MaterialProps> material_table;     // CompiledExpression { kx, ky, kz, rho, c }

    std::vector<CompiledExpression> heat_source_table;  // dedup; idx 0 = constant 0

    double initial_temperature = 300.0;
    StudyType study_type = StudyType::Steady;
    double transient_duration    = 0.0;
    double transient_time_step   = 1.0;
    std::vector<ProbePoint> observation_points;
};
```

## 设计要点

- **虚拟单元**：assembler 跳过 `index_map[idx]==invalidIndex`，postprocessor 用 `index_map` 展开，虚拟位置写 NaN
- **Cell-level BC**：消除投影歧义；虚拟邻居已在 `resolve_face_keys()` 阶段填好 `other_bc`
- **`other_bc` 在预处理阶段填充**，不在装配时
- **Ring buffer (`SolutionHistory`)**：BDF-k 多步法历史缓冲，容量 = `max_order + 1`。`accepted.current() == T` 在每步接受后成立。
- **各向异性 k**：`MaterialProps` 按 X / Y / Z 三轴分字段 `kx / ky / kz`，与装配时面法向 1:1 对应。面法向查表（`k_along` / `half_length_along` / `face_area` / `neighbor_*`）统一定义在 `mhs::utils`（`src/common/mesh_utils.hpp`），由装配器和预处理器共享。
