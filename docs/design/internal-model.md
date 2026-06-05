# 内部模型结构（SoA）

扁平 SoA，针对缓存局部性和向量化优化。`src/common/internal_model.hpp`。**所有几何 SI 米**。

## MeshGeometry

```cpp
struct MeshGeometry {
    int nx = 0, ny = 0, nz = 0;
    int total_cell_count = 0;            // nx * ny * nz

    std::vector<double> vertex_x, vertex_y, vertex_z;   // sizes nx+1, ny+1, nz+1
    std::vector<double> dx, dy, dz;                    // sizes nx, ny, nz
    std::vector<double> cx, cy, cz;                    // sizes nx, ny, nz (centers)
};
```

## CellFields（SoA）

```cpp
// Per-cell per-face BC (ADR-0005)
struct CellBC {
    std::array<BcType, FACE_COUNT> types;          // xm, xp, ym, yp, zm, zp
    std::array<uint16_t, FACE_COUNT> param_idxs;   // indices into BCParamTable
};

struct CellFields {
    int cell_count = 0;                            // = N_active

    // Full-grid (nx*ny*nz): virtual + active
    std::vector<size_t>  index_map;                // grid → compact; SIZE_MAX = virtual
    std::vector<uint8_t> valid_mask;               // 1 = active, 0 = virtual
    std::vector<size_t>  material_id, layer_id;

    // Compact (N_active): active only
    std::vector<CellBC>        cell_bcs;
    std::vector<uint16_t>      heat_source_idx;    // index into heat_source_table
};
```

### 虚拟单元

结构化网格 `nx × ny × nz` 包含大量无效单元（封装有空洞）。`valid_mask` + `index_map` 标记与映射；矩阵维度 = `N_active`。

### 热源字典化

`Block.ti_reyuan_expr` 字符串去重后编入 `InternalModel::heat_source_table`：

- 索引 `0` 保留为 `make_constant(0.0)`，未匹配到任何 block 的虚拟相邻单元指向 0
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

## GlobalState

```cpp
struct GlobalState {
    int    cell_count    = 0;     // = N_active
    double current_time  = 0.0;
    int    time_step     = 0;
    double dt            = 0.0;   // current transient step size

    std::vector<double> T;
    std::vector<double> T_prev;
    std::vector<double> residual;
};
```

## InternalModel

```cpp
struct InternalModel {
    MeshGeometry       mesh;
    CellFields         cells;
    BCParamTable       bc_params;
    std::vector<MaterialProps> material_table;     // CompiledExpression { kx, ky, kz, rho, c }

    std::vector<CompiledExpression> heat_source_table;  // dedup; idx 0 = constant 0

    double initial_temperature = 300.0;
    double ambient_temperature = 300.0;
    StudyType study_type = StudyType::Steady;
    double transient_duration    = 0.0;
    double transient_time_step   = 1.0;
};
```

## 设计要点

- **虚拟单元**：assembler 跳过 `valid_mask[idx]==0`，postprocessor 用 `index_map` 展开，虚拟位置写 NaN
- **Cell-level BC**：消除投影歧义；虚拟邻居已在 `resolve_face_keys()` 阶段填好 `other_bc`
- **`other_bc` 在预处理阶段填充**，不在装配时
- **无 ring buffer**：当前仅支持 Backward Euler；多步法（`T_history` / `dt_history`）尚未实现
- **各向异性 k**：`MaterialProps` 按 X / Y / Z 三轴分字段 `kx / ky / kz`，与装配时面法向 1:1 对应。面法向查表（`k_along` / `half_length_along` / `face_area` / `neighbor_*`）统一定义在 `mhs::core`（`src/common/face_dir_tables.hpp`），由装配器和预处理器共享。
