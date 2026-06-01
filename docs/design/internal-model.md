# 内部模型结构（SoA）

内部结构为扁平 SoA（Structure of Arrays）数组，针对缓存局部性和向量化优化。

---

## 3.1 网格几何

```cpp
namespace mhs::model {

struct MeshGeometry {
    int nx = 0, ny = 0, nz = 0;
    int total_cell_count = 0;  // nx * ny * nz

    std::vector<double> vertex_x;    // nx+1, from XML
    std::vector<double> vertex_y;    // ny+1, from XML
    std::vector<double> vertex_z;    // nz+1, from XML

    std::vector<double> dx;          // nx, computed from vertex_x
    std::vector<double> dy;          // ny, computed from vertex_y
    std::vector<double> dz;          // nz, computed from vertex_z

    std::vector<double> cx;         // nx, cell centers
    std::vector<double> cy;         // ny, cell centers
    std::vector<double> cz;         // nz, cell centers
};

} // namespace mhs::model
```

---

## 3.2 单元场（SoA）

### 3.2.1 Cell-Level BC

边界条件存储在单元级别，每个单元存储 6 个面的 BC 信息，解决面投影重叠问题。

```cpp
namespace mhs::model {

// Face direction indices
enum FaceDir : size_t { XM = 0, XP = 1, YM = 2, YP = 3, ZM = 4, ZP = 5 };

constexpr size_t FACE_COUNT = 6;

constexpr std::array<FaceDir, FACE_COUNT> FACE_DIRS = {
    FaceDir::XM, FaceDir::XP, FaceDir::YM, FaceDir::YP, FaceDir::ZM, FaceDir::ZP};

// Per-cell per-face BC
struct CellBC {
    std::array<BcType, FACE_COUNT> types;           // xm, xp, ym, yp, zm, zp
    std::array<uint16_t, FACE_COUNT> param_idxs;    // indices into BCParamTable
};

struct CellFields {
    int cell_count = 0;  // = N_active (valid cell count)

    // Full-grid size (nx*ny*nz): virtual + active
    std::vector<size_t> index_map;     // Maps old grid index → compact active index. SIZE_MAX = virtual
    std::vector<uint8_t> valid_mask;   // 1 = active cell, 0 = virtual
    std::vector<size_t> material_id;   // Full grid size
    std::vector<size_t> layer_id;      // Full grid size

    // Compact size (N_active): active cells only
    std::vector<CellBC> cell_bcs;
    std::vector<expr::CompiledExpression> heat_source;
};

} // namespace mhs::model
```

### 3.2.2 虚拟单元标记

结构化网格创建 `nx × ny × nz` 个单元，但并非所有单元都在有效几何区域内（电子封装有空洞）。

- `valid_mask`: 全网格大小，`1` = 有效单元，`0` = 虚拟单元
- `index_map`: 全网格大小，映射旧网格索引 → 紧凑活跃索引。`SIZE_MAX` = 虚拟单元
- `cell_count = N_active`: 矩阵维度为活跃单元数量

### 3.2.3 热源

每个单元的体热源 `Q(x,y,z,T,t)` [W/m³]，由 `Block.ti_reyuan_expr` 预编译而来。紧凑存储，大小为 N_active。

---

## 3.3 BC 参数表

```cpp
namespace mhs::model {

struct BCParamTable {
    std::vector<expr::CompiledExpression> dirichlet_T;  // N_dirichlet
    std::vector<expr::CompiledExpression> neumann_q;    // N_neumann
    std::vector<expr::CompiledExpression> cauchy_h;      // N_cauchy
    std::vector<expr::CompiledExpression> cauchy_T_inf;  // N_cauchy
};

} // namespace mhs::model
```

`FaceBCFields` 已移除。BC 参数通过 `CellBC.param_idxs` 索引到对应参数表。

---

## 3.4 全局状态缓冲

```cpp
namespace mhs::model {

struct GlobalState {
    int cell_count = 0;  // = N_active
    double current_time = 0.0;
    int time_step = 0;
    double dt = 0.0;     // current time step size for transient assembly
    
    std::vector<double> T;           // size = N_active
    std::vector<double> T_prev;       // size = N_active
    std::vector<double> residual;     // size = N_active

    std::deque<std::vector<double>> T_history;     // ring buffer
    std::deque<std::vector<double>> nl_history;    // ring buffer
    std::deque<double> dt_history;                 // ring buffer
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
    BCParamTable bc_params;

    std::vector<MaterialProps> material_table;

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

### Virtual Cell Handling

1. `LayerProcessor::resolve()` 生成 `valid_mask` + `index_map`
2. `GlobalState.cell_count = N_active`，T 向量紧凑存储
3. Assembler 跳过虚拟单元（`if (!valid_mask[idx]) continue;`）
4. Postprocessor 使用 `valid_mask` 展开 T 向量，虚拟区域填充 NaN

### Cell-Level BC Benefits

- **消除投影歧义**：每个单元的每个面独立 BC，不受面投影重叠影响
- **虚拟单元邻居**：预处理阶段处理，临面虚拟的有效单元自动获得 `other_bc`
- **一致性**：与虚拟单元标记机制一致

### BCParamTable

共享的 BC 参数（如多个面使用相同的 Dirichlet 温度）存储在 BCParamTable 中，通过 `param_idx` 索引。

`other_bc` 在预处理阶段填充，未显式指定的面自动设置为 default BC。
