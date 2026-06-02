# 模块接口

## 4.1 `io`

```cpp
namespace mhs::io {

model::IOStructure read_xml(const std::string& xml_path);

void write_vtu(const std::string& path,
    const model::InternalModel& model,
    const std::vector<double>& node_temperature);

void write_xml(const std::string& input_path,
    const std::string& output_path,
    const model::InternalModel& model,
    const std::vector<double>& node_temperature);

} // namespace mhs::io
```

---

## 4.2 `preprocessor`

```cpp
namespace mhs {

class Preprocessor {
public:
    Preprocessor() = default;
    ~Preprocessor() = default;

    std::unique_ptr<model::InternalModel> load(const model::IOStructure& ioStructure);
};

} // namespace mhs

namespace mhs::preprocessor {

// Convert length unit to SI (meters) scale factor
double length_unit_to_si(model::LengthUnit unit);

// Pre-evaluate all geometry expressions for all layers, including Z ranges
// Returns resolved geometry per layer: blocks with pre-evaluated rect coordinates
// and layer z_start/z_end
std::vector<ResolvedLayerGeometry> resolve_geometry(
    const std::vector<model::Layer>& layers,
    double si_scale);

// Determine which block a cell at (cx, cy, cz) belongs to in a resolved layer
// Uses pre-evaluated geometry values — no expression evaluation at runtime
// Traverses blocks in reverse order (last block wins in overlap regions)
// Returns block index or -1 if cell is virtual
int find_block_for_cell(const ResolvedLayerGeometry& resolved_layer,
    double cx, double cy, double cz);

// Resolve cell validity, layer assignment, and material assignment
// Populates valid_mask, index_map, layer_id, material_id in CellFields
void resolve_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers,
    const model::MeshGeometry& mesh,
    const std::unordered_map<std::string, size_t>& name_to_idx,
    model::CellFields& cells);

} // namespace mhs::preprocessor

namespace mhs::preprocessor {

struct FaceKeyInfo {
    char axis = 'Z';
    char side = 'E';
    double coord_value = 0.0;  // spatial coordinate of the boundary plane (SI meters)
    std::vector<std::array<double, 4>> rects; // {a_min, a_max, b_min, b_max} in SI
};

FaceKeyInfo parse_face_key(const std::string& key, double si_scale);
bool point_in_face_rects(const FaceKeyInfo& fk, double a, double b);

void resolve_face_keys(const std::vector<model::Boundary>& boundaries,
    model::ThermalBCType other_bc_type,
    const model::FirstTypeThermalBC& other_bc_first,
    const model::SecondTypeThermalBC& other_bc_second,
    const model::ThirdTypeThermalBC& other_bc_third,
    const model::MeshGeometry& mesh,
    model::CellFields& cells,
    model::BCParamTable& bc_params,
    double si_scale);

} // namespace mhs::preprocessor
```

### 预处理流程

```text
IOStructure（含字符串表达式 + mesh_vertex_x/y/z from XML）
  └─> Preprocessor::load()
        ├─> 转换单位（length_unit → SI），注册几何变量 → expr
        ├─> 注册材料函数、用户函数 → expr
        ├─> expr::clear_registry() （清除上次残留）
        ├─> Build MeshGeometry from mesh_vertex_x/y/z (×si_scale)
        │     └─> compute dx/dy/dz, cx/cy/cz
        ├─> resolve_geometry() — 预求解层几何（Z 范围 + Block XY 坐标）
        ├─> Build material_table (parse k/rho/c)
        ├─> resolve_layers()
        │     ├─> 生成 valid_mask + index_map
        │     └─> 分配 material_id, layer_id（全网格大小）
        ├─> Compile heat_source (find_block_for_cell per active cell)
        ├─> resolve_face_keys()
        │     ├─> 解析 face_key 字符串
        │     ├─> ThermalBCType → BcType conversion
        │     ├─> 为每个单元的每个面分配 CellBC
        │     ├─> 填充未指定面的 other_bc
        │     └─> 虚拟单元邻面也分配 other_bc
        └─> InternalModel ready
```

---

## 4.3 `assembler`

```cpp
namespace mhs::assembler {

struct LinearSystem {
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    Eigen::VectorXd residual;
};

class Assembler {
public:
    explicit Assembler(const model::InternalModel& model);
    ~Assembler() = default;

    LinearSystem assemble(const model::GlobalState& state);

private:
    const model::InternalModel& model_;
};

} // namespace mhs::assembler
```

### 组装热循环说明

对每个活跃单元 `cell_idx`：

1. 获取 `material_id[cell_idx]` → `material_table[id]` → `MaterialProps {k, rho, c}`
2. 对每个临面 `FaceDir`：
   - 查 `cell_bcs[compact_idx].types[dir]` → BC 类型
   - 查 `cell_bcs[compact_idx].param_idxs[dir]` → 参数索引
   - 调用 `bc_params.*[idx].eval(ctx)` 等
   - 施加 BC 贡献
3. 获取 `heat_source[compact_idx].eval(ctx)` → 体热源 Q
4. 组装扩散项 + 瞬态项

**注意**：虚拟单元已在预处理阶段标记，Assembler 只处理活跃单元。

---

## 4.4 `solver`

```cpp
namespace mhs {

enum class SolverType { SparseLU, BiCGSTAB };

struct SolverConfig {
    SolverType type = SolverType::BiCGSTAB;
    double tolerance = 1e-8;
    int max_iterations = 1000;
};

struct SolveResult {
    Eigen::VectorXd solution;
    bool success;
    double residual_norm;
    int iterations;
};

class Solver {
public:
    virtual ~Solver() = default;
    virtual SolveResult solve(const Eigen::SparseMatrix<double>& A,
                              const Eigen::VectorXd& b) = 0;
    static std::unique_ptr<Solver> create(SolverType type);
};

} // namespace mhs
```

---

## 4.5 `scheduler`

```cpp
namespace mhs {

struct SchedulerConfig {
    double transient_duration = 0.0;
    double time_step = 1.0;
    int max_nonlinear_iterations = 50;
    double nonlinear_tolerance = 1e-6;
    double underrelaxation = 1.0;
    bool is_steady = false;
    int ring_buffer_capacity = 5;
};

class Scheduler {
public:
    Scheduler() = default;
    explicit Scheduler(const SchedulerConfig& config);
    ~Scheduler() = default;

    void setModel(model::InternalModel* model);
    void setSolver(std::unique_ptr<Solver> solver);
    void run();
    const std::vector<double>& solution() const;

private:
    bool solve_nonlinear_step();
    void step_time(double dt);

    model::InternalModel* model_ = nullptr;
    std::unique_ptr<Solver> solver_;
    SchedulerConfig config_;
    model::GlobalState state_;
    std::vector<double> solution_;
    double current_time_ = 0.0;
    int current_step_ = 0;
};

} // namespace mhs
```

---

## 4.6 `postprocessor`

```cpp
namespace mhs {

class Postprocessor {
public:
    Postprocessor() = default;
    ~Postprocessor() = default;

    std::vector<double> interpolate_cell_to_node(const model::InternalModel& model,
        const std::vector<double>& cell_temperature) const;

    double max_temperature(const std::vector<double>& T) const;
    double min_temperature(const std::vector<double>& T) const;
};

} // namespace mhs
```

### 展开逻辑

```cpp
// 根据 nx, ny, nz 计算全网格大小
int total = mesh.nx * mesh.ny * mesh.nz;
std::vector<double> output_T(total);

for (int old_idx = 0; old_idx < total; old_idx++) {
    if (cells.valid_mask[old_idx]) {
        output_T[old_idx] = temperature[cells.index_map[old_idx]];
    } else {
        output_T[old_idx] = std::nan("");  // 虚拟区域 NaN
    }
}
````
