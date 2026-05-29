# 模块接口

## 4.1 `io`

```cpp
namespace mhs::io {

class Reader {
public:
    explicit Reader(const std::string& xml_path);
    model::IOStructure read_xml();
};

class Writer {
public:
    explicit Writer(const std::string& output_path);

    void write_result(const model::InternalModel& model,
                      const std::vector<double>& temperature);

    void write_vtu(const model::InternalModel& model,
                   const std::vector<double>& temperature,
                   const std::string& vtu_path);

private:
    std::string output_path_;
};

} // namespace mhs::io
```

---

## 4.2 `preprocessor`

```cpp
namespace mhs::preprocessor {

class ModelBuilder {
public:
    explicit ModelBuilder(const model::IOStructure& io_model);

    model::InternalModel build();

private:
    const model::IOStructure& io_model_;
};

// 处理层几何 → 生成 valid_mask, index_map, material_id, layer_id
class LayerProcessor {
public:
    static void resolve(
        const std::vector<model::Layer>& layers,
        const model::MeshGeometry& mesh,
        model::CellFields& cells);
};

// 解析面键字符串 → 为每个单元的每个面分配 CellBC
class FaceKeyProcessor {
public:
    static void resolve(
        const std::vector<model::Boundary>& boundaries,
        const model::MeshGeometry& mesh,
        model::CellFields& cells,
        model::BCParamTable& bc_params);
};

} // namespace mhs::preprocessor
```

### 预处理流程

```text
IOStructure（含字符串表达式 + mesh_vertex_x/y/z from XML）
  └─> ModelBuilder::build()
        ├─> 转换单位（length_unit → SI），注册几何变量 → expr
        ├─> 注册材料函数、用户函数 → expr
        ├─> LayerProcessor::resolve()
        │     ├─> 直接使用 mesh_vertex_x/y/z 构建顶点坐标
        │     ├─> 计算 dx/dy/dz, cx/cy/cz
        │     ├─> 生成 valid_mask + index_map
        │     └─> 分配 material_id, layer_id（全网格大小）
        ├─> FaceKeyProcessor::resolve()
        │     ├─> 解析 face_key 字符串
        │     ├─> 为每个单元的每个面分配 CellBC
        │     └─> 填充未指定面的 other_bc
        └─> 编译表达式
              ├─> 材料属性 k/rho/c → MaterialProps
              ├─> BC 参数 → BCParamTable
              └─> 热源 → CellFields.heat_source（紧凑）
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
namespace mhs::solver {

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

} // namespace mhs::solver
```

---

## 4.5 `scheduler`

```cpp
namespace mhs::scheduler {

struct SchedulerConfig {
    double transient_duration = 0.0;
    double time_step = 1.0;
    int max_newton_iterations = 50;
    double newton_tolerance = 1e-6;
    double underrelaxation = 1.0;
    bool is_steady = false;
    int ring_buffer_capacity = 5;
};

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);
    void setModel(model::InternalModel* model);
    void run();
    void initialize();
    bool advance_time_step();
    bool is_finished() const;
    const std::vector<double>& solution() const;

private:
    bool solve_nonlinear_step();

    model::InternalModel* model_ = nullptr;
    SchedulerConfig config_;
    double current_time_ = 0.0;
    int current_step_ = 0;
    std::unique_ptr<solver::Solver> solver_;
    std::vector<double> solution_;
};

} // namespace mhs::scheduler
```

---

## 4.6 `postprocessor`

```cpp
namespace mhs::postprocessor {

class PostProcessor {
public:
    explicit PostProcessor(const std::string& output_dir);

    // VTU 输出：展开 T 向量，虚拟区域填充 NaN
    void write_vtu(const model::InternalModel& model,
                   const std::vector<double>& temperature,
                   const std::string& filename);

    // XML 输出：展开 T 向量，虚拟区域填充 NaN
    void write_xml_result(const model::InternalModel& model,
                          const std::vector<double>& temperature,
                          const std::string& filename);

    double max_temperature(const std::vector<double>& T) const;
    double min_temperature(const std::vector<double>& T) const;

private:
    std::string output_dir_;
};

} // namespace mhs::postprocessor
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
```
