# 模块接口

## 4.1 `io`

```cpp
namespace mhs::io {

// XML → IO 模型（直接使用 tinyxml2）
class Reader {
public:
    explicit Reader(const std::string& xml_path);
    model::IOStructure read_xml();
};

// IO 模型 → XML（结果输出）
class Writer {
public:
    explicit Writer(const std::string& output_path);

    // 写出温度结果 XML（与原始 GUI 格式兼容）
    void write_result(const model::InternalModel& model,
                      const std::vector<double>& temperature);

    // 写出 VTU 文件（ParaView 可视化）
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

    // 构建内部模型
    // - 填充 expr 表达式注册表（变量、函数）
    // - 单位转换（IOStructure::length_unit → SI）
    // - 构建 MeshGeometry、CellFields、FaceBCFields
    // - 编译所有 CompiledExpression
    model::InternalModel build();

private:
    const model::IOStructure& io_model_;
};

// 处理层几何 → 填充每个单元的 material_id / layer_id
// 输入：IOStructure::layers，MeshGeometry
// 输出：CellFields::material_id, layer_id
class LayerProcessor {
public:
    static void resolve(
        const std::vector<model::Layer>& layers,
        const model::MeshGeometry& mesh,
        model::CellFields& cells);
};

// 解析面键字符串 → FaceBCFields + BCParamTable
// 输入：IOStructure::boundaries（包含 face_keys 字符串）
// 输出：FaceBCFields（索引数组），BCParamTable（编译后的表达式）
class FaceKeyProcessor {
public:
    static void resolve(
        const std::vector<model::Boundary>& boundaries,
        const model::MeshGeometry& mesh,
        model::FaceBCFields& face_bcs,
        model::BCParamTable& bc_params);
};

} // namespace mhs::preprocessor
```

### 预处理流程

```text
IOStructure（含字符串表达式）
  └─> ModelBuilder::build()
        ├─> 转换单位（length_unit → SI），注册几何变量 → expr
        ├─> 注册材料函数、用户函数 → expr
        ├─> LayerProcessor::resolve()
        │     ├─> 计算顶点坐标
        │     └─> 对每个单元判断属于哪个 Layer/Block → material_id, layer_id
        ├─> FaceKeyProcessor::resolve()
        │     ├─> 解析 face_key 字符串（世界坐标）
        │     └─> 坐标→索引映射 → face_bcs + bc_params
        └─> 编译所有 CompiledExpression
              ├─> 材料属性 k/rho/c → MaterialProps
              ├─> BC 参数 → BCParamTable
              └─> 热源 → CellFields.heat_source
```

---

## 4.3 `assembler`

```cpp
namespace mhs::assembler {

// 组装结果：线性系统 A * T = b
struct LinearSystem {
    Eigen::SparseMatrix<double> A;  // 稀疏 Jacobian
    Eigen::VectorXd b;                // RHS（含 BC 贡献）
    Eigen::VectorXd residual;         // 残差（用于收敛判断）
};

// 组装器（无状态，给定相同输入返回相同输出）
class Assembler {
public:
    explicit Assembler(const model::InternalModel& model);

    // 组装 Jacobian A 和 RHS b
    // GlobalState 包含：T, T_prev, dt_history, current_time, time_step
    // 内部读取 model.sparsity_pattern，只填充值
    LinearSystem assemble(const model::GlobalState& state);

private:
    const model::InternalModel& model_;
};

} // namespace mhs::assembler
```

### Sparsity Pattern

矩阵稀疏模式（7点 stencil）在预处理阶段预计算，存储在 `InternalModel` 中：

```cpp
struct InternalModel {
    // ...
    SparsityPattern sparsity;  // 预计算的非零位置
};
```

组装时只填充值，不重建结构。

### 组装热循环说明

对每个单元 `(i, j, k)`：

1. 获取 `material_id` → `material_table[id]` → `MaterialProps {k, rho, c}`
2. 对每个临面：
   - 查 `face_bcs.bc_type[face_idx]` → BC 类型
   - 查 `face_bcs.bc_param_idx[face_idx]` → 参数索引
   - 调用 `bc_params.dirichlet_T[idx].eval(ctx)` 等
   - 施加 BC 贡献
3. 获取 `heat_source[cell_idx].eval(ctx)` → 体热源 Q
4. 组装扩散项（邻居温度项）
5. 组装瞬态项（Crank-Nicolson，θ = 0.5，lumped mass）

### 并行化

组装热循环使用 TBB 并行化：`tbb::parallel_for` 对单元进行并行迭代。

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

// 求解结果（无状态，返回值携带所有信息）
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

### 求解器选择策略

| 规模             | 推荐求解器              | 原因                   |
| ---------------- | ----------------------- | ---------------------- |
| 小（< 10⁴ 单元） | `SparseLU`              | 直接法，无迭代收敛问题 |
| 大（> 10⁴ 单元） | `BiCGSTAB` + ILU 预条件 | 迭代法，内存效率高     |

---

## 4.5 `scheduler`

```cpp
namespace mhs::scheduler {

// 收敛状态（GlobalState 中使用）
enum class ConvergenceStatus { Running, Converged, Diverged, Stalled };

struct SchedulerConfig {
    double transient_duration = 0.0;    // 0 = 稳态
    double time_step = 1.0;
    int max_newton_iterations = 50;
    double newton_tolerance = 1e-6;
    double underrelaxation = 1.0;        // 1.0 = 无欠松弛
    bool is_steady = false;              // true 时跳过时间步进
    int ring_buffer_capacity = 5;        // T_history, nl_history 容量
};

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);

    void setModel(model::InternalModel* model);

    // 运行完整仿真
    void run();

    // 步进 API（用于测试）
    void initialize();          // 初始化 T = T_prev = initial_temperature
    bool advance_time_step();  // 返回是否收敛
    bool is_finished() const;

    const std::vector<double>& solution() const;

private:
    bool solve_nonlinear_step();

    model::InternalModel* model_ = nullptr;  // 非拥有权，外部管理
    SchedulerConfig config_;
    double current_time_ = 0.0;
    int current_step_ = 0;
    std::unique_ptr<solver::Solver> solver_;
    std::vector<double> solution_;
};

} // namespace mhs::scheduler
```

### 调度流程

**初始化**：

```cpp
state_.T.resize(cell_count, initial_temperature);
state_.T_prev = state_.T;
state_.T_history.push_back(state_.T);  // ring buffer
```

**稳态**（`is_steady=true`，`transient_duration=0`）：

```cpp
while (state_.status == ConvergenceStatus::Running && iter < max_newton):
    ls = assembler.assemble(state_);
    result = solver_->solve(ls.A, ls.b);
    state_.T += underrelaxation * result.solution;
    state_.residual = ls.residual;
    check_convergence();
postprocessor.write_vtu / write_xml_result
```

**瞬态**（自适应 Δt 解耦）：

```cpp
while (current_time < transient_duration):
    while (state_.status == ConvergenceStatus::Running && iter < max_newton):
        ls = assembler.assemble(state_);
        result = solver_->solve(ls.A, ls.b);
        state_.T += underrelaxation * result.solution;
        state_.residual = ls.residual;
        check_convergence();
    state_.T_history.push_back(state_.T);  // ring buffer push
    state_.T_prev = state_.T;
    dt_history.push_back(dt);
    current_time += dt;
postprocessor.write_vtu / write_xml_result
```

---

## 4.6 `postprocessor`

```cpp
namespace mhs::postprocessor {

class PostProcessor {
public:
    explicit PostProcessor(const std::string& output_dir);

    // 写出 VTU 文件（ParaView 可视化）
    void write_vtu(const model::InternalModel& model,
                   const std::vector<double>& temperature,
                   const std::string& filename);

    // 写出结果 XML（与原始 GUI 格式兼容）
    void write_xml_result(const model::InternalModel& model,
                          const std::vector<double>& temperature,
                          const std::string& filename);

    // 计算导出量
    double max_temperature(const std::vector<double>& T) const;
    double min_temperature(const std::vector<double>& T) const;

private:
    std::string output_dir_;
};

} // namespace mhs::postprocessor
```
