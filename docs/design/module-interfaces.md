# 模块接口

---

## 4.1 `xmlparser`

```cpp
namespace mhs::xmlparser {

// 将 XML 文件解析为通用 DOM 树
class XmlDocument {
public:
    static XmlDocument parse_file(const std::string& path);
    static XmlDocument parse_string(const std::string& xml);

    // DOM 树遍历接口
    class Node {
    public:
        std::string name() const;
        std::string text() const;
        std::string attr(const std::string& name) const;
        std::vector<Node> children() const;
        std::vector<Node> children(const std::string& name);
        Node first_child(const std::string& name) const;
        bool has_child(const std::string& name) const;
    };

    Node root() const;
};

} // namespace mhs::xmlparser
```

---

## 4.2 `io`

```cpp
namespace mhs::io {

// XML → IO 模型
class Reader {
public:
    explicit Reader(const std::string& xml_path);
    model::io::Structure read_structure();

private:
    xmlparser::XmlDocument doc_;
};

// IO 模型 → XML（结果输出）
class Writer {
public:
    explicit Writer(const std::string& output_path);

    // 写出温度结果 XML（与原始 GUI 格式兼容）
    void write_result(const model::internal::InternalModel& model,
                      const std::vector<double>& temperature);

    // 写出 VTU 文件（ParaView 可视化）
    void write_vtu(const model::internal::InternalModel& model,
                   const std::vector<double>& temperature,
                   const std::string& vtu_path);

private:
    std::string output_path_;
};

} // namespace mhs::io
```

---

## 4.4 `preprocessor`

```cpp
namespace mhs::preprocessor {

// 将 IO 模型转换为内部模型
class ModelBuilder {
public:
    explicit ModelBuilder(const model::io::Structure& io_model);

    // 返回完全组装好的内部模型（所有表达式已编译）
    model::internal::InternalModel build();

private:
    const model::io::Structure& io_model_;
};

// 处理层几何 → 填充每个单元的 material_id / layer_id
class LayerProcessor {
public:
    static void resolve_layer_geometry(
        const model::io::Structure& io_model,
        const std::vector<double>& vertex_x,
        const std::vector<double>& vertex_y,
        const std::vector<double>& vertex_z,
        model::internal::CellFields& cells);
};

// 解析面键字符串 → 每面 BC 数组
class FaceKeyProcessor {
public:
    static void resolve_face_keys(
        const std::vector<model::io::Boundary>& boundaries,
        const model::internal::MeshGeometry& mesh,
        model::internal::FaceBCFields& face_bcs,
        model::internal::BCParamTable& bc_params);
};

} // namespace mhs::preprocessor
```

### 预处理流程

```text
IO 模型（Structure，含字符串表达式）
  └─> ModelBuilder::build()
        ├─> 解析几何变量（w_top, h_middle 等）→ 具体数值
        ├─> 计算每层厚度 → 确定全局 Z 网格线
        ├─> 构建顶点坐标 vertex_x/y/z
        ├─> LayerProcessor::resolve_layer_geometry()
        │     └─> 对每个单元：判断属于哪个 Layer/Block → layer_id, material_id, heat_source
        ├─> FaceKeyProcessor::resolve_face_keys()
        │     └─> 解析 face_key 字符串 → face_bcs + bc_params
        └─> 编译所有 FieldExpression
              ├─> 材料属性 k/rho/c → MaterialProps（每种材料一个）
              ├─> BC 参数 → BCParamTable（每个边界参数一个）
              └─> 热源 → CellFields.heat_source（每个单元一个）
```

---

## 4.5 `assembler`

```cpp
namespace mhs::assembler {

// 组装上下文 — 组装器每次调用接收的信息
struct AssemblyContext {
    const model::internal::InternalModel& model;
    const model::internal::GlobalState& state;

    // 完整的表达式求值上下文（传递给 FieldExpression::eval）
    expr::FieldContext expr_ctx;
};

// 组装结果：线性系统 A * T = b
struct LinearSystem {
    Eigen::SparseMatrix<double> A;  // 稀疏 Jacobian
    Eigen::VectorXd b;                // RHS（含 BC 贡献）
    Eigen::VectorXd residual;        // 残差（用于收敛判断）
};

class Assembler {
public:
    explicit Assembler(const model::internal::InternalModel& model);

    // 组装 Jacobian A 和 RHS b（每次 Newton 迭代调用）
    LinearSystem assemble(const model::internal::GlobalState& state, double t);

    // 仅组装 RHS（用于定常迭代或 Picard）
    LinearSystem assemble_rhs_only(const model::internal::GlobalState& state, double t);

private:
    const model::internal::InternalModel& model_;
};

} // namespace mhs::assembler
```

### 组装热循环说明

对每个单元 `(i, j, k)`：

1. 获取 `material_id` → `material_table` → `MaterialProps {k, rho, c}`
2. 对每个临面：
   - 查 `face_bcs.bc_type[face_idx]` → BC 类型
   - 查 `face_bcs.bc_param_idx[face_idx]` → 参数
   - 调用 `bc_params.dirichlet_T[idx].eval(ctx)` 或类似函数
   - 施加 BC 贡献到 A 和 b
3. 获取 `heat_source[cell_idx].eval(ctx)` → 体热源 Q
4. 组装扩散项（邻居温度项）
5. 若为瞬态，组装瞬态项（ρc/Δt）(T - T_prev)

---

## 4.6 `solver`

```cpp
namespace mhs::solver {

enum class SolverType { SparseLU, BiCGSTAB };

struct SolverConfig {
    SolverType type = SolverType::BiCGSTAB;
    double tolerance = 1e-8;
    int max_iterations = 1000;
};

class SolverFactory {
public:
    static std::unique_ptr<SolverBase> create(const SolverConfig& config);
};

class SolverBase {
public:
    virtual ~SolverBase() = default;

    // 求解 A * x = b
    virtual Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A,
                                  const Eigen::VectorXd& b) = 0;

    // 求解后残差范数
    virtual double residual_norm() const = 0;
};

} // namespace mhs::solver
```

### 求解器选择策略

| 规模             | 推荐求解器              | 原因                   |
| ---------------- | ----------------------- | ---------------------- |
| 小（< 10⁴ 单元） | `SparseLU`              | 直接法，无迭代收敛问题 |
| 大（> 10⁴ 单元） | `BiCGSTAB` + ILU 预条件 | 迭代法，内存效率高     |

---

## 4.7 `scheduler`

```cpp
namespace mhs::scheduler {

struct SchedulerConfig {
    double transient_duration = 0.0;    // 0 = 稳态
    double time_step = 1.0;
    int max_newton_iterations = 50;
    double newton_tolerance = 1e-6;
    double underrelaxation = 1.0;        // 1.0 = 无欠松弛
    bool is_steady = false;              // true 时跳过时间步进
};

class Scheduler {
public:
    explicit Scheduler(const model::internal::InternalModel& model,
                       const SchedulerConfig& config);

    // 运行完整仿真，通过后处理器写出结果
    // - 稳态（is_steady=true）：t=0 时单次非线性迭代
    // - 瞬态：t=0 到 transient_duration 的时间步循环
    void run(postprocessor::PostProcessor& pp);

    // 步进 API（用于测试）
    void initialize();
    bool advance_time_step();       // 返回是否收敛
    bool is_finished() const;

    const model::internal::GlobalState& state() const { return state_; }

private:
    bool solve_nonlinear_step();
    bool check_convergence();

    const model::internal::InternalModel& model_;
    SchedulerConfig config_;
    model::internal::GlobalState state_;
    double current_time_ = 0.0;
    int current_step_ = 0;
    std::unique_ptr<assembler::Assembler> assembler_;
    std::unique_ptr<solver::SolverBase> linear_solver_;
};

} // namespace mhs::scheduler
```

### 调度流程

**稳态**（`is_steady=true`，`transient_duration=0`）：

```cpp
initialize()
while (!converged && iter < max_newton):
    ls = assembler.assemble(state, t=0)
    dT = solver.solve(ls.A, ls.b)
    state.T += underrelaxation * dT
    converged = check_convergence(ls.residual)
postprocessor.write_vtu / write_xml_result
```

**瞬态**：

```cpp
initialize()
while (current_time < transient_duration):
    while (!converged && iter < max_newton):
        ls = assembler.assemble(state, current_time)
        dT = solver.solve(ls.A, ls.b)
        state.T += underrelaxation * dT
        converged = check_convergence()
    state.T_prev = state.T
    current_time += time_step
postprocessor.write_vtu / write_xml_result
```

---

## 4.8 `postprocessor`

```cpp
namespace mhs::postprocessor {

class PostProcessor {
public:
    explicit PostProcessor(const std::string& output_dir);

    // 写出 VTU 文件（ParaView 可视化）
    void write_vtu(const model::internal::InternalModel& model,
                   const std::vector<double>& temperature,
                   const std::string& filename);

    // 写出结果 XML（与原始 GUI 格式兼容）
    void write_xml_result(const model::internal::InternalModel& model,
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
