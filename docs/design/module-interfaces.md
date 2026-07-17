# 模块接口

## `io`

```cpp
namespace mhs::io {
    mhs::core::ModelDefinition read_xml(const std::string& xml_path);
    bool merge_fluid_xml(const std::string& xml_path,
                         mhs::core::ModelDefinition& definition);

    void write_vtu(const std::string& path,
                   const mhs::core::Model& model,
                   const std::vector<double>& node_temperature);

    void write_xml(const std::string& input_path,
                   const std::string& output_path,
                   const mhs::core::Model& model,
                   const std::vector<double>& node_temperature,
                   const std::vector<mhs::core::ProbeTrace>& observation_traces = {});
}
```

## `preprocessor`

```cpp
namespace mhs::sim {
    mhs::core::Model build_model(
        const mhs::core::ModelDefinition& definition);
}

namespace mhs::utils {
double length_unit_to_si(mhs::core::LengthUnit unit);
}

namespace mhs::sim {

struct ResolvedRect         { bool add_sub; double x, y, width, height; };  // SI
struct ResolvedBlock        { std::vector<ResolvedRect> rects;
                              std::string material_name;
                              std::string ti_reyuan_expr; };  // 字符串，待 parse
struct ResolvedLayerGeometry{ std::vector<ResolvedBlock> blocks;
                              double z_start, z_end; };  // SI

std::vector<ResolvedLayerGeometry> resolve_geometry(
    const std::vector<mhs::core::Layer>& layers, double si_scale,
    const mhs::core::SymbolTable& symbols);

mhs::core::CellFields assign_cell_layers(
    const std::vector<ResolvedLayerGeometry>& resolved_layers,
    const mhs::core::MeshGeometry& mesh,
    const std::unordered_map<std::string, size_t>& name_to_idx,
    const std::vector<std::vector<uint16_t>>& block_hs_map);

void resolve_boundary_patches(
    const mhs::core::MeshGeometry& mesh,
    const mhs::core::CellFields& cells,
    const std::vector<ParsedFaceKey>& parsed_face_keys,
    const OtherBC& other_bc,
    std::vector<mhs::core::FaceBC>& face_bcs);

struct FaceKeyInfo { char axis = 'Z'; char side = 'E';
                     double coord_value = 0.0;
                     std::vector<std::array<double,4>> rects; };  // SI

FaceKeyInfo parse_face_key(const std::string& key, double si_scale);
bool point_in_face_rects(const FaceKeyInfo& fk, double a, double b);

std::vector<ParsedFaceKey> parse_all_face_keys(
    const std::vector<mhs::core::Boundary>& boundaries,
    mhs::core::BCParamTable& bc_params, double si_scale,
    const std::function<std::string(const std::string&)>& rewriter,
    const mhs::core::SymbolTable& symbols);
}
```

### 预处理流程

```text
mhs::core::ModelDefinition
  └─> build_model()
        ├─> 构造本地 mhs::core::SymbolTable（几何变量 + register_all_functions 注入 native）
        ├─> MeshGeometry from mesh_vertex_x/y/z (×si_scale)
        ├─> resolve_geometry(symbols)  // 预求层 Z 范围 + Block XY 坐标
        ├─> material_table             // 解析 k/rho/c，parse(formula, symbols)
        ├─> assign_cell_layers()       // grid_to_cell (full-grid), cell_to_grid + fields (compact)
        ├─> heat_source_table          // 按 Block 编译，idx 0 = constant(0)
        │     + cells.heat_source_idx[c_idx] = uint16_t
        ├─> parse_all_face_keys(symbols)  // 展平 (boundary, face_key) → ParsedFaceKey[]
        ├─> resolve_boundary_patches()    // 单次网格遍历写 face_bcs
        ├─> fluid::build_domain()      // 水力临时状态局部化，只输出热组装所需字段
        └─> mhs::core::Model ready
```

## `fluid`

`fluid_lib` 独立负责两件事：

1. `build_domain()` 在局部工作区中完成流体映射、通道几何、水力导通、边界解析和压力求解，最终只持久化冻结面流量、流固换热因子和边界热数据。
2. `assemble_increment()` 返回流固界面修正、流体内部迎风对流和入口/出口项。所有矩阵坐标均为已有对角或直接邻居位置，不改变基础热算子的稀疏模式。

流固界面在基础导热系数上追加差量：

```text
delta = conductance(interface convection) - conductance(base diffusion)
```

因此基础 Assembler 不需要流体分支，而总算子与原公式保持一致。

## `assembler`

```cpp
namespace mhs::sim {
    struct LinearSystem {
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
    };

    /// Result of a single assembler sweep over the active grid.
    /// All terms (diffusion, BC, source, mass) are evaluated at the
    /// temperature field and time passed in via AssembleContext.
    struct AssemblyResult {
        Eigen::SparseMatrix<double> K;
        Eigen::VectorXd f;
        Eigen::VectorXd M_diag;
    };

    /// Minimum data needed by Assembler::assemble to evaluate one cell sweep.
    /// Invariant (caller-enforced): T.size() == N_active.
    struct AssembleContext {
        Eigen::Ref<const Eigen::VectorXd> T;
        double current_time = 0.0;
    };

    class Assembler {
    public:
        explicit Assembler(const mhs::core::Model& model);
        AssemblyResult assemble(const AssembleContext& ctx) const;
    };
}
```

`assemble()` 首先用 `tbb::parallel_for(0, N_active)` 遍历 `cell_to_grid`，生成与物理类型无关的基础热算子；`fluid::assemble_increment()` 则只遍历 `fluid_to_global`。两条路径都不再扫描整个结构化网格。增量合并后只构造一次稀疏矩阵。面法向相关的几何查表（`k_along` / `face_area` / `half_length_along` / `neighbor_grid_index`）全部来自 `mhs::utils` 的 `mesh_utils`。基础组装项：

- 扩散项（与 `k` 求值，邻居平均传导率）
- 每面 BC（按 `face_bcs[cell * 6 + face]` 走 Dirichlet/Neumann/Cauchy 分支）
- 体热源 `Q = heat_source_table[hs_idx].eval(ctx)`，累入 RHS
- 质量项 `M_diag[c] = ρ·c·vol`，在 `ctx.T` 处求值

所有 `eval()` 走 TBB ETS，无锁。

## `time_scheme`

Three orthogonal components replace the old OOP `TimeScheme` hierarchy:

```cpp
namespace mhs::sim::time_scheme {

    // ── 1. Integrator (pure linear algebra) ──────────────────────────
    enum class IntegratorKind { Bdf1, Bdf2 };

    LinearSystem build_system(IntegratorKind kind, const AssemblyResult& ops,
                              const mhs::core::SolutionHistory& hist, double dt);

    // ── 2. Error controller (pure function) ─────────────────────────
    struct ErrorControlConfig {
        double abs_tol = 1e-4;
        double safety  = 0.9;
    };

    struct ErrorEstimate {
        double error_ratio = 0.0;       // LTE / abs_tol  (≤ 1 → accept)
        double suggested_factor = 1.0;  // PI-controller multiplier
    };

    ErrorEstimate estimate_error(const mhs::core::SolutionHistory& accepted,
                                 const std::vector<double>& trial_T, double trial_dt,
                                 const ErrorControlConfig& cfg);

    // ── 3. Step controller (strategy + output-grid) ─────────────────
    enum class StepStrategy {
        Free,         // dt purely error-driven; output via linear interpolation
        Strict,       // dt clamped to hit output times exactly
        Intermediate, // ensure at least one solve point between output times
        Manual        // fixed dt, no error-based adjustment
    };

    class StepController {
    public:
        StepController(StepStrategy strategy, double min_dt, double max_dt, double fixed_dt = 1.0);
        void rebuild(double duration, double output_dt);
        double prepare(double dt_suggested, double current_t, double duration);
        std::vector<double> flush_outputs(double current_t);
    };
}
```

The `solve()` main loop is:

```text
while (current_time < duration):
    dt      = step_ctrl.prepare(dt_sug, t, duration)
    ops     = assembler.assemble(ctx)
    ls      = build_system(kind, ops, accepted, dt)
    nonlinear_solve(provider, T, solver)
    est     = estimate_error(accepted, T, dt, cfg)
    if accepted:
        dt_sug = clamp(dt * est.suggested_factor, min, max)
        accepted.accept(T, t + dt)
        t     += dt
        for t_out in step_ctrl.flush_outputs(t):
            if strategy == Free: interpolate T → T_out
            else:               record directly at T
    else:
        dt_sug = dt * 0.5  // retry with smaller dt
```

The `SolutionHistory` (`mhs::core::SolutionHistory`) is a ring buffer storing
the most recently accepted (T, time) pairs. Capacity is explicit; `solve()` currently uses 2.

当前 `solve()` 使用 `StepStrategy::Free` 和 `IntegratorKind::Bdf1`；其余策略与 BDF2 由 `time_scheme` 接口提供。

## `linear_solver`

```cpp
namespace mhs::sim {
    enum class SolverType { Pardiso, EigenSparseLU, EigenBiCGSTAB };

    struct SolverConfig {
        double tolerance = 1e-8;
        int max_iterations = 1000;
    };

    struct SolverSpec {
        SolverType type = SolverType::Pardiso;
        SolverConfig config {};
    };

    class LinearSolver {
    public:
        virtual ~LinearSolver() = default;
        virtual void compute(const Eigen::SparseMatrix<double>& A) = 0;
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& b) = 0;
        void set_config(SolverConfig cfg);
        const SolverConfig& config() const;
        bool success() const;
        int iterations() const;
        double residual() const;
        static std::unique_ptr<LinearSolver> create(const SolverSpec& spec = {});
    };

    class EigenBiCGSTABSolver  : public LinearSolver { ... };
    class PardisoLUSolver   : public LinearSolver { ... };
    class EigenSparseLUSolver  : public LinearSolver { ... };
}
```

## `nonlinear`

```cpp
namespace mhs::sim {
    struct NonLinearResult { bool converged = false; int iterations = 0; };

    struct NonLinearConfig {
        double underrelaxation      = 1.0;
        int    max_iterations       = 200;
        double relative_tolerance   = 1e-6;
        double absolute_tolerance   = 1e-12;
    };

    NonLinearResult nonlinear_solve(LinearSystemProvider ls_provider,
                                    Eigen::Ref<Eigen::VectorXd> T,
                                    LinearSolver& solver,
                                    const NonLinearConfig& cfg = {});
}
```

`nonlinear_solve()` 通过 `LinearSystemProvider` 取得当前温度对应的线性系统，执行 Anderson 加速、欠松弛和收敛判断。控制参数由 `NonLinearConfig` 提供。

## `scheduler`

```cpp
namespace mhs::sim {
    struct SolveOptions {
        SolverSpec solver;
        NonLinearConfig nonlinear;
    };

    mhs::core::Solution solve(const mhs::core::Model& model,
                              const SolveOptions& options = {});

    // 探针局部采样与时序记录器是 solve() 的内部辅助，仅依赖 mhs::core。
    // `record(time, ...)` 把 `time` 透传到 `sample_one` 的 FieldContext.t，
    // 使时间依赖的 BC / 材料表达式（如 "500 + 100*t"）在正确的时刻被求值。
    class ProbeRecorder {
    public:
        void initialize(const mhs::core::Model& model);
        void record(double time, const std::vector<double>& cell_T);
        const std::vector<mhs::core::ProbeTrace>& traces() const;
    };
}
```

时间状态（`current_time`, `time_step`, `dt`）与已接受步历史（`SolutionHistory`）由 `solve()` 的局部状态持有，**不**在 `mhs::core::Model`，**不**在 `AssembleContext`。

`solve()` 从传入的 `model` 读取时间参数，从 `SolveOptions` 读取求解器和非线性配置：

- `study_type` — 决定是否进入时间循环
- `transient_duration` — 瞬态结束时间
- `transient_time_step` — 输出时间间隔；内部求解步长由误差控制器调整
- 非线性迭代参数 → `SolveOptions::nonlinear`

`solve()` 行为：

- `Steady`：跳过时间循环，单次调用 `nonlinear_solve()`
- `Transient`：循环至 `current_time >= duration`，每步 `assemble → build_system → nonlinear_solve → estimate_error`，接受后记录历史和探针。

返回的 `Solution` 包含最终温度、最终时刻和探针序列；调用前不存在需要 setter 补齐的半初始化状态。

## `postprocessor`

```cpp
namespace mhs::post {
    // `time` 注入 FieldContext.t，使时间依赖的 BC 表达式在正确的时刻被求值。
    // 稳态场景传 0.0 即可；瞬态由调用方提供当前求解时刻。
    std::vector<double> interpolate_cell_to_node(
        const mhs::core::Model& model,
        const std::vector<double>& cell_temperature,
        double time);

    double max_temperature(const std::vector<double>& T);
    double min_temperature(const std::vector<double>& T);
}
```

`interpolate_cell_to_node` 展开到全网格：虚拟位置写 NaN，由 `mhs::io::write_vtu` / `mhs::io::write_xml` 序列化。
