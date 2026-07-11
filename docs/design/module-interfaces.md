# 模块接口

## `io`

```cpp
namespace mhs::io {
    mhs::core::IOStructure read_xml(const std::string& xml_path);
    std::optional<mhs::core::FluidOverlay> read_fluid_overlay_xml(const std::string& xml_path);

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
    class Preprocessor {
    public:
        std::unique_ptr<mhs::core::Model> load(
            const mhs::core::IOStructure& ioStructure,
            const std::optional<mhs::core::FluidOverlay>& fluidOverlay = std::nullopt);
    };
}

namespace mhs::sim {

double length_unit_to_si(mhs::core::LengthUnit unit);

struct ResolvedRect         { bool add_sub; double x, y, width, height; };  // SI
struct ResolvedBlock        { std::vector<ResolvedRect> rects;
                              std::string material_name;
                              std::string ti_reyuan_expr; };  // 字符串，待 parse
struct ResolvedLayerGeometry{ std::vector<ResolvedBlock> blocks;
                              double z_start, z_end; };  // SI

std::vector<ResolvedLayerGeometry> resolve_geometry(
    const std::vector<mhs::core::Layer>& layers, double si_scale,
    const mhs::core::SymbolTable& symbols);

int find_block_for_cell(const ResolvedLayerGeometry& layer,
                        double cx, double cy, double cz);  // -1 = virtual

mhs::core::CellFields assign_cell_layers(
    const std::vector<ResolvedLayerGeometry>& resolved_layers,
    const mhs::core::MeshGeometry& mesh,
    const std::unordered_map<std::string, size_t>& name_to_idx,
    const std::vector<std::vector<uint16_t>>& block_hs_map);

void resolve_boundary_patches(
    const mhs::core::MeshGeometry& mesh,
    const mhs::core::CellFields& cells,
    const std::vector<ParsedFaceKey>& parsed_face_keys,
    mhs::core::BcType other_bc_enum,
    uint16_t other_bc_idx,
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
```

### 预处理流程

```text
mhs::core::IOStructure
  └─> Preprocessor::load()
        ├─> 构造本地 mhs::core::SymbolTable（几何变量 + register_all_functions 注入 native）
        ├─> MeshGeometry from mesh_vertex_x/y/z (×si_scale)
        ├─> resolve_geometry(symbols)  // 预求层 Z 范围 + Block XY 坐标
        ├─> material_table             // 解析 k/rho/c，parse(formula, symbols)
        ├─> assign_cell_layers()       // index_map (full-grid), material_id + heat_source_idx (compact)
        ├─> heat_source_table          // 去重 ti_reyuan_expr，idx 0 = constant(0)
        │     + cells.heat_source_idx[c_idx] = uint16_t
        ├─> parse_all_face_keys(symbols)  // 展平 (boundary, face_key) → ParsedFaceKey[]
        ├─> resolve_boundary_patches()    // 单次网格遍历写 face_bcs
        ├─> (可选) applyFluidOverlay(symbols)  // 由 Preprocessor::load 内部调用，传入同一 symbols
        └─> mhs::core::Model ready
```

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

`assemble()` 用 `tbb::parallel_for(0, total)` 扫描全网格，**跳过虚拟单元**。每线程独立 `tbb::enumerable_thread_specific<ThreadLocalData>` 持 triplet 列表 + RHS 向量 + 质量向量，并行结束后 `combine_each` 合并为 `AssemblyResult {K, f, M_diag}`。面法向相关的几何查表（`k_along` / `face_area` / `half_length_along` / `neighbor_grid_index`）全部来自 `mhs::utils` 的 `mesh_utils`，不再在 assembler 内定义 switch 分支。组装项：

- 扩散项（与 `k` 求值，邻居平均传导率）
- 每面 BC（按 `cell_bc.types[f]` 走 Dirichlet/Neumann/Cauchy 分支）
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

The `Scheduler::run()` main loop is:

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
the k most recently *accepted* (T, time) pairs.  Capacity is `max_order + 1`.

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

    struct SolveResult {
        Eigen::VectorXd solution;
        bool success;
        double residual_norm;
        int iterations;
    };

    class LinearSolver {
    public:
        virtual ~LinearSolver() = default;
        virtual SolveResult solve(const Eigen::SparseMatrix<double>& A,
                                  const Eigen::VectorXd& b) = 0;
        // Config lives on the base — every solver needs it, so subclasses no
        // longer override. The factory seeds `config_` from SolverSpec before
        // returning the instance.
        void set_config(SolverConfig cfg);
        const SolverConfig& config() const;
        static std::unique_ptr<LinearSolver> create(const SolverSpec& spec = {});
    };

    class EigenBiCGSTABSolver  : public LinearSolver { ... };
    class PardisoLUSolver   : public LinearSolver { ... };
    class EigenSparseLUSolver  : public LinearSolver { ... };
}
```

`Solver` 改名为 `LinearSolver`，与非线性迭代路径（`mhs::sim::nonlinear_solve`）区分。

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

`nonlinear_solve()` 是 Anderson 加速的定点迭代：通过 `LinearSystemProvider`（签名 `std::function<LinearSystem(Eigen::Ref<const Eigen::VectorXd>)>`）获取线性系统 → solve → underrelax 更新 → 收敛判据。Provider 显式接收当前迭代的温度场（按 `Eigen::Ref<const Eigen::VectorXd>`，零拷贝），让数据流在签名里可见；迭代状态由 `nonlinear_solve` 自身修改。接受可选的 `NonLinearConfig& cfg` 参数；**完整非线性循环在 `mhs::sim` 内**，`Scheduler` 不持有私有非线性逻辑。

非线性的控制参数（`underrelaxation` / `max_iterations` / 收敛容差）由 `NonLinearConfig` 持有，`nonlinear_solve` 通过可选参数接收；默认值见 `NonLinearConfig`，调整请直接改该结构体或在本函数中替换配置来源（模型字段、注册表等）。

## `scheduler`

```cpp
namespace mhs::sim {
    class Scheduler {
    public:
        void setModel(mhs::core::Model* model);
        void setSolver(std::unique_ptr<LinearSolver> solver);
        void run();
        const std::vector<double>& solution() const noexcept { return solution_; }
        double currentTime() const noexcept { return step_.current_time; }
        const std::vector<mhs::core::ProbeTrace>& probeTraces() const noexcept { return probe_recorder_.traces(); }
    };

    // 探针局部采样与时序记录器，专属于 Scheduler。仅依赖 mhs::core。
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

时间状态（`current_time`, `time_step`, `dt`）与已接受步历史（`SolutionHistory`）由 `Scheduler::run()` 内部持有，是 `Scheduler` 的私有成员，**不**在 `mhs::core`，**不**在 `AssembleContext`。

`Scheduler` 不持有任何专属配置；`run()` 全部从 `model_` 读取：

- `study_type` — 决定是否进入时间循环
- `transient_duration` / `transient_time_step` — 瞬态循环的 `duration` / `dt`
- 非线性迭代参数 → `mhs::sim::nonlinear_solve` 内部默认

`run()` 行为：

- `Steady`：跳过时间循环，单次调用 `nonlinear_solve()`
- `Transient`：按 `TimeScheme`（默认 `AdaptiveBdf`）循环至 `current_time >= duration`，每步 `assemble → build_system → nonlinear_solve → evaluate_step`，接受后 `history.push(T, t)`。

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
