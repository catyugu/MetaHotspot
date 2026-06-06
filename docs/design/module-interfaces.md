# 模块接口

## `io`

```cpp
namespace mhs::io {
    mhs::core::IOStructure read_xml(const std::string& xml_path);

    void write_vtu(const std::string& path,
                   const mhs::core::InternalModel& model,
                   const std::vector<double>& node_temperature);

    void write_xml(const std::string& input_path,
                   const std::string& output_path,
                   const mhs::core::InternalModel& model,
                   const std::vector<double>& node_temperature,
                   const std::vector<mhs::core::ProbeTrace>& observation_traces = {});
}
```

## `preprocessor`

```cpp
namespace mhs::sim {
    class Preprocessor {
    public:
        std::unique_ptr<mhs::core::InternalModel> load(const mhs::core::IOStructure& io);
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
    const std::vector<mhs::core::Layer>& layers, double si_scale);

int find_block_for_cell(const ResolvedLayerGeometry& layer,
                        double cx, double cy, double cz);  // -1 = virtual

void resolve_layers(const std::vector<ResolvedLayerGeometry>& layers,
                    const mhs::core::MeshGeometry& mesh,
                    const std::unordered_map<std::string, size_t>& name_to_idx,
                    mhs::core::CellFields& cells);

struct FaceKeyInfo { char axis = 'Z'; char side = 'E';
                     double coord_value = 0.0;
                     std::vector<std::array<double,4>> rects; };  // SI

FaceKeyInfo parse_face_key(const std::string& key, double si_scale);
bool point_in_face_rects(const FaceKeyInfo& fk, double a, double b);

void resolve_face_keys(const std::vector<mhs::core::Boundary>& boundaries,
                       mhs::core::ThermalBCType other_bc_type,
                       const mhs::core::FirstTypeThermalBC&  other_bc_first,
                       const mhs::core::SecondTypeThermalBC& other_bc_second,
                       const mhs::core::ThirdTypeThermalBC&  other_bc_third,
                       const mhs::core::MeshGeometry& mesh, mhs::core::CellFields& cells,
                       mhs::core::BCParamTable& bc_params, double si_scale);
```

### 预处理流程

```text
mhs::core::IOStructure
  └─> Preprocessor::load()
        ├─> mhs::core::clear_registry() + set_variable(几何变量) + mhs::sim::register_all_functions(ios.functions)
        ├─> MeshGeometry from mesh_vertex_x/y/z (×si_scale)
        ├─> resolve_geometry()     // 预求层 Z 范围 + Block XY 坐标
        ├─> material_table         // 解析 k/rho/c
        ├─> resolve_layers()       // valid_mask, index_map, material_id, layer_id
        ├─> heat_source_table      // 去重 ti_reyuan_expr，idx 0 = constant(0)
        │     + cells.heat_source_idx[c_idx] = uint16_t
        ├─> resolve_face_keys()    // 展平 (boundary, face_key) 后单次遍历网格：CellBC + BCParamTable + other_bc
        └─> mhs::core::InternalModel ready
```

## `assembler`

```cpp
namespace mhs::sim {
    struct LinearSystem {
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        Eigen::VectorXd residual;     // b - A * T  (snapshot at assemble time)
    };

    class Assembler {
    public:
        explicit Assembler(const mhs::core::InternalModel& model);
        LinearSystem assemble(const mhs::core::GlobalState& state);
    };
}
```

`assemble()` 用 `tbb::parallel_for(0, total)` 扫描全网格，**跳过虚拟单元**。每线程独立 `tbb::enumerable_thread_specific<ThreadLocalData>` 持 triplet 列表 + RHS 向量，并行结束后 `combine_each` 合并。面法向相关的几何查表（`k_along` / `face_area` / `half_length_along` / `neighbor_grid_index`）全部来自 `mhs::core` 的 `face_dir_tables`，不再在 assembler 内定义 switch 分支。组装项：

- 扩散项（与 `k` 求值，邻居平均传导率）
- 每面 BC（按 `cell_bc.types[f]` 走 Dirichlet/Neumann/Cauchy 分支）
- 体热源 `Q = heat_source_table[hs_idx].eval(ctx)`，累入 RHS
- 瞬态项（仅 `StudyType::Transient && dt > 0`）：`ρc·vol/dt·(T − T_prev)`

所有 `eval()` 走 TBB ETS，无锁。

## `linear_solver`

```cpp
namespace mhs::sim {
    enum class SolverType { Pardiso, SparseLU, BiCGSTAB };

    struct SolverConfig {
        SolverType type = SolverType::Pardiso;
        double tolerance = 1e-8;
        int max_iterations = 1000;
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
        static std::unique_ptr<LinearSolver> create(SolverType type);
    };

    class BiCGSTABSolver  : public LinearSolver { ... };
    class PardisoSolver   : public LinearSolver { ... };
    class SparseLUSolver  : public LinearSolver { ... };
}
```

`Solver` 改名为 `LinearSolver`，与非线性迭代路径（`mhs::sim::nonlinear_solve`）区分。

## `nonlinear`

```cpp
namespace mhs::sim {
    struct NonLinearResult { bool converged = false; int iterations = 0; };

    struct NonLinearConfig {
        double underrelaxation      = 1.0;
        int    max_iterations       = 50;
        double relative_tolerance   = 1e-6;
        double absolute_tolerance   = 1e-12;
    };

    NonLinearResult nonlinear_solve(const mhs::core::InternalModel& model,
                                    mhs::core::GlobalState& state,
                                    LinearSolver& solver);
}
```

`nonlinear_solve()` 是 Anderson 加速的定点迭代：assemble → solve → underrelax 更新 → 收敛判据。**完整非线性循环在 `mhs::sim` 内**，`Scheduler` 不持有私有非线性逻辑。

非线性的所有控制参数（`underrelaxation` / `max_iterations` / 收敛容差）都由本模块**自己持有**——`nonlinear_solve` 无 cfg 入参；默认值见 `NonLinearConfig`，调整请直接改该结构体或在本函数中替换配置来源（模型字段、注册表等）。

## `scheduler`

```cpp
namespace mhs::sim {
    class Scheduler {
    public:
        void setModel(mhs::core::InternalModel* model);
        void setSolver(std::unique_ptr<LinearSolver> solver);
        void run();
        const std::vector<double>& solution() const;
    };
}
```

时间状态（`current_time`, `time_step`, `dt`）在 `mhs::core::GlobalState`，非 `Scheduler` 私有成员。

`Scheduler` 不持有任何专属配置；`run()` 全部从 `model_` 读取：

- `study_type` — 决定是否进入时间循环
- `transient_duration` / `transient_time_step` — 瞬态循环的 `duration` / `dt`
- 非线性迭代参数 → `mhs::sim::nonlinear_solve` 内部默认

`run()` 行为：

- `Steady`：跳过时间循环，单次调用 `nonlinear_solve()`
- `Transient`：循环至 `current_time < duration`，每步 `T_prev = T` 后调用 `nonlinear_solve()`

## `postprocessor`

```cpp
namespace mhs::post {
    class Postprocessor {
    public:
        std::vector<double> interpolate_cell_to_node(
            const mhs::core::InternalModel& model,
            const std::vector<double>& cell_temperature) const;

        double sample_point(
            const std::vector<double>& node_T,
            const mhs::core::InternalModel& model,
            const mhs::core::ProbePoint& point) const;

        double max_temperature(const std::vector<double>& T) const;
        double min_temperature(const std::vector<double>& T) const;
    };
}
```

纯计算，无 IO。展开到全网格：虚拟位置写 NaN，由 `mhs::io::write_vtu` / `mhs::io::write_xml` 序列化。
