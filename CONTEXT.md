# MetaHotspot Context

热仿真引擎：电子封装多层堆叠结构的三维有限体积热分析。**结构化网格**，**单元中心 DOF**。

> 入口文档。要点索引；详情见 `docs/design/*` 与 `docs/adr/*`。

---

## 求解类型

- **Steady**: `mhs::core::StudyType::Steady`。视为 t=0 的单次非线性迭代，`solve()` 不做时间循环。
- **Transient**: `mhs::core::StudyType::Transient`。从 t=0 起按 `transient_time_step` 推进，每步 `assemble → build_system → nonlinear_solve → evaluate_step`。

## 网格

结构化 3D `nx × ny × nz`。当前不支持 2D，`ModelDefinition` 不保留未生效的维度字段。每个单元存温度 DOF 在中心；BC 走面积分，无面 DOF（ADR-0002）。

## 边界条件（face-level，ADR-0002）

| 类型       | 数学                    | 离散处理                        |
| ---------- | ----------------------- | ------------------------------- |
| FirstType  | `T = T₀`                | 鬼单元 `T_ghost = 2·T₀ − T_b`   |
| SecondType | `-k ∂T/∂n = q₀`         | 累入 RHS：`Σ q·A_face`          |
| ThirdType  | `-k ∂T/∂n = h(T − T_∞)` | 对角 `h·A` 系数 + RHS `h·A·T_∞` |

> 各项异性 `k`（ADR-0002 cell-level-bc 中讨论）：装配时按面法向选 `k_along(dir) ∈ {kx, ky, kz}`。

`other_bc` 在预处理阶段由 `resolve_boundary_patches` 填到所有未显式指定的面 + 虚拟邻居面（写入 `Model::face_bcs` 扁平数组）。

## 表达式（ADR-0004）

两种独立路径：

- **几何** — `mhs::core::eval_geometry(formula, symbols)`，依赖 `symbols.variables` 中的命名变量（`w_top`、`h_middle` 等，SI 米）
- **场 / BC / 热源** — `mhs::core::parse(formula, symbols)`，上下文 `{x, y, z, T, t}`，返回轻量句柄 `mhs::core::CompiledExpression`

`FieldContext` / `FieldEvaluator` / `SymbolTable` / `CompiledExpression` **定义在 `mhs::core`（`src/expr/expr.hpp`）**。依赖方向 `mhs::sim → mhs::core`，**从不超过此方向**。

**线程模型**：每次 `build_model()` 调用在 setup 阶段构造本地 `SymbolTable`、按值贯穿 setup 路径；`parse()` 主线程试编译；`eval()` **无锁** — TBB ETS 包装，每个工作线程懒构造独立 muparser 实例。`SymbolTable` 在构造时按值复制到 `MuCompiledTLS`，运行时不依赖任何外部状态。

复杂形式用 `mhs::sim::register_all_functions(symbols, fns)` 把 `ModelDefinition.functions` 写入 `SymbolTable::natives`。

## 求解流程

```text
XML → core::ModelDefinition via io::read_xml
  → sim::build_model → core::Model
    → sim::solve → core::Solution
        ├─ sim::time_scheme::StepController (Free/Strict/Intermediate/Manual)
        │   └─ adjust dt via strategy + output-time grid
        ├─ sim::Assembler::assemble(ctx)               [K, f, M_diag]
        │   ├─ base thermal assembly（无流体分支）
        │   └─ sim::fluid::assemble_increment（不改变稀疏模式）
        ├─ sim::time_scheme::build_system(kind, ops, hist, dt)
        │   └─ 纯函数: BDF1 / BDF2 stencil
        ├─ sim::nonlinear_solve(provider, T, *solver)  [Anderson 加速定点迭代]
        │   └─ sim::LinearSolver::solve(A, b) [EigenSparseLU / EigenBiCGSTAB]
        ├─ sim::time_scheme::estimate_error(…) → ErrorEstimate
        │   └─ 纯函数: LTE 估计 + PI 步长建议
        └─ post-step: probe_recorder.record()
            — Free 模式下先对 T 做线性插值
        → post::interpolate_cell_to_node
            → io::write_vtu + io::write_xml
```

## 关键设计原则

1. 内部模型不含原始字符串 — 表达式预编译为 `CompiledExpression`
2. 热源字典化 — `heat_source_table`（去重）+ 每单元 `uint16_t` 索引
3. 面级 BC — `Model::face_bcs[N_active * 6]` 扁平数组，无 `CellBC`
4. 含流体-固体耦合子系统 — `Model::fluid`（`FluidDomain`）
5. Precomputed sparsity — 组装只填值，不重建结构
6. Backward Euler 默认；BDF2 可选（`IntegratorKind` 枚举 + `build_system` 纯函数路由）
7. 算法与组装解耦 — `Assembler::assemble` 一次遍历返回 `AssemblyResult {K, f, M_diag}`；时间离散由 `time_scheme::build_system` 纯函数注入
8. 步长控制与时间积分完全解耦 — `StepController`（策略模式）+ `estimate_error`（纯函数）替代旧 OOP `TimeScheme` 层次
9. TBB 并行组装 — 跳虚拟单元，`enumerable_thread_specific<ThreadLocalData>` + 合并
10. 域类型定义在 `src/data/types.hpp` — 内部枚举 `mhs::core::StudyType` / `BcType` / `FaceDir` 的唯一真源
11. 无异常（仅 `bin/main.cpp` 边界 try/catch 捕获 std::exception → `mhs::logger::panic`）
12. POD / 纯函数优先

## 命名空间速查（领域驱动）

命名空间按**领域边界**划分，不与目录 1:1 映射。公共 API 最多两层 `mhs::领域`；第三层 `mhs::领域::detail` 仅隐藏实现。

| 命名空间                | 源目录                                                                  | 暴露类型 / 函数                                                                                                                                                                                                        |
| ----------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mhs::core`             | `data/` + `expr/`                                                       | ModelDefinition、Model、Solution、FluidDomain、SolutionHistory、StudyType、BcType、FaceBC、FaceDir、FluidBCType、CompiledExpression、Material、ProbePoint、CellFields、MeshGeometry                                  |
| `mhs::utils`            | `utils/`                                                                | mesh_utils 查表 + physics_utils + sampling（最小二乘+面外推）                                                                                                                                                          |
| `mhs::sim`              | `assembler/` `linear_solver/` `scheduler/` `nonlinear/` `preprocessor/` | build_model()、solve()、SolveOptions、LinearSolver、Assembler、AssemblyResult、LinearSystem、LinearSystemProvider、NonLinearConfig / NonLinearResult / nonlinear_solve() |
| `mhs::sim::fluid`       | `fluid/`                                                                | build_domain()、assemble_increment()、FluidAssemblyIncrement                                                                                                                |
| `mhs::sim::time_scheme` | `time_scheme/`                                                          | StepController (策略类) + IntegratorKind 枚举 + build_system/estimate_error 纯函数 + ErrorControlConfig / ErrorEstimate + StepStrategy 枚举（Free/Strict/Intermediate/Manual） + OutputTimeGrid                        |
| `mhs::io`               | `io/`                                                                   | read_xml / merge_fluid_xml / write_vtu / write_xml                                                                                                                                                              |
| `mhs::post`             | `postprocessor/`                                                        | interpolate_cell_to_node 及导出场函数 + 局部采样辅助 `mhs::utils`                                                                                                                                                      |
| `mhs::logger`           | `logger/`                                                               | init / flush / panic + 模板 debug/info/warn/error                                                                                                                                                                      |

### 铁律

1. **`mhs` 壳不含类型** — 不重导出、不定义；纯品牌前缀。
2. **core 不依赖兄弟** — core → 无 sim/io/post/logger include；sim/io/post 可依赖 core。依赖反转解决 core 需要 sim 行为的场景。
3. **`.hpp` 绝不 `using namespace`** — 全限定名。`.cpp` 保持现状。
4. **匿名 ns = 单文件私有，detail = 跨文件私有** — 公共符号不进第三层。

## 术语表

| 概念            | 中文     | 说明                                                                  |
| --------------- | -------- | --------------------------------------------------------------------- |
| StudyType       | 求解类型 | Steady / Transient                                                    |
| Layer           | 层       | 多 Block 的 Z 厚度堆叠                                                |
| Block           | 块       | XY 平面 add/sub Rect 几何；Z 范围继承父层                             |
| Rect            | 矩形     | 块几何的 add/sub 单元                                                 |
| FaceKey         | 面键     | 字符串 `Face\|Direction\|CoordValue\|RectList`，CoordValue 是空间坐标 |
| Material        | 材料     | 含 kx/ky/kz / ρ / c（均为字符串表达式）                               |
| BC / `other_bc` | 边界条件 | 三种类型 + 默认兜底                                                   |
| Daore Xishu     | 导热系数 | kx/ky/kz, W/(m·K); 1 或 3 段逗号分隔                                  |
| Midu            | 密度     | ρ, kg/m³                                                              |
| Bi Rerong       | 比热容   | c, J/(kg·K)                                                           |
| ti_reyuan_expr  | 体热源   | Block 的体热源密度表达式 [W/m³]                                       |

## 详细参考

- 内部数据结构 → `docs/design/internal-model.md`
- IO 数据结构 → `docs/design/io-structure.md`
- 模块接口 → `docs/design/module-interfaces.md`
- expr 模块 → `docs/design/expr-api.md`
- 数据流与流程 → `docs/design/data-flow.md`
- 项目结构 / Logger / 命名空间 → `docs/design/project-structure.md`
- ADR 决策记录 → `docs/adr/0001-…0004`
