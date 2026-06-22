# MetaHotspot Context

热仿真引擎：电子封装多层堆叠结构的三维有限体积热分析。**结构化网格**，**单元中心 DOF**。

> 入口文档。要点索引；详情见 `docs/design/*` 与 `docs/adr/*`。

---

## 求解类型

- **Steady**: `mhs::core::StudyType::Steady`。视为 t=0 的单次非线性迭代，scheduler 不做时间循环。
- **Transient**: `mhs::core::StudyType::Transient`。从 t=0 起按 `transient_time_step` 推进，每步 `assemble → build_system → nonlinear_solve → evaluate_step`。

## 网格

结构化 3D `nx × ny × nz`。**当前不支持 Dimension2D**（IO 会解析但预处理未实现 2D 路径）。每个单元存温度 DOF 在中心；BC 走面积分，无面 DOF（ADR-0002）。

## 边界条件（cell-level，ADR-0005）

| 类型       | 数学                    | 离散处理                        |
| ---------- | ----------------------- | ------------------------------- |
| FirstType  | `T = T₀`                | 鬼单元 `T_ghost = 2·T₀ − T_b`   |
| SecondType | `-k ∂T/∂n = q₀`         | 累入 RHS：`Σ q·A_face`          |
| ThirdType  | `-k ∂T/∂n = h(T − T_∞)` | 对角 `h·A` 系数 + RHS `h·A·T_∞` |

> 各项异性 `k`（ADR-0005 cell-level-bc 中讨论）：装配时按面法向选 `k_along(dir) ∈ {kx, ky, kz}`。

`other_bc` 在预处理阶段填到所有未显式指定的面 + 虚拟邻居面。

## 表达式（ADR-0004）

两种独立路径：

- **几何** — `mhs::core::eval_geometry()`，依赖已注册的命名变量（`w_top`、`h_middle` 等，SI 米）
- **场 / BC / 热源** — `mhs::core::parse()`，上下文 `{x, y, z, T, t}`，返回轻量句柄 `mhs::core::CompiledExpression`

`FieldContext` / `FieldEvaluator` / `CompiledExpression` **定义在 `mhs::core`（`src/expr/expr.hpp`）**。依赖方向 `mhs::sim → mhs::core`，**从不超过此方向**。

**线程模型**：注册表变动互斥锁；`parse()` 主线程持锁试编译；`eval()` **无锁** — TBB ETS 包装，每个工作线程懒构造独立 muparser 实例。

复杂形式用 `register_native(name, FieldEvaluator)` 注册 C++ 函数字段。`IOStructure.functions` 是当前 native 入口。

## 求解流程

```text
XML → core::IOStructure via io::read_xml
  → sim::Preprocessor::load → core::InternalModel
    → sim::Scheduler::run
        ├─ sim::time_scheme::TimeScheme (Bdf1Scheme | Bdf2Scheme | AdaptiveBdfScheme)
        │   ├─ select_step(history, t, duration) → (dt, order)
        │   ├─ build_system(ops, accepted, order, dt) → LinearSystem
        │   └─ evaluate_step(accepted, trial_T, dt) → StepResult
        ├─ sim::Assembler::assemble(state)            [K, f, M_diag] (单次 TBB 遍历)
        └─ sim::nonlinear_solve(provider, state, *solver_) [Anderson 加速定点迭代]
            → sim::LinearSolver::solve(A, b) [SparseLU / BiCGSTAB]
        → post::interpolate_cell_to_node
            → io::write_vtu + io::write_xml
```

## 关键设计原则

1. 内部模型不含原始字符串 — 表达式预编译为 `CompiledExpression`
2. 热源字典化 — `heat_source_table`（去重）+ 每单元 `uint16_t` 索引
3. Cell-level BC — 每单元存 6 面 BC（`CellBC`）
4. Precomputed sparsity — 组装只填值，不重建结构
5. Backward Euler 默认；可选 BDF2 / 自适应 BDF（`TimeScheme` 抽象）
6. 算法与组装解耦 — `Assembler::assemble` 一次遍历返回 `AssemblyResult {K, f, M_diag}`；时间离散由 `TimeScheme::build_system` 注入（`K` 用当前 T，`M_diag` 仍取 `accepted.current()` 保持 Newton 内冻结）
7. TBB 并行组装 — 跳虚拟单元，`enumerable_thread_specific<ThreadLocalData>` + 合并
8. 域类型定义在 `src/data/types.hpp` — 内部枚举 `mhs::core::StudyType` / `BcType` / `FaceDir` 的唯一真源
9. 无虚函数（`mhs::sim::LinearSolver` 与 `mhs::sim::time_scheme::TimeScheme` 除外）；无异常（仅 `bin/main.cpp` 边界 try/catch 捕获 std::exception → `mhs::logger::panic`）
10. POD / 纯函数优先

## 命名空间速查（领域驱动）

命名空间按**领域边界**划分，不与目录 1:1 映射。公共 API 最多两层 `mhs::领域`；第三层 `mhs::领域::detail` 仅隐藏实现。

| 命名空间                | 源目录                                                                  | 暴露类型 / 函数                                                                                                                                                                                            |
| ----------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mhs::core`             | `data/` + `expr/`                                                       | InternalModel、IOModel、GlobalState（含 `SolutionHistory accepted`）、StudyType、BcType、FaceDir、CompiledExpression、FieldEvaluator、Material                                                             |
| `mhs::utils`            | `common/`                                                               | mesh_utils 查表                                                                                                                                                                                            |
| `mhs::sim`              | `assembler/` `linear_solver/` `scheduler/` `nonlinear/` `preprocessor/` | LinearSolver、BiCGSTABSolver、PardisoSolver、SparseLUSolver、Assembler、AssemblyResult、LinearSystem、LinearSystemProvider、Scheduler、Preprocessor、NonLinearConfig / NonLinearResult / nonlinear_solve() |
| `mhs::sim::time_scheme` | `time_scheme/`                                                          | TimeScheme 抽象接口 + Bdf1Scheme / Bdf2Scheme / AdaptiveBdfScheme + TimeSchemeConfig / StepDecision / StepResult                                                                                           |
| `mhs::io`               | `io/`                                                                   | read_xml / write_vtu / write_xml                                                                                                                                                                           |
| `mhs::post`             | `postprocessor/`                                                        | interpolate_cell_to_node 及导出场函数 + sample_point 局部采样辅助                                                                                                                                          |
| `mhs::logger`           | `common/logger.*`                                                       | init / flush / panic + 模板 debug/info/warn/error                                                                                                                                                          |

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
- IO 数据结构 → `docs/design/io-model.md`
- 模块接口 → `docs/design/module-interfaces.md`
- expr 模块 → `docs/design/expr-api.md`
- 数据流与流程 → `docs/design/data-flow.md`
- 项目结构 / Logger / 命名空间 → `docs/design/project-structure.md`
- ADR 决策记录 → `docs/adr/0001-…0005`
