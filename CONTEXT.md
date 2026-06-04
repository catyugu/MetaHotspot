# MetaHotspot Context

热仿真引擎：电子封装多层堆叠结构的三维有限体积热分析。**结构化网格**，**单元中心 DOF**。

> 入口文档。要点索引；详情见 `docs/design/*` 与 `docs/adr/*`。

---

## 求解类型

- **Steady**: `StudyType::Steady`。视为 t=0 的单次非线性迭代，scheduler 不做时间循环。
- **Transient**: `StudyType::Transient`。从 t=0 起按 `transient_time_step` 推进，每步 `T_prev = T` + `nonlinear::solve()`。

## 网格

结构化 3D `nx × ny × nz`。**不支持 Dimension2D**（panic）。每个单元存温度 DOF 在中心；BC 走面积分，无面 DOF（ADR-0002）。

## 边界条件（cell-level，ADR-0005）

| 类型       | 数学                    | 离散处理                        |
| ---------- | ----------------------- | ------------------------------- |
| FirstType  | `T = T₀`                | 鬼单元 `T_ghost = 2·T₀ − T_b`   |
| SecondType | `-k ∂T/∂n = q₀`         | 累入 RHS：`Σ q·A_face`          |
| ThirdType  | `-k ∂T/∂n = h(T − T_∞)` | 对角 `h·A` 系数 + RHS `h·A·T_∞` |

`other_bc` 在预处理阶段填到所有未显式指定的面 + 虚拟邻居面。

## 表达式（ADR-0004）

两种独立路径：

- **几何** — `mhs::expr::eval_geometry()`，依赖已注册的命名变量（`w_top`、`h_middle` 等，SI 米）
- **场 / BC / 热源** — `mhs::expr::parse()`，上下文 `{x, y, z, T, t}`，返回轻量句柄 `mhs::CompiledExpression`

`FieldContext` / `FieldEvaluator` / `CompiledExpression` **定义在 `mhs::expr`（`src/expr/expr.hpp`）**，`mhs::*` 是 `src/common/types.hpp` 的 `using` 重导出。依赖方向 `common → expr`，**从不超过此方向**。

**线程模型**：注册表变动互斥锁；`parse()` 主线程持锁试编译；`eval()` **无锁** — TBB ETS 包装，每个工作线程懒构造独立 ExprTK AST。

复杂形式用 `register_native(name, FieldEvaluator)` 注册 C++ 函数字段。`IOStructure.functions` 是当前 native 入口。

## 求解流程

```text
XML → io::read_xml → IOStructure
  → Preprocessor::load → InternalModel
    → Scheduler::run
        └─ Assembler::assemble(state)   [tbb::parallel_for + ETS, 锁无关]
            → nonlinear::solve()        [Anderson 加速定点迭代]
                → Solver::solve(A, b)   [SparseLU / BiCGSTAB]
        → Postprocessor::interpolate_cell_to_node
            → io::write_vtu + io::write_xml
```

时间状态（`current_time` / `time_step` / `dt`）存于 `GlobalState`，不在 `Scheduler` 私有成员。

## 关键设计原则

1. 内部模型不含原始字符串 — 表达式预编译为 `CompiledExpression`
2. 热源字典化 — `heat_source_table`（去重）+ 每单元 `uint16_t` 索引
3. Cell-level BC — 每单元存 6 面 BC（`CellBC`）
4. Precomputed sparsity — 组装只填值，不重建结构
5. Backward Euler — 瞬态项 `ρc·vol/dt·(T − T_prev)`，θ=1.0
6. TBB 并行组装 — 跳虚拟单元，`enumerable_thread_specific<ThreadLocalData>` + 合并
7. 域类型定义在 `src/common/types.hpp`（重导出 expr 类型）— 内部枚举 `StudyType` / `BcType` / `FaceDir` 的唯一真源
8. 无虚函数（`Solver` 除外）；无异常（仅 `bin/main.cpp` 边界 try/catch 捕获 std::exception → `panic`）
9. POD / 纯函数优先

## 命名空间速查

| 命名空间            |     | 暴露类型 / 函数                                                            |
| ------------------- | --- | -------------------------------------------------------------------------- |
| `mhs`               |     | Preprocessor、Solver、Scheduler、Postprocessor、IOStructure、InternalModel |
| `mhs::io`           |     | `read_xml` / `write_vtu` / `write_xml`                                     |
| `mhs::expr`         |     | `CompiledExpression` / `parse` / 注册表                                    |
| `mhs::preprocessor` |     | 自由函数 `resolve_*` / `parse_face_key` / `point_in_face_rects`            |
| `mhs::assembler`    |     | `Assembler` / `LinearSystem`                                               |
| `mhs::nonlinear`    |     | `solve()` / `NonLinearConfig` / `NonLinearResult`                          |
| `mhs::logger`       |     | `init` / `flush` / `panic` + 模板 debug/info/warn/error                    |

> `solver` / `scheduler` / `postprocessor` 没有独立子命名空间 — 类型直接挂在 `mhs::`。

## 术语表

| 概念            | 中文     | 说明                                                                  |
| --------------- | -------- | --------------------------------------------------------------------- |
| StudyType       | 求解类型 | Steady / Transient                                                    |
| Layer           | 层       | 多 Block 的 Z 厚度堆叠                                                |
| Block           | 块       | XY 平面 add/sub Rect 几何；Z 范围继承父层                             |
| Rect            | 矩形     | 块几何的 add/sub 单元                                                 |
| FaceKey         | 面键     | 字符串 `Face\|Direction\|CoordValue\|RectList`，CoordValue 是空间坐标 |
| Material        | 材料     | 含 k / ρ / c（均为字符串表达式）                                      |
| BC / `other_bc` | 边界条件 | 三种类型 + 默认兜底                                                   |
| Daore Xishu     | 导热系数 | k, W/(m·K)                                                            |
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
