# 时间步进重构 — 完整实现路径

> **Status**: needs-triage
> **背景**: 当前 `mhs::sim` 仅支持硬编码 BDF1，固定步长，单 `T_prev` 缓冲。算法与组装耦合在 `assembler.cpp` 末段。无法支持 BDF2 / 自适应步长。
> **目标**: 把"瞬态项累入 A、b"从 `assembler.cpp` 抽到 `mhs::sim::time_scheme::TimeScheme`；配套 `mhs::core::TimeStepBuffer` 撑起 BDFk 的历史窗口；`Scheduler` 持有"算法+控制器"两个策略对象。
> **原则**: 不向后兼容。C++ 源代码层零兼容。`assemble()` 与 `T_prev` 一起删。任何"老接口保留→转发"的桥接代码一律不写。IO 输入层默认值策略不算兼容代码（缺 `<Scheme>` 时默认 BDF1）。
> **替代方案文档**: `docs/adr/0006-time-stepping.md`（记录架构与权衡）

---

## 1. 总览

10 个独立提交，每片按 red→green→refactor 走完。**禁止**跨切片"先实现后补测试"。

| #   | 提交                             | 关键产物                            | 红测试                                 | 关键风险              |
| --- | -------------------------------- | ----------------------------------- | -------------------------------------- | --------------------- |
| 0   | `TimeStepBuffer` 数据结构        | `mhs::core::TimeStepBuffer`         | `TimeStepBufferTest.*`                 | 环形索引边界          |
| 1   | 拆 `assemble()`                  | `assemble_static` + `assemble_mass` | `AssemblerTest.*`                      | Newton 内环调用方迁移 |
| 2   | `TimeScheme` 抽象 + `Bdf1Scheme` | `mhs::sim::time_scheme::*`          | `Bdf1SchemeTest.*`                     | 接口边界定不下来      |
| 3   | `Scheduler` 切换到 `TimeScheme`  | `Scheduler::run()` 主循环           | `SchedulerTest.Bdf1ReproducesLegacy*`  | 行为回归              |
| 4   | `Bdf2Scheme`                     | `Bdf2Scheme` + 起步降阶             | `Bdf2SchemeTest.*`                     | 起步降阶漏实现        |
| 5   | IO `TimeSchemeSpec`              | `IOStructure::TimeSchemeSpec`       | `IOStructureTest.*`                    | 老 XML 默认行为       |
| 6   | `AdaptiveBdfScheme` + 控制器     | `AdaptiveBdfScheme`                 | `AdaptiveBdfTest.*`                    | 控制器稳定性          |
| 7   | 输出时刻线性插值回退             | clamp + 线性插值                    | `AdaptiveBdfTest.LinearInterpFallback` | 浮点比较容差          |
| 8   | 集成测试 + 参考数据              | `cases/bdf2_transient_tests/`       | `run_cases.py` 通过                    | 数值基线              |
| 9   | 清理遗留                         | 删 `T_prev` / `assemble()`          | 全绿                                   | grep 漏掉引用方       |
| 10  | 文档更新                         | `CONTEXT.md` / `docs/design/*`      | n/a                                    | 表述与代码脱节        |

---

## 2. 详细切片

### 切片 0 — `TimeStepBuffer`（独立，无依赖）

**目标**：在 `mhs::core` 引入环形时间步历史缓冲。**不引 Eigen、不引 sim**。

**新建文件**：

- `src/data/time_step_buffer.hpp`
- `src/data/time_step_buffer.cpp`（内联实现也行；若全 header-only 也 OK）

**接口**（参考 `docs/adr/0006-time-stepping.md` §3.2.1）：

```cpp
namespace mhs::core {
class TimeStepBuffer {
public:
    explicit TimeStepBuffer(std::size_t cell_count, std::size_t capacity);
    void reset(const std::vector<double>& T_initial);
    void push(const std::vector<double>& T_new);
    const std::vector<double>& latest() const noexcept;
    const std::vector<double>& at(std::size_t i) const noexcept;
    double time_at(std::size_t i) const noexcept;
    double dt_to(std::size_t i) const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
private:
    std::vector<std::vector<double>> slots_;
    std::vector<double>              times_;
    std::size_t head_  = 0;
    std::size_t stored_ = 0;
    std::size_t cap_;
};
}
```

**修改**：

- `src/data/internal_model.hpp::GlobalState` 不变（buffer 在切片 2 引入到 GlobalState；本切片只独立验证 buffer）

**测试**（红→绿）：

- `tests/test_time_step_buffer.cpp`（新文件）
    - `PushThenLatest`：push 一次，`latest()` 等于 push 的 T
    - `AtRelative`：push 两次，`at(0)==T_2`、`at(1)==T_1`
    - `WrapAround`：capacity=3，push 5 次，`at(0)` 是第 5 次、`at(2)` 是第 3 次
    - `TimeAtAndDtTo`：push 时配 time；`time_at(0)` 是最新 time；`dt_to(1)` = `time_at(0) - time_at(1)`
    - `Reset`：reset 后 `size()==1`、`latest()==reset 值`
    - `EmptyBuffer`：构造后 `latest()` 行为？决策：构造时 `size()==0`；`latest()` 抛或返回空引用？**提案**：构造后未 push 时 `latest()` 返回空引用 + UB 警告，文档明示"必须 push 一次后才能 latest()"

**验证命令**：

```bash
conda activate cpp_env
python run_tests.py
```

预期：所有 TimeStepBuffer 测试通过，其它测试无变化。

---

### 切片 1 — 拆 `assemble()`

**目标**：把 `Assembler::assemble()` 拆成 `assemble_static()` + `assemble_mass()`。**删除**原 `assemble()`。

**修改**：

- `src/assembler/assembler.hpp`
    - 删除 `assemble()` 声明
    - 新增 `assemble_static()`、`assemble_mass()` 声明
- `src/assembler/assembler.cpp`
    - 删除 `assemble()` 实现
    - 把原 assemble 末段的瞬态项分支（`study_type==Transient && dt>0`）**整段删除**（不是"暂时挪到外面"，是直接删）
    - 实现 `assemble_static()`：原 assemble 的非瞬态部分
    - 实现 `assemble_mass()`：返回 `Eigen::VectorXd` 每单元 ρc·vol（lumped 对角）
    - 引入新结构体 `StaticOpsResult { K, f_static }` 和 `MassOpsResult { M_diag }`
- `src/nonlinear/nonlinear_solver.cpp`
    - `nonlinear_solve` **不**直接调 `assemble()`；改为接收外部传入的 `LinearSystem`（或继续自构但 `nonlinear_solve` 持有 `assembler` 用于装配？见 1.1 决策）

**1.1 关键决策**：`nonlinear_solve` 是否保留自构 LinearSystem 的能力？

- **选项 A**：`nonlinear_solve` 仅接收 `LinearSystem`（不自构）。`Scheduler` 在 Newton 每次迭代前调 `assembler.assemble_static + assemble_mass + scheme.build_system`。
- **选项 B**：`nonlinear_solve` 仍持 `assembler`，但通过"装配策略"回调生成 LinearSystem。
- **提案**：选 A。最简单，依赖方向最干净。`nonlinear_solve` 不再知道"如何装配"，只负责 Newton/Anderson 外层循环。

**测试**（红→绿）：

- `tests/test_assembler.cpp`（修改）
    - **先删** `AssembleReturnsCorrectSize` 等旧测试
    - 新增 `AssembleStaticReturnsKAndFStatic`：调用 `assemble_static`，验证 `K` 大小正确、`f_static` 大小正确
    - 新增 `AssembleStaticHasNoTransient`：**关键** —— 验证 `K(c,c)` 不含 `ρc·vol/dt` 项（即稳态 / 瞬态调用结果一致）
    - 新增 `AssembleMassReturnsDiag`：调用 `assemble_mass`，验证 `M_diag(c) == ρc·vol(c)`
    - 新增 `AssembleStaticReadsTemperature`（暂留作选测）：用不同 T 输入，`K` 应变化（材料非线性）

**编译错暴露**：`nonlinear_solver.cpp` 还在调 `assembler.assemble()` → 编译失败 → 在切片 3 修复（切片 1 不修复 `nonlinear_solver`）。**注**：这一步会让 build 红。

**变通**：切片 1 提交时**只**改 `assembler.hpp` 暴露新接口，**暂留**旧 `assemble()` 作为 deprecated 但可用的桥接。切片 9 才删。**这是唯一允许的"分阶段删除"窗口**——为防止 build 长红，跨切片提交可以暂留。

> 等等。原则是"零兼容"。`assemble()` 直接删。但 build 会红到切片 3 吗？
> **是的**。切片 1 → 2 → 3 是连续的小切片，CI 短红是允许的（feature branch）。**最终切片 3 提交时 build 转绿**。如果团队约定"main 必须绿"，则切片 1 暂留旧接口是务实折中。
>
> **决策**：切片 1 删 `assemble()`。Feature branch 允许短红。切片 3 同步把 `nonlinear_solver.cpp` 迁移到新接口。如果中间其它人拉取会破，那就在 `WIP/` 分支做。
>
> **替代方案**：切片 1 删 `assemble()` 但在 `assembler.cpp` 内**先留个 stub** 抛 `std::runtime_error("moved to assemble_static + assemble_mass; see TimeScheme")`。这不算"兼容代码"，只是个清晰错误信息。**不**调用旧逻辑。**不**返回旧 LinearSystem。
>
> **最终决策**：切片 1 直接删 `assemble()`。`nonlinear_solver.cpp` 立即改为调用 `assemble_static` + `assemble_mass` + 临时 `LinearSystem` 拼装（**仅切片 1 的临时拼装**）。切片 2 引入 `TimeScheme` 后，把拼装逻辑挪到 `Bdf1Scheme::build_system()`，切片 1 的临时拼装也删除。
>
> 这意味着切片 1 的 `nonlinear_solve` 内部包含"BDF1 拼装"代码，**仅做切片 1 的过渡**。它是"业务代码"而非"兼容桥接"——因为它用的是新接口（assemble_static / assemble_mass），不是转发给旧接口。

**验证命令**：

```bash
cmake --build build --parallel
python run_tests.py
```

预期：所有 gtest 绿；老的 `assemble()` 调用已迁移。

---

### 切片 2 — `TimeScheme` 抽象 + `Bdf1Scheme`

**目标**：建立算法抽象与第一个实现。

**新建**：

- `src/time_scheme/time_scheme.hpp`
    - `enum class TimeSchemeKind { Bdf1, Bdf2, AdaptiveBdf }`
    - `struct TimeSchemeConfig`
    - `struct StepDecision { double dt; std::size_t order; }`
    - `enum class AcceptDecision { Accept, Reject }`
    - `class TimeScheme` 抽象接口
    - `class Bdf1Scheme` 实现
    - `class StaticSchemeFactory::create(const TimeSchemeConfig&)` 工厂
- `src/time_scheme/time_scheme.cpp`
- `src/time_scheme/bdf1_scheme.cpp`（或合并）
- `tests/test_time_scheme.cpp`
- `tests/test_bdf1_scheme.cpp`

**修改**：

- `src/data/internal_model.hpp::GlobalState`
    - 删除 `T_prev`
    - 加 `TimeStepBuffer history;`（容量 = `TimeSchemeConfig::max_order`）
    - 加 `int output_step = 0;`
- `src/scheduler/scheduler.hpp`
    - 加 `#include "time_scheme/time_scheme.hpp"`
    - 加 `std::unique_ptr<time_scheme::TimeScheme> scheme_;`
    - 加 `time_scheme::TimeSchemeConfig scheme_cfg_;`
    - 加 `void setTimeSchemeConfig(const TimeSchemeConfig&);`
    - 加 `std::unique_ptr<TimeScheme> schemeFactory(const TimeSchemeConfig&);`
- `src/scheduler/scheduler.cpp`
    - 在 `run()` 初始化时按 `scheme_cfg_.max_order` 构造 history
    - 临时不调 `scheme_`（切片 3 才调）—— 本切片只验证 TimeScheme 抽象独立可用

**关键不变量（写在 GlobalState 注释）**：

- `T == history.latest()`；由 `Scheduler::accept` 时同步更新
- `T_prev` 删除；改用 `history.at(1)`

**测试**（红→绿）：

- `test_time_scheme.cpp`
    - `Bdf1SchemeSelectStepReturnsInitialDt`：第一次 select 给 initial_dt
    - `Bdf1SchemeBuildSystemCoefficient`：对单元 cell c，调 `build_system`，验证 `A(c,c) == K(c,c) + M_diag(c)/dt`、`b(c) == f_static(c) + M_diag(c) * T_history.latest()[c] / dt`
    - `Bdf1SchemeAcceptOrRejectAlwaysAccepts`：固定步长无拒绝
- `test_bdf1_scheme.cpp`
    - `MatchesLegacyAt1ms`：**关键** —— 与 `cases/simple_transient_tests/case1.xml` 的旧数值在 `tol=1e-9` 内一致
        - 实现方法：在 `Bdf1SchemeTest.MatchesLegacyAt1ms` 中，**直接对比**用旧 assemble 路径（如果还可用）与新 Bdf1Scheme 路径的输出；若旧路径已删，则对比 `run_cases.py` 历史数据
    - 起步用例：`Initialize` 后 `history.size()==1`

**验证命令**：

```bash
python run_tests.py
```

预期：TimeScheme 单元测试绿。

---

### 切片 3 — `Scheduler` 切换到 `TimeScheme`

**目标**：`Scheduler::run()` 不再调 `assembler.assemble()`；按 TimeScheme 的接口构建方程。

**修改**：

- `src/scheduler/scheduler.cpp::run()`
    - 删除：`state_.T_prev = state_.T`
    - 删除：原来的固定 dt 循环
    - 改为：调用 `scheme_->initialize()`，进入 `while (t < duration)` 循环
    - 主循环内：
    1. `scheme_->select_step(state_.history, t)` → `(dt_internal, order)`
    2. clamp `dt_internal` 到 `min(dt_internal, t_end - t)`
    3. clamp `dt_internal` 到 `min(dt_internal, t_next_out - t)`（output_dt>0 时）
    4. `auto K_fs = assembler.assemble_static(state_)`
    5. `auto M    = assembler.assemble_mass(state_)`
    6. `auto ls   = scheme_->build_system(K_fs, M, state_.history, order, dt_internal)`
    7. `nonlinear_solve_with_external_ls(ls, state_, *solver_)` —— `nonlinear_solve` 的新签名
    - 写探针：仅在 output_step 推进时
- `src/nonlinear/nonlinear_solver.hpp`
    - 新增 `nonlinear_solve_with_external_ls(const LinearSystem&, ...)`；保留旧 `nonlinear_solve` 作为 deprecated（切片 9 删）
    - 或：直接改 `nonlinear_solve` 签名（旧调用方只剩 `assembler` 内的旧拼装；切片 1 已删除）

**3.1 关键决策**：`nonlinear_solve` 签名怎么改？

- **方案 1（推荐）**：保留 `nonlinear_solve(model, state, solver)` 但内部不再自构 ls —— 改为在 `Scheduler` 主循环内每次 Newton 迭代前 `assemble_static + assemble_mass + scheme.build_system`，把 ls 作为参数传入
- **方案 2**：把 `nonlinear_solve` 拆成 `newton_iterate(ls_provider, state, solver)`，ls_provider 是一个"每 Newton 迭代重新生成 ls"的回调
- **提案**：方案 1。`nonlinear_solve` 签名变为 `nonlinear_solve(const LinearSystem& initial_ls, GlobalState&, LinearSolver&, ...)`. Newton 第一次用 `initial_ls`，后续迭代用同样公式（重算 ls）—— **这点需要选 A 还是选 B**：
    - **A. 每次 Newton 重算 ls**（最简单，性能略差）
    - **B. 第一次 Newton 算 ls，后续冻结 ls**（Frozen operator，仿 CFD 实践；性能好但破坏 K=f(T) 的耦合）
    - **提案**：A。第一版正确优先。

**3.2 关键决策**：`state.dt` 与 `history.time_at(0) - history.time_at(1)` 的关系？

- `state.dt` 是"最近一次推进的 dt"；history 是历史窗口
- 切片 3 起 `state.dt` 由 `scheme_->select_step` 返回的 dt 决定；不再写死

**测试**（红→绿）：

- `test_scheduler.cpp`（修改）
    - **新增** `Bdf1ReproducesLegacyTransientCase1`：在 `cases/simple_transient_tests/case1.xml` 上跑，输出与现状**逐位**一致
    - **新增** `TransientWithoutTimeSchemeThrows` 或 graceful degrade？提案：若无 `scheme_`，`run()` 报 fatal 并 `MHS_LOG_ERROR`
    - 修改现有 `SchedulerTest.*`（如果它们调过 `T_prev`）
- `run_cases.py`：
    - 在切换前后跑同一 case，确认 `compare_lib.py` 返回 0

**验证命令**：

```bash
python run_tests.py
python run_cases.py
```

预期：`cases/simple_transient_tests/case1.xml` 数值基线维持。

---

### 切片 4 — `Bdf2Scheme` + 起步降阶

**目标**：固定步长 BDF2；起步时降阶到 BDF1（`history.size()==1` 时）。

**新建**：

- `src/time_scheme/bdf2_scheme.hpp` / `.cpp`

**公式**（变步长/固定步长统一）：

```text
h_n = t_n - t_{n-1}
h_{n-1} = t_{n-1} - t_{n-2}
δ = h_n / h_{n-1}
α0 = (1 + 2δ) / (h_n·(1+δ))
α1 = -(1+δ) / (h_n·δ)
α2 = δ / (h_n·(1+δ))
A = α0·M + K
b = α0·M·T_n + α1·M·T_{n-1} + α2·M·T_{n-2} + f_static
```

**固定步长退化**（δ=1）：`α0=3/(2h)`, `α1=-2/h`, `α2=1/(2h)`，与教科书一致。

**起步逻辑**：

- `select_step`：若 `history.size() < 2`，返回 `order=1`；否则 `order=2`
- `build_system`：根据传入的 `order` 走 BDF1 或 BDF2

**测试**（红→绿）：

- `test_bdf2_scheme.cpp`
    - `CoefficientsFixedStep`：固定 h 调 build_system，验证 `A(c,c) == K(c,c) + α0·M_diag(c)`、`b(c) == f_static(c) + α0·M_diag(c)·T_n + α1·M_diag(c)·T_{n-1} + α2·M_diag(c)·T_{n-2}`
    - `CoefficientsVariableStep`：手动设 δ=2，验证 α 与公式一致
    - `StartsAsOrder1`：**关键** —— `history.size()==1` 时 `select_step` 返回 `order=1`；build_system 按 BDF1 拼
    - `HandlesTwoStepHistory`：`history.size()==3` 时 BDF2 正常返回
    - `Bdf2ConvergesToAnalytic`：1D 棒材、初值温阶跃、两端 T=0、无热源；解析解为 Fourier 级数；取 `t_end` 处中心点温度，比较 `dt=0.1, 0.05, 0.025` 的对数斜率 → 接近 2

**验证命令**：

```bash
python run_tests.py
```

---

### 切片 5 — IO `TimeSchemeSpec`

**目标**：XML 配置支持 `<Scheme>` 节点；老 XML 缺节点时默认 BDF1。

**修改**：

- `src/data/io_model.hpp`
    - 新增 `enum class TimeSchemeKind { Bdf1, Bdf2, AdaptiveBdf }`
    - 新增 `struct TimeSchemeSpec { kind; initial_dt; min_dt; max_dt; abs_tol; rel_tol; max_order; output_dt; }`
    - `IOStructure` 加 `TimeSchemeSpec time_scheme;` 字段
- `src/io/io.cpp`（XML 解析）
    - 解析新 `<Transient>` 节点下的 `<Scheme>`、`<InitialDt>`、`<MinDt>`、`<MaxDt>`、`<AbsTol>`、`<RelTol>`、`<MaxOrder>`、`<OutputDt>`
    - 缺节点时：`kind=Bdf1`、`initial_dt=transient_time_step`、`output_dt=transient_duration`（每"duration 一次"）
- `src/preprocessor/preprocessor.cpp`
    - 把 `IOStructure::time_scheme` 翻译为 `InternalModel::time_scheme`
    - 新增 `InternalModel::TimeSchemeSpec time_scheme;` 字段
- `src/data/internal_model.hpp`
    - 加 `TimeSchemeSpec time_scheme;`
    - 保留 `transient_time_step`、`transient_duration`（**仅作为 IO 输出**读取用，scheduler 不依赖）

**5.1 关键决策**：`transient_time_step` 字段保留吗？

- **是**。但 scheduler 不再读它。
- 保留原因：现有 IO 输出可能引用；`run_cases.py` 的 `compare_lib.py` 也可能读 case XML。
- **新加注释**明确标记 deprecated on algorithm side, kept for IO compat only.

**测试**（红→绿）：

- `test_io.cpp`（修改）
    - `SchemeDefaultsToBdf1`：老 XML（无 `<Scheme>`）解析后 `time_scheme.kind == Bdf1`
    - `ParsesSchemeNode`：新 XML `<Scheme>AdaptiveBdf</Scheme>` 解析正确
    - `ParsesAllKnobs`：所有新节点解析正确
    - `OutputDtDefaultsToDuration`：`output_dt` 缺省时 = `transient_duration`

**验证命令**：

```bash
python run_tests.py
python run_cases.py  # 验证老 XML 仍跑通
```

---

### 切片 6 — `AdaptiveBdfScheme` + 控制器

**目标**：自适应阶变步长 BDF。

**新建**：

- `src/time_scheme/adaptive_bdf_scheme.hpp` / `.cpp`
- `src/time_scheme/step_controller.hpp` / `.cpp`（控制器可独立单元测试）

**控制器逻辑**（参考 Hairer-Norsett-Wanner 卷 II §V）：

1. 用 order k 与 order k-1 两次预测的差 `e = T^{(k)} - T^{(k-1)}` 作为局部截断误差估计
2. 决策：
   - `||e||_∞ ≤ tol` → 接受；`order_new = min(k+1, max_order)`，`dt_new = safety · dt · (tol/||e||)^{1/(k+1)}`
   - `||e|| > tol` → 拒绝；`order_new = k`（保持或降 1），`dt_new = safety · dt · (tol/||e||)^{1/k}`，clamp 到 `[min_dt, max_dt]`

**δ 范围约束**：默认 `0.5 ≤ δ ≤ 2.0` 软约束；超出时按公式自动衰减（不必显式拒绝）。

**输出时刻对齐**（本切片**只**做 clamp，**不做**插值回退——回退放切片 7）：

- 维护 `t_next_output = output_step · output_dt`
- 每步确定 `dt_internal` 后，clamp 到 `dt_internal = min(dt_internal, t_next_out - t_current)`
- 当 `t ≈ t_next_output`（容差 `1e-9·max(1, t)`）时：写探针 + `output_step++` + `t_next_output += output_dt`

**测试**（红→绿）：

- `test_adaptive_bdf_scheme.cpp`
    - `ShrinksOnLargeError`：构造 T 使得 k=2 预测与 k=1 预测差大 → `accept_or_reject == Reject`
    - `GrowsOnSmallError`：构造 T 使得误差小 → `accept_or_reject == Accept` + 下次 select_step dt 增大
    - `ClampsToMinDt`：`dt_too_small` 情况 clamp 到 `min_dt` 并终止（如触发"硬卡步长"则记录事件）
    - `ClampsToMaxDt`：同上 `max_dt`
    - `StepLogRecordsEverything`：step_log 字段记录每步的 (t, dt, order, verdict)
- `test_step_controller.cpp`
    - 单元测试控制器对误差 dt 选择的对数斜率正确

**验证命令**：

```bash
python run_tests.py
```

---

### 切片 7 — 输出时刻线性插值回退

**目标**：当控制器把 dt clamp 到 `min_dt` 仍不足以命中 t_out 时，回退到线性插值。

**修改**：

- `src/scheduler/scheduler.cpp::run()`
    - 跟踪 `t_last` 与 `T_at_t_last`（上一步的 t 与 T）
    - 当 `t_next_out` 在 `(t_last, t_last + dt_internal]` 区间内时：
        - 计算插值 `T(t_out) = T_last + (T_new - T_last) · (t_out - t_last) / dt_internal`
        - 在 `t_out` 写探针与快照

**测试**（红→绿）：

- `test_adaptive_bdf_scheme.cpp`（补充）
    - `LinearInterpFallback`：构造一个 case，`min_dt > output_dt` 强制线性插值路径
    - `ProbeRecordedAtOutputTimesOnly`：构造 3 个 output_time，验证探针 times 长度 == 3
    - `OutputTimesAreExact`：验证探针 times 等于 `[0, output_dt, 2·output_dt, …, < duration]`

**验证命令**：

```bash
python run_tests.py
```

---

### 切片 8 — 集成测试 + 参考数据

**目标**：真实 case XML 跑通自适应 BDF；签入参考数据。

**新建**：

- `cases/bdf2_transient_tests/case1.xml` —— 简单瞬态，`<Scheme>Bdf2</Scheme>`
- `cases/adaptive_transient_tests/case1.xml` —— 边界温阶跃，触发自适应加密
- `cases/adaptive_transient_tests/case2.xml` —— 稳态附近，触发自适应放大
- `cases/adaptive_transient_tests/case3.xml` —— `output_dt` 与内部 dt 互不整除
- `cases/adaptive_transient_tests/expected/` 目录存首次运行生成的参考

**修改**：

- `scripts/compare_lib.py`（如果新 case 需不同容差）
- `run_cases.py`（添加新 case 目录到 `CASE_GROUPS`）

**测试**（红→绿）：

- 跑 `python run_cases.py`，所有 case 通过 `compare_lib.py`
- 首次生成参考：`python run_cases.py` + 手动 `git add` 参考
- 第二轮：删 build + 重跑，确认仍通过

**验证命令**：

```bash
conda activate cpp_env  # 测例运行
cmake --build build --parallel
python run_cases.py
```

---

### 切片 9 — 清理遗留

**目标**：删除 `T_prev`、删除旧 `assemble()`、删除 `nonlinear_solve` 旧签名、删除 deprecated 标记。

**修改**：

- `src/data/internal_model.hpp::GlobalState`
    - 删除 `T_prev` 字段
- `src/assembler/assembler.hpp` / `.cpp`
    - 确认 `assemble()` 早已删除（切片 1 完成）
    - 清理 `assemble()` 任何残留引用
- `src/nonlinear/nonlinear_solver.hpp` / `.cpp`
    - 删除旧 `nonlinear_solve(model, state, solver)` 签名
    - 只保留接收 `LinearSystem` 的版本
- 全代码库 grep `T_prev` 找残留
- 全代码库 grep `assemble(state)` 找残留

**测试**：所有测试仍绿。

**验证命令**：

```bash
# 找残留
grep -rn "T_prev" src/ tests/
grep -rn "assembler.assemble(" src/ tests/  # 不应有匹配
python run_tests.py
python run_cases.py
```

---

### 切片 10 — 文档更新

**目标**：反映新架构；标记 deprecated 字段。

**修改**：

- `CONTEXT.md`
    - "求解流程" 章节：更新 `Scheduler::run()` 流程图，加入 `TimeScheme` 与 `TimeStepBuffer`
    - "关键设计原则" 加一条：算法与组装解耦
- `docs/design/module-interfaces.md`
    - 更新 `assembler` 段：暴露 `assemble_static` + `assemble_mass`
    - 新增 `time_scheme` 段：TimeScheme、Bdf1Scheme、Bdf2Scheme、AdaptiveBdfScheme
- `docs/design/data-flow.md`
    - 更新"求解"段：主循环走 TimeScheme
- `docs/adr/0006-time-stepping.md`
    - 状态从 "Proposed" 改为 "Accepted"
- `docs/design/internal-model.md`
    - `GlobalState` 字段更新（`T_prev` → `history`）
- `src/data/internal_model.hpp::GlobalState` 注释
    - 明确 `transient_time_step`、`transient_duration` 仅供 IO 输出

**验证**：文档与代码一致。

---

## 3. 通用规则

### 3.1 TDD 纪律

- 每个 commit 前**先写红测试**。红测试的失败信息要贴进 commit message
- 实现后再变绿
- 重构 commit（不改行为，只改结构）单独提交

### 3.2 命名空间边界

- `mhs::core::TimeStepBuffer`（无 Eigen 依赖）
- `mhs::sim::time_scheme::TimeScheme`、`Bdf1Scheme`、`Bdf2Scheme`、`AdaptiveBdfScheme`、`TimeSchemeConfig`
- `mhs::sim::assembler::StaticOpsResult`、`MassOpsResult`、`Assembler`

### 3.3 不写兼容代码

- 切片 1 临时 `nonlinear_solve` 内拼装 BDF1：不是兼容桥接，因为它用的是新接口（`assemble_static` + `assemble_mass`），不是转发给旧 `assemble()`
- 切片 9 必须全删；不允许"留 deprecated 标记"
- 老 XML 缺 `<Scheme>` → 默认 BDF1：这是输入层默认值策略，不是兼容代码

### 3.4 性能取舍

- 第一版允许每次 Newton 重算 K/M（性能退化）—— 正确优先
- "Frozen operator" 优化放后续 PR

### 3.5 错误处理

- 控制器拒绝超过 `max_internal_steps` 次未收敛 → 终止 + `state.diverged = true` + `MHS_LOG_ERROR`
- dt 触 `min_dt` 下界仍超容差 → 终止 + 记录"硬卡步长"事件
- `t_next_out` 与 `t` 浮点不一致 → 用 `1e-9·max(1, t)` 容差比较

---

## 4. 关键里程碑

| 切片  | 验证                                                  | 阻塞/非阻塞          |
| ----- | ----------------------------------------------------- | -------------------- |
| 0 完  | `test_time_step_buffer.cpp` 绿                        | 非阻塞               |
| 1 完  | `test_assembler.cpp` 绿；老 case 数值不变             | **阻塞后续**         |
| 2 完  | `test_time_scheme.cpp` 绿                             | 非阻塞               |
| 3 完  | `cases/simple_transient_tests/case1.xml` 数值基线维持 | **阻塞后续**         |
| 4 完  | `test_bdf2_scheme.cpp` 绿；BDF2 收敛阶 ≈ 2            | 非阻塞               |
| 5 完  | 老 XML 仍跑通；新 XML 解析正确                        | 非阻塞               |
| 6 完  | `test_adaptive_bdf_scheme.cpp` 绿                     | 非阻塞               |
| 7 完  | 探针 times == 规定 t_out 序列                         | 非阻塞               |
| 8 完  | `run_cases.py` 全部通过                               | 非阻塞               |
| 9 完  | 全代码 grep 干净；测试全绿                            | **合并主分支前必做** |
| 10 完 | 文档与代码一致                                        | 合并前必做           |

---

## 5. 切片依赖图

```text
0 (TimeStepBuffer)
 │
1 (拆 assemble)
 │
2 (TimeScheme + Bdf1)
 │   ↘
3 (Scheduler 切换) ←── 阻塞合并
 │
4 (Bdf2)
 │
5 (IO)
 │
6 (AdaptiveBdf)
 │
7 (线性插值)
 │
8 (集成测试 + 参考)
 │
9 (清理)
 │
10 (文档)
```

0 → 1 → 2 → 3 必须**串行**（每步 build 需绿才能进下一步）。
4 → 5 → 6 → 7 可以**局部并行**（只改不同文件），但建议串行便于 review。
8 → 9 → 10 严格串行。

---

## 6. 风险登记

| 风险                                                      | 触发          | 对策                                     |
| --------------------------------------------------------- | ------------- | ---------------------------------------- |
| Newton 内环每步重算 K/M → 性能退化                        | 切片 1–9 全程 | 接受；后续加 `CachePolicy` 单独 PR       |
| BDF2 起步降阶漏实现 → 启动震荡                            | 切片 4        | `Bdf2SchemeTest.StartsAsOrder1` 强制     |
| 自适应 dt 与 output_dt 不同步 → 探针缺帧                  | 切片 6        | clamp + 切片 7 线性插值回退              |
| `T_prev` 删除后 grep 漏掉引用方                           | 切片 9        | 编译失败即查；`nonlinear_solve` 已不依赖 |
| IO `transient_time_step` 与 `time_scheme.initial_dt` 重复 | 切片 5        | grep + 注释"两个字段同义, IO 层别名"     |
| 旧 `test_assembler.cpp` 直接调 `assemble()` 编译不过      | 切片 1        | 切片 1 同步改测试；非"兼容代码"          |
| BDF2 变步长 δ 极端 → 系数失稳                             | 切片 4        | 控制器 δ 范围软约束                      |
| 切 0 实现的环形索引 off-by-one                            | 切片 0        | `WrapAround` 测试覆盖                    |
