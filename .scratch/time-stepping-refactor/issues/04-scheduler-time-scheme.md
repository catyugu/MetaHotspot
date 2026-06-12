# 切片 3 — `Scheduler` 切换到 `TimeScheme`

> **Status**: needs-triage
> **依赖**: 切片 0、1、2
> **阻塞合并**: 是（行为基线必须维持）

## 目标

`Scheduler::run()` 不再调 `assembler.assemble()`；按 TimeScheme 的接口构建方程。`nonlinear_solve` 接收外部 `LinearSystem`。

## 修改

### `src/scheduler/scheduler.cpp::run()`

- 删除 `state_.T_prev = state_.T`
- 改为 TimeScheme 主循环
- 主循环：
  1. `scheme_->select_step(history, t)` → `(dt, order)`
  2. clamp `dt` 到 `min(dt, t_end - t)`
  3. clamp `dt` 到 `min(dt, t_next_out - t)`（output_dt>0）
  4. `assembler.assemble_static(state_)` → K + f_static
  5. `assembler.assemble_mass(state_)` → M
  6. `scheme_->build_system(K, f_static, M, history, order, dt)` → LinearSystem
  7. `nonlinear_solve(ls, state_, *solver_)`
- 写探针：仅在 output_step 推进时

### `src/nonlinear/nonlinear_solver.hpp` / `.cpp`

- `nonlinear_solve` 签名改为接收 `LinearSystem` 而非自构
- Newton 内每次迭代前重新 `build_system`（**选 A：每次 Newton 重算 ls**，第一版正确优先）
- 切片 1 的临时拼装代码删除（被 `Bdf1Scheme::build_system` 替代）

### 关于 `state.dt`

- `state.dt` 不再是 Scheduler 写死的标量；由 `scheme_->select_step` 返回
- 切片 3 起 `state.dt` 由主循环控制

## 测试

### `tests/test_scheduler.cpp`

- **新增** `Bdf1ReproducesLegacyTransientCase1`：**关键** —— 在 `cases/simple_transient_tests/case1.xml` 上跑，输出与现状逐位一致
- `TransientWithoutTimeSchemeFails`：无 scheme 时 `run()` 报错
- 修改/删除依赖 `T_prev` 的旧测试

## 验证

```bash
cmake --build build --parallel
python run_tests.py
python run_cases.py  # 关键：case1.xml 数值基线
```
