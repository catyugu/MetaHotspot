# ADR-0001: Transient-First Architecture

## Status

Accepted

## Context

Cases include both steady and transient studies. CLAUDE.md mandates treating all problems as nonlinear natively.

## Decision

The whole system is designed for transient simulation. Steady state is a single nonlinear solve at `t = 0`.

- `Scheduler::run()` 根据 `InternalModel::study_type` 分支：`Steady` 跳过时间循环，调用一次 `mhs::sim::nonlinear_solve()`；`Transient` 进入时间步循环至 `current_time >= transient_duration`。
- 瞬态使用 `mhs::sim::time_scheme::StepController`（策略模式）结合 `build_system` 纯函数和 `estimate_error` 纯函数进行步长控制、LTE 估计和时间输出。
- Nonlinear iteration lives inside each time step.
- `GlobalState` 始终携带 `T`、`accepted`（`SolutionHistory`）和 `dt`，以支持未来时间导数项。

## Notes

- **Steady evaluation context**: when `study_type == Steady`, expressions are evaluated with `t = 0`. Steady means equilibrium, not time advancing.
- **Steady behavior**: `Scheduler::run()` 在 `study_type == Steady` 分支下跳过时间循环，仅对初始 `T = initial_temperature` 调用一次 `mhs::sim::nonlinear_solve()` 至收敛。
- **Transient behavior**: 标准时间步进 `t₀ → t₁ → … → t_end`，每步 `assemble → build_system → nonlinear_solve → evaluate_step`，接受后 `accepted.accept(T, t)`，收敛后 `current_time += dt`。
