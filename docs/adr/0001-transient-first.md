# ADR-0001: Transient-First Architecture

## Status

Accepted

## Context

Cases include both steady and transient studies. CLAUDE.md mandates treating all problems as nonlinear natively.

## Decision

The whole system is designed for transient simulation. Steady state is a single nonlinear solve at `t = 0`.

- `Scheduler::run()` 根据 `InternalModel::study_type` 分支：`Steady` 跳过时间循环，调用一次 `mhs::sim::nonlinear_solve()`；`Transient` 进入时间步循环至 `current_time >= transient_duration`。
- Nonlinear iteration lives inside each time step.
- `GlobalState` always carries `T`, `T_prev`, and `dt` so future time-derivative terms fit without structural change.

## Notes

- **Steady evaluation context**: when `study_type == Steady`, expressions are evaluated with `t = 0`. Steady means equilibrium, not time advancing.
- **Steady behavior**: `Scheduler::run()` 在 `study_type == Steady` 分支下跳过时间循环，仅对初始 `T = initial_temperature` 调用一次 `mhs::sim::nonlinear_solve()` 至收敛。
- **Transient behavior**: 标准时间步进 `t₀ → t₁ → … → t_end`，每步 `T_prev = T` 后调用 `mhs::sim::nonlinear_solve()`，收敛后 `current_time += dt`。
