# ADR-0001: Transient-First Architecture

## Status

Accepted

## Context

Cases include both steady and transient studies. CLAUDE.md mandates treating all problems as nonlinear natively.

## Decision

The whole system is designed for transient simulation. Steady state is a single nonlinear solve at `t = 0`.

- The scheduler always enters a time-stepping loop; steady just sets `transient_duration = 0`.
- Nonlinear iteration lives inside each time step.
- `GlobalState` always carries `T`, `T_prev`, and `dt` so future time-derivative terms fit without structural change.

## Notes

- **Steady evaluation context**: when `study_type == Steady`, expressions are evaluated with `t = 0`. Steady means equilibrium, not time advancing.
- **Steady behavior**: the scheduler's time loop is skipped when `transient_duration == 0`; exactly one call to `nonlinear::solve()` runs to convergence.
- **Transient behavior**: standard stepping `t₀ → t₁ → … → t_end`, each step calls `nonlinear::solve()` then `T_prev = T`.
