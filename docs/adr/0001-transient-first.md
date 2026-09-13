# ADR-0001: Transient-First Architecture

## Status

Accepted.

## Context

Cases include both steady and transient studies. CLAUDE.md mandates treating all problems as nonlinear natively.

## Decision

The whole system is designed for transient simulation. Steady state is a single nonlinear solve at `t = 0`.

- `solve_system()` branches on an explicit `Study`:
    - `Steady` — skip the time loop and call `mhs::sim::nonlinear_solve()` once, starting from a uniform `initial_temperature` vector (or an explicitly provided initial state).
    - `Transient` — step from `t = 0` up to `transient_duration`. Each step runs `assemble → build_system → nonlinear_solve → estimate_error`; on accept, `accepted.accept(T, t)` and `current_time += dt`.
- Time stepping composes three orthogonal pieces:
    - `mhs::sim::time_scheme::StepController` (strategy: Adaptive / Fixed) drives output-time alignment and step sizing. Both strategies shorten a step at output/final boundaries, even below the nominal `min_dt`, so observers only receive states actually solved at those times.
    - `time_scheme::build_system(kind, …)` is a pure function that injects the BDF1/BDF2 stencil on top of an `Operators`.
    - `time_scheme::estimate_error(…)` is a pure function that returns an LTE estimate and a PI-style step-size suggestion.
- Nonlinear iteration lives inside each step. The fixed-point iteration in `nonlinear_solve` uses Anderson acceleration with a divergence guard and warm-up; on guard trip it falls back to a damped Picard step.
- `solve_system()` receives one `SystemAssembler(state, time)` callback. Each
  nonlinear iteration requests a complete current linearization; the solve loop
  does not know state partitions or coupling topology.
- Time interpolation of the global state is forbidden. Modal coefficients are
  coordinates in a reduced trial space, not physical nodal values for which a
  time-step-level interpolation policy can be assumed.

## Rationale

- Steady and transient share assembly, nonlinear solve, linear solve, and probe recording; only transient uses time integration and error control.
- Treating every case as nonlinear natively means Cauchy BCs, temperature-dependent `k`, and fluid-solid coupling all go through the same fixed-point loop.
- Splitting strategy / algebra / error-control into three orthogonal pieces (vs. an OOP `TimeScheme` hierarchy) keeps the inner loops free of virtual calls and lets the integrator kind be selected per step without subclass gymnastics.

## Notes

- **Steady evaluation context.** When `study_type == Steady`, expressions are evaluated with `t = 0`. Steady means equilibrium, not time advancing.
- **Step history.** The accepted-solution ring buffer (`mhs::core::SolutionHistory`) carries the snapshots needed for the BDF stencil. BDF2 is selected by the integrator kind in `build_system`; the buffer's capacity (currently 2) is sized to its needs.
- **Time-step loop in `solve_system()`.** At each nonlinear iteration the
  solver calls the aggregate `SystemAssembler`, then passes the returned
  operators to `build_system`. The LTE estimate from `estimate_error` drives
  the next step's `dt` via `StepController::prepare`.
