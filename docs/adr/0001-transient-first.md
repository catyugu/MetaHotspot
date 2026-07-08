# ADR-0001: Transient-First Architecture

## Status

Accepted.

## Context

Cases include both steady and transient studies. CLAUDE.md mandates treating all problems as nonlinear natively.

## Decision

The whole system is designed for transient simulation. Steady state is a single nonlinear solve at `t = 0`.

- `Scheduler::run()` branches on `Model::study_type`:
    - `Steady` — skip the time loop and call `mhs::sim::nonlinear_solve()` once, starting from `T = initial_temperature`.
    - `Transient` — step from `t = 0` up to `transient_duration`. Each step runs `assemble → build_system → nonlinear_solve → estimate_error`; on accept, `accepted.accept(T, t)` and `current_time += dt`.
- Time stepping composes three orthogonal pieces:
    - `mhs::sim::time_scheme::StepController` (strategy: Free / Strict / Intermediate / Manual) drives output-time alignment and step sizing.
    - `time_scheme::build_system(kind, …)` is a pure function that injects the BDF1/BDF2 stencil on top of an `AssemblyResult`.
    - `time_scheme::estimate_error(…)` is a pure function that returns an LTE estimate and a PI-style step-size suggestion.
- Nonlinear iteration lives inside each step. The fixed-point iteration in `nonlinear_solve` uses Anderson acceleration with a divergence guard and warm-up; on guard trip it falls back to a damped Picard step.

## Rationale

- One code path for time-stepping. Steady is the degenerate one-step case, so steady and transient share the entire solver, error estimator, and probe pipeline.
- Treating every case as nonlinear natively means Cauchy BCs, temperature-dependent `k`, and fluid-solid coupling all go through the same fixed-point loop.
- Splitting strategy / algebra / error-control into three orthogonal pieces (vs. an OOP `TimeScheme` hierarchy) keeps the inner loops free of virtual calls and lets the integrator kind be selected per step without subclass gymnastics.

## Notes

- **Steady evaluation context.** When `study_type == Steady`, expressions are evaluated with `t = 0`. Steady means equilibrium, not time advancing.
- **Step history.** The accepted-solution ring buffer (`mhs::core::SolutionHistory`) carries the snapshots needed for the BDF stencil. BDF2 is selected by the integrator kind in `build_system`; the buffer's capacity (currently 2) is sized to its needs.
- **Time-step loop in `Scheduler::run()`.** The transient branch builds a `LinearSystemProvider` lambda per step that calls `assembler.assemble(ctx)` and then `build_system(IntegratorKind::Bdf1, ops, accepted, dt)`. The LTE estimate from `estimate_error` drives the next step's `dt` via `StepController::prepare`.
