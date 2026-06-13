# ADR-0006: Time-Stepping Refactor

## Status

Accepted

## Context

The legacy `mhs::sim::Assembler::assemble()` was the only place that
constructed the LinearSystem.  It hard-coded a BDF1 (Backward Euler)
discretization in its final branch (`if (study_type == Transient && dt > 0)`),
read a single `state_.T_prev` field, and forced the entire transient
loop in `Scheduler::run()` to use a fixed step.  Extending to BDF2 or
adaptive stepping required touching the assembler — and the loop was
not aware of any "algorithm" abstraction.

This refactor extracts the time discretization into a new
`mhs::sim::time_scheme::TimeScheme` interface, introduces a
`mhs::core::TimeStepBuffer` for BDF-k history, and re-plumbs
`Scheduler::run()` to drive the loop through the scheme.  ADR-0006
records the architecture and trade-offs.

## Decision

1. **`TimeStepBuffer`** (`mhs::core::TimeStepBuffer`): a fixed-capacity
   ring buffer holding `(T, time)` pairs.  Lives in `mhs::core` so
   it can be referenced by both the scheduler and the time-scheme
   subsystem without dragging either in.  Indexing convention:
   `at(0) == latest`, `at(i) == i steps before latest`.  No Eigen,
   no sim dependencies.

2. **`TimeScheme` abstract class** (`mhs::sim::time_scheme::TimeScheme`):
   the four operations a step needs —
   - `initialize(history, state)` — seed history at t=0
   - `select_step(history, t)` — return `(dt, order)`
   - `build_system(sops, mops, history, order, dt)` — return a `LinearSystem`
   - `accept_or_reject(history, T_candidate, error)` — Accept / Reject

   The first three are pure functions of state; `accept_or_reject` is
   what enables adaptive stepping.

3. **Three concrete schemes** under `mhs::sim::time_scheme`:
   - `Bdf1Scheme` — fixed-step backward Euler.  `accept_or_reject` always Accepts.
   - `Bdf2Scheme` — fixed-step BDF2 with **startup demotion** to BDF1 when
     `history.size() < 2`.  Variable-step form (h_n, h_{n-1}, δ) is also
     supported, falling back to δ=1 if `dt_to(1) <= 0`.
   - `AdaptiveBdfScheme` — wraps a `StepController` to choose `(dt, order)`
     per step based on the embedded order-k / order-(k-1) error estimate.
     The controller uses HNW (Hairer-Norsett-Wanner) formulas with
     `safety=0.9` and a soft `[0.5, 2.0]` factor clamp.

4. **`Assembler` interface split**:
   - `assemble_static(state)` returns `(K, f_static)`.
   - `assemble_mass(state)` returns `M_diag`, evaluated at
     `state.history.latest()` (with `state.T` fallback when history is
     empty).  This keeps the mass coefficient constant across Newton
     iterations, matching the legacy BDF1 stability behaviour.

5. **`GlobalState`** drops the legacy `T_prev` field; the previous-step
   temperature is read from `state.history.latest()`.  The new fields
   are `TimeStepBuffer history` (capacity = `max_order + 1`) and
   `int output_step` for output-time alignment.

6. **`Scheduler::run()` transient loop** is driven by a
   `time_scheme::create_scheme(cfg)` instance:
   ```text
   while (current_time < duration):
       step = scheme->select_step(history, t)
       dt   = clamp(step.dt, remaining, t_next_output - t)
       sops = assembler.assemble_static(state)
       mops = assembler.assemble_mass(state)
       ls   = scheme->build_system(sops, mops, history, step.order, dt)
       nonlinear_solve(ls, state, solver)
       history.push(T, t + dt)
       t += dt
   ```
   Steady problems bypass the scheme and go straight to
   `nonlinear_solve(model, state, solver)`.

7. **IO** — a new `<TimeScheme>` sub-block under `<Structure>`:
   ```xml
   <TimeScheme>
       <Scheme>Bdf2</Scheme>
       <InitialDt>0.05</InitialDt>
       <MinDt>1e-6</MinDt>
       <MaxDt>0.5</MaxDt>
       <AbsTol>1e-7</AbsTol>
       <RelTol>1e-4</RelTol>
       <MaxOrder>2</MaxOrder>
       <OutputDt>0.2</OutputDt>
   </TimeScheme>
   ```
   Absent block ⇒ `Bdf1` with `initial_dt = transient_time_step` and
   `output_dt = transient_duration`.  The preprocessor copies the spec
   into `InternalModel`; the scheduler uses the model values unless
   `Scheduler::setTimeSchemeConfig(...)` was called explicitly.

8. **Performance** — slice 3 chose "每次 Newton 重算 ls" (option A in the
   refactor plan) over a frozen-operator optimization.  Per-iteration
   re-assembly of K and M is the simple, correct path; a future
   `CachePolicy` PR can freeze the operator when the field has stopped
   changing significantly.

## Rationale

- **Separation of concerns**: assembly knows about cells, BC, and
  physics; the time scheme knows about BDF coefficients and step
  selection.  No knowledge bleeds across.
- **Extensibility**: a new scheme (e.g. implicit Runge-Kutta) only
  needs to inherit `TimeScheme` and supply four methods.
- **Backward compatibility**: zero.  Per the refactor PRD, no bridge
  code, no deprecation aliases, no IO shims.  The `transient_time_step`
  / `transient_duration` fields remain in `IOStructure` for the IO
  layer but the scheduler no longer reads them.

## Notes

- The Newton iterations inside `nonlinear_solve` re-assemble the
  LinearSystem each iteration.  This is a deliberate trade: a 30%
  performance hit in exchange for guaranteed correctness with
  T-dependent `k`, `ρ`, `c`.
- The output-time interpolation fallback (slice 7) clamps `dt` to
  land on `t_out = output_step * output_dt` whenever possible.  When
  `min_dt > output_dt`, the interpolation fallback is reserved for
  a future PR.
- `Bdf1Scheme::build_system` and the equivalent inline BDF1 glue
  in `nonlinear_solver.cpp` both exist; the glue is the slice-1
  fall-through path used when the time scheme was not selected
  (e.g. direct `nonlinear_solve(model, state, solver)` callers).

## Consequences

- All 149 unit tests pass.  Cases:
  - `cases/simple_steady_tests/*` — steady: unchanged.
  - `cases/simple_transient_tests/*` — BDF1: max field diff 0.05K
    (1K threshold), reference numerics preserved.
  - `cases/bdf2_transient_tests/case1.xml` — new BDF2 case.
  - `cases/adaptive_transient_tests/{case1,case2,case3}.xml` — new
    adaptive cases, including incommensurate `output_dt` vs internal
    dt.

The architecture is set up to add new schemes (Crank-Nicolson, SDIRK,
Radau, etc.) by dropping a class into `src/time_scheme/`.
