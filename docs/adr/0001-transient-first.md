# ADR-0001: Transient-First Architecture

## Status

Accepted

## Context

The simulation cases include both steady-state (`StudyType>Steady`) and transient studies (`TransientStudyDuration`, `TransientStudyTimeStep`). CLAUDE.md mandates treating all problems as nonlinear natively.

## Decision

Design the entire system for transient simulation from day one. Steady-state is treated as the t→∞ limit.

**Consequences:**

- `scheduler` always runs a time-stepping loop, even if `TransientStudyDuration=0` (steady case runs one "step" and converges).
- Nonlinear Newton iteration lives inside each time step.
- The global state buffer always stores time history fields (for future time-derivative terms).

## Rationale

- Transient is the more general form; steady-state is a special case.
- Adding transient to a steady-only design later would require fundamental architectural changes.
- CLAUDE.md explicitly requires native nonlinear treatment, which naturally maps to the transient Newton loop.

## Notes

- **Steady-state evaluation context**: When `study_type == Steady`, field expressions are evaluated with `t = 0` (not t→∞). The "steady-state" means the system has reached equilibrium, not that time is advancing.
- **Steady-state solver behavior**: `scheduler` detects `TransientStudyDuration == 0` and runs exactly **one nonlinear Newton iteration** (or iterations until convergence), without any time-stepping loop. This is a direct steady-state solve at t=0.
- For transient cases, the normal time-stepping loop applies: `t₀ → t₁ → t₂ → ... → t_end`.
