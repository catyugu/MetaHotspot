# ADR-0001: Transient-First Architecture

## Status

Accepted.

## Decision

Steady and transient studies share assembly, nonlinear iteration, linear solves,
and probe recording. Steady state is one nonlinear solve at `t = 0`.

Transient integration uses AdaptiveTimeStepper 2.1.0 through a private CPM
source dependency. The controller owns BDF history, adaptive order (1–2), local
error estimation, rejected-step retries, and dense output. Public headers and
installed CMake targets expose no AdaptiveTimeStepper dependency.

`SolveOptions` exposes `time_tolerance` (default `1e-4`) and optional
`max_time_step` in seconds. An unset maximum uses the controller's automatic
limit. C represents an unset maximum by zero; Python uses `None`. The relative
error test acts on unscaled temperatures, with absolute tolerance equal to
`0.1 * time_tolerance`. Step growth, minimum step, and order selection remain
internal policies.

Each callback assembles `C`, `K`, and `f` at the requested time and nonlinear
iterate, then solves `(K + cj C) T = f - C history`. A failed linear or nonlinear
solve returns false so the controller can shrink and retry. Exhausting retries
raises a numerical failure, which the C API reports through its error channel.

Thermal assembly has positive diagonal capacity. The initial derivative is
computed as `C^-1 (f - K T)` to preserve the supplied initial temperature exactly.
General internal callback systems with coupled or algebraic capacity use the
controller's consistent initialization.

Output sampling is independent of solver steps. Positive output intervals use
BDF dense interpolation; an interval of zero records every accepted step.
Snapshots and probes share this timeline, including the initial value and final
time exactly once. The controller must land on the final time, so the returned
final state is an accepted solution.

A step can miss an unresolved narrow pulse or discontinuity in a forcing
function. `max_time_step` allows the caller to resolve its shortest time scale.
Automatic extraction of forcing discontinuities is left for future work.
