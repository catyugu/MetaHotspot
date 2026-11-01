# ADR-0002: External Boundary Traces

## Status

Accepted.

## Context

Boundary laws act at the physical surface. Using the adjacent cell-center temperature to evaluate a temperature-dependent heat flux or HTC changes that law and gives postprocessing a second, inconsistent boundary-temperature formula.

## Decision

Each compiled external `BoundaryFace` has an algebraically independent temperature trace. Its index in `Model::boundary_faces` is its trace index. `BoundaryFace` references a parent topological `FaceId`, a shared `BoundaryCondition`, and tangential bounds; subdividing a boundary does not modify `DiscreteMesh`.

For cell temperature `T`, trace `lambda`, normal conductivity `k`, owner distance `d`, and area `A`, the conductive flux is

`F = k A (T - lambda) / d`.

Dirichlet prescribes `lambda`; Neumann satisfies `F + A q(lambda) = 0`, where positive `q` is inward; convection equates `F` to `A h(lambda) (lambda - T_inf(lambda))`. Boundary expressions therefore receive `lambda` as `T`. The same local equation supplies assembly, gradient reconstruction, probes, output sampling, and the final `Solution::boundary_temperature` array.

The trace equations are statically condensed face by face. Constant laws use their analytic solution; a temperature-dependent law uses a bracketed scalar solve. The global state, `K/C/f`, nonlinear driver, time integration, and public solve entry remain cell-sized. Internal faces retain TPFA.

This decision does not introduce HMM, MPFA, non-matching interfaces, general polyhedral cells, or EROM coupling.

## Rationale

- Boundary temperature has one numerical definition everywhere.
- Temperature-dependent laws are evaluated at the surface they describe.
- Local condensation preserves the existing sparse matrix size and pattern.
- Contiguous POD boundary records add only `O(number of boundary faces)` result storage.
