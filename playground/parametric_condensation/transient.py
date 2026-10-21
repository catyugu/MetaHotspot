#!/usr/bin/env python3
"""Nonlinear steady and transient solves for a :class:`assemble.System`.

The only nonlinearity is the ambient-port flux (radiation), so the Jacobian is
``K + C/dt + sum_g trace^T diag(slope) trace`` and stays symmetric positive
definite.  Every solve is AMG-preconditioned CG: one hierarchy is built at the
linearisation point and reused as a *frozen preconditioner* for all Newton steps
and all time steps, while CG always runs on the current Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from linalg import Cost, SPDSolver

NEWTON_RTOL = 1.0e-9
NEWTON_LIMIT = 25


@dataclass(frozen=True)
class History:
    times: np.ndarray
    probe_temperature: np.ndarray  # (steps, n_probe)
    port_temperature: dict[str, np.ndarray]


def steady(system, *, ambient: float, power, cost: Cost | None = None) -> np.ndarray:
    """Nonlinear steady state by Newton with an AMG-CG Jacobian."""
    cost = cost if cost is not None else Cost()
    rhs = system.source @ np.asarray(power, dtype=np.float64)
    state = system.initial(ambient)
    solver = SPDSolver(system.jacobian(state), cost=cost)
    for _ in range(NEWTON_LIMIT):
        residual = system.K @ state + system.flux(state) - rhs
        jacobian = system.jacobian(state)
        delta = solver.solve_on(jacobian, -residual, label=f"{system.label} steady")
        state = state + delta
        if np.linalg.norm(delta) <= NEWTON_RTOL * max(1.0, np.linalg.norm(state)):
            return state
    raise RuntimeError(f"{system.label}: steady Newton did not converge")


def integrate(system, *, ambient: float, power, dt: float, duration: float,
              initial=None, cost: Cost | None = None) -> History:
    """Implicit Euler with Newton on the ambient-port flux.

    ``initial`` defaults to a uniform ``ambient`` field.  Starting from a steady
    state instead is the physically relevant BCI scenario (the boundary
    condition changes, the component responds) and avoids exciting the
    microsecond through-thickness modes, whose implicit-Euler error otherwise
    dominates any step-convergence study.
    """
    cost = cost if cost is not None else Cost()
    steps = int(round(duration / dt))
    state = system.initial(ambient) if initial is None else np.array(initial, dtype=np.float64)
    rhs = system.source @ np.asarray(power, dtype=np.float64)
    capacity = system.C.tocsc()

    probes = np.empty((steps + 1, system.probe_map.shape[0]))
    probes[0] = system.probe_map @ state
    ports = {label: np.empty((steps + 1, trace.shape[0]))
             for label, trace in system.port_groups.items()}
    for label, trace in system.port_groups.items():
        ports[label][0] = trace @ state

    base = (capacity / dt + system.K).tocsc()
    solver = SPDSolver(base + system.jacobian(state) - system.K, cost=cost)
    for step in range(1, steps + 1):
        target = state.copy()
        for _ in range(NEWTON_LIMIT):
            residual = (
                capacity @ (target - state) / dt
                + system.K @ target
                + system.flux(target)
                - rhs
            )
            jacobian = (base + system.jacobian(target) - system.K).tocsc()
            delta = solver.solve_on(jacobian, -residual, label=f"{system.label} step {step}")
            target = target + delta
            if np.linalg.norm(delta) <= NEWTON_RTOL * max(1.0, np.linalg.norm(target)):
                break
        else:
            raise RuntimeError(f"{system.label}: Newton did not converge at step {step}")
        state = target
        probes[step] = system.probe_map @ state
        for label, trace in system.port_groups.items():
            ports[label][step] = trace @ state

    return History(times=np.arange(steps + 1) * dt, probe_temperature=probes,
                   port_temperature=ports)
