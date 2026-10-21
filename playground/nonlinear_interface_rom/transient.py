#!/usr/bin/env python3
"""Implicit-Euler nonlinear transients, full order and reduced.

Both solvers integrate the same physics:

    C T' + K T + f_nl(T) = B u(t)

with ``f_nl`` supported on the interface DOFs.  The reduced solver evaluates
``f_nl`` at the *retained physical port temperatures*, so the interface law is
never approximated -- only its projected contribution to the interior
coordinates is.

Full-order steps use AMG-preconditioned CG.  The Jacobian changes every Newton
iteration because the interface slope depends on temperature, so the AMG
hierarchy is built once at the linearization point and reused as a frozen
preconditioner for all steps and all Newton iterations; the CG iteration itself
always runs on the true Jacobian.  Setups and iterations are counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from linalg import Cost, SPDSolver, dense_spd_solve

NEWTON_RTOL = 1.0e-9
NEWTON_LIMIT = 20


@dataclass(frozen=True)
class History:
    times: np.ndarray
    port_temperature: np.ndarray
    probe_temperature: np.ndarray
    interface_power: np.ndarray


def _allocate(steps, port_count, probe_count):
    return (
        np.empty((steps + 1, port_count)),
        np.empty((steps + 1, probe_count)),
        np.empty((steps + 1, port_count)),
    )


def steady(operators, interface_law, source, probes, power, *, ambient, cost=None):
    """Full-order nonlinear steady state by Newton with an AMG-CG Jacobian."""
    cost = cost if cost is not None else Cost()
    K = operators.K.tocsc()
    n = K.shape[0]
    rhs = source @ np.asarray(power, dtype=np.float64)
    state = np.full(n, ambient)
    solver = SPDSolver((K + sp.diags(interface_law.slope(state, n))).tocsc(), cost=cost)
    for _ in range(NEWTON_LIMIT):
        residual = K @ state + interface_law.flux(state, n) - rhs
        jacobian = (K + sp.diags(interface_law.slope(state, n))).tocsc()
        delta = solver.solve_on(jacobian, -residual, label="steady newton")
        state = state + delta
        if np.linalg.norm(delta) <= NEWTON_RTOL * max(1.0, np.linalg.norm(state)):
            return state
    raise RuntimeError("steady Newton did not converge")


def steady_reduced(model, interface_law, source, probes, power, *, ambient, cost=None):
    """Reduced nonlinear steady state; the interface law sees the true ports."""
    cost = cost if cost is not None else Cost()
    m = model.port_count
    n = model.V.shape[0]
    reduced_rhs = model.V.T @ (source @ np.asarray(power, dtype=np.float64))
    reduced = np.zeros(model.dimension)
    reduced[:m] = ambient
    for _ in range(NEWTON_LIMIT):
        full = model.V @ reduced
        residual = model.K @ reduced + model.V.T @ interface_law.flux(full, n) - reduced_rhs
        jacobian = model.K.copy()
        # model.interface is the full-model index of reduced port i, so this
        # slice of the slope vector is already in reduced port order.
        jacobian[np.arange(m), np.arange(m)] += interface_law.slope(full, n)[model.interface]
        delta = dense_spd_solve(jacobian, -residual, cost=cost)
        reduced = reduced + delta
        if np.linalg.norm(delta) <= NEWTON_RTOL * max(1.0, np.linalg.norm(reduced)):
            return reduced
    raise RuntimeError("reduced steady Newton did not converge")


def integrate(operators, interface_law, source, probes, power, *, dt, duration, port_cells, ambient, cost=None):
    """Full-order implicit Euler with Newton on the diagonal interface law."""
    cost = cost if cost is not None else Cost()
    K = operators.K.tocsc()
    C = operators.C.tocsc()
    n = K.shape[0]
    steps = int(round(duration / dt))
    state = np.full(n, ambient)
    rhs = source @ np.asarray(power, dtype=np.float64)
    port_history, probe_history, power_history = _allocate(steps, port_cells.size, probes.size)
    port_history[0] = state[port_cells]
    probe_history[0] = state[probes]
    power_history[0] = interface_law.flux(state, n)[port_cells]

    base = (C / dt + K).tocsc()
    frozen = (base + sp.diags(interface_law.slope(state, n))).tocsc()
    solver = SPDSolver(frozen, cost=cost)
    for step in range(1, steps + 1):
        target = state.copy()
        for _ in range(NEWTON_LIMIT):
            residual = C @ (target - state) / dt + K @ target + interface_law.flux(target, n) - rhs
            jacobian = (base + sp.diags(interface_law.slope(target, n))).tocsc()
            delta = solver.solve_on(jacobian, -residual, label=f"newton {step}")
            target = target + delta
            if np.linalg.norm(delta) <= NEWTON_RTOL * max(1.0, np.linalg.norm(target)):
                break
        else:
            raise RuntimeError(f"Newton did not converge at step {step}")
        state = target
        port_history[step] = state[port_cells]
        probe_history[step] = state[probes]
        power_history[step] = interface_law.flux(state, n)[port_cells]
    return History(
        times=np.arange(steps + 1) * dt,
        port_temperature=port_history,
        probe_temperature=probe_history,
        interface_power=power_history,
    )


def integrate_reduced(model, interface_law, source, probes, power, *, dt, duration, port_cells, ambient, cost=None):
    """Reduced implicit Euler; the interface law sees the true port temperatures."""
    cost = cost if cost is not None else Cost()
    m = model.port_count
    n = model.V.shape[0]
    steps = int(round(duration / dt))
    reduced = np.zeros(model.dimension)
    reduced[:m] = ambient
    reduced_rhs = model.V.T @ (source @ np.asarray(power, dtype=np.float64))
    port_history, probe_history, power_history = _allocate(steps, m, probes.size)
    port_history[0] = reduced[:m]
    probe_history[0] = model.V[probes, :] @ reduced
    power_history[0] = interface_law.flux(model.V @ reduced, n)[port_cells]

    base = model.C / dt + model.K
    for step in range(1, steps + 1):
        target = reduced.copy()
        for _ in range(NEWTON_LIMIT):
            full = model.V @ target
            residual = (
                model.C @ (target - reduced) / dt + model.K @ target + model.V.T @ interface_law.flux(full, n) - reduced_rhs
            )
            jacobian = base.copy()
            # The interface block of V is the identity, so the diagonal slope
            # enters the reduced Jacobian exactly and nothing else does.
            jacobian[np.arange(m), np.arange(m)] += interface_law.slope(full, n)[port_cells]
            delta = dense_spd_solve(jacobian, -residual, cost=cost)
            target = target + delta
            if np.linalg.norm(delta) <= NEWTON_RTOL * max(1.0, np.linalg.norm(target)):
                break
        else:
            raise RuntimeError(f"reduced Newton did not converge at step {step}")
        reduced = target
        full = model.V @ reduced
        port_history[step] = reduced[:m]
        probe_history[step] = full[probes]
        power_history[step] = interface_law.flux(full, n)[port_cells]
    return History(
        times=np.arange(steps + 1) * dt,
        port_temperature=port_history,
        probe_temperature=probe_history,
        interface_power=power_history,
    )
