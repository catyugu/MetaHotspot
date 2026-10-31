"""Independent monolithic FOM and ROM+TIM/lid transient solves."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from assembly import AMBIENT_C, SOURCE_NAMES, sampling_matrix
from cms import factorize
from external import EXTERNAL_SPECS, assemble_cooler, attachment, connect

POWER_TIMES = np.array([0.0, 0.5, 3.0])
POWER_LEVELS = np.array([[10.0, 10.0], [65.0, 55.0], [0.0, 0.0]])
END_TIME = 6.0
SNAPSHOT_TIMES = np.array([0.5, 1.0, 3.0, 6.0])
PROBES_MM = {
    "A junction": (6.75, 9.0, 0.65),
    "B junction": (17.25, 9.0, 0.65),
    "substrate below A": (6.75, 9.0, 0.30),
}
LID_PROBE_MM = (6.75, 9.0, 1.80)


def power_at(t):
    return POWER_LEVELS[np.searchsorted(POWER_TIMES, t, side="right") - 1]


def time_grid(max_step):
    """Land on jumps and snapshots; restart from 20 us at every event."""
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("time step must be finite and positive")
    events = np.unique(np.r_[POWER_TIMES, SNAPSHOT_TIMES, END_TIME])
    grid = [0.0]
    for left, right in zip(events[:-1], events[1:]):
        t, step = left, min(2e-5, max_step)
        while t + step < right - 1e-12:
            t = round(t + step, 12)
            grid.append(t)
            step = min(2 * step, max_step)
        grid.append(float(right))
    return np.array(grid)


def probe_matrix(model):
    return sampling_matrix(model, np.asarray(list(PROBES_MM.values())) * 1e-3)


def reconstruct(basis, state):
    m = basis.model.m
    return np.r_[state[:m], basis.lift @ state[:m] + basis.directions @ state[m:]]


def simulate(model, basis, capacity, stiffness, max_step=0.005, cooling_scale=1.0):
    """Connect the same extracted component to a full spatial external FOM.

    Both the reference and reduced assembly share silicon/TIM interface nodes,
    independently solve all coupled equations, and include the external thermal
    mass. cooling_scale changes only the lid Robin condition after extraction.
    """
    started = time.perf_counter()
    grid = time_grid(max_step)
    cooler = assemble_cooler(model, cooling_scale)
    order = stiffness.shape[0]
    sf, sr = attachment(model.N, cooler), attachment(order, cooler)
    kf, kr = (
        connect(model.K, cooler, model.m),
        connect(sp.csc_matrix(stiffness), cooler, model.m),
    )
    full_capacity = (
        np.r_[model.capacity, np.zeros(cooler.n)]
        + np.asarray(sf.T @ cooler.capacity).ravel()
    )
    cr = (
        sp.block_diag((capacity, sp.csc_matrix((cooler.n, cooler.n))), format="csc")
        + sr.T @ sp.diags(cooler.capacity) @ sr
    ).tocsc()
    br = np.vstack(
        [
            basis.project_inputs(sp.csc_matrix(model.B_P)),
            np.zeros((cooler.n, len(SOURCE_NAMES))),
        ]
    )
    bf = np.vstack([model.B_P, np.zeros((cooler.n, len(SOURCE_NAMES)))])
    component_probes = probe_matrix(model)
    pf = sp.vstack(
        [component_probes, sp.csc_matrix((cooler.n, component_probes.shape[1]))]
    )
    pr = sp.csc_matrix(
        np.vstack(
            [
                basis.project_inputs(component_probes),
                np.zeros((cooler.n, component_probes.shape[1])),
            ]
        )
    )
    lid_probe = sampling_matrix(cooler, [np.asarray(LID_PROBE_MM) * 1e-3])
    pf, pr = (
        sp.hstack([pf, sf.T @ lid_probe], format="csc"),
        sp.hstack([pr, sr.T @ lid_probe], format="csc"),
    )
    xf, xr = np.zeros(kf.shape[0]), np.zeros(kr.shape[0])
    yf, yr = np.zeros((grid.size, pf.shape[1])), np.zeros((grid.size, pr.shape[1]))
    averages_f, averages_r = (
        np.zeros((grid.size, len(SOURCE_NAMES))),
        np.zeros((grid.size, len(SOURCE_NAMES))),
    )
    losses = np.asarray(kf @ np.ones(kf.shape[0]))
    snapshots, cooler_snapshots = [], []
    energy_error, previous_step = 0.0, None
    max_power = float(POWER_LEVELS.sum(axis=1).max())
    full_seconds, reduced_seconds = 0.0, 0.0
    for it, (left, right) in enumerate(zip(grid[:-1], grid[1:]), start=1):
        step = round(right - left, 12)
        changed = step != previous_step
        p = power_at((left + right) / 2)
        t0 = time.perf_counter()
        if changed:
            full_factor = factorize(kf + sp.diags(full_capacity / step, format="csc"))
        old = xf
        xf = full_factor.solve(full_capacity / step * old + bf @ p)
        full_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        if changed:
            reduced_factor = factorize(kr + cr / step)
        xr = reduced_factor.solve(cr @ xr / step + br @ p)
        reduced_seconds += time.perf_counter() - t0
        previous_step = step
        yf[it], yr[it] = pf.T @ xf, pr.T @ xr
        averages_f[it], averages_r[it] = bf.T @ xf, br.T @ xr
        balance = full_capacity @ ((xf - old) / step) + losses @ xf - p.sum()
        energy_error = max(energy_error, abs(float(balance)) / max_power)
        if np.any(np.isclose(right, SNAPSHOT_TIMES, rtol=0, atol=1e-12)):
            snapshots.append(
                (float(right), xf[: model.N].copy(), reconstruct(basis, xr[:order]))
            )
            cooler_snapshots.append((float(right), sf @ xf, sr @ xr))
    peak_power = POWER_LEVELS[POWER_LEVELS.sum(axis=1).argmax()]
    steady_full = factorize(kf).solve(bf @ peak_power)
    steady_reduced = factorize(kr).solve(br @ peak_power)
    steady_f, steady_r = (
        steady_full[: model.N],
        reconstruct(basis, steady_reduced[:order]),
    )
    cooler_steady = sf @ steady_full, sr @ steady_reduced
    snapshot_error = max(float(np.max(np.abs(r - f))) for _, f, r in snapshots)
    cooler_error = max(float(np.max(np.abs(r - f))) for _, f, r in cooler_snapshots)
    peak_rise, peak_time = max((float(f.max()), t) for t, f, _ in snapshots)
    summary = {
        "ambient_C": AMBIENT_C,
        "connection": "3D_TIM_and_lid",
        "lid_cooling_scale": cooling_scale,
        "external_FOM_dofs": cooler.N,
        "external_FOM_interior_dofs": cooler.n,
        "assembled_FOM_order": kf.shape[0],
        "assembled_ROM_order": kr.shape[0],
        "external_layers": EXTERNAL_SPECS,
        "external_layer_z_cells": cooler.layer_cells,
        "external_actual_max_cell_size_m": [
            float(np.diff(a).max()) for a in cooler.axes
        ],
        "interface": "shared silicon-top/TIM-bottom temperatures",
        "max_step_s": max_step,
        "steps": grid.size - 1,
        "power_source_names": SOURCE_NAMES,
        "power_switch_times_s": POWER_TIMES.tolist(),
        "power_levels_W": POWER_LEVELS.tolist(),
        "peak_total_power_W": max_power,
        "probes_mm": {**PROBES_MM, "lid above A": LID_PROBE_MM},
        "max_probe_error_K": float(np.max(np.abs(yr - yf))),
        "max_source_average_error_K": float(np.max(np.abs(averages_r - averages_f))),
        "max_snapshot_field_error_K": snapshot_error,
        "external_max_snapshot_field_error_K": cooler_error,
        "snapshot_peak_FOM_C": AMBIENT_C + peak_rise,
        "snapshot_peak_time_s": peak_time,
        "steady_peak_FOM_C": AMBIENT_C + float(steady_f.max()),
        "steady_peak_ROM_C": AMBIENT_C + float(steady_r.max()),
        "steady_field_error_K": float(np.max(np.abs(steady_r - steady_f))),
        "external_steady_field_error_K": float(
            np.max(np.abs(cooler_steady[1] - cooler_steady[0]))
        ),
        "steady_power_balance_relative": abs(
            float(losses @ steady_full) / max_power - 1
        ),
        "transient_energy_balance_relative": energy_error,
        "seconds": {
            "FOM_step_solves": full_seconds,
            "ROM_step_solves": reduced_seconds,
            "total": time.perf_counter() - started,
        },
    }
    return summary, {
        "cooler": cooler,
        "cooler_snapshots": cooler_snapshots,
        "cooler_steady": cooler_steady,
        "probe_names": [*PROBES_MM, "lid above A"],
        "times": grid,
        "FOM_probes": yf,
        "ROM_probes": yr,
        "FOM_averages": averages_f,
        "ROM_averages": averages_r,
        "snapshots": snapshots,
        "steady": (steady_f, steady_r),
    }
