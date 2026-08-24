r"""Reduce-order extraction cost vs. full-solve amortization (break-even) at 1.0 mm.

Metric: a ROM is only worthwhile once
    T_extract + N*T_rom < N*T_detailed
i.e. once it has served ``N_break-even`` transient steps (or that many single
full runs).  Runs at the finer 1.0 mm mesh of Case-1 (134,640 cells).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyamg
import scipy.sparse as sp

PROJECT = Path(__file__).resolve().parents[3]  # repo root
CASE = PROJECT / "playground" / "bci_rom_testcase1"
MACRO = PROJECT / "playground" / "macromodel"
sys.path[:0] = [str(CASE), str(MACRO), str(PROJECT / "python")]
from model_case1 import Case1Config, Case1Model  # noqa: E402
from utils import (
    assemble_reduced_k,
    build_parametric_basis,
    project_bci,
    solve_rom_steady,
    solve_rom_transient,
)  # noqa: E402

OUT = PROJECT / "results" / "weekly_0825"
H = (50.0, 1000.0)
POWER = np.array([0.1, 0.2, 0.3, 0.4])
DURATION_S = 1000.0
DT_S = 10.0


def amg_transient(C, K, G, power, dt, duration, rtol=1.0e-9, maxiter=2000):
    """Full-domain transient by Ruge-Stueben AMG-preconditioned warm-started CG.

    The operator ``(C/dt + K)`` is constant over the trace, so the AMG hierarchy
    is built once and reused; each step warm-starts CG from the previous state.
    Returns ``(elapsed_s, per_step_s)``.
    """
    lhs = (C / dt + K).tocsc()
    ml = pyamg.ruge_stuben_solver(lhs.tocsr())
    M = ml.aspreconditioner(cycle="V")
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    theta = np.zeros(K.shape[0])
    rhs = G @ np.asarray(power, dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(1, times.size):
        b = (C @ theta) / dt + rhs
        theta, info = sp.linalg.cg(
            lhs, b, x0=theta, rtol=rtol, atol=0.0, maxiter=maxiter, M=M
        )
        if info != 0:
            raise RuntimeError(f"AMG-CG transient did not converge: info={info}")
    elapsed = time.perf_counter() - t0
    return elapsed, elapsed / max(times.size - 1, 1)


def main(mm: float):
    model = Case1Model(
        Case1Config(
            max_xy_cell_mm=mm,
            max_z_cell_mm=mm,
            duration_s=DURATION_S,
            dt_s=DT_S,
        )
    )
    cells = int(model.full_cell_count)

    core = model.core_operators()
    G = model.source_shape()
    terms = model.boundary_terms()
    h_ranges = model.h_ranges()
    steps = int(round(DURATION_S / DT_S))

    # ---- full-domain operators at effective boundary h ------------------
    K_h = core.K.tocsc().copy()
    for pk, Hk in zip(model.physical_to_effective(H), terms):
        K_h = K_h + float(pk) * Hk.tocsc()
    K_h = (0.5 * (K_h + K_h.T)).tocsc()
    C = core.C.tocsc()

    # ---- detailed (full FVM) transient with AMG preconditioner ----------
    full_transient_s, full_per_step = amg_transient(C, K_h, G, POWER, DT_S, DURATION_S)

    # ---- ROM extraction ------------------------------------------------
    t = time.perf_counter()
    basis, summary = build_parametric_basis(
        core,
        G,
        terms,
        h_ranges,
        tolerance=1e-3,
        max_order=1024,
        probe_rounds=2,
        seed=20260805,
    )
    extract = time.perf_counter() - t
    order = int(basis.shape[1])

    # ---- online (reduced) solve timing ---------------------------------
    C_hat, K0, F_hat, F_b, A_b = project_bci(core, G, terms, basis)
    p = model.physical_to_effective(H)
    K_hat = assemble_reduced_k(K0, F_b, A_b, p)
    t = time.perf_counter()
    solve_rom_steady(K_hat, F_hat, POWER)
    rom_steady = time.perf_counter() - t
    t = time.perf_counter()
    _, _ = solve_rom_transient(C_hat, K_hat, F_hat, lambda _: POWER, DT_S, DURATION_S)
    rom_transient = time.perf_counter() - t
    rom_per_step = rom_transient / steps if steps else 0.0

    # ---- break-even ----------------------------------------------------
    denom = max(full_per_step - rom_per_step, 1e-12)
    n_break_steps = extract / denom
    n_break_runs = n_break_steps / steps
    total_saved_per_run = full_transient_s - rom_transient

    result = dict(
        mesh_mm=mm,
        cells=cells,
        transient_steps=steps,
        full_transient_s=full_transient_s,
        full_per_step_s=full_per_step,
        solver="Ruge-Stueben AMG-preconditioned warm-started CG",
        extraction_s=extract,
        basis_order=order,
        extraction_candidates=summary["processed_candidate_count"],
        rom_steady_s=rom_steady,
        rom_transient_s=rom_transient,
        rom_per_step_s=rom_per_step,
        break_even_steps=float(n_break_steps),
        break_even_runs=float(n_break_runs),
        saved_per_run_s=total_saved_per_run,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"breakeven_{mm:g}mm.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    mm = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    main(mm)
