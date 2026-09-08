#!/usr/bin/env python3
r"""E1 — AMG preconditioner diagnostic on the real BCI extraction systems.

Question: 88.5 s extraction on Case-1 (122,400 cells) spends ~54 s (61%) in
AMG *setup* (154 Ruge-Stueben builds, one per enrich solve) and ~19 s in CG
iterations.  Which preconditioner configuration minimizes (setup + iterations)
for the *same* operator family the extraction actually solves,

    A(h, sigma) = K + sigma*C + sum_k h_k H_k,   h ~ random in effective range,

with RHS g_port (unit-power source shape of one die)?

This script is standalone: it builds the Case-1 operators once, replays the
extraction's (shift, h) plan for port 0 (same seeds, same sub-seed formula as
``_BasisBuilder.run``), and solves a representative subset of those systems
with several candidate preconditioners:

  * Ruge-Stueben, default classical strength/interp (current production path)
  * RS with direct interpolation
  * RS with strength threshold theta in {0.1, 0.5}
  * smoothed aggregation (standard)
  * SA with classical strength
  * sparse direct LU (scipy splu) — lower-bound reference for the solve phase
  * unpreconditioned CG on the easiest systems only (cap) — sanity baseline

No library code is modified; this only *selects* the candidate for the E1b
full-extraction A/B run.

Outputs (JSON + printed table + convergence curves):
    results/optimization/amg_diagnostic.json
    results/optimization/amg_diagnostic_convergence.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import pyamg

CASE = Path(__file__).resolve().parents[2]  # playground/bci_rom_testcase1
ROOT = CASE.parents[1]  # repo root
sys.path[:0] = [str(CASE)]

from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
    port_eigenvalue_bounds,
    random_h,
)

OUT = ROOT / "playground" / "bci_rom_testcase1" / "results" / "optimization"

ENRICH_RTOL = 1.0e-6
MAXITER = 2000
SEED = 20260805
PORT = 0  # the 0.1 W die — same spectral plan for every port here


def assemble(model, shift, h_vec):
    """Replica of _BasisBuilder._candidate_A."""
    A = model.core.K.tocsc() + shift * model.core.C.tocsc()
    for hk, Hk in zip(h_vec, model.boundary_terms):
        A = A + hk * Hk.tocsc()
    return A.tocsc()


def plan_shifts(model, g):
    """Deterministic port-0 shift plan (same code path as the extraction)."""
    t0 = time.perf_counter()
    lambda_min, lambda_max = port_eigenvalue_bounds(model.core.K, model.core.C, g)
    t_bounds = time.perf_counter() - t0
    kappa = lambda_max / max(lambda_min, np.finfo(float).tiny)
    count = mpmm_elliptic_shift_count(1.0e-3, lambda_min, lambda_max)
    shifts = mpmm_elliptic_shifts(count, lambda_max, kappa)
    return lambda_min, lambda_max, shifts, t_bounds


def pick_candidates(shifts):
    """12 representative (shift, h_draw_index) systems, cost-weighted.

    The extraction's enrich solves concentrate at the low end of the spectrum
    (many h draws there), so sample those densest.
    """
    cands = []
    # every second low shift x 2 h draws
    low = list(shifts[-6:])
    for i, s in enumerate(low):
        cands.append((float(s), 0))
        if i % 2 == 1:
            cands.append((float(s), 1))
    mid = shifts[len(shifts) // 2]
    cands.append((float(mid), 0))
    cands.append((float(mid), 1))
    hi = shifts[0]
    cands.append((float(hi), 0))
    return cands


def sub_seed(port, shift, h_sample):
    return SEED + 1003 * port + int(round(float(shift) * 1.0e6)) + h_sample


def solve_with(precond_factory, A, b):
    """Return (iters, time_s, converged, residual_history)."""
    t0 = time.perf_counter()
    M = precond_factory(A)
    t_setup = time.perf_counter() - t0
    hist = []

    def cb(xk):
        hist.append(float(np.linalg.norm(b - A @ xk)))

    t0 = time.perf_counter()
    x, info = spla.cg(
        A, b, x0=None, rtol=ENRICH_RTOL, atol=0.0, maxiter=MAXITER, M=M, callback=cb
    )
    t_iter = time.perf_counter() - t0
    conv = info == 0
    return dict(
        setup_s=t_setup,
        iter_s=t_iter,
        iters=len(hist) if conv else MAXITER,
        converged=bool(conv),
        residual_history=hist,
    )


# ---------------------------------------------------------------- factories


def rs_factory(theta=0.25, interp="classical", **kw):
    def build(A):
        ml = pyamg.ruge_stuben_solver(
            A,
            strength=("classical", {"theta": theta}),
            interpolation=interp,
            max_coarse=10,
            keep=True,
        )
        return ml.aspreconditioner(cycle="V")

    build.label = f"RS theta={theta} {interp} interp"
    return build


def sa_factory(strength="symmetric", **kw):
    def build(A):
        ml = pyamg.smoothed_aggregation_solver(
            A,
            strength=strength,
            aggregate="standard",
            smooth=("jacobi", {"omega": 4.0 / 3.0}),
            max_coarse=10,
            keep=True,
        )
        return ml.aspreconditioner(cycle="V")

    build.label = f"SA strength={strength}"
    return build


def solve_lu(A, b):
    t0 = time.perf_counter()
    lu = spla.splu(A.tocsc())
    t_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    x = lu.solve(b)
    t_solve = time.perf_counter() - t0
    res = float(np.linalg.norm(b - A @ x)) / max(float(np.linalg.norm(b)), 1e-300)
    return dict(
        setup_s=t_setup,
        iter_s=t_solve,
        iters=1,
        converged=bool(res < ENRICH_RTOL),
        residual_history=[res],
    )


def cg_only(A, b):
    t0 = time.perf_counter()
    hist = []

    def cb(xk):
        hist.append(float(np.linalg.norm(b - A @ xk)))

    x, info = spla.cg(
        A, b, x0=None, rtol=ENRICH_RTOL, atol=0.0, maxiter=500, callback=cb
    )
    t = time.perf_counter() - t0
    return dict(
        setup_s=0.0,
        iter_s=t,
        iters=len(hist),
        converged=bool(info == 0),
        residual_history=hist,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = Case1Model(
        Case1Config(max_xy_cell_mm=1.0, max_z_cell_mm=1.0, duration_s=100.0, dt_s=5.0)
    )
    n = model.full_cell_count
    G = model.source_shape
    g = G[:, PORT : PORT + 1]
    h_ranges = model.h_ranges()
    print(f"cells={n}  source port={PORT}  h_ranges={h_ranges.tolist()}")

    lambda_min, lambda_max, shifts, t_bounds = plan_shifts(model, g)
    kappa = lambda_max / max(lambda_min, 1e-300)
    print(
        f"lambda_min={lambda_min:.4e} lambda_max={lambda_max:.3f} kappa={kappa:.2e} "
        f"shift_count={shifts.size} (eigenbounds {t_bounds:.2f}s)"
    )
    cands = pick_candidates(shifts)
    print(f"candidate systems: {len(cands)}")

    # build all systems once
    systems = []
    for shift, hs in cands:
        h_vec = random_h(h_ranges, sub_seed(PORT, shift, hs))
        A = assemble(model, shift, h_vec)
        systems.append((shift, h_vec, A, g.ravel()))

    # ---------------------------------------------------------------- sweep
    # Note: scipy SuperLU on a 122k-cell 3D operator is very expensive
    # (large fill-in); LU is run on a *subset* (the 3 largest-shift systems)
    # purely as an order-of-magnitude direct-solve reference.
    LU_SUBSET = 3
    factories = {
        "rs_default": rs_factory(theta=0.25, interp="classical"),
        "rs_direct": rs_factory(theta=0.25, interp="direct"),
        "rs_theta01": rs_factory(theta=0.1, interp="classical"),
        "rs_theta05": rs_factory(theta=0.5, interp="classical"),
        "sa_std": sa_factory(strength="symmetric"),
        "sa_classical": sa_factory(strength="classical"),
    }

    summary_rows = []
    conv_csv = []

    # LU reference first (subset only; direct solve on 3D grids is expensive)
    lu_rows = []
    for shift, h_vec, A, b in systems[:LU_SUBSET]:
        r = solve_lu(A, b)
        r["shift"] = float(shift)
        r["h"] = h_vec
        lu_rows.append(r)
    lu_tot = dict(
        setup_s=sum(r["setup_s"] for r in lu_rows),
        iter_s=sum(r["iter_s"] for r in lu_rows),
        iters=sum(r["iters"] for r in lu_rows),
        converged=all(r["converged"] for r in lu_rows),
    )

    all_results = {"lu": {"per_system": lu_rows, "totals": lu_tot}}
    print(
        f"\n{'cfg':<14}{'setup_s':>9}{'iter_s':>9}{'iters':>7}  conv  "
        f"{'setup+iter':>11}"
    )
    print(
        f"{'LU (direct)':<14}{lu_tot['setup_s']:>9.2f}{lu_tot['iter_s']:>9.2f}"
        f"{lu_tot['iters']:>7}  {str(lu_tot['converged']):<6}"
        f"{lu_tot['setup_s'] + lu_tot['iter_s']:>11.2f}"
    )

    for name, factory in factories.items():
        rows = []
        tot = dict(setup_s=0.0, iter_s=0.0, iters=0, converged=True)
        for shift, h_vec, A, b in systems:
            r = solve_with(factory, A, b)
            r["shift"] = float(shift)
            r["h"] = [float(x) for x in h_vec]
            rows.append(r)
            tot["setup_s"] += r["setup_s"]
            tot["iter_s"] += r["iter_s"]
            tot["iters"] += r["iters"]
            tot["converged"] &= r["converged"]
        all_results[name] = {"per_system": rows, "totals": tot}
        total_s = tot["setup_s"] + tot["iter_s"]
        print(
            f"{name:<14}{tot['setup_s']:>9.2f}{tot['iter_s']:>9.2f}"
            f"{tot['iters']:>7}  {str(tot['converged']):<6}{total_s:>11.2f}"
        )
        # convergence curve of the hardest system (smallest shift) per config
        hard = rows[-1]
        for it, res in enumerate(hard["residual_history"]):
            conv_csv.append([name, f"{hard['shift']:.6e}", it, res])

    # unpreconditioned CG on the 3 easiest (largest-shift) systems
    cg_rows = []
    for shift, h_vec, A, b in systems[:2] + systems[-1:]:
        r = cg_only(A, b)
        r["shift"] = float(shift)
        r["h"] = [float(x) for x in h_vec]
        cg_rows.append(r)
    all_results["cg_noprecond"] = {"per_system": cg_rows}
    for r in cg_rows:
        print(
            f"{'cg_noprecond':<14}{'0.00':>9}{r['iter_s']:>9.2f}{r['iters']:>7}"
            f"  {str(r['converged']):<6}  shift={r['shift']:.4g}"
        )

    # model/plan metadata
    meta = {
        "cells": int(n),
        "lambda_min": float(lambda_min),
        "lambda_max": float(lambda_max),
        "kappa": float(kappa),
        "shift_count": int(shifts.size),
        "candidate_count": len(systems),
        "rtol": ENRICH_RTOL,
        "systems": [
            {"shift": float(s), "h": [float(x) for x in h]} for s, h, _, _ in systems
        ],
    }

    payload = {"meta": meta, "results": all_results}
    (OUT / "amg_diagnostic.json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8"
    )
    with open(OUT / "amg_diagnostic_convergence.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "shift", "iter", "residual"])
        w.writerows(conv_csv)
    print("\njson ->", OUT / "amg_diagnostic.json")


if __name__ == "__main__":
    main()
