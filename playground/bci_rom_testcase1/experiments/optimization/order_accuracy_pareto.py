#!/usr/bin/env python3
r"""E3 — tolerance -> (order, extraction time, steady/transient accuracy) Pareto.

The BCI-FANTASTIC extraction tolerance epsilon controls (a) the elliptic shift
count per source (hence pre-SVD order and extraction cost), (b) the residual
probe threshold, and (c) the closing SVD cut (epsilon/10 . ||s||_2).  This
experiment sweeps epsilon over {1e-2, 3e-3, 1e-3, 3e-4, 1e-4} on the Case-1
122,400-cell model, and scores each extracted ROM at the reference scenario
(H_crown=50, H_fr4=1000, P=[0.1..0.4]) on steady + transient junction and
full-field error vs the monolithic FVM solve — so the Pareto point is
order/cost vs matched *holdout* accuracy, not the training residual.

Outputs: results/optimization/order_accuracy_pareto.json/.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parents[2]
ROOT = CASE.parents[1]
sys.path[:0] = [str(CASE)]

from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    accuracy_summary,
    assemble_reduced_k,
    build_parametric_basis,
    project_bci,
    solve_rom_steady,
    solve_rom_transient,
)

OUT = ROOT / "playground" / "bci_rom_testcase1" / "results" / "optimization"

AMBIENT = 308.15
H_SCENARIO = (50.0, 1000.0)
POWER_W = np.array([0.1, 0.2, 0.3, 0.4])
DT_S = 50.0
DURATION_S = 2000.0
TOLERANCES = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4]
SEED = 20260805


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = Case1Model(
        Case1Config(
            max_xy_cell_mm=1.0, max_z_cell_mm=1.0, dt_s=DT_S, duration_s=DURATION_S
        )
    )
    core = model.core
    G = model.source_shape
    terms = model.boundary_terms
    h_ranges = model.h_ranges()

    full = model.full_reference(H_SCENARIO)
    print(
        f"cells={model.full_cell_count}  full steady+transient ref "
        f"{full.steady_s + full.transient_s:.1f}s"
    )
    ref_ss = full.steady_temperature
    ref_hist = full.history
    junc_ref = model.junction_temperature(ref_ss)

    rows = []
    for tol in TOLERANCES:
        t0 = time.perf_counter()
        basis, summary = build_parametric_basis(
            core,
            G,
            terms,
            h_ranges,
            tolerance=tol,
            max_order=2048,
            probe_rounds=3,
            seed=SEED,
        )
        extract_s = time.perf_counter() - t0
        order = int(basis.shape[1])

        C_hat, K0, F_hat, F_b, A_b = project_bci(core, G, terms, basis)
        p_vec = model.physical_to_effective(H_SCENARIO)
        K_hat = assemble_reduced_k(K0, F_b, A_b, p_vec)

        theta_ss = solve_rom_steady(K_hat, F_hat, POWER_W)
        rec_ss = AMBIENT + basis @ theta_ss
        t = time.perf_counter()
        _, theta_hist = solve_rom_transient(
            C_hat, K_hat, F_hat, lambda _: POWER_W, DT_S, DURATION_S
        )
        rom_tr_s = time.perf_counter() - t
        rec_hist = AMBIENT + theta_hist @ basis.T
        acc = accuracy_summary(ref_ss, rec_ss, ref_hist, rec_hist, AMBIENT)

        row = {
            "tolerance": tol,
            "basis_order": order,
            "pre_svd_order": int(summary["pre_svd_order"]),
            "extract_s": round(extract_s, 1),
            "rom_transient_s": round(rom_tr_s, 2),
            "shift_count_per_port": [
                p["shift_count"] for p in summary["per_port_plans"]
            ],
            "steady_max_abs_rise_err_K": acc["steady_max_absolute_rise_error_K"],
            "steady_max_rel_rise_err": acc["steady_max_relative_rise_error"],
            "transient_final_max_abs_rise_err_K": acc[
                "transient_final_max_absolute_rise_error_K"
            ],
            "transient_final_max_rel_rise_err": acc[
                "transient_final_max_relative_rise_error"
            ],
            "accuracy_passed": acc["accuracy_passed"],
        }
        rows.append(row)
        print(row, flush=True)

    (OUT / "order_accuracy_pareto.json").write_text(
        json.dumps(rows, indent=1, default=float), encoding="utf-8"
    )
    with open(OUT / "order_accuracy_pareto.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n->", OUT / "order_accuracy_pareto.json")


if __name__ == "__main__":
    main()
