#!/usr/bin/env python3
r"""E1b — End-to-end extraction A/B: RS classical vs direct interpolation.

E1 showed direct interpolation halves per-solve AMG setup with unchanged CG
iteration count.  This harness proves the win end-to-end through the *real*
production extraction (``build_parametric_basis``): same tolerance/seed/mesh,
only the ``interpolation`` argument of ``pyamg.ruge_stuben_solver`` differs
(classical = production default, direct = candidate).  Each variant produces a
basis that is then projected and scored at the reference scenario, so the
comparison is order + accuracy at matched settings with extraction wall-time.

Because the two extractions share everything except the AMG setup kernel,
the wall-time delta is the setup saving multiplied by the solve count
(expected ~154 x 0.18 s ~ 28 s, i.e. 88 -> ~60 s).

Outputs: results/optimization/amg_interpolation_ab.json/.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import pyamg

CASE = Path(__file__).resolve().parents[2]
ROOT = CASE.parents[1]
sys.path[:0] = [str(CASE)]

from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel import utils  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    ENRICH_RTOL,
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


def run_variant(model, core, G, terms, h_ranges, *, interpolation, tag):
    """Full production extraction with an AMG interpolation override."""
    orig = pyamg.ruge_stuben_solver
    calls = {"n": 0}

    def patched(A, **kw):
        calls["n"] += 1
        kw["interpolation"] = interpolation
        return orig(A, **kw)

    pyamg.ruge_stuben_solver = patched
    try:
        t0 = time.perf_counter()
        basis, summary = build_parametric_basis(
            core,
            G,
            terms,
            h_ranges,
            tolerance=1.0e-3,
            max_order=1024,
            probe_rounds=3,
            seed=20260805,
        )
        extract_s = time.perf_counter() - t0
    finally:
        pyamg.ruge_stuben_solver = orig

    C_hat, K0, F_hat, F_b, A_b = project_bci(core, G, terms, basis)
    p_vec = model.physical_to_effective(H_SCENARIO)
    K_hat = assemble_reduced_k(K0, F_b, A_b, p_vec)
    theta_ss = solve_rom_steady(K_hat, F_hat, POWER_W)
    rec_ss = AMBIENT + basis @ theta_ss
    _, theta_hist = solve_rom_transient(
        C_hat, K_hat, F_hat, lambda _: POWER_W, DT_S, DURATION_S
    )
    rec_hist = AMBIENT + theta_hist @ basis.T
    full = model.full_reference(H_SCENARIO)
    acc = accuracy_summary(
        full.steady_temperature, rec_ss, full.history, rec_hist, AMBIENT
    )

    return {
        "tag": tag,
        "interpolation": interpolation,
        "basis_order": int(basis.shape[1]),
        "pre_svd_order": int(summary["pre_svd_order"]),
        "amg_setup_calls": calls["n"],
        "extract_s": round(extract_s, 2),
        "relative_response_error": float(summary["relative_response_error"]),
        "steady_max_abs_rise_err_K": acc["steady_max_absolute_rise_error_K"],
        "steady_max_rel_rise_err": acc["steady_max_relative_rise_error"],
        "transient_final_max_abs_rise_err_K": acc[
            "transient_final_max_absolute_rise_error_K"
        ],
        "transient_final_max_rel_rise_err": acc[
            "transient_final_max_relative_rise_error"
        ],
    }


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
    print(f"cells={model.full_cell_count}")

    rows = []
    for interp, tag in (
        ("classical", "production_classical"),
        ("direct", "candidate_direct"),
    ):
        print(f"\n=== {tag} ({interp}) ===", flush=True)
        row = run_variant(
            model, core, G, terms, h_ranges, interpolation=interp, tag=tag
        )
        print(row, flush=True)
        rows.append(row)

    (OUT / "amg_interpolation_ab.json").write_text(
        json.dumps(rows, indent=1, default=float), encoding="utf-8"
    )
    with open(OUT / "amg_interpolation_ab.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n->", OUT / "amg_interpolation_ab.json")


if __name__ == "__main__":
    main()
