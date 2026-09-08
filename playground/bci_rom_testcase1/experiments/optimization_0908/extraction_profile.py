#!/usr/bin/env python3
r"""E1 — sub-stage profile of the production BCI extraction (0908).

Baseline: Case-1 122,400-cell model, tol=1e-3, probe_rounds=3, seed=20260805
(identical to expr0907 E1b so the profile is comparable).  Stage timers are
installed by monkey-patching ``metahotspot.macromodel.utils`` module globals —
no production code touched:

  * _rs_preconditioner  -> AMG setup (count + wall)
  * spd_solve           -> enrich full solves (wall) [called only from _enrich]
  * port_eigenvalue_bounds -> spectral planning per port
  * _BasisBuilder._probe_residual / _extend_projected / _reduced_solve
                         -> probe / incremental projection / reduced solves

The closing SVD is timed separately on the captured snapshot matrix, so the
profile reports every sub-stage of "spectral planning, probe, full sparse
solves (setup vs CG), incremental projections, final SVD" plus the wall clock
of the full extraction.  Output: results/0908/extraction_profile.json
(+ summary CSV row appended to results/0908/summary_extraction0908.csv).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

import numpy as np

CASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CASE)]

from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel import utils  # noqa: E402
from metahotspot.macromodel.utils import build_parametric_basis  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "0908"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- timers ----

T = {"amg_setup": 0.0, "spd_solve": 0.0, "spectral": 0.0,
     "probe": 0.0, "extend": 0.0, "reduced": 0.0}
AMG_SETUPS = [0]
SPD_SOLVES = [0]


def wrap(obj, target, key):
    fn = getattr(obj, target)
    setattr(obj, target, _wrap(fn, key))
    return None


def _wrap(fn, key):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        T[key] += time.perf_counter() - t0
        if key == "amg_setup":
            AMG_SETUPS[0] += 1
        if key == "spd_solve":
            SPD_SOLVES[0] += 1
        return r
    return wrapper


def main():
    model = Case1Model(Case1Config(max_xy_cell_mm=1.0, max_z_cell_mm=1.0))
    core = model.core
    G = model.source_shape
    terms = model.boundary_terms
    h_ranges = model.h_ranges()
    n = core.K.shape[0]
    print(f"model: {model.name}  full cells={n}  ports={len(model.source_ports())}")

    # instrument utils module globals + _BasisBuilder methods
    _orig_setup = utils._rs_preconditioner
    _orig_spd = utils.spd_solve
    utils._rs_preconditioner = _wrap(_orig_setup, "amg_setup")
    utils.spd_solve = _wrap(_orig_spd, "spd_solve")
    utils.port_eigenvalue_bounds = _wrap(utils.port_eigenvalue_bounds, "spectral")
    wrap(utils._BasisBuilder, "_probe_residual", "probe")
    wrap(utils._BasisBuilder, "_extend_projected", "extend")
    wrap(utils._BasisBuilder, "_reduced_solve", "reduced")

    t0 = time.perf_counter()
    basis, summary = build_parametric_basis(
        core, G, terms, h_ranges,
        tolerance=1.0e-3, max_order=1024, probe_rounds=3, seed=20260805,
    )
    wall = time.perf_counter() - t0

    # closing SVD: the operation costs depend only on the matrix shape
    # (n x pre_svd_order), so time a standalone SVD of a matrix of the exact
    # same shape as the snapshot matrix the production run() truncates.
    import scipy.linalg
    s_dim = int(summary["pre_svd_order"])
    t0 = time.perf_counter()
    scipy.linalg.svd(
        np.random.default_rng(0).standard_normal((n, s_dim)),
        compute_uv=False, check_finite=False,
    )
    svd_s = time.perf_counter() - t0

    enrich_total = T["spd_solve"]
    payload = {
        "scenario": "Case-1 122,400 cells; 4 source ports; 2 boundary groups; "
                    "tol=1e-3; probe_rounds=3; seed=20260805",
        "wall_time_s": round(wall, 3),
        "basis_order": int(basis.shape[1]),
        "pre_svd_order": int(summary["pre_svd_order"]),
        "candidate_count": int(summary["candidate_count"]),
        "processed_candidate_count": int(summary["processed_candidate_count"]),
        "amg_setup_count": AMG_SETUPS[0],
        "enrich_solve_count": SPD_SOLVES[0],
        "relative_response_error": float(summary["relative_response_error"]),
        "timers_s": {
            "spectral_planning": round(T["spectral"], 3),
            "probe": round(T["probe"], 3),
            "enrich_solves_total": round(enrich_total, 3),
            "amg_setup": round(T["amg_setup"], 3),
            "incremental_projection": round(T["extend"], 3),
            "reduced_solve": round(T["reduced"], 3),
            "closing_svd_standalone_nx4": round(svd_s, 3),
            "sum_of_stages": round(T["spectral"] + T["probe"] + enrich_total
                                   + T["extend"] + T["reduced"] + svd_s, 3),
        },
        "shares": {
            "spectral": round(T["spectral"] / wall, 4),
            "probe": round(T["probe"] / wall, 4),
            "enrich": round(enrich_total / wall, 4),
            "amg_setup_of_enrich": round(T["amg_setup"] / enrich_total, 4),
            "incremental_projection": round(T["extend"] / wall, 4),
        },
        "solver": {
            "amg_interpolation": "direct",
            "cg_rtol": utils.ENRICH_RTOL,
            "preconditioner": "Ruge-Stueben AMG V-cycle, direct interpolation",
        },
        "per_port_plans": summary["per_port_plans"],
    }
    (OUT / "extraction_profile.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(payload, indent=1))

    csv_path = OUT / "summary_extraction0908.csv"
    header = ["family", "tag_or_mesh_mm", "cells", "basis_order", "pre_svd_order",
              "candidates", "processed", "wall_time_s", "probe_s", "enrich_s",
              "amg_setup_s", "amg_setup_count", "spectral_s",
              "relative_response_error"]
    row = ["profile_122400", "1.0", n, basis.shape[1], summary["pre_svd_order"],
           summary["candidate_count"], summary["processed_candidate_count"],
           round(wall, 3), round(T["probe"], 4), round(enrich_total, 4),
           round(T["amg_setup"], 4), AMG_SETUPS[0], round(T["spectral"], 4),
           f"{summary['relative_response_error']:.6e}"]
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


if __name__ == "__main__":
    main()
