#!/usr/bin/env python3
r"""E2 — per-shift moment-continuation A/B (algorithm-level extraction variant).

Baseline is the faithful Extended FANTASTIC 2021 Algorithm 1 as implemented in
``metahotspot.macromodel.utils`` (one response per failed probe, one full solve
per response).  The variant keeps the exact same certification (random-h probe,
``probe_rounds`` consecutive passes) and the exact same closing SVD, but at a
failed probe it computes not one but ``moment_rounds`` transfer-function moments
at the SAME operator ``A(sigma, h)``:

    v_1 = A^{-1} g                (the physical response, as in the baseline)
    v_{j+1} = A^{-1} (C v_j)      (j-th moment continuation, same AMG setup)

This is the classical multipoint-moment-matching continuation (Codecasa et al.
2003/2005; the mechanism behind FloTHERM RomCore's per-(source,shift) Krylov
moment iteration, cf. expr0907 survey).  Because every moment in one chain
reuses the operator, ONE Ruge-Stueben AMG hierarchy serves ``moment_rounds``
full solves, cutting the AMG-setup-dominated part of the enrich stage.

Metrics (matched tolerance/seed/h-range, never a parameter sweep):
  * amg_setup_count, enrich_solve_count, cg wall, candidate count
  * pre-SVD order, kept order, relative_response_error, extraction wall time
  * holdout accuracy at the reproduce_case1 scenario (h=50/1000 W/m2K,
    P=[0.1,0.2,0.3,0.4] W, dt=50 s, 2000 s): steady + transient junction
    errors vs full FVM, full-field recovery errors.

Reproducibility: the extracted bases are saved as .npz next to the JSON so the
accuracy numbers can be re-derived without re-extracting.

Output: results/0908/moment_enrich_ab.json (+ .npz bases).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CASE)]

import scipy.linalg  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel import utils  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    _BasisBuilder,
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
    orthonormalize_block,
    port_eigenvalue_bounds,
    random_h,
    spd_solve,
)

OUT = Path(__file__).resolve().parents[2] / "results" / "0908"
OUT.mkdir(parents=True, exist_ok=True)

ROM_TOL = 1.0e-3
PROBE_ROUNDS = 3
SEED = 20260805
MAX_ORDER = 1024
H_SCENARIO = (5.0e1, 1.0e3)  # physical HTC: crowns, FR4 bottom
POWER = np.array([0.1, 0.2, 0.3, 0.4])
DURATION_S = 2000.0
DT_S = 50.0


# ------------------------------------------------------------- timers --------

class StageClock:
    """Wall-time accumulation for the enrich stage (setup vs CG iterations)."""

    def __init__(self):
        self.setup_s = 0.0
        self.setup_n = 0
        self.solve_s = 0.0
        self.solve_n = 0
        self._orig_setup = utils._rs_preconditioner
        self._orig_spd = utils.spd_solve

    def install(self):
        utils._rs_preconditioner = self._setup
        utils.spd_solve = self._solve

    def _setup(self, A):
        t0 = time.perf_counter()
        r = self._orig_setup(A)
        self.setup_s += time.perf_counter() - t0
        self.setup_n += 1
        return r

    def _solve(self, A, b, x0=None, rtol=utils.ENRICH_RTOL):
        t0 = time.perf_counter()
        r = self._orig_spd(A, b, x0=x0, rtol=rtol)
        self.solve_s += time.perf_counter() - t0
        self.solve_n += 1
        return r


# ------------------------------------------------ moment-continuation driver

def run_extract(core, G, terms, h_ranges, *, moment_rounds, clock: StageClock):
    """Certified basis (Extended FANTASTIC Algorithm 1) with moment continuation.

    ``moment_rounds = 1`` reproduces the production path exactly (one moment =
    the physical response); larger values add the (A^-1 C) Krylov chain at each
    failed probe's operator.
    """
    builder = _BasisBuilder(
        core, G, terms, h_ranges,
        tolerance=ROM_TOL, max_order=MAX_ORDER,
        probe_rounds=PROBE_ROUNDS, seed=SEED,
    )

    def enrich_moment(h_vec, shift, x0):
        """Solve the (sigma,h) operator; append moment_rounds directions.

        Returns the physical response (first moment) as the warm start for the
        next operator, mirroring the production ``_enrich`` contract.
        """
        if builder.processed_count >= builder.order_limit:
            builder.converged = False
            return x0 if x0 is not None else np.zeros(builder.g_vec.size)
        A = builder._candidate_A(h_vec, shift)
        prev = x0
        rhs = builder.g_vec.ravel()
        first_response = None
        for k in range(moment_rounds):
            x = np.asarray(spd_solve(A, rhs, x0=prev)).ravel()
            if first_response is None:
                first_response = x
            builder.snapshots.append(x.reshape(-1, 1))
            block = orthonormalize_block(builder.port_basis, x)
            if not block.shape[1]:
                break  # rank exhausted (moment chain saturated)
            B_old = builder.port_basis
            builder._extend_projected(block, B_old)
            builder.basis = np.column_stack((builder.basis, block))
            builder.processed_count += 1
            prev = x
            rhs = builder.C @ x  # next moment: A^{-1} C (A^{-1}C)^{k-1} g
        reference = max(float(first_response @ builder.g_vec), np.finfo(float).tiny)
        score_after = utils.response_score(
            first_response, builder.port_basis,
            builder._reduced_solve(h_vec, shift)[:, None], A, reference,
        )
        builder.worst_score = max(builder.worst_score, score_after)
        return first_response

    started = time.perf_counter()
    clock.install()
    for port in range(builder.n_src):
        g = builder.G[:, port:port + 1]
        lam_min, lam_max = port_eigenvalue_bounds(builder.K, builder.C, g)
        kappa = lam_max / max(lam_min, np.finfo(float).tiny)
        count = mpmm_elliptic_shift_count(ROM_TOL, lam_min, lam_max)
        shifts = mpmm_elliptic_shifts(count, lam_max, kappa)
        builder.per_port_plans.append({
            "port": int(port), "lambda_min": float(lam_min),
            "lambda_max": float(lam_max), "kappa": float(kappa),
            "shift_count": int(count), "shifts_per_s": shifts.tolist(),
        })
        builder._init_port(g)
        x0 = None
        for shift in shifts:
            if not builder.converged:
                break
            passes = 0
            h_samples = 0
            while passes < builder.probe_rounds and builder.converged:
                sub_seed = (builder.seed + 1003 * port
                            + int(round(float(shift) * 1.0e6)) + h_samples)
                h_vec = random_h(builder.h_ranges, sub_seed)
                h_samples += 1
                builder.candidate_total += 1
                if builder._probe_residual(h_vec, shift) <= builder.tolerance:
                    passes += 1
                    continue
                passes = 0
                x0 = enrich_moment(h_vec, shift, x0)  # warm start = response
            builder.outer_idx += 1
            builder.history.append({
                "outer_idx": builder.outer_idx, "port": int(port),
                "shift": float(shift), "h_samples": h_samples,
            })
        if not builder.converged:
            break

    # closing SVD (production semantics: tol/10 cutoff on the snapshot matrix)
    pre_svd_order = int(builder.basis.shape[1])
    snapshot_matrix = np.column_stack(builder.snapshots)
    U_s, s_s, _ = scipy.linalg.svd(
        snapshot_matrix, full_matrices=False, check_finite=False)
    svd_tol = ROM_TOL / 10.0
    cutoff = svd_tol * float(np.linalg.norm(s_s))
    keep = np.flatnonzero(s_s >= cutoff)
    basis = np.ascontiguousarray(U_s[:, keep])

    constant = np.ones((builder.internal_order, 1))
    constant /= np.linalg.norm(constant)
    constant = orthonormalize_block(basis, constant)
    if constant.shape[1]:
        basis = np.column_stack((basis, constant))
    orth_err = float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1])))) \
        if basis.shape[1] else 0.0
    if orth_err > 1.0e-10:
        raise RuntimeError("moment basis lost orthogonality")

    summary = {
        "per_port_plans": builder.per_port_plans,
        "tolerance": ROM_TOL, "probe_rounds": PROBE_ROUNDS,
        "candidate_count": builder.candidate_total,
        "processed_candidate_count": builder.processed_count,
        "outer_count": builder.outer_idx,
        "basis_order": int(basis.shape[1]), "pre_svd_order": pre_svd_order,
        "maximum_order": int(builder.order_limit),
        "orthogonality_error": orth_err,
        "relative_response_error": float(builder.worst_score),
        "converged": bool(builder.converged),
        "history": builder.history,
        "seconds": time.perf_counter() - started,
        "moment_rounds": moment_rounds,
        "amg_setup_count": clock.setup_n,
        "enrich_solve_count": clock.solve_n,
        "amg_setup_s": round(clock.setup_s, 4),
        "enrich_cg_s": round(clock.solve_s, 4),
    }
    return basis, summary


# ------------------------------------------------------------ accuracy -------

def evaluate(model, basis, summary):
    from metahotspot.macromodel.utils import (
        assemble_reduced_k, project_bci, solve_rom_steady, solve_rom_transient,
        accuracy_summary,
    )
    core = model.core
    G = model.source_shape
    terms = model.boundary_terms
    h_ranges = model.h_ranges()
    del h_ranges

    t0 = time.perf_counter()
    full = model.full_reference(H_SCENARIO)
    ref_s = time.perf_counter() - t0
    Tf_ss = full.steady_temperature
    junc_full_ss = model.junction_temperature(Tf_ss)
    junc_full_hist = model.junction_temperature(full.history)

    C_hat, K0, F_hat, F_bdry, A_bdry = project_bci(core, G, terms, basis)
    p_vec = model.physical_to_effective(H_SCENARIO)
    K_hat = assemble_reduced_k(K0, F_bdry, A_bdry, p_vec)

    t0 = time.perf_counter()
    theta_ss = solve_rom_steady(K_hat, F_hat, POWER)
    r_times, theta_hist = solve_rom_transient(
        C_hat, K_hat, F_hat, lambda t: POWER, DT_S, DURATION_S)
    rom_s = time.perf_counter() - t0
    junc_rom_ss = model.ambient_K + F_hat.T @ theta_ss
    junc_rom_hist = model.ambient_K + theta_hist @ F_hat
    rec_ss = model.ambient_K + basis @ theta_ss
    rec_hist = model.ambient_K + theta_hist @ basis.T
    acc = accuracy_summary(Tf_ss, rec_ss, full.history, rec_hist, model.ambient_K)

    def pct_err(a, b):
        denom = np.abs(junc_full_ss - model.ambient_K)
        return 100.0 * np.max(np.abs(a - b), axis=0) / np.maximum(denom, 1e-9)

    err_rom = pct_err(junc_rom_hist, junc_full_hist)
    return {
        "order": int(basis.shape[1]),
        "full_reference_s": round(ref_s, 2),
        "rom_online_s": round(rom_s, 3),
        "steady_junction_maxerr_K": float(np.max(np.abs(junc_rom_ss - junc_full_ss))),
        "steady_junction_maxerr_pct": float(
            100.0 * np.max(np.abs(junc_rom_ss - junc_full_ss))
            / np.max(np.abs(junc_full_ss - model.ambient_K))),
        "transient_junction_maxerr_pct": [float(v) for v in err_rom],
        "transient_junction_maxerr_K": float(
            np.max(np.abs(junc_rom_hist - junc_full_hist))),
        "steady_field_max_abs_rise_err_K": acc["steady_max_absolute_rise_error_K"],
        "steady_field_max_relative_rise_err": acc["steady_max_relative_rise_error"],
        "transient_field_max_abs_rise_err_K":
            acc["transient_final_max_absolute_rise_error_K"],
        "accuracy_passed": bool(acc["accuracy_passed"]),
    }


def main():
    model = Case1Model(Case1Config(max_xy_cell_mm=1.0, max_z_cell_mm=1.0))
    core = model.core
    G = model.source_shape
    terms = model.boundary_terms
    h_ranges = model.h_ranges()
    print(f"model cells={core.K.shape[0]}, ports={G.shape[1]}, groups={len(terms)}")

    runs = {}
    for k in (1, 2, 3):
        print(f"--- moment_rounds={k}")
        clock = StageClock()
        t0 = time.perf_counter()
        basis, summary = run_extract(core, G, terms, h_ranges, moment_rounds=k,
                                     clock=clock)
        wall = time.perf_counter() - t0
        print(f"  wall={wall:.1f}s  order={basis.shape[1]}  "
              f"pre_svd={summary['pre_svd_order']}  "
              f"solves={summary['enrich_solve_count']}  "
              f"amg_setups={summary['amg_setup_count']}")
        acc = evaluate(model, basis, summary)
        print("  " + json.dumps(acc))
        np.savez(OUT / f"moment_enrich_k{k}_basis.npz", basis=basis,
                 summary_json=json.dumps(summary))
        runs[k] = {
            "wall_time_s": round(wall, 2),
            "basis_order": int(basis.shape[1]),
            "pre_svd_order": int(summary["pre_svd_order"]),
            "candidate_count": int(summary["candidate_count"]),
            "processed_candidate_count": int(summary["processed_candidate_count"]),
            "amg_setup_count": int(summary["amg_setup_count"]),
            "amg_setup_s": summary["amg_setup_s"],
            "enrich_cg_s": summary["enrich_cg_s"],
            "relative_response_error": float(summary["relative_response_error"]),
            "converged": bool(summary["converged"]),
            "accuracy": acc,
        }

    payload = {
        "scenario": "Case-1 122,400 cells; 4 ports; 2 groups; tol=1e-3; "
                    "probe_rounds=3; seed=20260805; holdout h=(50,1000), "
                    "P=[0.1,0.2,0.3,0.4]W, dt=50s, 2000s; ambient 308.15K",
        "meaning": "moment_rounds=1 is the faithful Algorithm 1 (production "
                   "semantics); k>1 appends (A^-1 C) moments at each failed "
                   "probe's operator, sharing one AMG hierarchy per chain",
        "runs": runs,
    }
    (OUT / "moment_enrich_ab.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
