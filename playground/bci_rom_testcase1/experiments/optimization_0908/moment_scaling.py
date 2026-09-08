#!/usr/bin/env python3
r"""E2b — moment-continuation scaling at the coarsest resolution (0.75 mm).

The expr0907 scaling data shows the enrich stage growing super-linearly with
mesh size (122k -> 329k cells: 54s -> 183s); AMG setup dominates it.  The
moment-continuation variant shares one Ruge-Stueben hierarchy across k solves
at the same operator, so its benefit should INCREASE with mesh size.  This
script runs the k=1 baseline and the k=2 variant on the 0.75 mm mesh
(329,280 cells) at the same tolerance/seed and reports the same counters as
moment_enrich_ab (extraction only — accuracy is order-verified by the
122k-cell A/B).

Output: results/0908/moment_scaling.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CASE)]

from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel import utils  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    _BasisBuilder,
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
    orthonormalize_block,
    port_eigenvalue_bounds,
    random_h,
)

OUT = Path(__file__).resolve().parents[2] / "results" / "0908"
OUT.mkdir(parents=True, exist_ok=True)

ROM_TOL = 1.0e-3
PROBE_ROUNDS = 3
SEED = 20260805
MAX_ORDER = 1024


class Clock:
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


def run_extract(core, G, terms, h_ranges, *, moment_rounds, clock: Clock):
    import scipy.linalg
    builder = _BasisBuilder(
        core, G, terms, h_ranges,
        tolerance=ROM_TOL, max_order=MAX_ORDER,
        probe_rounds=PROBE_ROUNDS, seed=SEED,
    )

    def enrich_moment(h_vec, shift, x0):
        if builder.processed_count >= builder.order_limit:
            builder.converged = False
            return x0 if x0 is not None else np.zeros(builder.g_vec.size)
        A = builder._candidate_A(h_vec, shift)
        prev = x0
        rhs = builder.g_vec.ravel()
        first_response = None
        for k in range(moment_rounds):
            x = np.asarray(utils.spd_solve(A, rhs, x0=prev)).ravel()
            if first_response is None:
                first_response = x.reshape(-1, 1)
            builder.snapshots.append(x.reshape(-1, 1))
            block = orthonormalize_block(builder.port_basis, x.reshape(-1, 1))
            if not block.shape[1]:
                break
            B_old = builder.port_basis
            builder._extend_projected(block, B_old)
            builder.basis = np.column_stack((builder.basis, block))
            builder.processed_count += 1
            prev = x
            rhs = builder.C @ x
        reference = max(float(first_response.ravel() @ builder.g_vec),
                        np.finfo(float).tiny)
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
            "shift_count": int(count),
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
                x0 = enrich_moment(h_vec, shift, x0)
            builder.outer_idx += 1
        if not builder.converged:
            break

    pre_svd_order = int(builder.basis.shape[1])
    snapshot_matrix = np.column_stack(builder.snapshots)
    U_s, s_s, _ = scipy.linalg.svd(
        snapshot_matrix, full_matrices=False, check_finite=False)
    cutoff = (ROM_TOL / 10.0) * float(np.linalg.norm(s_s))
    keep = np.flatnonzero(s_s >= cutoff)
    basis = np.ascontiguousarray(U_s[:, keep])
    constant = np.ones((builder.internal_order, 1))
    constant /= np.linalg.norm(constant)
    constant = orthonormalize_block(basis, constant)
    if constant.shape[1]:
        basis = np.column_stack((basis, constant))
    return basis, {
        "basis_order": int(basis.shape[1]),
        "pre_svd_order": pre_svd_order,
        "candidate_count": builder.candidate_total,
        "processed_candidate_count": builder.processed_count,
        "relative_response_error": float(builder.worst_score),
        "converged": bool(builder.converged),
        "moment_rounds": moment_rounds,
        "amg_setup_count": clock.setup_n,
        "enrich_solve_count": clock.solve_n,
        "amg_setup_s": round(clock.setup_s, 3),
        "enrich_cg_s": round(clock.solve_s, 3),
        "seconds": round(time.perf_counter() - started, 2),
    }


def main():
    model = Case1Model(Case1Config(max_xy_cell_mm=0.75, max_z_cell_mm=0.75))
    core = model.core
    G = model.source_shape
    terms = model.boundary_terms
    h_ranges = model.h_ranges()
    print(f"model cells={core.K.shape[0]} (0.75 mm mesh)")

    runs = {}
    for k in (1, 2):
        print(f"--- moment_rounds={k}")
        clock = Clock()
        basis, summary = run_extract(core, G, terms, h_ranges,
                                     moment_rounds=k, clock=clock)
        print(json.dumps(summary))
        runs[k] = summary
        np.savez(OUT / f"moment_scaling_k{k}_basis.npz", basis=basis)

    (OUT / "moment_scaling.json").write_text(
        json.dumps({
            "scenario": "Case-1 0.75 mm mesh (329,280 cells); tol=1e-3; "
                        "probe_rounds=3; seed=20260805",
            "cells": int(core.K.shape[0]),
            "runs": runs,
        }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
