#!/usr/bin/env python3
"""Verification of the three claims written up in the report.

Everything else that was tried in this directory has been removed; this is the
set that backs the numbers in section 2 of weekly_report_0923.md.

1. exactness -- the Woodbury form reproduces full-order solves;
2. conditioning -- the small system is ill-conditioned near the reference point
   but the field error stays at solver accuracy;
3. expansion -- the pole series sums to the same solution.

Run: python playground/adaptive_bci_sampling/verify.py 2.5
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "bci_rom_testcase1"))

from model_case1 import Case1Config, Case1Model  # noqa: E402
from rational_surrogate import BoundaryRationalModel  # noqa: E402
from metahotspot.macromodel.utils import spd_solve  # noqa: E402

MM = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5

model = Case1Model(Case1Config(max_xy_cell_mm=MM, max_z_cell_mm=MM))
K = model.core.K.tocsc()
terms = list(model.boundary_terms)
G = model.source_shape
P = model.nominal_power()
ranges = np.asarray(model.h_ranges(), dtype=np.float64)
b = G @ P
k_par = ranges.shape[0]
p_ref = ranges[:, 1].copy()
sur = BoundaryRationalModel(K, terms, G, p_ref)
print(
    f"mesh={MM} mm  cells={K.shape[0]}  groups={k_par}  m_b={sur.m_b}  "
    f"reference solves={sur.m_b + G.shape[1]}"
)


def operator(p):
    A = K.copy()
    for pk, H in zip(np.asarray(p, dtype=np.float64), terms):
        A = A + float(pk) * H
    return A.tocsc()


def exact(p):
    return np.asarray(spd_solve(operator(p), b), dtype=np.float64).ravel()


# ---------------------------------------------------------------- 1. exactness
rng = np.random.default_rng(20260922)
log_lo, log_hi = np.log10(ranges[:, 0]), np.log10(ranges[:, 1])
probes = [("box corner min", ranges[:, 0]), ("box corner max", ranges[:, 1])]
for _ in range(8):
    probes.append(
        ("log-uniform", 10.0 ** (log_lo + rng.random(k_par) * (log_hi - log_lo)))
    )
probes.append(("box face", np.array([ranges[0, 0], p_ref[1]])))
print("\n=== 1. exactness of the Woodbury form ===")
worst_x = worst_j = 0.0
for name, p in probes:
    x_sur = sur.response(p) @ P
    x_dir = exact(p)
    worst_x = max(worst_x, float(np.linalg.norm(x_sur - x_dir) / np.linalg.norm(x_dir)))
    jd = np.asarray(G.T @ x_dir).ravel()
    js = np.asarray(G.T @ x_sur).ravel()
    worst_j = max(worst_j, float(np.max(np.abs(js - jd)) / np.max(np.abs(jd))))
print(f"{'worst relative field error':>34}{worst_x:14.2e}")
print(f"{'worst relative junction error':>34}{worst_j:14.2e}")


# ------------------------------------------------------------ 2. conditioning
def condition_of_M(p):
    lam = sur.lambdas(p)
    act = np.flatnonzero(lam != 0.0)
    if act.size == 0:
        return 1.0
    M = sur.Phi[np.ix_(act, act)].copy()
    M[np.diag_indices(act.size)] += 1.0 / lam[act]
    return float(np.linalg.cond(M))


# worst where a parameter sits closest to the reference point, since 1/lambda
# blows up there; sweep towards it and keep the worst pair
worst_cond, worst_cond_at, worst_cond_err = 0.0, None, 0.0
for eps in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
    for k in range(k_par):
        p = ranges[:, 0].copy()
        p[k] = p_ref[k] * (1.0 - eps)
        c = condition_of_M(p)
        if c > worst_cond:
            worst_cond = c
            worst_cond_at = (k, eps)
            worst_cond_err = float(
                np.linalg.norm(sur.response(p) @ P - exact(p))
                / np.linalg.norm(exact(p))
            )
print(
    f"\n{'cond(M) at the hardest corner':>34}" f"{condition_of_M(ranges[:, 0]):14.2e}"
)
print(
    f"{'worst cond(M) over probes':>34}{worst_cond:14.2e}"
    f"   (group {worst_cond_at[0]}, 1-{worst_cond_at[1]:g} of p_ref)"
)
print(f"{'field error at that worst cond':>34}{worst_cond_err:14.2e}")


# ------------------------------------------------------------------ 3. series
print("\n=== 3. the pole series sums to the same solution ===")
target = ranges[:, 0].copy()
t_pole, coefs = sur.ray_pole_weights(target)
worst = 0.0
for t in (0.1, 0.3, 0.5, 0.7, 0.9):
    p = p_ref + t * (target - p_ref)
    s = (-t * coefs / (1.0 - t * (1.0 / t_pole))[None, :, None]).sum(axis=1)
    x_exp = (sur.x_ref - sur.G_b @ s) @ P
    worst = max(
        worst, float(np.linalg.norm(x_exp - exact(p)) / np.linalg.norm(exact(p)))
    )
print(f"{'worst relative error of the series (diagonal ray)':>46}{worst:12.2e}")
