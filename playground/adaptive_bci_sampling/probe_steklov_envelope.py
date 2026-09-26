#!/usr/bin/env python3
"""Falsify the whole-box Steklov tail proposal on the Case-1 meshes.

Run from the repository root with PYTHONPATH=python and a built C API.  This
probe deliberately does not change the extraction or claim a dynamic bound.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bci_rom_testcase1"))
from model_case1 import Case1Config, Case1Model  # noqa: E402
from deterministic_design import shared_frequency_plan  # noqa: E402


def probe(mesh: float, count: int = 30, *, steady_only: bool = False,
          full_dc_spectrum: bool = False) -> dict:
    started = time.perf_counter()
    model = Case1Model(Case1Config(max_xy_cell_mm=mesh, max_z_cell_mm=mesh))
    k = sp.csc_matrix(model.core.K)
    c = sp.csc_matrix(model.core.C)
    g = np.asarray(model.source_shape, dtype=float)
    terms = [sp.csc_matrix(h) for h in model.boundary_terms]
    ranges = np.asarray(model.h_ranges(), dtype=float)
    assert all((h - sp.diags(h.diagonal())).nnz == 0 for h in terms)
    hdiag = sum((hi - lo) * h.diagonal() for (lo, hi), h in zip(ranges, terms))
    active = np.flatnonzero(hdiag > 0)
    plan = shared_frequency_plan(k, c, g, 1e-3)
    shifts = sorted(set([0., min(plan["shifts"]), plan["shifts"][len(plan["shifts"]) // 2], max(plan["shifts"])]))
    if steady_only:
        shifts = [0.]
    output = {"mesh_mm": mesh, "n": k.shape[0], "boundary_dofs": len(active),
              "ranges_effective": ranges.tolist(), "shifts": shifts,
              "elapsed_setup_s": time.perf_counter() - started, "rows": []}
    print(f"mesh={mesh:g} n={k.shape[0]} boundary={len(active)} shifts={shifts}", flush=True)
    for shift in shifts:
        clock = time.perf_counter()
        base = k + shift * c
        low = (base + sum(float(p) * h for p, h in zip(ranges[:, 0], terms))).tocsc()
        upper = (low + sp.diags(hdiag, format="csc")).tocsc()
        factor = spla.splu(low, permc_spec="MMD_AT_PLUS_A")
        xlo = factor.solve(g)
        xhi = spla.spsolve(upper, g)
        zlo = g.T @ xlo
        zhi = g.T @ xhi
        kappa = float(la.eigh(zlo, zhi, eigvals_only=True)[-1])
        calls = 0
        weights = np.sqrt(hdiag[active])

        def action(v):
            nonlocal calls
            calls += 1
            b = np.zeros((k.shape[0], v.shape[1] if v.ndim == 2 else 1))
            b[active] = weights[:, None] * np.asarray(v).reshape(len(active), -1)
            result = weights[:, None] * factor.solve(b)[active]
            return result if v.ndim == 2 else result[:, 0]

        operator = spla.LinearOperator((len(active), len(active)), matvec=action,
                                        matmat=action, dtype=float)
        nev = min(count + 1, len(active) - 2)
        mu, vectors = spla.eigsh(operator, k=nev, which="LM", tol=1e-7)
        mu, vectors = mu[::-1], vectors[:, ::-1]
        threshold = 1e-3
        required = next((m for m in range(len(mu)) if mu[m] * kappa <= threshold), None)
        row = {"shift": shift, "kappa": kappa, "eigenvalues": mu.tolist(),
               "tail_times_kappa_at_m": {str(m): float(mu[m] * kappa)
                                         for m in (0, 3, 8, 20, 30) if m < len(mu)},
               "first_m_below_1e-3": required, "inverse_rhs_for_eigs": calls,
               "corner_source_rhs": 2 * g.shape[1],
               "elapsed_s": time.perf_counter() - clock}
        if shift == 0.:
            # Diagnostic only: finite corners are not a whole-box certificate.
            correction_rhs = np.zeros((k.shape[0], nev))
            correction_rhs[active] = weights[:, None] * vectors
            modes = factor.solve(correction_rhs) / np.sqrt(mu)[None, :]
            rows = {}
            for m in (0, 3, 8, 30):
                if m > nev:
                    continue
                trial = la.orth(np.column_stack((xlo, modes[:, :m])))
                maximum = 0.
                for bounds in itertools.product((0, 1), repeat=len(terms)):
                    point = ranges[np.arange(len(terms)), bounds]
                    a = (base + sum(float(p) * h for p, h in zip(point, terms))).tocsc()
                    x = spla.spsolve(a, g)
                    ar = trial.T @ (a @ trial)
                    xr = trial @ la.solve(ar, trial.T @ g, assume_a="pos")
                    error = (x - xr).T @ (a @ (x - xr))
                    exact = g.T @ x
                    maximum = max(maximum, float(la.eigh((error + error.T) / 2,
                                                            (exact + exact.T) / 2,
                                                            eigvals_only=True)[-1]))
                rows[str(m)] = maximum
            row["sampled_corner_relative_error"] = rows
            row["elapsed_s"] = time.perf_counter() - clock
        if shift == 0. and full_dc_spectrum:
            if len(active) > 1500:
                raise ValueError("full DC spectrum is intended for the 5/2.5 mm meshes")
            rhs = np.zeros((k.shape[0], len(active)))
            rhs[active, np.arange(len(active))] = weights
            gram = weights[:, None] * factor.solve(rhs)[active]
            entire = la.eigvalsh((gram + gram.T) * 0.5)
            row["entire_spectrum"] = {
                "smallest_mu": float(entire[0]),
                "required_modes_at_1e-3": int(np.count_nonzero(entire * kappa > threshold)),
                "additional_inverse_rhs": len(active),
            }
            row["elapsed_s"] = time.perf_counter() - clock
        output["rows"].append(row)
        print(f"shift={shift:.5g} kappa={kappa:.4g} tail={row['tail_times_kappa_at_m']} "
              f"eigen-RHS={calls} time={row['elapsed_s']:.2f}s", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=float)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steady-only", action="store_true")
    parser.add_argument("--full-dc-spectrum", action="store_true")
    args = parser.parse_args()
    result = probe(args.mesh, args.count, steady_only=args.steady_only,
                   full_dc_spectrum=args.full_dc_spectrum)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
