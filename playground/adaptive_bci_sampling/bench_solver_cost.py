"""Cost benchmark at 1 mm: Ruge-Stuben AMG setup+CGR versus direct LU.

Both solver paths are timed on the same representative operators of the Case 1
HTC family: the four RHS of the four sources share one setup in either solver,
so the comparison is setup count times setup cost plus right-hand sides times
solve cost.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

import pyamg  # noqa: E402
from deterministic_design import shared_frequency_plan  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402

mesh = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3

clock = time.perf_counter()
model = Case1Model(Case1Config(max_xy_cell_mm=mesh, max_z_cell_mm=mesh))
kernel = model.core.K.tocsc()
mass = model.core.C.tocsc()
source = np.asarray(model.source_shape, dtype=np.float64)
terms = [term.tocsc() for term in model.boundary_terms]
ranges = np.asarray(model.h_ranges(), dtype=np.float64)
print(f"mesh={mesh} mm  n={kernel.shape[0]}  nnz(K)={kernel.nnz}  nnz(C)={mass.nnz}  "
      f"ports={source.shape[1]}  built in {time.perf_counter() - clock:.1f}s", flush=True)

plan = shared_frequency_plan(kernel, mass, source, 1e-3)
shifts = [float(value) for value in plan["shifts"]]
print(f"frequency plan: count={plan['count']}  lambda=[{plan['lower']:.6g}, "
      f"{plan['upper']:.6g}]  shifts={len(shifts)}", flush=True)

diagonal = sum(np.asarray(term.diagonal()).ravel() for term in terms)
print(f"boundary cells m_b={int((diagonal > 0).sum())}", flush=True)

seed = np.sqrt(ranges[:, 0] * ranges[:, 1])


def operator(shift, point):
    value = kernel if not shift else kernel + float(shift) * mass
    for factor, term in zip(point, terms):
        value = value + float(factor) * term
    return sp.csc_matrix(value)


cases = [("steady  s=0", 0.0), ("mid     s=%.3g" % shifts[len(shifts) // 2], shifts[len(shifts) // 2]),
         ("high    s=%.3g" % shifts[0], shifts[0])]
rows = []
for label, shift in cases:
    matrix = operator(shift, seed)
    csr = matrix.tocsr()
    rhs4 = np.ascontiguousarray(source)

    timings = {"amg_setup": [], "amg_cg4": [], "lu_factor": [], "lu_solve4": []}
    iterations = []
    residuals = []
    for attempt in range(repeats + 1):
        started = time.perf_counter()
        ml = pyamg.ruge_stuben_solver(csr, interpolation="direct")
        preconditioner = ml.aspreconditioner(cycle="V")
        setup = time.perf_counter() - started
        # One Ruge-Stuben setup, then one CG run per right-hand side: a
        # shared preconditioner cannot be applied to a block in scipy, and
        # spd_solve builds its own setup per call anyway.
        started = time.perf_counter()
        solution = np.empty_like(rhs4)
        info = 0
        for column in range(rhs4.shape[1]):
            solution[:, column], part = spla.cg(
                csr, np.ascontiguousarray(rhs4[:, column]), rtol=1e-6, atol=0.0,
                maxiter=2000, M=preconditioner,
            )
            info = info or int(part)
        cg = time.perf_counter() - started
        started = time.perf_counter()
        factor = spla.splu(matrix, permc_spec="MMD_AT_PLUS_A")
        factorisation = time.perf_counter() - started
        started = time.perf_counter()
        direct = factor.solve(rhs4)
        direct_time = time.perf_counter() - started
        if attempt:  # first round is warm-up
            timings["amg_setup"].append(setup)
            timings["amg_cg4"].append(cg)
            timings["lu_factor"].append(factorisation)
            timings["lu_solve4"].append(direct_time)
            residuals.append(float(np.max(np.abs(csr @ solution - rhs4))
                                   / np.max(np.abs(rhs4))))
        assert info == 0, info
        if attempt == repeats:
            agreement = float(np.max(np.abs(solution - direct)) / np.max(np.abs(direct)))
    row = {name: statistics.median(values) for name, values in timings.items()}
    print(f"{label}: AMG setup {row['amg_setup'] * 1e3:9.1f} ms | CGR(4 rhs) {row['amg_cg4'] * 1e3:9.1f} ms "
          f"| LU factor {row['lu_factor'] * 1e3:9.1f} ms | LU solve(4 rhs) {row['lu_solve4'] * 1e3:8.1f} ms "
          f"| CG vs LU {agreement:.2e}", flush=True)
