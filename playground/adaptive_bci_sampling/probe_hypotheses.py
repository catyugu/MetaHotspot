"""Check the hypotheses the monotonicity and weak-greedy arguments need.

Point 7 of the review: dY_ab/dp_k = -x_a^T H_k x_b <= 0 needs x >= 0, which holds
only if A(p) is a symmetric M-matrix and the sources are nonnegative -- H_k >= 0
alone gives the diagonal case only.

Point 3: with A_- = A(p_min) and Gamma = max_i p_max,i / p_min,i one has
A(p)^-1 <= A_-^-1 <= Gamma A(p)^-1, hence
    ||e_i||_{A(p)} <= sqrt(r_i^T A_-^-1 r_i) <= sqrt(Gamma) ||e_i||_{A(p)}.
The constant is mesh independent, but Gamma is 1e4 here, so the equivalence is
loose; a branch and bound would use a cell-local anchor instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

from deterministic_design import full_operator, zolotarev_seed  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402

mesh = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
model = Case1Model(Case1Config(max_xy_cell_mm=mesh, max_z_cell_mm=mesh))
kernel = model.core.K.tocsc()
source = np.asarray(model.source_shape, dtype=np.float64)
terms = [term.tocsc() for term in model.boundary_terms]
ranges = np.asarray(model.h_ranges(), dtype=np.float64)
n = kernel.shape[0]
print(f"mesh={mesh} n={n} ports={source.shape[1]} ranges={ranges.tolist()}", flush=True)

print(f"[sources] min entry {source.min():.3e}", flush=True)
for index, term in enumerate(terms):
    diagonal = np.asarray(term.diagonal()).ravel()
    off = term - sp.diags(diagonal)
    print(f"[H_{index}] min diagonal {diagonal.min():.3e} | off-diagonal nnz {off.nnz} | "
          f"support {int((diagonal > 0).sum())}", flush=True)

off_diagonal = kernel - sp.diags(np.asarray(kernel.diagonal()).ravel())
print(f"[kernel] off-diagonal nnz {off_diagonal.nnz} | max off-diagonal value "
      f"{off_diagonal.max() if off_diagonal.nnz else 0.0:.3e} | asymmetry nnz "
      f"{(kernel - kernel.T).nnz}", flush=True)

seed, _spectra = zolotarev_seed(kernel, terms, ranges)
minimum_operator = full_operator(kernel, terms, ranges[:, 0])
minimum_factor = spla.splu(sp.csc_matrix(minimum_operator).tocsc())
minimum_images = minimum_factor.solve(source)
print(f"[A(p_min)^-1 G] min entry {minimum_images.min():.3e} "
      f"-> nonnegative: {bool(minimum_images.min() >= 0.0)}", flush=True)

seed_images = spla.splu(
    sp.csc_matrix(full_operator(kernel, terms, seed)).tocsc()
).solve(source)
constant = np.ones((n, 1)) / np.sqrt(n)
basis = np.linalg.qr(np.column_stack([seed_images, constant]), mode="reduced")[0]

gamma = float(np.max(ranges[:, 1] / ranges[:, 0]))
print(f"[Gamma] max_i p_max/p_min = {gamma:.6g} -> sqrt = {np.sqrt(gamma):.6g}", flush=True)
rng = np.random.default_rng(20260926)
low, high = np.inf, 0.0
for _ in range(12):
    parameter = np.exp(rng.uniform(np.log(ranges[:, 0]), np.log(ranges[:, 1])))
    operator = sp.csc_matrix(full_operator(kernel, terms, parameter))
    factor = spla.splu(operator.tocsc())
    coefficients = np.linalg.solve(basis.T @ (operator @ basis), basis.T @ source)
    residual = source - operator @ (basis @ coefficients)
    truth = np.einsum("ij,ij->j", residual, factor.solve(residual))
    surrogate = np.einsum("ij,ij->j", residual, minimum_factor.solve(residual))
    ratio = surrogate / np.maximum(truth, np.finfo(float).tiny)
    low, high = min(low, float(ratio.min())), max(high, float(ratio.max()))
print(f"[sandwich] eta^2 / ||e||^2: [{low:.6g}, {high:.6g}]  predicted [1, {gamma:.6g}]",
      flush=True)
print(f"[sandwich] eta / ||e||    : [{np.sqrt(low):.6g}, {np.sqrt(high):.6g}]  "
      f"predicted [1, {np.sqrt(gamma):.6g}]", flush=True)
print(f"[sandwich] observed spread {high / low:.3g} vs predicted {gamma:.3g}", flush=True)
