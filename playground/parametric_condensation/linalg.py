#!/usr/bin/env python3
"""Sparse SPD solves by AMG-preconditioned conjugate gradients.

No direct factorization is used anywhere in this sandbox.  Every full-order
solve is a preconditioned CG iteration on a symmetric positive definite
operator, matching the production solver's contract:

* one Ruge-Stuben AMG hierarchy per operator (setup counted once),
* the hierarchy is reused for every right-hand side and every Newton step at
  that operator, which is where the cost actually is,
* solves, iterations and setups are counted so every table can report them.

Reduced systems are small and dense; they are solved by Cholesky, which is the
correct online operation for an ``r x r`` SPD system and is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DEFAULT_RTOL = 1.0e-10
MAX_ITERATIONS = 4000


@dataclass
class Cost:
    """Accumulated solver cost; reported instead of wall-clock guesses."""

    amg_setups: int = 0
    cg_solves: int = 0
    cg_iterations: int = 0
    dense_cholesky: int = 0

    def add(self, other: Cost) -> None:
        self.amg_setups += other.amg_setups
        self.cg_solves += other.cg_solves
        self.cg_iterations += other.cg_iterations
        self.dense_cholesky += other.dense_cholesky

    def as_dict(self) -> dict:
        return {
            "amg_setups": self.amg_setups,
            "cg_solves": self.cg_solves,
            "cg_iterations": self.cg_iterations,
            "dense_cholesky": self.dense_cholesky,
        }


class SPDSolver:
    """One AMG hierarchy, reusable for any number of right-hand sides.

    ``solve_on`` accepts a *different* matrix and runs CG on that matrix with
    this hierarchy as a frozen preconditioner.  That is the only correct way to
    reuse an AMG setup across Newton iterations: the preconditioner may be stale,
    but the Krylov iteration must always see the current Jacobian.  Solving a
    frozen matrix by mistake turns Newton into a slow fixed-point iteration.
    """

    def __init__(
        self, operator, *, rtol: float = DEFAULT_RTOL, cost: Cost | None = None
    ):
        self.matrix = sp.csr_matrix(operator)
        hierarchy = pyamg.ruge_stuben_solver(
            self.matrix, interpolation="direct", coarse_solver="gauss_seidel"
        )
        self.preconditioner = hierarchy.aspreconditioner(cycle="V")
        self.rtol = float(rtol)
        self.cost = cost if cost is not None else Cost()
        self.cost.amg_setups += 1

    def solve(self, rhs, *, x0=None, label: str = "solve") -> np.ndarray:
        return self.solve_on(self.matrix, rhs, x0=x0, label=label)

    def solve_on(self, operator, rhs, *, x0=None, label: str = "solve") -> np.ndarray:
        """CG on ``operator`` (any SPD matrix) with this hierarchy's preconditioner."""
        matrix = sp.csr_matrix(operator)
        columns = np.asarray(rhs, dtype=np.float64)
        vector = columns.ndim == 1
        if vector:
            columns = columns[:, None]
        initial = None
        if x0 is not None:
            initial = np.asarray(x0, dtype=np.float64)
            initial = initial[:, None] if initial.ndim == 1 else initial
        out = np.empty_like(columns)
        for index in range(columns.shape[1]):
            guess = None if initial is None else initial[:, index]
            counter = {"iterations": 0}

            def count(_vector, counter=counter):
                counter["iterations"] += 1

            solution, info = spla.cg(
                matrix,
                columns[:, index],
                x0=guess,
                rtol=self.rtol,
                atol=0.0,
                maxiter=MAX_ITERATIONS,
                M=self.preconditioner,
                callback=count,
            )
            self.cost.cg_solves += 1
            self.cost.cg_iterations += counter["iterations"]
            if info != 0:
                raise RuntimeError(
                    f"AMG-CG failed ({label}, column {index}): info={info}"
                )
            out[:, index] = solution
        return out[:, 0] if vector else out


def dense_spd_solve(matrix, rhs, *, cost: Cost | None = None):
    """Cholesky solve of a small dense SPD system (the online ROM operation)."""
    import scipy.linalg

    factor = scipy.linalg.cho_factor(np.asarray(matrix), lower=True, check_finite=False)
    if cost is not None:
        cost.dense_cholesky += 1
    return scipy.linalg.cho_solve(factor, np.asarray(rhs), check_finite=False)
