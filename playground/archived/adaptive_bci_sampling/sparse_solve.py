"""Reusable AMG-preconditioned CG solves for the fixed SPD operators here.

Every operator in this sandbox is symmetric positive definite, and the same
operator is reused for many right-hand sides (one per port, one per cell corner,
one per shift).  A sparse direct factorisation is therefore not the right tool:
its cost grows with the mesh and it is re-run for every new parameter point.  The
solver below builds one Ruge-Stuben AMG hierarchy per operator and reuses it for
all of that operator's right-hand sides.
"""

from __future__ import annotations

import numpy as np
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

RTOL = 1.0e-10
MAXITER = 5000


class AmgSolver:
    """AMG-preconditioned CG for one fixed SPD operator, many right-hand sides."""

    def __init__(self, operator, rtol=RTOL):
        self.operator = sp.csr_matrix(operator)
        self.rtol = float(rtol)
        self.hierarchy = pyamg.ruge_stuben_solver(
            self.operator, interpolation="direct"
        )
        self.preconditioner = self.hierarchy.aspreconditioner(cycle="V")

    def solve(self, rhs):
        """Solve for one vector or every column of a two-dimensional array."""
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.ndim == 1:
            return self._column(rhs)
        return np.column_stack([self._column(rhs[:, index]) for index in range(rhs.shape[1])])

    def _column(self, column):
        solution, info = spla.cg(
            self.operator,
            column,
            rtol=self.rtol,
            atol=0.0,
            maxiter=MAXITER,
            M=self.preconditioner,
        )
        if info != 0:
            raise RuntimeError(f"AMG-CG did not converge: info={info}")
        return solution
