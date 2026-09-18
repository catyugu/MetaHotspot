"""Comparison-principle enclosures for a stored backward-Euler thermal system.

This is an experiment, not a continuous-PDE or formally interval-arithmetic
certificate. Matrices are fixed, capacities positive, and A is a strictly
row-diagonally-dominant symmetric Z matrix. Floating-point guards are included;
no claim of fully verified arithmetic is made.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


class Certificate:
    """Intersect positive comparison-vector majorants of temporal state error.

    If |e_previous| <= prior and A w_j > 0, then
        beta_j = max_i (D_i prior_i + |r_i|) / (A w_j)_i
    implies |e_current| <= beta_j w_j. Intersect the resulting componentwise
    bounds, not the supersolution inequalities. A @ min_j(beta_j w_j) need NOT
    dominate the error source, but the minimum is still a valid error bound.
    """

    def __init__(self, A, D, weights=None):
        self.A = sp.csr_matrix(A, dtype=float)
        self.A.sum_duplicates()
        self.A.eliminate_zeros()
        self.D = np.asarray(D, dtype=float).reshape(-1)
        n = len(self.D)
        if self.A.shape != (n, n) or np.any(self.D <= 0) or not np.isfinite(self.D).all():
            raise ValueError("A must be square and D=C/dt strictly positive")
        if not np.isfinite(self.A.data).all():
            raise ValueError("nonfinite operator")
        diagonal = self.A.diagonal()
        off = self.A - sp.diags(diagonal)
        if off.nnz and np.max(off.data) > 0:
            raise ValueError("positive offdiagonal: comparison principle unavailable")
        asym = self.A - self.A.T
        if asym.nnz and np.max(np.abs(asym.data)) > 1e-12 * np.max(diagonal):
            raise ValueError("this PCG experiment requires symmetry")
        self.abs_A = abs(self.A)
        self.guard = 64 * np.finfo(float).eps
        row_sum = np.asarray(self.A.sum(axis=1)).ravel()
        row_abs = np.asarray(self.abs_A.sum(axis=1)).ravel()
        if np.any(row_sum - self.guard * row_abs <= 0):
            raise ValueError("strict row diagonal dominance not established")
        W = np.ones((n, 1)) if weights is None else np.asarray(weights, dtype=float)
        if W.ndim != 2 or W.shape[0] != n or W.shape[1] == 0:
            raise ValueError("weights must be N by m")
        if np.any(W <= 0) or not np.isfinite(W).all():
            raise ValueError("comparison vectors must be strictly positive")
        self.W = np.ascontiguousarray(W / W.max(axis=0))
        self.AW_lower = np.nextafter(self.A @ self.W - self.guard * (self.abs_A @ self.W), -np.inf)
        if np.any(self.AW_lower <= 0):
            raise ValueError("A w > 0 not established for all comparison vectors")

    def bound(self, estimate, previous, prior, forcing):
        estimate = np.asarray(estimate, dtype=float)
        previous = np.asarray(previous, dtype=float)
        prior = np.asarray(prior, dtype=float)
        forcing = np.asarray(forcing, dtype=float)
        if any(a.shape != self.D.shape for a in (estimate, previous, prior, forcing)):
            raise ValueError("state, forcing and radius must have shape N")
        if np.any(prior < 0) or not all(np.isfinite(a).all() for a in (estimate, previous, prior, forcing)):
            raise ValueError("invalid enclosure input")
        residual = forcing + self.D * previous - self.A @ estimate
        magnitude = np.abs(forcing) + self.D * np.abs(previous) + self.abs_A @ np.abs(estimate)
        drive = self.D * prior + np.abs(residual) + self.guard * magnitude
        drive = np.nextafter(drive * (1 + self.guard), np.inf)
        betas = np.max(drive[:, None] / self.AW_lower, axis=0)
        b = np.min(self.W * betas[None, :], axis=1)
        return np.nextafter(b * (1 + self.guard), np.inf)

    @staticmethod
    def peak(estimate, radius):
        """The maximizing cell may differ at the lower and upper endpoints."""
        return (float(np.max(np.nextafter(estimate - radius, -np.inf))),
                float(np.max(np.nextafter(estimate + radius, np.inf))))


def decide(lower: float, upper: float, threshold: float) -> str:
    if upper < threshold:
        return "safe"
    if lower > threshold:
        return "unsafe"
    return "unknown"


@dataclass
class CheckedStep:
    state: np.ndarray
    radius: np.ndarray
    lower: float
    upper: float
    iterations: int
    accepted: bool


def solve_checked(cert, previous, prior, forcing, guess, precondition,
                  *, width, threshold=None, maxiter=100, check_every=2):
    """Perform more PCG work only while the requested interval is inadequate.

    With a threshold, any strict decision is sufficient. If the threshold lies
    inside a sufficiently narrow interval, return UNKNOWN, never pretend it is
    safe. Inherited uncertainty is never reset by a current-step solve.
    """
    if width <= 0 or check_every < 1:
        raise ValueError("width and check interval must be positive")
    x = np.asarray(guess, dtype=float).copy()
    rhs = forcing + cert.D * previous
    residual = rhs - cert.A @ x
    scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    direction = None
    rz_old = None
    for iteration in range(maxiter + 1):
        converged = float(np.linalg.norm(residual)) <= 1e-12 * scale
        if iteration % check_every == 0 or converged or iteration == maxiter:
            b = cert.bound(x, previous, prior, forcing)
            lower, upper = cert.peak(x, b)
            accepted = upper - lower <= width
            if threshold is not None:
                accepted = accepted or decide(lower, upper, threshold) != "unknown"
            if accepted or converged or iteration == maxiter:
                return CheckedStep(x, b, lower, upper, iteration, accepted)
        z = np.asarray(precondition(residual))
        rz = float(residual @ z)
        if rz <= 0:
            raise RuntimeError("PCG preconditioner lost positive definiteness")
        direction = z.copy() if direction is None else z + (rz / rz_old) * direction
        Ad = cert.A @ direction
        pAp = float(direction @ Ad)
        if pAp <= 0:
            raise RuntimeError("PCG operator lost positive definiteness")
        alpha = rz / pAp
        x += alpha * direction
        residual -= alpha * Ad
        rz_old = rz
    raise AssertionError("unreachable")


def replay(cert, forcings, checkpoint_index, end_index, checkpoint_state,
           checkpoint_radius, precondition):
    """Recompute from a trustworthy checkpoint, preserving its error radius.

    Indices count completed time steps. Forcings is a sized sequence supporting
    integer indexing. No reference trajectory is accepted or consulted here.
    """
    x = checkpoint_state.copy()
    b = checkpoint_radius.copy()
    total_iterations = 0
    M = sla.LinearOperator(cert.A.shape, matvec=precondition, dtype=float)
    for step in range(checkpoint_index, end_index):
        f = forcings[step]
        previous = x
        count = [0]
        def callback(_):
            count[0] += 1
        x, info = sla.cg(cert.A, f + cert.D * previous, x0=previous, M=M,
                         rtol=1e-11, atol=0., maxiter=1000, callback=callback)
        if info != 0:
            raise RuntimeError(f"checkpoint replay CG failed: {info}")
        b = cert.bound(x, previous, b, f)
        total_iterations += count[0]
    return x, b, end_index - checkpoint_index, total_iterations
