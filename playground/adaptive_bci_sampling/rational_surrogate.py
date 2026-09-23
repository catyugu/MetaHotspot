#!/usr/bin/env python3
"""Exact rational parametrization of the affine BCI solution family.

The full-order steady problem is affine in the boundary coefficients,

    A(p) x = g,    A(p) = K + sum_k p_k H_k,

with ``H_k = diag(a_k)`` supported on the boundary cells of group ``k``.  Let
``E`` select the union of those cells (``m_b`` of them, typically a few hundred
to a few thousand, against ``N`` cells in the mesh).  At a reference point
``p_ref`` with ``A_ref = A(p_ref)``,

    A(p) = A_ref + E diag(lambda) E^T,   lambda_c = sum_k (p_k - p_ref_k) a_k[c].

Restricting to the *active* cells (those with ``lambda_c != 0``), Woodbury's
identity gives, exactly,

    x(p) = x_ref - G_b[:, act] ( Phi[act,act] + diag(1/lambda[act]) )^{-1} c[act]

    x_ref = A_ref^{-1} g,  G_b = A_ref^{-1} E,  Phi = E^T A_ref^{-1} E,  c = E^T x_ref.

Nothing here is an approximation.  The consequence is the one the sampling
strategy rests on:

  * the whole parametric solution family is fixed by ``m_b + n_src`` reference
    solves at ONE parameter value, plus an ``m_b x m_b`` dense system whose only
    parameter-dependent part is the diagonal ``1/lambda``;
  * every parameter value, and every derivative with respect to it, is then
    available without a full-order solve;
  * the parameter dependence is a rational map of dimension ``k`` (the number of
    boundary groups, normally 2-3) whose poles are the eigenvalues of the fixed
    small pencil ``diag(a) Phi diag(a)``-type problem -- computable once.

The last point is what makes the sampling question a question about *poles*
rather than about the geometry of the parameter box.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import splu


class BoundaryRationalModel:
    """Exact rational parametrization of ``A(p)^{-1} G`` around ``p_ref``."""

    def __init__(self, K, terms, source_shape, p_ref):
        self.K = K.tocsc()
        self.terms = [t.tocsc() for t in terms]
        self.G = np.asarray(source_shape, dtype=np.float64)
        self.p_ref = np.asarray(p_ref, dtype=np.float64).ravel()
        if self.p_ref.size != len(self.terms):
            raise ValueError("p_ref must have one entry per boundary group")
        n = self.K.shape[0]

        A_ref = self.K.copy()
        for pk, H in zip(self.p_ref, self.terms):
            A_ref = A_ref + float(pk) * H
        self.A_ref = A_ref.tocsc()

        stacked = np.column_stack(
            [np.asarray(H.diagonal()).ravel() for H in self.terms]
        )
        self.cells = np.flatnonzero(np.abs(stacked).sum(axis=1) > 0.0)
        self.m_b = int(self.cells.size)
        self.areas = np.ascontiguousarray(stacked[self.cells, :])

        n_src = self.G.shape[1]
        rhs = np.zeros((n, n_src + self.m_b), dtype=np.float64)
        rhs[:, :n_src] = self.G
        rhs[self.cells, n_src:] = np.eye(self.m_b)
        solved = splu(self.A_ref).solve(rhs)
        self.x_ref = np.ascontiguousarray(solved[:, :n_src])
        self.G_b = np.ascontiguousarray(solved[:, n_src:])
        self.Phi = np.ascontiguousarray(self.G_b[self.cells, :])
        self.c = np.ascontiguousarray(self.x_ref[self.cells, :])

    # ------------------------------------------------------------------ core

    def lambdas(self, p) -> np.ndarray:
        """``lambda_c = sum_k (p_k - p_ref_k) a_k[c]``."""
        return self.areas @ (np.asarray(p, dtype=np.float64).ravel() - self.p_ref)

    def _system(self, p):
        lam = self.lambdas(p)
        active = np.flatnonzero(lam != 0.0)
        if active.size == 0:
            return None, None, None
        M = self.Phi[np.ix_(active, active)].copy()
        M[np.diag_indices(active.size)] += 1.0 / lam[active]
        return active, lam, M

    def response(self, p) -> np.ndarray:
        """Exact ``A(p)^{-1} G``, shape ``(N, n_src)``.

        Real ``p`` uses a symmetric solver; complex ``p`` (needed to evaluate
        the solution on a Bernstein ellipse) falls back to the general one.
        """
        active, lam, M = self._system(p)
        if active is None:
            return self.x_ref.copy()
        if np.iscomplexobj(M):
            s = np.linalg.solve(M, self.c[active, :])
        else:
            s = scipy.linalg.solve(M, self.c[active, :], assume_a="sym")
        return self.x_ref - self.G_b[:, active] @ s

    # ------------------------------------------------------- pole structure

    def ray_poles(self, target):
        """Poles of the rational map along the segment ``p_ref -> target``.

        Along ``p(t) = p_ref + t (target - p_ref)`` the per-cell perturbation is
        ``lambda_c = t b_c`` with ``b_c = sum_k (target_k - p_ref_k) a_k[c]``, so
        the whole parameter vector enters through the single scalar ``t`` and the
        map is again a sum of simple poles.  On the cells that move, writing
        ``B = diag(-b_c)`` (positive, since ``target`` lies on the lower side),

            s(t) = -t B^{1/2} (I - t Psi)^{-1} B^{1/2} c,
            Psi  = B^{1/2} Phi B^{1/2},

        hence poles at ``t = 1/mu_j`` for the positive eigenvalues of ``Psi``.
        Since ``A(p)`` is definite throughout the box, no pole lies in
        ``(0, 1]``; the nearest one beyond the segment is ``T = 1/max mu_j``.

        This is the k-parameter statement.  The pole structure is
        one-dimensional along *every* ray out of the reference point, and a
        coordinate ray -- one group moving with the rest pinned -- is only one
        such ray, so its ratio is an optimistic bound on the box.
        """
        b = self.areas @ (np.asarray(target, dtype=np.float64).ravel() - self.p_ref)
        active = np.flatnonzero(b < 0.0)
        if active.size == 0:
            raise ValueError("target does not move any boundary cell away from p_ref")
        sq = np.sqrt(-b[active])
        Phi = self.Phi[np.ix_(active, active)]
        Psi = (sq[:, None] * Phi) * sq[None, :]
        mu = np.linalg.eigvalsh(0.5 * (Psi + Psi.T))
        mu = mu[mu > 0.0]
        return 1.0 / mu, mu

    @staticmethod
    def segment_rho(t_pole):
        """Zolotarev ratio for the segment ``t in [0, 1]`` and its nearest pole.

        For an interval ``[0, 1]`` and a pole at ``T > 1`` the ratio of the
        general formula collapses to ``2T - 1 + 2 sqrt(T (T - 1))``.
        """
        T = float(np.min(t_pole))
        if T <= 1.0:
            raise ValueError(f"pole inside the segment: t = {T:g}")
        return 2.0 * T - 1.0 + 2.0 * np.sqrt(T * (T - 1.0))

    def ray_pole_weights(self, target):
        """Pole expansion along a ray: returns ``(t_poles, coefs)``.

        ``s(t) = -t sum_j coefs[:, j, :] / (1 - t mu_j)``, with ``coefs`` indexed
        like :attr:`cells` on its first axis and over sources on its third.
        """
        b = self.areas @ (np.asarray(target, dtype=np.float64).ravel() - self.p_ref)
        active = np.flatnonzero(b < 0.0)
        if active.size == 0:
            raise ValueError("target does not move any boundary cell away from p_ref")
        sq = np.sqrt(-b[active])
        Phi = self.Phi[np.ix_(active, active)]
        Psi = (sq[:, None] * Phi) * sq[None, :]
        Psi = 0.5 * (Psi + Psi.T)
        mu, V = np.linalg.eigh(Psi)
        keep = mu > 0.0
        mu, V = mu[keep], V[:, keep]
        roots = sq[:, None] * V
        beta = V.T @ (sq[:, None] * self.c[active, :])
        coefs = np.zeros((self.m_b, mu.size, self.c.shape[1]), dtype=np.float64)
        coefs[active, :, :] = np.einsum("ij,js->ijs", roots, beta)
        return 1.0 / mu, coefs
