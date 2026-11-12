"""Spectral planning and projection variants; production extraction stays intact."""

import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from metahotspot.macromodel import utils


def global_bounds(model):
    """AMG inverse Lanczos and positive-vector envelope at weak HTC corner."""
    started = time.perf_counter()
    K, C = model.core.K, model.core.C
    c = C.diagonal()
    assert np.all(c > 0) and (C - sp.diags(c)).nnz == 0
    p_range = model.h_ranges()

    def corner(p):
        A = K.copy()
        for v, H in zip(p, model.boundary_terms, strict=True):
            A = A + v * H
        return A.tocsr()

    low, high = corner(p_range[:, 0]), corner(p_range[:, 1])
    off = low - sp.diags(low.diagonal())
    assert np.max(np.abs((low - low.T).data), initial=0.0) < 1.0e-12
    assert np.max(off.data, initial=0.0) <= 0.0
    D = sp.diags(1.0 / np.sqrt(c))
    A = (D @ low @ D).tocsr()
    pre = utils._rs_preconditioner(A)
    calls = []

    def inverse(b):
        t = time.perf_counter()
        x, info = sla.cg(A, b, M=pre, rtol=1.0e-12, atol=0.0, maxiter=2000)
        if info:
            raise RuntimeError(f"eigen inverse CG failed: {info}")
        calls.append(time.perf_counter() - t)
        return x

    val, vec = sla.eigsh(
        A,
        k=1,
        sigma=0.0,
        which="LM",
        OPinv=sla.LinearOperator(A.shape, matvec=inverse),
        v0=np.sqrt(c),
        tol=1.0e-10,
        ncv=16,
    )
    u = vec[:, 0] / np.sqrt(c)
    if u.sum() < 0:
        u = -u
    Au, Cu = low @ u, c * u
    residual = np.linalg.norm(Au - val[0] * Cu) / np.linalg.norm(Au)
    assert np.all(u > 0)
    quot = Au / Cu
    upper = float(np.max(np.asarray(np.abs(D @ high @ D).sum(axis=1))))
    # Floating-point outward margin; this is not an interval-arithmetic proof.
    lower = float(quot.min() * (1.0 - 1.0e-5))
    record = {
        "lambda_min": lower,
        "ritz": float(val[0]),
        "lambda_max": upper,
        "enclosure": [float(quot.min()), float(quot.max())],
        "relative_residual": float(residual),
        "inverse_calls": len(calls),
        "seconds": time.perf_counter() - started,
    }
    return (lower, upper), record


def svd_basis(snapshots, tol):
    X = snapshots / np.linalg.norm(snapshots, axis=0)
    U, s, _ = la.svd(X, full_matrices=False, check_finite=False)
    V = U[:, s >= tol * s[0]]
    one = np.ones((V.shape[0], 1)) / np.sqrt(V.shape[0])
    q = utils.orthonormalize_block(V, one)
    if q.shape[1]:
        V = np.column_stack([V, q])
    return V, s


def direct_projection(model, V):
    return (
        V.T @ (model.core.C @ V),
        V.T @ (model.core.K @ V),
        V.T @ model.source_shape,
        [V.T @ (H @ V) for H in model.boundary_terms],
    )
