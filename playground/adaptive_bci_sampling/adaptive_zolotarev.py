"""Deterministic Zolotarev-seeded greedy HTC sampling at each frequency shift.

The stopping criterion is the reproduction's relative *field residual* on a
finite deterministic HTC candidate grid.  It is neither a box-wide nor a
junction-output certificate; closing SVD can also invalidate it.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg

from metahotspot.macromodel.utils import (
    _snapshot_svd_basis,
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
    orthonormalize_block,
    port_eigenvalue_bounds,
    spd_solve,
)


def residual_scores(K, C, terms, g, shift, basis, candidates):
    """Exact Euclidean Galerkin residuals via an affine residual Gram matrix."""
    if basis.shape[1] == 0:
        raise ValueError("a seed snapshot is required before scoring")
    candidates = np.asarray(candidates, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64).ravel()
    offset = (K + shift * C) @ basis
    images = [term @ basis for term in terms]
    span = np.column_stack([g, offset, *images])
    gram = span.T @ span
    projected = basis.T @ offset
    terms_hat = [basis.T @ image for image in images]
    source_hat = basis.T @ g
    coefficients = []
    residual_coordinates = []
    for point in candidates:
        reduced = projected.copy()
        for value, projected_term in zip(point, terms_hat):
            reduced += float(value) * projected_term
        c = scipy.linalg.solve(
            reduced, source_hat, assume_a="pos", check_finite=False
        )
        coefficients.append(c)
        residual_coordinates.append(
            np.concatenate(([1.0], -c, *(-float(h) * c for h in point)))
        )
    coordinates = np.asarray(residual_coordinates)
    squared = np.einsum("bi,ij,bj->b", coordinates, gram, coordinates)
    relative = np.sqrt(np.maximum(squared, 0.0)) / np.linalg.norm(g)
    return relative, np.asarray(coefficients)


def greedy_shift(
    K, C, terms, g, shift, seeds, candidates, *, tolerance, max_extra, basis
):
    """Seed a port's local space then enrich at the largest grid residual."""
    started = time.perf_counter()
    basis = np.ascontiguousarray(basis)
    snapshots = []
    chosen = []

    def take_snapshot(point, x0=None):
        nonlocal basis
        A = K + float(shift) * C
        for value, term in zip(point, terms):
            A = A + float(value) * term
        response = np.asarray(
            spd_solve(A.tocsc(), g, x0=x0), dtype=np.float64
        ).ravel()
        snapshots.append(response)
        chosen.append(tuple(float(v) for v in point))
        addition = orthonormalize_block(basis, response[:, None])
        if addition.shape[1]:
            basis = np.ascontiguousarray(np.column_stack((basis, addition)))

    for point in seeds:
        take_snapshot(point)

    extra = 0
    evaluations = 0
    while True:
        scores, coefficients = residual_scores(
            K, C, terms, g, shift, basis, candidates
        )
        evaluations += len(candidates)
        index = int(np.argmax(scores))
        worst = float(scores[index])
        if worst <= tolerance or extra >= max_extra:
            break
        point = candidates[index]
        if tuple(float(v) for v in point) in chosen:
            break
        take_snapshot(point, x0=basis @ coefficients[index])
        extra += 1

    return {
        "basis": basis,
        "snapshots": snapshots,
        "extra_solves": extra,
        "maximum_residual": worst,
        "converged": worst <= tolerance,
        "candidate_evaluations": evaluations,
        "selected": chosen,
        "seconds": time.perf_counter() - started,
    }


def adaptive_basis(model, seeds, candidates, tolerance, max_extra, closing=None):
    """Extract all four ports with the same frequency plan as reproduction."""
    started = time.perf_counter()
    K = model.core.K.tocsc()
    C = model.core.C.tocsc()
    G = np.asarray(model.source_shape, dtype=np.float64)
    terms = [term.tocsc() for term in model.boundary_terms]
    snapshots = []
    plans = []
    history = []
    for port in range(G.shape[1]):
        g = G[:, port]
        low, high = port_eigenvalue_bounds(K, C, g)
        shift_count = mpmm_elliptic_shift_count(tolerance, low, high)
        shifts = mpmm_elliptic_shifts(shift_count, high, high / low)
        plans.append(shift_count)
        local = np.empty((K.shape[0], 0), dtype=np.float64)
        for shift in shifts:
            result = greedy_shift(
                K, C, terms, g, shift, seeds, candidates,
                tolerance=tolerance, max_extra=max_extra, basis=local,
            )
            local = result["basis"]
            snapshots.extend(result["snapshots"])
            history.append({
                "port": port,
                "shift": float(shift),
                "seed_solves": len(seeds),
                "extra_solves": result["extra_solves"],
                "maximum_residual": result["maximum_residual"],
                "converged": result["converged"],
                "candidate_evaluations": result["candidate_evaluations"],
                "selected": result["selected"],
            })
    solver_seconds = time.perf_counter() - started
    cutoff = tolerance if closing is None else closing
    started_svd = time.perf_counter()
    basis, singular_values = _snapshot_svd_basis(
        np.column_stack(snapshots), cutoff
    )
    constant = np.ones((K.shape[0], 1), dtype=np.float64)
    constant /= np.linalg.norm(constant)
    constant = orthonormalize_block(basis, constant)
    if constant.shape[1]:
        basis = np.column_stack((basis, constant))
    svd_seconds = time.perf_counter() - started_svd
    return basis, {
        "full_rhs_solves": len(snapshots),
        "basis_order": int(basis.shape[1]),
        "svd_kept_order": int(np.count_nonzero(
            singular_values >= cutoff * singular_values[0]
        )),
        "shift_counts": plans,
        "max_pre_svd_grid_residual": max(x["maximum_residual"] for x in history),
        "unresolved_shifts": sum(not x["converged"] for x in history),
        "extra_solves": sum(x["extra_solves"] for x in history),
        "candidate_evaluations": sum(x["candidate_evaluations"] for x in history),
        "frequency_plan_and_solves_s": solver_seconds,
        "svd_s": svd_seconds,
        "seconds_without_spectrum": solver_seconds + svd_seconds,
        "history": history,
    }
