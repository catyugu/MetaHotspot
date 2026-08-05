#!/usr/bin/env python3
"""Shared operator-level utilities for macromodel (MOR) experiments.

This module is deliberately *model-agnostic*: it operates only on the generic
`Operators` interface (K, C, f sparse matrices + rhs) and plain numpy/scipy
data.  No geometry, mesh, material or config type appears here, so any
reduction method (BCI-FANTASTIC, tangential rational Krylov, ...) can be
built on top of whatever model supplies its operators — the method only sees
the K/C/f interface, not the concrete model internals.

Contents
--------
* dense helpers:            symmetric_dense, eigenpairs_descending
* MPMM frequency sampling:  mpmm_elliptic_shift_count, mpmm_elliptic_shifts
* Krylov enrichment:        orthonormalize_block, reduced_response, response_error
* sparse operator helpers:  normalized_operators
* boundary-port machinery:  extract_boundary_groups, closure_diagonal,
                            closure_diagonal_multi, project_closure_group
* Galerkin projection:      project_exact_ports
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy import special

from metahotspot.compiled import Operators


# ---------------------------------------------------------------------------
# dense linear algebra
# ---------------------------------------------------------------------------


def symmetric_dense(matrix) -> np.ndarray:
    """Symmetric part of a dense matrix (0.5*(M + M^T))."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def eigenpairs_descending(matrix):
    """Symmetric eigen-decomposition, eigenpairs sorted by descending value."""
    values, vectors = scipy.linalg.eigh(symmetric_dense(matrix), check_finite=False)
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0.0), vectors[:, order]


# ---------------------------------------------------------------------------
# MPMM elliptic-optimal frequency sampling  (FANTASTIC 2014 eqs. 4-5)
# ---------------------------------------------------------------------------


def mpmm_elliptic_shift_count(
    relative_epsilon: float, lambda_min: float, lambda_max: float
):
    """Smallest m satisfying 4*exp(-m^2*pi^2/log(4/k')) <= eps (FANTASTIC 2014 eq. 4)."""
    kappa = lambda_max / max(lambda_min, np.finfo(float).tiny)
    k_prime = lambda_min / lambda_max
    log_term = max(math.log(4.0 / k_prime), np.finfo(float).tiny)
    for m in range(1, 200):
        if 4.0 * math.exp(-(m * m) * math.pi * math.pi / log_term) <= relative_epsilon:
            return m
    raise RuntimeError("MPMM elliptic shift count did not converge")


def mpmm_elliptic_shifts(count: int, lambda_max: float, kappa: float) -> np.ndarray:
    """Elliptic-dn distributed real shifts on [0, lambda_max] (FANTASTIC 2014 eq. 5)."""
    modulus = 1.0 - 1.0 / (kappa * kappa)
    modulus = float(np.clip(modulus, 0.0, 1.0 - 1.0e-12))
    k_complete = special.ellipk(modulus)
    theta = (2.0 * np.arange(1, count + 1) - 1.0) * k_complete / count
    _, _, dn_values, _ = special.ellipj(theta, modulus)
    shifts = lambda_max * np.asarray(dn_values, dtype=np.float64)
    return np.sort(shifts)[::-1]


# ---------------------------------------------------------------------------
# residual-driven Krylov enrichment
# ---------------------------------------------------------------------------


def orthonormalize_block(basis, vectors):
    """Orthonormalize ``vectors`` against the current column-orthonormal ``basis``.

    Returns the new orthonormal columns (a basis for the span of the input that
    is orthogonal to ``basis``).  ``basis`` may be empty (shape (n, 0)).
    """
    block = np.asarray(vectors, dtype=np.float64).copy()
    if not block.size:
        return np.empty((block.shape[0], 0), dtype=np.float64)

    for _ in range(2):
        if basis.shape[1]:
            block -= basis @ (basis.T @ block)

    q, r, _ = scipy.linalg.qr(
        block,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((block.shape[0], 0), dtype=np.float64)
    keep = diagonal > np.finfo(float).eps * max(block.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, keep])


def reduced_response(basis, A, B_dense):
    """Solve the projected system (basis^T A basis) x = -(basis^T B_dense)."""
    if not basis.shape[1]:
        return np.empty((0, B_dense.shape[1]), dtype=np.float64)
    reduced_A = symmetric_dense(basis.T @ (A @ basis))
    factor = scipy.linalg.cho_factor(
        reduced_A,
        lower=True,
        overwrite_a=False,
        check_finite=False,
    )
    return scipy.linalg.cho_solve(
        factor,
        -(basis.T @ B_dense),
        overwrite_b=False,
        check_finite=False,
    )


def response_error(response, basis, reduced, A, reference):
    """Residual of the current reduced model against the full response.

    Returns ``(error_response, error_eigenvalues, error_tangents, score)`` where
    ``score`` is the A-energy norm of the worst residual direction relative to
    ``reference`` (the response Gramian's leading eigenvalue).
    """
    error_response = response - basis @ reduced if basis.shape[1] else response
    error_gram = symmetric_dense(error_response.T @ (A @ error_response))
    values, tangents = eigenpairs_descending(error_gram)
    score = math.sqrt(float(values[0]) / reference)
    return error_response, values, tangents, score


# ---------------------------------------------------------------------------
# sparse operator helpers
# ---------------------------------------------------------------------------


def normalized_operators(K, C, f) -> Operators:
    """CSC-normalize K, C (eliminate explicit zeros) and copy f."""
    K = sp.csc_matrix(K)
    C = sp.csc_matrix(C)
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(f, dtype=np.float64).copy())


# ---------------------------------------------------------------------------
# boundary-port machinery
# ---------------------------------------------------------------------------


def extract_boundary_groups(merged: Operators, interface_ports: int, group_sizes):
    """Extract per-group boundary coupling in a consistent internal frame.

    ``merged`` is the full DtN operator ``[interface ports | boundary group
    ports | FVM cells]``, ``interface_ports`` the count of interface ports,
    and ``group_sizes`` the list of port counts per boundary group (in order).
    Each group's coupled cells are returned in the internal-block frame
    (0-based within the FVM cells), regardless of where the group sits among
    the ports.
    """
    internal_base = interface_ports + sum(group_sizes)
    groups = []
    offset = interface_ports
    for size in group_sizes:
        rows = merged.K[offset : offset + size, :].tocsr()
        cells = np.empty(size, dtype=np.int64)
        conductance = np.empty(size, dtype=np.float64)
        for k in range(size):
            row = rows[k]
            negative = [col for col in row.indices if row[0, col] < 0.0]
            if len(negative) != 1:
                raise RuntimeError("boundary port must couple to exactly one cell")
            cells[k] = negative[0] - internal_base
            conductance[k] = -row[0, negative[0]]
        groups.append((cells, conductance))
        offset += size
    return groups


def closure_diagonal(h: float, boundary_cells, boundary_g, boundary_areas, n_cell):
    """Diagonal closure K_ii(h) = K_ii + diag(closure) after port elimination.

    Each boundary port k couples cell c through conductance g_k; attaching the
    ambient heat exchange h*A_k at the port and eliminating the port adds
        closure_c = g_k * h * A_k / (g_k + h * A_k)
    to the cell diagonal — the exact (saturating) series combination of the
    in-cell conduction g_k and the external convection h*A_k.
    """
    closure = np.zeros(n_cell)
    for cell, g, area in zip(boundary_cells, boundary_g, boundary_areas):
        closure[cell] += g * h * area / (g + h * area)
    return closure


def closure_diagonal_multi(h_values, boundary_groups, boundary_areas, n_cell):
    """Sum of per-group saturation closures, one heat-exchange coefficient each.

    ``boundary_groups`` is a list of ``(cells, g)`` pairs (one per boundary
    group, in port order) as returned by :func:`extract_boundary_groups`, and
    ``boundary_areas`` the per-group face area arrays.  ``h_values`` the
    per-group coefficient.  Because each group couples disjoint cells, the
    total closure is the per-cell sum.
    """
    if len(h_values) != len(boundary_groups):
        raise ValueError("h_values must match boundary group count")
    if len(boundary_areas) != len(boundary_groups):
        raise ValueError("boundary_areas must match boundary group count")
    closure = np.zeros(n_cell)
    for h_value, (cells, g), areas in zip(h_values, boundary_groups, boundary_areas):
        for cell, g_k, area in zip(cells, g, areas):
            closure[cell] += g_k * h_value * area / (g_k + h_value * area)
    return closure


def project_closure_group(cells, g, areas, n_cell, basis):
    """Return ``h_k -> B^T diag(closure_k(h_k)) B`` for one boundary group.

    The projected closure matrix is what the online model adds to the reduced
    interior block for a given boundary coefficient of that group.
    """

    def closure(h_k):
        closure = np.zeros(n_cell)
        for cell, g_k, area in zip(cells, g, areas):
            closure[cell] += g_k * h_k * area / (g_k + h_k * area)
        weighted = closure[:, None] * basis
        return sp.csc_matrix(weighted.T @ basis)

    return closure


# ---------------------------------------------------------------------------
# Galerkin projection (exact ports + reduced interior)
# ---------------------------------------------------------------------------


def project_exact_ports(
    operators: Operators, ports: int, basis, ambient_K: float | None = None
) -> Operators:
    """Project ``[ports | interior]`` onto ``[ports | basis]``.

    Physical ports stay exact (identity on the leading block); the interior is
    projected with ``basis``.  When ``ambient_K`` is given, the rhs is shifted
    so the interior modes are *rise* coordinates above ambient.
    """
    source = np.asarray(operators.f, dtype=np.float64)
    if ambient_K is not None:
        offset = np.full(operators.K.shape[0] - ports, ambient_K)
        source = np.asarray(source - operators.K[:, ports:] @ offset).ravel()

    def project(matrix):
        reduced = sp.bmat(
            (
                (
                    sp.csc_matrix(matrix[:ports, :ports]),
                    sp.csc_matrix(matrix[:ports, ports:] @ basis),
                ),
                (
                    sp.csc_matrix(basis.T @ matrix[ports:, :ports]),
                    sp.csc_matrix(basis.T @ matrix[ports:, ports:] @ basis),
                ),
            ),
            format="csc",
        )
        reduced = (0.5 * (reduced + reduced.T)).tocsc()
        reduced.eliminate_zeros()
        return reduced

    return Operators(
        project(operators.K),
        project(operators.C),
        np.r_[source[:ports], np.asarray(basis.T @ source[ports:]).ravel()],
    )
