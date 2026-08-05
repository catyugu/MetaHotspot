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
* sparse operator helpers:  normalized_operators, internal_blocks
* boundary-port machinery:  extract_boundary_groups, closure_diagonal_multi,
                            project_closure_group
* parametric basis:         random_parameter_vectors, build_parametric_basis
* Galerkin projection:      project_exact_ports
* accuracy metrics:         temperature_error_metrics, accuracy_summary, format_accuracy
* mesh helpers:             grid_cells, coordinate_map

The accuracy metrics and mesh helpers were folded in from the former
``experiment_setup.py`` when that module was absorbed here; everything remains
model-agnostic (works on ``Compiled`` objects / plain arrays, never on a
concrete model config).
"""

from __future__ import annotations

import math
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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


def internal_blocks(operators: Operators, ports: int):
    """Interior/port CSC blocks of a ``[ports | interior]`` DtN operator.

    Returns ``(K_ii, C_ii, B_io, D_io)`` — the interior-interior K/C and the
    interior-port K/C coupling blocks used to assemble rational-Krylov
    candidate operators.  Parameter-count agnostic: ``ports`` is the leading
    block size, whatever the number of boundary groups behind it.
    """
    return (
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, ports:].tocsc(),
        operators.K[ports:, :ports].tocsc(),
        operators.C[ports:, :ports].tocsc(),
    )


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
# parametric basis construction (FANTASTIC BCI 2015 Algorithm 1)
# ---------------------------------------------------------------------------

TARGET_RELATIVE_EPSILON = 5.0e-3  # elliptic shift-count target (FANTASTIC 2014 eq. 4)
RESIDUAL_TOLERANCE = 5.0e-3  # residual-driven enrichment stop tolerance
MAX_ORDER = 2048
RANDOM_PARAMETER_SAMPLES = 20  # random h-vectors for training (Algorithm 1)
RANDOM_SEED = 20260805


def random_parameter_vectors(h_ranges, sample_count, seed, boundaries=None):
    """Random admissible h-vectors (one h per group), log-uniform.  No greedy.

    ``h_ranges`` is an ``(n_groups, 2)`` array of admissible ``(lo, hi)``
    ranges, one row per affine parameter; the log-uniform draws are vectorized
    across groups.  FANTASTIC BCI 2015 Algorithm 1: parameters chosen at
    random to avoid reduced-basis greedy stagnation.  ``boundaries`` (geometric
    holdout) are appended so the certified range is covered at its extremes.
    """
    h_ranges = np.asarray(h_ranges, dtype=np.float64)
    lows = np.log10(h_ranges[:, 0])
    highs = np.log10(h_ranges[:, 1])
    rng = np.random.default_rng(seed)
    draws = 10.0 ** rng.uniform(lows, highs, size=(sample_count, h_ranges.shape[0]))
    vectors = [tuple(row) for row in draws]
    for b in boundaries or ():
        vectors.append(tuple(b))
    seen, out = set(), []
    for v in vectors:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_parametric_basis(
    core,
    ports,
    boundary_groups,
    boundary_areas,
    *,
    h_ranges,
    boundaries,
    residual_tolerance=RESIDUAL_TOLERANCE,
    max_order=MAX_ORDER,
    target_relative_epsilon=TARGET_RELATIVE_EPSILON,
    sample_count=RANDOM_PARAMETER_SAMPLES,
    seed=RANDOM_SEED,
):
    """Certified interior basis by residual-driven enrichment (Algorithm 1).

    Model- and parameter-count-agnostic: ``boundary_groups`` is a list of
    ``(cells, g)`` pairs (one per boundary group, in port order) as returned by
    :func:`extract_boundary_groups`, ``boundary_areas`` the per-group face area
    arrays, and ``h_ranges`` an ``(n_groups, 2)`` array of admissible
    ``(lo, hi)`` ranges, one row per affine parameter.

    Candidates are ``(h_vec, shift)``: random admissible boundary-coefficient
    vectors crossed with the FANTASTIC-2014 elliptic-optimal complex shifts.
    The candidate operator is
        A(h_vec, shift) = K_ii + shift*C_ii + diag(closure_multi(h_vec))
    with the exact saturating per-group closure (the operators K_ii, C_ii never
    contain h — only the boundary-port closure does).  Every candidate streams
    one frequency-domain solve; residual directions above tolerance are inserted
    immediately, so no full-state response is retained.  The basis is kept
    column-orthonormal throughout.
    """
    started = time.perf_counter()
    K0, C0, B0, D0 = internal_blocks(core, ports)
    h_ranges = np.asarray(h_ranges, dtype=np.float64)
    h_vectors = random_parameter_vectors(h_ranges, sample_count, seed, boundaries)

    # Eigenvalue bounds of the h-free interior operator K_ii (generalized
    # eigenvalue problem K_ii v = lambda C_ii v) drive both the elliptic
    # shift distribution and the per-candidate shift count.
    eigenvalue_scale = max(float(np.max(np.abs(C0.diagonal()))), np.finfo(float).tiny)
    eigenvalue_ratio = max(
        math.sqrt(np.linalg.cond(K0.todense().astype(np.float64))),
        1.0,
    )
    kappa = eigenvalue_ratio**2
    lambda_min = float(eigenvalue_scale / kappa)
    lambda_max = float(eigenvalue_scale)
    if kappa > 1.0e6:
        lambda_min = max(lambda_min, lambda_max / 1.0e6)
        kappa = lambda_max / lambda_min
    elliptic_count = mpmm_elliptic_shift_count(
        target_relative_epsilon, lambda_min, lambda_max
    )
    shifts = np.r_[0.0, mpmm_elliptic_shifts(elliptic_count, lambda_max, kappa)]

    raw_points = [(hv, float(shift)) for hv in h_vectors for shift in shifts]
    internal_order = K0.shape[0]
    order_limit = min(max_order, internal_order)
    basis = np.empty((internal_order, 0), dtype=np.float64)
    history = []
    worst_score = 0.0
    converged = True

    for h_vec, shift in raw_points:
        closure = closure_diagonal_multi(
            h_vec, boundary_groups, boundary_areas, internal_order
        )
        A = (K0 + shift * C0 + sp.diags(closure)).tocsc()
        A = (0.5 * (A + A.T)).tocsc()
        B_dense = (B0 + shift * D0).toarray()

        response = np.asarray(spla.splu(A).solve(-B_dense))
        response_gram = symmetric_dense(-response.T @ B_dense)
        response_values, _ = eigenpairs_descending(response_gram)
        reference = max(float(response_values[0]), np.finfo(float).tiny)

        order_before = basis.shape[1]
        reduced = reduced_response(basis, A, B_dense)
        error_response, error_values, tangents, score_before = response_error(
            response,
            basis,
            reduced,
            A,
            reference,
        )
        requested = int(
            np.count_nonzero(error_values > residual_tolerance**2 * reference)
        )
        available = order_limit - basis.shape[1]
        count = min(requested, available)

        added = 0
        if count:
            block = orthonormalize_block(
                basis,
                error_response @ tangents[:, :count],
            )
            if not block.shape[1]:
                raise RuntimeError("rational Krylov enrichment stalled")
            basis = np.column_stack((basis, block))
            added = block.shape[1]

        if count == requested and added == count:
            score_after = (
                math.sqrt(float(error_values[count]) / reference)
                if count < error_values.size
                else 0.0
            )
        else:
            reduced = reduced_response(basis, A, B_dense)
            _, _, _, score_after = response_error(
                response,
                basis,
                reduced,
                A,
                reference,
            )

        worst_score = max(worst_score, score_after)
        history.append(
            {
                "order_before": int(order_before),
                "order_after": int(basis.shape[1]),
                "score_before": float(score_before),
                "score_after": float(score_after),
                "h_vec": h_vec,
                "shift": float(shift),
                "requested": int(requested),
                "added": int(added),
            }
        )
        if requested > available or score_after > residual_tolerance:
            converged = False
            break

    if basis.shape[1]:
        orthogonality = basis.T @ basis - np.eye(basis.shape[1])
        orthogonality_error = float(np.max(np.abs(orthogonality)))
    else:
        orthogonality_error = 0.0
    if orthogonality_error > 1.0e-10:
        raise RuntimeError("rational Krylov basis lost orthogonality")

    return basis, {
        "parameter_vectors": h_vectors,
        "elliptic_shift_count": elliptic_count,
        "elliptic_shifts_per_s": shifts[1:].tolist(),
        "eigenvalue_ratio_kappa": kappa,
        "eigenvalue_bounds_per_s": [lambda_min, lambda_max],
        "target_relative_epsilon": target_relative_epsilon,
        "parameter_sampling": "random (FANTASTIC BCI 2015 Algorithm 1)",
        "candidate_count": len(raw_points),
        "processed_candidate_count": len(history),
        "basis_order": int(basis.shape[1]),
        "maximum_order": int(order_limit),
        "orthogonality_error": orthogonality_error,
        "relative_response_error": float(worst_score),
        "residual_tolerance": residual_tolerance,
        "converged": bool(converged and len(history) == len(raw_points)),
        "history": history,
        "seconds": time.perf_counter() - started,
        "memory_strategy": (
            "stream one full response per candidate; form the residual Gramian "
            "directly; never cache candidates or repeat global scans"
        ),
    }


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


# ---------------------------------------------------------------------------
# accuracy metrics and mesh helpers  (folded in from experiment_setup.py)
# ---------------------------------------------------------------------------

MAX_RELATIVE_RISE_ERROR = 0.01


def temperature_error_metrics(reference, approximation, ambient_K: float) -> dict:
    reference = np.asarray(reference)
    approximation = np.asarray(approximation)
    absolute_error = float(np.max(np.abs(approximation - reference)))
    reference_rise = float(np.max(np.abs(reference - ambient_K)))
    relative_error = (
        absolute_error / reference_rise
        if reference_rise
        else float(absolute_error != 0.0)
    )
    return {
        "reference_temperature_range_K": [
            float(reference.min()),
            float(reference.max()),
        ],
        "max_absolute_rise_error_K": absolute_error,
        "max_relative_rise_error": relative_error,
        "passed": relative_error < MAX_RELATIVE_RISE_ERROR,
    }


def accuracy_summary(
    reference_steady,
    reduced_steady,
    reference_history,
    reduced_history,
    ambient_K: float,
) -> dict:
    steady = temperature_error_metrics(reference_steady, reduced_steady, ambient_K)
    transient = temperature_error_metrics(
        reference_history[-1], reduced_history[-1], ambient_K
    )
    return {
        "steady_reference_temperature_range_K": steady["reference_temperature_range_K"],
        "transient_final_reference_temperature_range_K": transient[
            "reference_temperature_range_K"
        ],
        "steady_max_absolute_rise_error_K": steady["max_absolute_rise_error_K"],
        "steady_max_relative_rise_error": steady["max_relative_rise_error"],
        "transient_final_max_absolute_rise_error_K": transient[
            "max_absolute_rise_error_K"
        ],
        "transient_final_max_relative_rise_error": transient["max_relative_rise_error"],
        "accuracy_passed": steady["passed"] and transient["passed"],
    }


def format_accuracy(summary: dict) -> str:
    steady_range = summary["steady_reference_temperature_range_K"]
    transient_range = summary["transient_final_reference_temperature_range_K"]
    return (
        f"reference range steady={steady_range[0]:.3f}..{steady_range[1]:.3f} K, "
        f"transient final={transient_range[0]:.3f}..{transient_range[1]:.3f} K; "
        f"rise error steady={summary['steady_max_absolute_rise_error_K']:.5f} K/"
        f"{summary['steady_max_relative_rise_error']:.3%}, transient final="
        f"{summary['transient_final_max_absolute_rise_error_K']:.5f} K/"
        f"{summary['transient_final_max_relative_rise_error']:.3%}"
    )


def grid_cells(compiled) -> np.ndarray:
    return compiled.grid_to_cell.reshape(compiled.nx, compiled.ny, compiled.nz)


def coordinate_map(source, target, z_offset: int, label: str) -> np.ndarray:
    if source.nx != target.nx or source.ny != target.ny:
        raise RuntimeError(f"{label}: lateral meshes differ")
    source_grid = grid_cells(source)
    target_grid = grid_cells(target)[:, :, z_offset : z_offset + source.nz]
    if target_grid.shape != source_grid.shape:
        raise RuntimeError(f"{label}: z range differs")
    valid = source_grid >= 0
    if not np.array_equal(valid, target_grid >= 0):
        raise RuntimeError(f"{label}: geometry occupancy differs")

    source_ids = source_grid[valid]
    target_ids = target_grid[valid]
    if (
        source_ids.size != source.cell_count
        or np.unique(source_ids).size != source.cell_count
    ):
        raise RuntimeError(f"{label}: source cell IDs are incomplete")
    mapping = np.empty(source.cell_count, dtype=np.int64)
    mapping[source_ids] = target_ids
    if np.unique(mapping).size != mapping.size:
        raise RuntimeError(f"{label}: target mapping is not one-to-one")
    return mapping
