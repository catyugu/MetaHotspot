#!/usr/bin/env python3
"""Model-agnostic operator-level utilities for thermal reduced-order models."""

from __future__ import annotations

import math
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import special
import pyamg

from metahotspot._compiled_data import Operators


# ---------------------------------------------------------------------------
# MPMM elliptic-optimal frequency sampling
# ---------------------------------------------------------------------------


def mpmm_elliptic_shift_count(
    relative_epsilon: float, lambda_min: float, lambda_max: float
):
    """Smallest m satisfying 4 exp(-m pi^2/log(4/k')) <= epsilon."""
    k_prime = lambda_min / lambda_max
    log_term = math.log(4.0 / k_prime)
    for m in range(1, 400):
        if 4.0 * math.exp(-m * math.pi * math.pi / log_term) <= relative_epsilon:
            return m
    raise RuntimeError("MPMM elliptic shift count did not converge")


def mpmm_elliptic_shifts(count: int, lambda_max: float, kappa: float) -> np.ndarray:
    """Elliptic-dn distributed positive real matching points."""
    modulus = np.sqrt(1.0 - 1.0 / (kappa * kappa))
    k_complete = special.ellipk(modulus**2)
    theta = (2.0 * np.arange(1, count + 1) - 1.0) * k_complete / (2.0 * count)
    _, _, dn_values, _ = special.ellipj(theta, modulus**2)
    return np.sort(lambda_max * np.asarray(dn_values, dtype=np.float64))[::-1]


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------


def orthonormalize_block(basis, vectors):
    """Return an orthonormal basis for ``vectors`` modulo ``basis``."""
    block = np.asarray(vectors, dtype=np.float64).copy()
    if not block.size:
        return np.empty((block.shape[0], 0), dtype=np.float64)
    input_norm = float(np.linalg.norm(block))

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
    threshold = np.finfo(float).eps * max(block.shape) * input_norm
    keep = diagonal > threshold
    return np.ascontiguousarray(q[:, keep])


def _snapshot_svd_basis(snapshot_matrix, tolerance):
    """Compress full responses by direction using the FANTASTIC SVD cutoff."""
    snapshots = np.asarray(snapshot_matrix, dtype=np.float64)
    snapshots = snapshots / np.linalg.norm(snapshots, axis=0)
    U, singular_values, _ = scipy.linalg.svd(
        snapshots,
        full_matrices=False,
        check_finite=False,
    )
    keep = singular_values >= tolerance * singular_values[0]
    return np.ascontiguousarray(U[:, keep]), singular_values


# ---------------------------------------------------------------------------
# SPD linear solve
# ---------------------------------------------------------------------------

ENRICH_RTOL = 1.0e-6


def _rs_preconditioner(A):
    ml = pyamg.ruge_stuben_solver(A, interpolation="direct")
    return ml.aspreconditioner(cycle="V")


def spd_solve(A, b, x0=None, rtol=ENRICH_RTOL):
    """Solve ``A x = b`` by Ruge-Stuben AMG-preconditioned CG."""
    b = np.asarray(b, dtype=np.float64).ravel()
    if x0 is None:
        x0 = np.zeros_like(b)
    M = _rs_preconditioner(A.tocsr())
    x, info = spla.cg(
        A,
        b,
        x0=np.asarray(x0, dtype=np.float64).ravel(),
        rtol=rtol,
        atol=0.0,
        maxiter=2000,
        M=M,
    )
    if info != 0:
        raise RuntimeError(f"AMG-CG did not converge: info={info}")
    return x


# ---------------------------------------------------------------------------
# sparse operator helpers
# ---------------------------------------------------------------------------


def normalized_operators(K, C, f) -> Operators:
    """CSC-normalize K and C and copy f."""
    K = sp.csc_matrix(K)
    C = sp.csc_matrix(C)
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(f, dtype=np.float64).copy())


# ---------------------------------------------------------------------------
# per-port spectral bounds
# ---------------------------------------------------------------------------

EIGENBOUND_SUBSPACE_CAP = 64
EIGENBOUND_RTOL = 1.0e-8


def port_eigenvalue_bounds(
    K,
    C,
    g,
    *,
    shift=1.0e-6,
    cap=EIGENBOUND_SUBSPACE_CAP,
) -> tuple[float, float]:
    """Estimate the generalized spectral interval used for one source port."""
    K = K.tocsc()
    C = C.tocsc()
    c_diag = np.asarray(C.diagonal()).ravel()
    if np.any(c_diag <= 0.0):
        raise ValueError("port_eigenvalue_bounds: C must have a positive diagonal")
    g = np.asarray(g, dtype=np.float64).ravel()
    n = K.shape[0]
    g0 = g / np.linalg.norm(g)
    scale = float(np.median(c_diag))
    s0 = max(float(shift), scale * 1.0e-6)

    high = spla.eigsh(
        K,
        k=2,
        M=C,
        which="LM",
        tol=EIGENBOUND_RTOL,
        ncv=min(cap, n - 1),
        v0=g0,
        return_eigenvectors=False,
    )
    lambda_max = float(np.max(high))

    A_shift = (K + s0 * C).tocsc()
    low, _ = spla.lobpcg(
        A_shift,
        np.column_stack((g0, np.ones(n) / np.sqrt(n))),
        B=C,
        M=_rs_preconditioner(A_shift.tocsr()),
        largest=False,
        tol=EIGENBOUND_RTOL,
        maxiter=cap,
    )
    positive = np.asarray(low) - s0
    positive = positive[positive > 1.0e-9]
    if not positive.size:
        raise RuntimeError("port_eigenvalue_bounds resolved no positive eigenvalue")
    return float(positive.min()), lambda_max


# ---------------------------------------------------------------------------
# BCI Galerkin projection
# ---------------------------------------------------------------------------


def project_bci(
    operators: Operators,
    source_shape: np.ndarray,
    boundary_terms,
    basis,
    boundary_epsilon=1.0e-3,
):
    """Project the full-domain operators onto ``basis``, exposing BCI ports."""

    def project(matrix):
        reduced = sp.csc_matrix(basis.T @ matrix @ basis)
        reduced.eliminate_zeros()
        return reduced

    C_hat = project(operators.C)
    K_hat0 = project(operators.K)
    F_hat = np.asarray(basis.T @ source_shape, dtype=np.float64)

    n_cell = operators.K.shape[0]
    b_diag = np.zeros(n_cell)
    for term in boundary_terms:
        b_diag += np.asarray(term.diagonal()).ravel()
    b_cells = np.flatnonzero(b_diag > 0.0)
    if b_cells.size == 0:
        raise ValueError("no boundary cells found for BCI port extraction")

    V_d = np.asarray(basis[b_cells, :])
    U_d, s_d, Wt_d = np.linalg.svd(V_d, full_matrices=False)
    keep = np.flatnonzero(s_d > boundary_epsilon * s_d[0])
    if not keep.size:
        raise RuntimeError("boundary SVD retained no modes")
    U_t = np.ascontiguousarray(U_d[:, keep])
    S_t = s_d[keep]
    W_t = np.ascontiguousarray(Wt_d[keep, :].T)
    F_bdry = W_t * S_t

    A_bdry = []
    for term in boundary_terms:
        Hk_b = np.asarray(term.diagonal()).ravel()[b_cells]
        A_bdry.append((U_t.T @ (Hk_b[:, None] * U_t)).astype(np.float64))

    return C_hat, K_hat0, F_hat, F_bdry, A_bdry


def assemble_reduced_k(K_hat0, F_bdry, A_bdry, h_vec) -> sp.csc_matrix:
    """Assemble K_hat(h) = K_hat0 + sum h_k F_bdry A_k F_bdry^T."""
    K = K_hat0.tocsc()
    for h, A in zip(h_vec, A_bdry):
        K = K + h * sp.csc_matrix(F_bdry @ A @ F_bdry.T)
    return K.tocsc()


def solve_rom_steady(K_hat, F_hat, power) -> np.ndarray:
    """Solve K_hat theta = F_hat power by AMG-preconditioned CG."""
    rhs = F_hat @ np.asarray(power, dtype=np.float64)
    A = K_hat.tocsc().tocsr()
    theta, info = spla.cg(
        A,
        rhs,
        rtol=1.0e-8,
        atol=0.0,
        maxiter=10000,
        M=_rs_preconditioner(A),
    )
    if info != 0:
        raise RuntimeError(f"steady CG did not converge: info={info}")
    return theta.ravel()


TRANSIENT_RTOL = 1.0e-8
TRANSIENT_MAXITER = 10000


def solve_rom_transient(
    C_hat,
    K_hat,
    F_hat,
    power_t,
    dt: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step BDF1 transient of the reduced system."""
    n_modes = C_hat.shape[0]
    lhs = (C_hat / dt + K_hat.tocsc()).tocsc()
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    history = np.empty((times.size, n_modes), dtype=np.float64)
    theta = np.zeros(n_modes)
    history[0] = theta

    lhs_csr = lhs.tocsr()
    preconditioner = _rs_preconditioner(lhs_csr)
    for i in range(1, times.size):
        t = times[i]
        rhs = (C_hat @ theta) / dt + F_hat @ np.asarray(power_t(t), dtype=np.float64)
        theta, info = spla.cg(
            lhs_csr,
            rhs,
            x0=theta,
            rtol=TRANSIENT_RTOL,
            atol=0.0,
            maxiter=TRANSIENT_MAXITER,
            M=preconditioner,
        )
        if info != 0:
            raise RuntimeError(f"transient CG did not converge at t={t:g}: info={info}")
        history[i] = theta
    return times, history


# ---------------------------------------------------------------------------
# Extended-FANTASTIC parametric basis extraction
# ---------------------------------------------------------------------------

ROM_TOLERANCE = 1.0e-3
MAX_ORDER = 2048
PROBE_ROUNDS = 3
RANDOM_SEED = 20260805


def random_h(h_ranges, seed) -> tuple[float, ...]:
    """Draw one log-uniform parameter vector from the admissible HTC box."""
    return _draw_h(h_ranges, np.random.default_rng(seed))


def _draw_h(h_ranges, rng) -> tuple[float, ...]:
    h_ranges = np.asarray(h_ranges, dtype=np.float64)
    if h_ranges.size == 0:
        return ()
    low = np.log10(h_ranges[:, 0])
    high = np.log10(h_ranges[:, 1])
    return tuple(float(v) for v in 10.0 ** rng.uniform(low, high))


def _local_projection(K, C, boundary_terms, g, basis):
    K_hat = basis.T @ (K @ basis)
    C_hat = basis.T @ (C @ basis)
    H_hat = [basis.T @ (H @ basis) for H in boundary_terms]
    g_hat = basis.T @ g
    return K_hat, C_hat, H_hat, g_hat


def _local_response(projected, basis, h_vec, shift):
    K_hat, C_hat, H_hat, g_hat = projected
    A_hat = K_hat + shift * C_hat
    for h, H in zip(h_vec, H_hat):
        A_hat = A_hat + h * H
    coefficients = scipy.linalg.solve(
        A_hat,
        g_hat,
        assume_a="pos",
        check_finite=False,
    )
    return basis @ coefficients


def build_parametric_basis(
    operators,
    source_shape,
    boundary_terms,
    h_ranges,
    *,
    tolerance=ROM_TOLERANCE,
    max_order=MAX_ORDER,
    probe_rounds=PROBE_ROUNDS,
    seed=RANDOM_SEED,
):
    """Extract an Extended-FANTASTIC basis for an affine HTC family.

    Each (source, matching-point) pair owns an independent response space
    ``S_m(sigma)``.  A failed random-parameter residual test is solved exactly
    at that same parameter value, warm-started by the projected response that
    failed the test.  All exact responses are then collected in one snapshot
    matrix.  Each full response is scaled to unit norm before the closing SVD,
    so the FANTASTIC singular-value-ratio criterion measures redundancy of
    response directions rather than the amplitude variation across matching
    points.
    """
    started = time.perf_counter()
    K = operators.K.tocsc()
    C = operators.C.tocsc()
    G = np.asarray(source_shape, dtype=np.float64)
    h_ranges = np.asarray(h_ranges, dtype=np.float64)
    rng = np.random.default_rng(seed)

    snapshots = []
    plans = []
    history = []
    validation_count = 0
    max_accepted_residual = 0.0

    def full_operator(h_vec, shift):
        A = K + shift * C
        for h, H in zip(h_vec, boundary_terms):
            A = A + h * H
        return A.tocsc()

    for port in range(G.shape[1]):
        g = G[:, port].copy()
        g_norm = np.linalg.norm(g)
        lambda_min, lambda_max = port_eigenvalue_bounds(K, C, g)
        kappa = lambda_max / lambda_min
        shift_count = mpmm_elliptic_shift_count(
            tolerance,
            lambda_min,
            lambda_max,
        )
        shifts = mpmm_elliptic_shifts(shift_count, lambda_max, kappa)
        plans.append(
            {
                "port": int(port),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "kappa": float(kappa),
                "shift_count": int(shift_count),
                "shifts_per_s": shifts.tolist(),
            }
        )

        for shift in shifts:
            local_basis = np.empty((K.shape[0], 0), dtype=np.float64)
            h_vec = _draw_h(h_ranges, rng)
            initial_guess = np.zeros(K.shape[0], dtype=np.float64)
            full_solves = 0
            trials = 0

            while True:
                if len(snapshots) >= max_order:
                    raise RuntimeError("FANTASTIC extraction reached max_order")

                A = full_operator(h_vec, shift)
                response = np.asarray(
                    spd_solve(A, g, x0=initial_guess),
                    dtype=np.float64,
                ).ravel()
                snapshots.append(response)
                full_solves += 1

                block = orthonormalize_block(local_basis, response[:, None])
                if block.shape[1]:
                    local_basis = np.column_stack((local_basis, block))
                projected = _local_projection(
                    K,
                    C,
                    boundary_terms,
                    g,
                    local_basis,
                )

                accepted = True
                accepted_residual = 0.0
                for _ in range(probe_rounds):
                    trial_h = _draw_h(h_ranges, rng)
                    estimate = _local_response(
                        projected,
                        local_basis,
                        trial_h,
                        shift,
                    )
                    trial_A = full_operator(trial_h, shift)
                    residual = np.linalg.norm(trial_A @ estimate - g) / g_norm
                    validation_count += 1
                    trials += 1
                    accepted_residual = max(accepted_residual, float(residual))
                    if residual > tolerance:
                        h_vec = trial_h
                        initial_guess = estimate
                        accepted = False
                        break

                if accepted:
                    max_accepted_residual = max(
                        max_accepted_residual,
                        accepted_residual,
                    )
                    history.append(
                        {
                            "port": int(port),
                            "shift": float(shift),
                            "full_solves": int(full_solves),
                            "validation_count": int(trials),
                            "accepted_residual": float(accepted_residual),
                        }
                    )
                    break

    snapshot_matrix = np.column_stack(snapshots)
    basis, singular_values = _snapshot_svd_basis(snapshot_matrix, tolerance)
    svd_order = basis.shape[1]

    # The h-free BCI operator has the uniform-temperature null mode; preserve it exactly.
    constant = np.ones((K.shape[0], 1), dtype=np.float64)
    constant /= np.linalg.norm(constant)
    constant = orthonormalize_block(basis, constant)
    if constant.shape[1]:
        basis = np.column_stack((basis, constant))

    orthogonality_error = float(
        np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1])))
    )

    return np.ascontiguousarray(basis), {
        "per_port_plans": plans,
        "tolerance": float(tolerance),
        "probe_rounds": int(probe_rounds),
        "candidate_count": int(len(snapshots) + validation_count),
        "processed_candidate_count": int(len(snapshots)),
        "validation_count": int(validation_count),
        "pre_svd_order": int(len(snapshots)),
        "svd_kept_order": int(svd_order),
        "basis_order": int(basis.shape[1]),
        "maximum_order": int(max_order),
        "orthogonality_error": orthogonality_error,
        "max_accepted_residual": float(max_accepted_residual),
        "history": history,
        "seconds": time.perf_counter() - started,
        "singular_value_ratios": (singular_values / singular_values[0]).tolist(),
    }


# ---------------------------------------------------------------------------
# accuracy metrics
# ---------------------------------------------------------------------------

MAX_RELATIVE_RISE_ERROR = 0.01


def _field_error_metrics(reference, approximation, ambient_K: float) -> dict:
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
    """Steady and transient-final field accuracy versus the reference."""
    steady = _field_error_metrics(reference_steady, reduced_steady, ambient_K)
    transient = _field_error_metrics(
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
