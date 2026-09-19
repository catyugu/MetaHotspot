"""Gauss-Radau certificate for the exact Extended-FANTASTIC energy residual."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from energy_residual import CertificateResult, residual_and_base_energy


def _solve_symmetric_tridiagonal(diagonal, off_diagonal, rhs):
    """Solve a symmetric tridiagonal system in O(n) work."""
    diagonal = np.asarray(diagonal, dtype=np.float64)
    off_diagonal = np.asarray(off_diagonal, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    n = diagonal.size
    ab = np.zeros((3, n), dtype=np.float64)
    ab[1] = diagonal
    if n > 1:
        ab[0, 1:] = off_diagonal
        ab[2, :-1] = off_diagonal
    return scipy.linalg.solve_banded(
        (1, 1), ab, rhs, check_finite=False
    )


def certify_energy_ratio_radau(
    A,
    anchor_diag,
    b,
    estimate,
    tolerance,
    *,
    max_iterations=None,
):
    """Certify the exact squared relative A-energy error.

    anchor_diag defines a positive diagonal P satisfying A >= P.
    In the thermal experiment,

        P = shift*C + sum_j h_j H_j

    and the remaining h-free conductance K is positive semidefinite.
    Hence B=P^{-1/2} A P^{-1/2} >= I.

    Gauss quadrature gives a lower bound for z^T B^{-1}z and left
    Gauss-Radau, with the known lower spectral endpoint 1, gives an
    upper bound. The inverse quadratic form is never explicitly solved
    merely to classify an accepted probe.
    """
    if not (0.0 < tolerance < 1.0):
        raise ValueError("tolerance must lie in (0,1)")
    anchor_diag = np.asarray(anchor_diag, dtype=np.float64).ravel()
    if np.any(anchor_diag <= 0.0):
        raise ValueError("anchor diagonal must be positive")

    b = np.asarray(b, dtype=np.float64).ravel()
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    residual, base = residual_and_base_energy(A, b, estimate)
    scale = max(
        abs(float(np.dot(b, estimate))),
        abs(float(np.dot(estimate, A @ estimate))),
        1.0,
    )
    if base < -1.0e-11 * scale:
        raise ValueError(
            "estimate is incompatible with the energy decomposition"
        )
    base = max(base, 0.0)
    threshold = float(tolerance / (1.0 - tolerance) * base)

    inv_sqrt = 1.0 / np.sqrt(anchor_diag)
    z = inv_sqrt * residual
    beta0_sq = float(np.dot(z, z))

    if beta0_sq <= threshold:
        return CertificateResult(
            "accept",
            0,
            0.0,
            beta0_sq,
            threshold,
            np.zeros_like(estimate),
        )
    if beta0_sq == 0.0:
        return CertificateResult(
            "accept", 0, 0.0, 0.0, threshold, np.zeros_like(estimate)
        )

    n = b.size
    if max_iterations is None:
        max_iterations = n

    beta0 = float(np.sqrt(beta0_sq))
    vector = z / beta0
    previous = np.zeros_like(vector)
    beta_previous = 0.0
    diagonal = []
    off_diagonal = []
    vectors = []
    endpoint = np.nextafter(1.0, 0.0)

    for iteration in range(1, int(max_iterations) + 1):
        vectors.append(vector.copy())
        original_vector = inv_sqrt * vector
        work = inv_sqrt * np.asarray(A @ original_vector).ravel()
        if iteration > 1:
            work -= beta_previous * previous
        alpha = float(np.dot(vector, work))
        work -= alpha * vector

        if iteration > 1:
            work -= previous * float(np.dot(previous, work))
        work -= vector * float(np.dot(vector, work))

        beta_next = float(np.linalg.norm(work))
        diagonal.append(alpha)
        diag = np.asarray(diagonal, dtype=np.float64)
        off = np.asarray(off_diagonal, dtype=np.float64)

        rhs = np.zeros(iteration, dtype=np.float64)
        rhs[0] = 1.0
        gauss_coeff = _solve_symmetric_tridiagonal(diag, off, rhs)
        lower = beta0_sq * float(gauss_coeff[0])

        if beta_next <= np.finfo(float).eps * max(abs(alpha), 1.0):
            upper = lower
        else:
            endpoint_rhs = np.zeros(iteration, dtype=np.float64)
            endpoint_rhs[-1] = beta_next * beta_next
            endpoint_solution = _solve_symmetric_tridiagonal(
                diag - endpoint, off, endpoint_rhs
            )
            final_diagonal = endpoint + float(endpoint_solution[-1])

            radau_diag = np.concatenate((diag, [final_diagonal]))
            radau_off = np.concatenate((off, [beta_next]))
            radau_rhs = np.zeros(iteration + 1, dtype=np.float64)
            radau_rhs[0] = 1.0
            radau_coeff = _solve_symmetric_tridiagonal(
                radau_diag, radau_off, radau_rhs
            )
            upper = beta0_sq * float(radau_coeff[0])
            upper = np.nextafter(upper, np.inf)

        decided = (
            upper <= threshold
            or lower > threshold
            or upper == lower
        )
        if decided:
            if upper == lower:
                decision = (
                    "accept" if lower <= threshold else "reject"
                )
            else:
                decision = (
                    "accept" if upper <= threshold else "reject"
                )

            correction_scaled = np.zeros_like(z)
            for coefficient, lanczos_vector in zip(
                gauss_coeff, vectors
            ):
                correction_scaled += coefficient * lanczos_vector
            correction = inv_sqrt * (
                beta0 * correction_scaled
            )
            return CertificateResult(
                decision,
                iteration,
                lower,
                upper,
                threshold,
                correction,
            )

        previous = vector
        vector = work / beta_next
        beta_previous = beta_next
        off_diagonal.append(beta_next)

    correction_scaled = np.zeros_like(z)
    for coefficient, lanczos_vector in zip(gauss_coeff, vectors):
        correction_scaled += coefficient * lanczos_vector
    return CertificateResult(
        "undecided",
        int(max_iterations),
        lower,
        upper,
        threshold,
        inv_sqrt * (beta0 * correction_scaled),
    )
