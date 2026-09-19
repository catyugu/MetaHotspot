"""Certified energy-residual decisions for Extended FANTASTIC experiments.

The exact paper-style indicator used here is the squared relative A-energy
error

    rho_A = (r.T A^{-1} r) / (b.T A^{-1} b),

for A x = b and a Galerkin approximation x_hat. The certificate avoids
forming A^{-1} r. It exploits A >= shift * C and diagonal C, which is exact
for MetaHotspot's finite-volume thermal operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class CertificateResult:
    decision: str
    iterations: int
    lower_q: float
    upper_q: float
    threshold_q: float
    correction: np.ndarray | None = None


def residual_and_base_energy(A, b, estimate):
    """Return residual and the part of exact energy known from estimate."""
    b = np.asarray(b, dtype=np.float64).ravel()
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    Aestimate = np.asarray(A @ estimate, dtype=np.float64).ravel()
    residual = b - Aestimate
    base = float(2.0 * np.dot(b, estimate) - np.dot(estimate, Aestimate))
    return residual, base


def exact_energy_ratio(A, b, estimate, *, residual=None):
    """Compute the exact squared relative A-energy error by a sparse solve."""
    if residual is None:
        residual, base = residual_and_base_energy(A, b, estimate)
    else:
        residual = np.asarray(residual, dtype=np.float64).ravel()
        _, base = residual_and_base_energy(A, b, estimate)
    correction = np.asarray(spla.spsolve(A.tocsc(), residual), dtype=np.float64)
    q = max(float(np.dot(residual, correction)), 0.0)
    denom = base + q
    ratio = 0.0 if q == 0.0 else q / denom
    return float(ratio), correction, float(q), float(denom)


def mass_upper_bound(residual, c_diag, shift):
    """Strict q upper bound from A >= shift*C."""
    residual = np.asarray(residual, dtype=np.float64).ravel()
    c_diag = np.asarray(c_diag, dtype=np.float64).ravel()
    if shift <= 0.0:
        raise ValueError("shift must be positive")
    if np.any(c_diag <= 0.0):
        raise ValueError("C diagonal must be positive")
    return float(np.dot(residual / c_diag, residual) / shift)


def certify_energy_ratio(
    A,
    c_diag,
    b,
    estimate,
    shift,
    tolerance,
    *,
    max_iterations=None,
):
    """Certify rho_A <= tolerance or rho_A > tolerance.

    With q=r^T A^{-1}r and
      g=2 b^T x_hat-x_hat^T A x_hat,
    rho_A=q/(g+q), so acceptance is exactly
      q <= tolerance/(1-tolerance) * g.

    With B=C^{-1/2} A C^{-1/2}, z=C^{-1/2}r and A>=shift*C,
    B>=shift*I. CG from zero gives a lower bound L_k for
    z^T B^{-1}z, while the unresolved energy is at most
    ||r_k||^2/shift. Thus [L_k,U_k] is a certified interval for q.
    """
    if not (0.0 < tolerance < 1.0):
        raise ValueError("tolerance must lie in (0,1)")
    if shift <= 0.0:
        raise ValueError("shift must be positive")
    c_diag = np.asarray(c_diag, dtype=np.float64).ravel()
    if np.any(c_diag <= 0.0):
        raise ValueError("C diagonal must be positive")

    b = np.asarray(b, dtype=np.float64).ravel()
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    residual, base = residual_and_base_energy(A, b, estimate)
    scale = max(
        abs(float(np.dot(b, estimate))),
        abs(float(np.dot(estimate, A @ estimate))),
        1.0,
    )
    if base < -1.0e-11 * scale:
        raise ValueError("estimate is incompatible with the Galerkin energy decomposition")
    base = max(base, 0.0)
    threshold = float(tolerance / (1.0 - tolerance) * base)

    inv_sqrt_c = 1.0 / np.sqrt(c_diag)
    z = inv_sqrt_c * residual
    initial_norm_sq = float(np.dot(z, z))
    upper = initial_norm_sq / shift
    if upper <= threshold:
        return CertificateResult(
            "accept", 0, 0.0, upper, threshold, np.zeros_like(estimate)
        )
    if initial_norm_sq == 0.0:
        return CertificateResult(
            "accept", 0, 0.0, 0.0, threshold, np.zeros_like(estimate)
        )

    n = b.size
    if max_iterations is None:
        max_iterations = n
    y = np.zeros_like(z)
    cg_residual = z.copy()
    direction = z.copy()
    rr = initial_norm_sq
    lower = 0.0

    for iteration in range(1, int(max_iterations) + 1):
        original_direction = inv_sqrt_c * direction
        Bdirection = inv_sqrt_c * np.asarray(A @ original_direction).ravel()
        curvature = float(np.dot(direction, Bdirection))
        if curvature <= 0.0:
            raise RuntimeError("mass-scaled operator lost positive definiteness")
        alpha = rr / curvature
        y += alpha * direction
        cg_residual -= alpha * Bdirection
        rr_new = float(np.dot(cg_residual, cg_residual))

        lower = float(np.dot(z, y))
        upper = lower + rr_new / shift
        correction = inv_sqrt_c * y

        if upper <= threshold:
            return CertificateResult(
                "accept", iteration, lower, upper, threshold, correction
            )
        if lower > threshold:
            return CertificateResult(
                "reject", iteration, lower, upper, threshold, correction
            )

        if rr_new <= np.finfo(float).eps**2 * initial_norm_sq:
            decision = "accept" if lower <= threshold else "reject"
            return CertificateResult(
                decision, iteration, lower, lower, threshold, correction
            )

        beta = rr_new / rr
        direction = cg_residual + beta * direction
        rr = rr_new

    return CertificateResult(
        "undecided",
        int(max_iterations),
        lower,
        upper,
        threshold,
        inv_sqrt_c * y,
    )
