"""Dynamic bridge toy: does an inexact moment space cost O(delta) or O(delta^2)?

The open question is whether finite matching-resolvent defects can be turned into
a dynamic guarantee (the Phi_Sigma bridge).  The exact MPMM space has a Hermite
structure at the matching shifts, so the value defect there is O(delta^2), but a
perturbed space need not inherit that.  This bench answers the order question on
a dense artificial problem before any of the heavier machinery is justified.

Model (mass-whitened, so C = I and the pencil has one symmetric parameter matrix)

    B = B^T > 0,      H(s) = f^T (s I + B)^-1 f,      x(sigma) = (B + sigma I)^-1 f

with eigenvalues lambda_i = kappa ** u_i, kappa = lambda_max / lambda_min and
u the family below; the frequency scale is fixed by a = 1, b = kappa.

Two modes:

  rotation   Take the exact moment space U = orth[x(sigma_j)] and a controlled
             rotation V_theta = U cos(theta) + Q sin(theta) into an orthogonal
             complement Q.  Report the matching defects delta_* of V_theta and the
             relative dynamics perturbation between the two ROMs,

                 e_H2    = || H_U - H_{V_theta} ||_H2 / || H ||_H2,
                 e_state = || U e^{-B_U t} f_U - V_theta e^{-B_V t} f_{V_theta}
                             ||_L2(0, inf; B) / || e^{-B t} f ||_L2(0, inf; B).

             Both are exact (Lyapunov), never a frequency grid.  The verdict is
             the local exponent of e_H2 against delta_*: one means the quadratic
             Phi_Sigma is refuted on this instance, two means it survives.

  skeleton   The full plan's cardinal functions, the Massei--Robol identity
             1 - (s + lambda) I_Sigma[(. + lambda)^-1](s) = r_Sigma(lambda)/r_Sigma(-s),
             and the SISO sample-propagation constant K_samp.

Everything is a 4x4 or 8x8 dense solve; this is a counterexample search, not a
production accuracy test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, solve_lyapunov
from scipy.optimize import minimize_scalar
from scipy.special import ellipk

try:
    import mpmath
except ImportError:  # the 80-digit reference is optional, the invariants are not
    mpmath = None

from metahotspot.macromodel.utils import mpmm_elliptic_shift_count, mpmm_elliptic_shifts

RELATIVE_EPSILON = 1e-3

SPECTRUM_FAMILIES = {
    "log-uniform": {
        4: [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        8: list(np.linspace(0.0, 1.0, 8)),
        26: list(np.linspace(0.0, 1.0, 26)),
    },
    "endpoint-cluster": {
        4: [0.0, 1e-3, 1.0 - 1e-3, 1.0],
        8: [0.0, 1e-6, 1e-4, 1e-2, 1.0 - 1e-2, 1.0 - 1e-4, 1.0 - 1e-6, 1.0],
        26: [0.0, 1e-8, 1e-6, 1e-4, 1e-2] + list(np.linspace(0.05, 0.95, 16))
            + [1.0 - 1e-2, 1.0 - 1e-4, 1.0 - 1e-6, 1.0 - 1e-8, 1.0],
    },
    "low-cluster": {
        4: [0.0, 1e-4, 1e-2, 1.0],
        8: [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 0.3, 0.6, 1.0],
        26: [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1] + list(np.linspace(0.15, 0.95, 19)) + [1.0],
    },
    "repeated": {
        4: [0.0, 0.0, 1.0, 1.0],
        8: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        26: [0.0] * 13 + [1.0] * 13,
    },
}

ROTATIONS = np.logspace(-1.0, -6.0, 11)


def spectrum(kappa: float, family: str, size: int) -> np.ndarray:
    """Eigenvalues lambda_i = kappa ** u_i for a prerregistered family."""
    return np.array([kappa**u for u in SPECTRUM_FAMILIES[family][size]])


def source(eigenvalues: np.ndarray, family: str) -> np.ndarray:
    """Right-hand side: flat, or weighted so no single time scale dominates H2."""
    if family == "flat":
        return np.full(eigenvalues.size, 1.0 / math.sqrt(eigenvalues.size))
    if family == "h2-balanced":
        values = eigenvalues**0.25
        return values / np.linalg.norm(values)
    raise ValueError(f"unknown source family {family!r}")


def plan_shifts(kappa: float) -> np.ndarray:
    """The delivered plan's matching points for [1, kappa], descending."""
    count = mpmm_elliptic_shift_count(RELATIVE_EPSILON, 1.0, kappa)
    return np.asarray(mpmm_elliptic_shifts(count, kappa, kappa), dtype=np.float64)


def moment_space(b_matrix, f_vector, shifts, tolerance=1e-10):
    """Orthonormal basis of the moment space spanned by the snapshots.

    The snapshot matrix can be rank deficient (a repeated eigenvalue family has
    fewer distinct snapshots than shifts), so the basis comes from a truncated
    SVD: the extra columns a bare QR would return are numerical padding, not part
    of the exact moment space.  The singular values are returned for the record.
    """
    snapshots = np.column_stack([np.linalg.solve(b_matrix + sigma * np.eye(b_matrix.shape[0]), f_vector)
                                 for sigma in shifts])
    left, values, _ = np.linalg.svd(snapshots, full_matrices=False)
    rank = int(np.sum(values > tolerance * values[0])) if values[0] > 0.0 else 0
    return left[:, :rank], values, rank


def complement_space(basis, seed: int) -> np.ndarray:
    """Orthonormal basis of the orthogonal complement, of the same dimension."""
    size, rank = basis.shape
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal((size, rank))
    probe -= basis @ (basis.T @ probe)
    complement, _ = np.linalg.qr(probe)
    return complement


def matching_defects(b_matrix, f_vector, shifts, space) -> np.ndarray:
    """delta_j = ||x_j - x_V,j||_{B + sigma_j I} / ||x_j||_{B + sigma_j I}."""
    defects = []
    for sigma in shifts:
        operator = b_matrix + sigma * np.eye(b_matrix.shape[0])
        exact = np.linalg.solve(operator, f_vector)
        reduced = space @ np.linalg.solve(space.T @ operator @ space, space.T @ f_vector)
        error = exact - reduced
        defects.append(math.sqrt((error @ operator @ error) / (exact @ operator @ exact)))
    return np.array(defects)


def h2_norm(generator, b_vector, c_vector) -> float:
    """||c (s I - G)^-1 b||_H2 for a stable generator G, via the Gramian.

    The generator carries the sign: H(s) = f^T (s I + B)^-1 f has G = -B, and the
    semigroup e^{-B t} is e^{G t}.  The Gramian solves G P + P G^T + b b^T = 0.
    """
    gramian = solve_lyapunov(generator, -np.outer(b_vector, b_vector))
    return math.sqrt(max(float(c_vector @ gramian @ c_vector), 0.0))


def difference_h2(left_generator, left_source, right_generator, right_source) -> float:
    """||G_left - G_right||_H2 for two SISO transfers with the same input."""
    system = block_diag(left_generator, right_generator)
    input_vector = np.concatenate([left_source, right_source])
    output_vector = np.concatenate([left_source, -right_source])
    return h2_norm(system, input_vector, output_vector)


def hankel_norm(generator, b_vector, c_vector) -> float:
    """Largest Hankel singular value of c (s I - G)^-1 b for a stable G."""
    controllable = solve_lyapunov(generator, -np.outer(b_vector, b_vector))
    observable = solve_lyapunov(generator.T, -np.outer(c_vector, c_vector))
    return math.sqrt(max(float(np.max(np.linalg.eigvals(controllable @ observable).real)), 0.0))


def difference_hankel(left_generator, left_source, right_generator, right_source) -> float:
    """Hankel norm of the difference of two SISO transfers with the same input."""
    system = block_diag(left_generator, right_generator)
    return hankel_norm(system, np.concatenate([left_source, right_source]),
                       np.concatenate([left_source, -right_source]))


def state_impulse_energy(b_matrix, f_vector, space, other) -> float:
    """||W e^{-B_W t} f_W - W' e^{-B_W' t} f_W'||_L2(0, inf; B) over ||e^{-B t}f||.

    The numerator is one Lyapunov solve on the block system that carries both
    semigroups; the denominator is the same quantity for the full model.
    """
    left = -(space.T @ b_matrix @ space)
    right = -(other.T @ b_matrix @ other)
    left_source, right_source = space.T @ f_vector, other.T @ f_vector
    embedding = np.hstack([space, -other])
    weight = embedding.T @ b_matrix @ embedding
    joint = block_diag(left, right)
    gramian = solve_lyapunov(joint.T, -weight)
    generator = np.concatenate([left_source, right_source])
    numerator = float(generator @ gramian @ generator)
    reference = solve_lyapunov(-b_matrix.T, -b_matrix)
    denominator = float(f_vector @ reference @ f_vector)
    return math.sqrt(max(numerator, 0.0) / denominator)


def polynomial_order(delta, error, floor=1e-12) -> float:
    """Local exponent of error against delta over the smallest rotations.

    Rotations that push the defect below the double-precision floor of the
    difference carry no information, so they are dropped rather than fitted.
    """
    delta = np.asarray(delta, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    usable = (delta > 0.0) & (error > floor)
    if usable.sum() < 3:
        return float("nan")
    tail = slice(max(int(usable.sum()) - 6, 0), int(usable.sum()))
    slope = np.polyfit(np.log(delta[usable][tail]), np.log(error[usable][tail]), 1)[0]
    return float(slope)


def rotation_rows(arguments) -> list[dict]:
    """Layer 1: one row per (kappa, family, source, size, theta, branch).

    ``branch`` is the sign of the rotation, ``+`` for V = U cos theta + Q sin
    theta and ``-`` for V = U cos theta - Q sin theta.  Both are needed for the
    sharp-baseline gate: the first-order Frechet derivative changes sign with
    Q -> -Q, so if it is nonzero at least one branch must increase the total
    error over the exact-MPMM baseline.
    """
    rows = []
    for kappa in arguments.kappas:
        shifts_all = plan_shifts(kappa)
        for family in arguments.families:
            eigenvalues = spectrum(kappa, family, arguments.size)
            if not np.all(np.diff(eigenvalues) >= 0.0):
                raise ValueError(f"family {family!r} is not nondecreasing at {arguments.size}")
            b_matrix = np.diag(eigenvalues)
            for source_family in arguments.sources:
                f_vector = source(eigenvalues, source_family)
                count = arguments.size // 2
                picks = [0, len(shifts_all) - 1] if count == 2 else [
                    int(round(index * (len(shifts_all) - 1) / (count - 1))) for index in range(count)
                ]
                shifts = shifts_all[picks]
                exact, singular_values, rank = moment_space(b_matrix, f_vector, shifts)
                other = complement_space(exact, arguments.seed)
                exact_b, exact_f = exact.T @ b_matrix @ exact, exact.T @ f_vector
                reference_h2 = h2_norm(-b_matrix, f_vector, f_vector)
                reference_hankel = hankel_norm(-b_matrix, f_vector, f_vector)
                baseline_h2 = difference_h2(-exact_b, exact_f, -b_matrix, f_vector) / reference_h2
                block = []
                for theta in ROTATIONS:
                    spaces = {
                        "+": exact * math.cos(theta) + other * math.sin(theta),
                        "-": exact * math.cos(theta) - other * math.sin(theta),
                    }
                    totals = {}
                    branch_rows = {}
                    for branch, space in spaces.items():
                        reduced_b, reduced_f = space.T @ b_matrix @ space, space.T @ f_vector
                        error_h2 = difference_h2(-reduced_b, reduced_f, -exact_b, exact_f) / reference_h2
                        hankel = difference_hankel(-reduced_b, reduced_f, -exact_b, exact_f) / reference_hankel
                        total = difference_h2(-reduced_b, reduced_f, -b_matrix, f_vector) / reference_h2
                        delta = float(matching_defects(b_matrix, f_vector, shifts, space).max())
                        error_state = float(state_impulse_energy(b_matrix, f_vector, exact, space))
                        totals[branch] = total
                        branch_rows[branch] = {
                            "mode": "rotation",
                            "kappa": float(kappa),
                            "spectrum_family": family,
                            "source_family": source_family,
                            "size": arguments.size,
                            "shifts": [float(value) for value in shifts],
                            "snapshot_rank": rank,
                            "snapshot_singular_values": [float(value) for value in singular_values],
                            "theta": float(theta),
                            "branch": branch,
                            "delta_star": delta,
                            "rel_h2_UV": float(error_h2),
                            "rel_state_UV": error_state,
                            "rel_hankel_UV": float(hankel),
                            "rel_h2_to_true": float(total),
                            "rel_h2_of_U": float(baseline_h2),
                            "h2_over_delta": float(error_h2 / delta),
                            "h2_over_delta2": float(error_h2 / delta**2),
                            "hankel_over_delta": float(hankel / delta),
                            "hankel_over_delta2": float(hankel / delta**2),
                            "state_over_delta": float(error_state / delta),
                            "K_sample": None,
                            "K_vec": None,
                            "rho": None,
                            "rho2": None,
                        }
                    excess = max(totals.values()) - baseline_h2
                    for row in branch_rows.values():
                        row["total_excess"] = float(excess)
                        row["excess_over_delta"] = float(excess / row["delta_star"])
                    block.extend(branch_rows.values())
                orders = {}
                for name, key in (("h2", "rel_h2_UV"), ("state", "rel_state_UV"),
                                  ("hankel", "rel_hankel_UV"), ("excess", "total_excess")):
                    orders[name] = polynomial_order([row["delta_star"] for row in block],
                                                    [row[key] for row in block])
                for row in block:
                    for name in orders:
                        row[f"{name}_order"] = orders[name]
                resolved = [row for row in block if row["rel_h2_UV"] > 1e-12]
                best = resolved[-1]
                print(f"kappa={kappa:.3e} family={family} source={source_family} "
                      f"size={arguments.size} rank={rank}/{len(shifts)} "
                      f"delta_*={block[0]['delta_star']:.3e}..{block[-1]['delta_star']:.3e} "
                      f"order h2={orders['h2']:.3f} hankel={orders['hankel']:.3f} "
                      f"state={orders['state']:.3f} excess={orders['excess']:.3f} "
                      f"h2/δ*={best['h2_over_delta']:.3e} hankel/δ*={best['hankel_over_delta']:.3e} "
                      f"st/δ*={best['state_over_delta']:.3e} exc/δ*={best['excess_over_delta']:.3e} "
                      f"e_U={baseline_h2:.3e}", flush=True)
                rows.extend(block)
    return rows


def signed_log_product(values) -> tuple[float, float]:
    """Sign and log magnitude of a product of nonzero reals, without overflow."""
    sign, total = 1.0, 0.0
    for value in values:
        if value == 0.0:
            return 0.0, -math.inf
        if value < 0.0:
            sign = -sign
        total += math.log(abs(float(value)))
    return sign, total


def plan_ratio(shifts, point: float) -> float:
    """r_Sigma(z) = prod_k (z - sigma_k) / (z + sigma_k) for a real argument."""
    sign, total = signed_log_product([(point - sigma) / (point + sigma) for sigma in shifts])
    if sign == 0.0:
        return 0.0
    return sign * math.exp(total) if total > -700.0 else 0.0


def cardinal_values(shifts, point: float) -> np.ndarray:
    """Cardinal functions at a real argument, in the Massei--Robol product form.

        ell_j(s) = 2 sigma_j / (s + sigma_j)
                   prod_{k != j} (s - sigma_k)/(s + sigma_k) * (sigma_j + sigma_k)/(sigma_j - sigma_k)
    """
    values = []
    for j, sigma_j in enumerate(shifts):
        sign, total = signed_log_product([(point - sigma) / (point + sigma)
                                          for k, sigma in enumerate(shifts) if k != j])
        sign_k, total_k = signed_log_product([(sigma_j + sigma) / (sigma_j - sigma)
                                              for k, sigma in enumerate(shifts) if k != j])
        sign_lead, total_lead = signed_log_product([2.0 * sigma_j / (point + sigma_j)])
        sign = sign * sign_k * sign_lead
        total = total + total_k + total_lead
        values.append(0.0 if sign == 0.0 else sign * math.exp(total))
    return np.array(values)


def high_precision_cardinal(shifts, point: float) -> list:
    """80-digit reference values of the cardinal functions at a real argument."""
    with mpmath.workdps(80):
        values = []
        for j, sigma_j in enumerate(shifts):
            target, current = mpmath.mpf(str(point)), mpmath.mpf(str(sigma_j))
            value = 2 * current / (target + current)
            for k, sigma in enumerate(shifts):
                if k == j:
                    continue
                other = mpmath.mpf(str(sigma))
                value *= (target - other) / (target + other)
                value *= (current + other) / (current - other)
            values.append(value)
        return values


def plan_invariants(shifts, arguments) -> dict:
    """The three algebraic identities the whole skeleton layer rests on.

    I1  ell_j(sigma_k) = delta_jk.
    I2  1 - (s + lambda) sum_j ell_j(s)/(sigma_j + lambda) = r_Sigma(lambda)/r_Sigma(-s),
        with the mixed relative tolerance 1e-11 (1 + |L| + |R|).
    I3  |r_Sigma(-i omega)| = 1 on the imaginary axis.
    Plus a 80-digit mpmath reference for the cardinal values themselves.
    """
    count = len(shifts)
    identity_error = 0.0
    for k, sigma in enumerate(shifts):
        values = cardinal_values(shifts, sigma)
        for j in range(count):
            identity_error = max(identity_error, abs(values[j] - (1.0 if j == k else 0.0)))
    residual_error = 0.0
    tests = [shifts[0], 0.5 * (shifts[0] + shifts[-1]), shifts[-1],
             math.sqrt(shifts[0] * shifts[-1])]
    lambdas = [shifts[-1], math.sqrt(shifts[0] * shifts[-1]), shifts[0], 2.0 * shifts[0]]
    for point in tests:
        values = cardinal_values(shifts, point)
        for lam in lambdas:
            left = 1.0 - (point + lam) * float(
                np.sum([values[j] / (shifts[j] + lam) for j in range(count)]))
            right = plan_ratio(shifts, lam) / plan_ratio(shifts, -point)
            residual_error = max(residual_error, abs(left - right) / (1.0 + abs(left) + abs(right)))
    axis_error = 0.0
    for omega in np.logspace(math.log10(shifts[-1]) - 3.0, math.log10(shifts[0]) + 3.0, 200):
        value = np.prod([(1j * omega - sigma) / (1j * omega + sigma) for sigma in shifts])
        axis_error = max(axis_error, abs(abs(value) - 1.0))
    reference = None
    if mpmath is not None:
        reference = 0.0
        for point in (shifts[0], math.sqrt(shifts[0] * shifts[-1])):
            exact = high_precision_cardinal(shifts, float(point))
            double = cardinal_values(shifts, float(point))
            for j in range(count):
                scale = 1.0 + abs(float(exact[j])) + abs(double[j])
                reference = max(reference, abs(float(exact[j]) - double[j]) / scale)
    return {
        "delta_jk_error": identity_error,
        "skeleton_identity_error": residual_error,
        "imaginary_axis_error": axis_error,
        "mpmath_reference_error": reference,
    }


def plan_geometry(shifts, kappa: float) -> dict:
    """rho_Sigma, the elliptic upper bound q_m, and the elliptic constant.

    ``rho_Sigma`` is the maximum of ``|r_Sigma|`` over the delivered interval
    ``[1, kappa]``; ``q_m = 4 exp(-m pi^2 / log(4 kappa))`` is the elliptic
    error-bound construction, whose square should land on the same scale.  The
    interval is the *plan's* interval, not the span of the extreme shifts: the
    elliptic set does not touch the interval endpoints.
    """
    lower, upper = 1.0, float(kappa)
    grid = np.geomspace(lower, upper, 4001)
    rho = max(abs(plan_ratio(shifts, float(point))) for point in grid)
    modulus = math.sqrt(1.0 - 1.0 / (upper / lower) ** 2)
    k_complete = float(ellipk(modulus**2))
    return {
        "rho": float(rho),
        "rho2": float(rho**2),
        "q_m": float(4.0 * math.exp(-len(shifts) * math.pi**2 / math.log(4.0 * upper / lower))),
        "elliptic_k": k_complete,
    }


def partial_fractions(shifts) -> np.ndarray:
    """Residues a_jk of ell_j(s) = sum_k a_jk / (s + sigma_k).

    Both the numerator products and the pole denominators are ratios of numbers
    that span six decades, so they are accumulated as signed logarithms.  The
    weights are *not* all of one sign: on the delivered plan they reach
    magnitudes of 2e+07 and cancel, so the sanity condition is not positivity but
    ``fraction_reconstruction``, which rebuilds the cardinal functions from the
    residues at the band endpoints.
    """
    count = len(shifts)
    weights = np.empty((count, count))
    for j, sigma_j in enumerate(shifts):
        sign_c, total_c = signed_log_product([(sigma_j + sigma) / (sigma_j - sigma)
                                              for k, sigma in enumerate(shifts) if k != j])
        for k, sigma_k in enumerate(shifts):
            sign_n, total_n = signed_log_product([-(sigma_k + sigma) for i, sigma in enumerate(shifts)
                                                  if i != j])
            sign_d, total_d = signed_log_product([sigma - sigma_k for i, sigma in enumerate(shifts)
                                                  if i != k])
            sign = sign_c * sign_n * sign_d
            total = (math.log(2.0 * sigma_j) + total_c + total_n - total_d)
            weights[j, k] = sign * math.exp(total)
    return weights


def sample_constant(shifts, weights, transfer_values, norm) -> float:
    """K_sample: worst skeleton interpolation of matching-value errors.

        K_sample = max over z_j in {0, h_j} of sqrt(z^T Q z) / ||H||_H2,
        Q_ij = sum_{k,l} a_ik a_jl / (sigma_k + sigma_l),

    and because Q is positive semidefinite the maximum of a convex quadratic over
    the box is attained at a vertex, so all 2^m vertices are enumerated exactly.
    """
    poles = 1.0 / (shifts[:, None] + shifts[None, :])
    gram = weights @ poles @ weights.T
    gram = 0.5 * (gram + gram.T)
    worst = 0.0
    for mask in range(1 << len(shifts)):
        z = transfer_values * np.array([(mask >> index) & 1 for index in range(len(shifts))])
        worst = max(worst, math.sqrt(max(float(z @ gram @ z), 0.0)))
    return worst / norm


def plan_ratio_max(shifts, interval: float) -> float:
    """``max_{lambda in [1, interval]} |r_Sigma(lambda)| = rho_Sigma``."""
    grid = np.geomspace(1.0, float(interval), 4001)
    return float(max(abs(plan_ratio(shifts, float(point))) for point in grid))


def skeleton_transfer(shifts, time: float, interval: float, ratio_max: float = None) -> tuple[float, float]:
    """``(K_vec(t), gamma_Sigma(t))`` for one time argument.

    ``interval`` is the plan's ``lambda_max``, so the band is ``[1, interval]``:
    this is the box spectral interval the plan was built on, and both the maximum
    in ``c_j`` and the maximum of ``|r_Sigma|`` are taken there.  For ``t >= 0``
    the identity ``sup_{t >= 0} gamma_Sigma(t) = rho_Sigma`` holds in closed form
    because ``|r_Sigma(-t)| = prod_j (t + sigma_j)/|t - sigma_j| >= 1``.
    """
    values = cardinal_values(shifts, float(time))
    weights = [max((1.0 + time) / (1.0 + sigma), (interval + time) / (interval + sigma))
               for sigma in shifts]
    k_vec = float(np.sum(np.abs(values) * np.array(weights)))
    if ratio_max is None:
        ratio_max = plan_ratio_max(shifts, interval)
    denominator = abs(plan_ratio(shifts, -float(time))) if time > 0.0 else 1.0
    return k_vec, ratio_max / max(denominator, 1e-300)


def hp_vector_constant(shifts, interval: float, time: float) -> float:
    """``K_vec(t)`` at 80 digits, for the interval-wise extremum search."""
    with mpmath.workdps(80):
        hp = mpmath.mpf
        low, high, argument = hp(1), hp(repr(float(interval))), hp(repr(float(time)))
        values = high_precision_cardinal(shifts, float(time))
        total = hp(0)
        for value, sigma_j in zip(values, shifts):
            shift_value = hp(repr(float(sigma_j)))
            branch = (low + argument) / (low + shift_value)
            other = (high + argument) / (high + shift_value)
            total += abs(value) * (branch if branch > other else other)
        return float(total)


def hp_vector_tail(shifts, interval: float) -> float:
    """``K_vec(infinity) = sum_j 2 sigma_j c_j / (interval + sigma_j)``."""
    with mpmath.workdps(80):
        hp = mpmath.mpf
        high = hp(repr(float(interval)))
        total = hp(0)
        for j, sigma_j in enumerate(shifts):
            shift_value = hp(repr(float(sigma_j)))
            constant = hp(2) * shift_value
            for k, sigma in enumerate(shifts):
                if k == j:
                    continue
                other = hp(repr(float(sigma)))
                constant *= (shift_value + other) / (shift_value - other)
            total += abs(constant) / (high + shift_value)
        return float(total)


def vector_maximum(shifts, interval: float) -> dict:
    """``sup_{t >= 0} K_vec(t)`` on each smooth branch, numerically resolved.

    ``|ell_j|`` and the branch of ``c_j`` both break at the shifts, so the
    half-line splits at ``0, sigma_m, ..., sigma_1`` plus the tail; on every open
    interval ``K_vec`` is a smooth rational function, and the maximum is located
    on a dense grid, refined locally, and cross-checked at 80 digits - which
    verifies the value found, not the absence of another stationary point.
    """
    ordered = sorted(float(value) for value in shifts)
    edges: list[float | None] = [0.0] + ordered + [None]
    ratio_max = plan_ratio_max(shifts, interval)
    best = {"K_vec": 0.0, "t": 0.0, "interval": None, "hp": 0.0}
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        upper = float(ordered[0]) * 1e12 if right is None else float(right)
        grid = np.geomspace(max(float(left), 1e-12) * (1 + 1e-12) + 1e-300, upper, 400)
        values = [skeleton_transfer(shifts, float(point), interval, ratio_max)[0] for point in grid]
        position = int(np.argmax(values))
        window = (float(grid[max(position - 1, 0)]), float(grid[min(position + 1, len(grid) - 1)]))
        refined = minimize_scalar(
            lambda value: -skeleton_transfer(shifts, float(value), interval, ratio_max)[0],
            bounds=window, method="bounded", options={"xatol": 1e-15},
        )
        candidate = float(refined.x) if refined.fun < -values[position] else float(grid[position])
        hp_value = hp_vector_constant(shifts, interval, candidate)
        if right is None:
            tail = hp_vector_tail(shifts, interval)
            if tail > hp_value:
                candidate, hp_value = float("inf"), tail
        if hp_value > best["hp"]:
            best = {"K_vec": hp_value, "t": candidate, "interval": index, "hp": hp_value}
    return best


def vector_constant(shifts, interval: float) -> dict:
    """The rigorous constant ``sup_{t >= 0} K_vec(t)`` plus the grid diagnostics."""
    worst = vector_maximum(shifts, interval)
    lower, upper = float(shifts[-1]), float(shifts[0])
    grid = np.concatenate([[0.0], np.geomspace(lower * 1e-4, upper * 1e4, 400)])
    sampled = {"K_vec": 0.0, "gamma": 0.0, "t": 0.0}
    at_zero = None
    ratio_max = plan_ratio_max(shifts, interval)
    for time in grid:
        k_vec, gamma = skeleton_transfer(shifts, float(time), interval, ratio_max)
        if time == 0.0:
            at_zero = {"K_vec": k_vec, "gamma": gamma}
        if k_vec > sampled["K_vec"]:
            sampled = {"K_vec": k_vec, "gamma": gamma, "t": float(time)}
        sampled["gamma"] = max(sampled["gamma"], gamma)
    sampled["at_zero"] = at_zero
    sampled["K_vec_sup"] = worst["K_vec"]
    sampled["K_vec_sup_at"] = worst["t"]
    sampled["K_vec_sup_interval"] = worst["interval"]
    sampled["K_vec_sup_over_grid"] = worst["K_vec"] / sampled["K_vec"]
    return sampled


def fraction_reconstruction(shifts, weights) -> float:
    """I4: the residues must rebuild the cardinal functions.

    ``ell_j(s) = sum_k a_jk / (s + sigma_k)`` is checked at the interval
    endpoints and the shift geometric mean with the mixed relative tolerance;
    the residues have both signs and large magnitudes, so this is the only cheap
    way to know whether the double-precision partial fractions are usable.
    """
    lower, upper = float(shifts[-1]), float(shifts[0])
    tests = [lower, math.sqrt(lower * upper), upper, 0.5 * (lower + upper)]
    error = 0.0
    for point in tests:
        values = cardinal_values(shifts, point)
        rebuilt = np.array([float(np.sum(weights[j] / (point + shifts))) for j in range(len(shifts))])
        for j in range(len(shifts)):
            scale = 1.0 + abs(values[j]) + abs(rebuilt[j])
            error = max(error, abs(values[j] - rebuilt[j]) / scale)
    return error


def skeleton_rows(arguments) -> list[dict]:
    """Layer 2: skeleton invariants, rho_Sigma, K_sample and K_vec."""
    rows = []
    for kappa in arguments.kappas:
        shifts = plan_shifts(kappa)
        invariants = plan_invariants(shifts, arguments)
        geometry = plan_geometry(shifts, kappa)
        weights = partial_fractions(shifts)
        invariants["fraction_reconstruction_error"] = fraction_reconstruction(shifts, weights)
        poles = 1.0 / (shifts[:, None] + shifts[None, :])
        gram = weights @ poles @ weights.T
        invariants["sample_gram_min_eigenvalue"] = float(np.min(np.linalg.eigvalsh(0.5 * (gram + gram.T))))
        print(f"kappa={kappa:.6e} shifts={len(shifts)} "
              f"invariants: delta_jk={invariants['delta_jk_error']:.3e} "
              f"skeleton={invariants['skeleton_identity_error']:.3e} "
              f"axis={invariants['imaginary_axis_error']:.3e} "
              f"mpmath={invariants['mpmath_reference_error']} "
              f"fractions={invariants['fraction_reconstruction_error']:.3e} "
              f"gram_min_eig={invariants['sample_gram_min_eigenvalue']:.3e} "
              f"rho={geometry['rho']:.6e} rho2={geometry['rho2']:.6e} q_m={geometry['q_m']:.6e} "
              f"residue_range=[{weights.min():.3e},{weights.max():.3e}]", flush=True)
        for family in arguments.families:
            eigenvalues = spectrum(kappa, family, arguments.size)
            for source_family in arguments.sources:
                f_vector = source(eigenvalues, source_family)
                b_matrix = np.diag(eigenvalues)
                transfer = np.array([
                    float(f_vector @ np.linalg.solve(b_matrix + sigma * np.eye(b_matrix.shape[0]), f_vector))
                    for sigma in shifts
                ])
                norm = h2_norm(-b_matrix, f_vector, f_vector)
                constant = sample_constant(shifts, weights, transfer, norm)
                vector = vector_constant(shifts, kappa)
                rows.append({
                    "mode": "skeleton",
                    "kappa": float(kappa),
                    "spectrum_family": family,
                    "source_family": source_family,
                    "size": arguments.size,
                    "shifts": [float(value) for value in shifts],
                    "K_sample": constant,
                    "K_vec": vector["K_vec"],
                    "K_vec_sup": vector["K_vec_sup"],
                    "K_vec_sup_at": vector["K_vec_sup_at"],
                    "K_vec_sup_over_grid": vector["K_vec_sup_over_grid"],
                    "K_vec_at_zero": vector["at_zero"]["K_vec"],
                    "K_vec_worst_t": vector["t"],
                    "gamma": vector["gamma"],
                    "gamma_at_zero": vector["at_zero"]["gamma"],
                    "rho": geometry["rho"],
                    "rho2": geometry["rho2"],
                    "q_m": geometry["q_m"],
                    **invariants,
                })
                print(f"kappa={kappa:.3e} family={family} source={source_family} "
                      f"K_sample={constant:.6e} K_vec={vector['K_vec']:.6e} "
                      f"K_vec_sup={vector['K_vec_sup']:.6e} at t={vector['K_vec_sup_at']:.3e} "
                      f"K_vec(0)={vector['at_zero']['K_vec']:.6e} gamma={vector['gamma']:.6e} "
                      f"gamma(0)={vector['at_zero']['gamma']:.6e}", flush=True)
    return rows


def linear_rows(arguments) -> list[dict]:
    """Layer 3: measure the linear candidate on the full plan.

    For the whole 13-shift plan, a controlled rotation V_theta of the exact moment
    space, and time arguments t away from the matching points, the derived bound is

        delta(t; V) <= gamma_Sigma(t) + K_vec(t) delta_*,

    so the two pieces are measured separately as well:

        skeleton piece   ||e_skel(t)||_(A_t) / ||x(t)||_(A_t)   against gamma_Sigma(t),
        sample piece     sum_j |ell_j(t)| ||x_j - x_V,j||_(A_t) / ||x(t)||_(A_t)
                         against K_vec(t) delta_*,

    with e_skel(t) = x(t) - sum_j ell_j(t) x(sigma_j).  The dimension has to exceed
    the shift count here, otherwise the 13 snapshots span everything and the
    experiment is vacuous.
    """
    rows = []
    for kappa in arguments.kappas:
        shifts = plan_shifts(kappa)
        if arguments.size <= len(shifts):
            raise SystemExit(f"--mode linear needs --size above the shift count {len(shifts)}")
        for family in arguments.families:
            eigenvalues = spectrum(kappa, family, arguments.size)
            b_matrix = np.diag(eigenvalues)
            for source_family in arguments.sources:
                f_vector = source(eigenvalues, source_family)
                exact, singular_values, rank = moment_space(b_matrix, f_vector, shifts)
                other = complement_space(exact, arguments.seed)
                if rank < len(shifts):
                    print(f"kappa={kappa:.3e} family={family} source={source_family} "
                          f"snapshot rank {rank} < {len(shifts)} shifts: skipping", flush=True)
                    continue
                ratio_max = plan_ratio_max(shifts, kappa)
                times = [0.0] + list(ordered := sorted(float(value) for value in shifts))
                times += [math.sqrt(ordered[j] * ordered[j + 1]) for j in range(len(ordered) - 1)]
                times += list(np.geomspace(1e-6, 1e6 * kappa, 40)) + [1e9 * kappa]
                for theta in arguments.thetas:
                    space = exact * math.cos(theta) + other * math.sin(theta)
                    delta_star = float(matching_defects(b_matrix, f_vector, shifts, space).max())
                    for time in times:
                        operator = b_matrix + float(time) * np.eye(b_matrix.shape[0])
                        target = np.linalg.solve(operator, f_vector)
                        reduced = space @ np.linalg.solve(space.T @ operator @ space, space.T @ f_vector)
                        scale = math.sqrt(float(target @ operator @ target))
                        measured = math.sqrt(max(float((target - reduced) @ operator @ (target - reduced)), 0.0)) / scale
                        skeleton = target - np.sum(
                            [cardinal_values(shifts, float(time))[j] *
                             np.linalg.solve(b_matrix + shifts[j] * np.eye(b_matrix.shape[0]), f_vector)
                             for j in range(len(shifts))], axis=0)
                        skeleton_piece = math.sqrt(max(float(skeleton @ operator @ skeleton), 0.0)) / scale
                        sample_piece = 0.0
                        values = cardinal_values(shifts, float(time))
                        for j, sigma in enumerate(shifts):
                            local = sigma * np.eye(b_matrix.shape[0]) + b_matrix
                            snapshot = np.linalg.solve(local, f_vector)
                            projected = space @ np.linalg.solve(
                                space.T @ local @ space, space.T @ f_vector)
                            error = snapshot - projected
                            sample_piece += abs(values[j]) * math.sqrt(
                                max(float(error @ operator @ error), 0.0)) / scale
                        k_vec, gamma = skeleton_transfer(shifts, float(time), kappa, ratio_max)
                        rows.append({
                            "mode": "linear",
                            "kappa": float(kappa),
                            "spectrum_family": family,
                            "source_family": source_family,
                            "size": arguments.size,
                            "shifts": [float(value) for value in shifts],
                            "snapshot_rank": rank,
                            "theta": float(theta),
                            "delta_star": delta_star,
                            "time": float(time),
                            "delta_t_measured": float(measured),
                            "bound_total": float(gamma + k_vec * delta_star),
                            "bound_over_measured": float((gamma + k_vec * delta_star) / measured)
                            if measured > 0.0 else None,
                            "skeleton_piece": float(skeleton_piece),
                            "skeleton_bound": float(gamma),
                            "sample_piece": float(sample_piece),
                            "sample_bound": float(k_vec * delta_star),
                            "K_vec": float(k_vec),
                            "gamma": float(gamma),
                            "K_sample": None,
                            "rho": None,
                        })
                    block = [row for row in rows if row["theta"] == theta
                             and row["kappa"] == kappa and row["spectrum_family"] == family
                             and row["source_family"] == source_family]
                    ratios = sorted(row["bound_over_measured"] for row in block)
                    worst = max(block, key=lambda row: row["bound_over_measured"] or -1.0)
                    violations = sum(1 for row in block
                                     if row["delta_t_measured"] > row["bound_total"] * (1 + 1e-12))
                    print(f"kappa={kappa:.3e} family={family} source={source_family} "
                          f"theta={theta:.0e} delta_*={delta_star:.3e} times={len(block)} "
                          f"max bound/measured={worst['bound_over_measured']:.3f} at t={worst['time']:.3e} "
                          f"median={ratios[len(ratios) // 2]:.3f} "
                          f"skeleton {worst['skeleton_piece']:.3e}<={worst['skeleton_bound']:.3e} "
                          f"sample {worst['sample_piece']:.3e}<={worst['sample_bound']:.3e} "
                          f"violations={violations}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("rotation", "skeleton", "linear"), default="rotation")
    parser.add_argument("--size", type=int, choices=(4, 8, 26), default=4)
    parser.add_argument("--kappas", default=None,
                        help="comma separated; default 1e2,1e4,1e6 and the 5 mm box plan's kappa")
    parser.add_argument("--families", default="log-uniform,endpoint-cluster,low-cluster,repeated")
    parser.add_argument("--sources", default="flat,h2-balanced")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--thetas", default="1e-2,1e-4,1e-6",
                        help="rotation magnitudes for --mode linear")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.kappas is None:
        arguments.kappas = [1e2, 1e4, 1e6, 4.1400e1 / 4.2795e-5]
    else:
        arguments.kappas = [float(value) for value in arguments.kappas.split(",")]
    arguments.families = arguments.families.split(",")
    arguments.sources = arguments.sources.split(",")
    arguments.thetas = [float(value) for value in arguments.thetas.split(",")]
    if arguments.mode == "rotation":
        rows = rotation_rows(arguments)
    elif arguments.mode == "skeleton":
        rows = skeleton_rows(arguments)
    else:
        rows = linear_rows(arguments)
    report = {"mode": arguments.mode, "size": arguments.size, "rows": rows}
    if arguments.output:
        arguments.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
