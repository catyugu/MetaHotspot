"""Frequency-major adaptation of utils.py at deb981297d1353d9005e0a042657e1d8eef0e595."""
import time
import numpy as np
from metahotspot.macromodel import utils as u

def build_frequency_major_basis(
    operators,
    source_shape,
    boundary_terms,
    h_ranges,
    *,
    tolerance=u.ROM_TOLERANCE,
    max_order=u.MAX_ORDER,
    probe_rounds=u.PROBE_ROUNDS,
    seed=u.RANDOM_SEED,
):
    """Released response-space algorithm with frequency-major scheduling.

    Every source retains its own basis and Galerkin projection. Only the loop
    order changes: finish all source ports before advancing the frequency.
    This permits a single overwriteable AMG hierarchy in the solve adapter.
    """
    started = time.perf_counter()
    K = operators.K.tocsc()
    C = operators.C.tocsc()
    G = np.asarray(source_shape, dtype=np.float64)
    h_ranges = np.asarray(h_ranges, dtype=np.float64)
    rng = np.random.default_rng(seed)

    K_min, K_max = K.copy(), K.copy()
    for H, (h_min, h_max) in zip(boundary_terms, h_ranges, strict=True):
        K_min = K_min + h_min * H
        K_max = K_max + h_max * H

    snapshots = []
    plans = []
    history = []
    validation_count = 0
    max_accepted_residual = 0.0

    def full_operator(h_vec, shift):
        A = K + shift * C
        for h, H in zip(h_vec, boundary_terms, strict=True):
            A = A + h * H
        return A.tocsc()

    lambda_min, lambda_max = u.port_eigenvalue_bounds(
        K_min, K_max, C, np.sqrt(C.diagonal())
    )
    kappa = lambda_max / lambda_min
    shift_count = u.mpmm_elliptic_shift_count(tolerance, lambda_min, lambda_max)
    shifts = u.mpmm_elliptic_shifts(shift_count, lambda_max, kappa)
    plans = [dict(port=port, lambda_min=float(lambda_min),
                  lambda_max=float(lambda_max), kappa=float(kappa),
                  shift_count=int(shift_count), shifts_per_s=shifts.tolist())
             for port in range(G.shape[1])]
    local_bases = [np.empty((K.shape[0], 0), dtype=np.float64)
                   for _ in range(G.shape[1])]
    projections = [u._AffineGalerkinProjection((K, C, *boundary_terms), G[:, port])
                   for port in range(G.shape[1])]

    for shift in shifts:
        for port in range(G.shape[1]):
            g = G[:, port].copy()
            g_norm = np.linalg.norm(g)
            local_basis = local_bases[port]
            projection = projections[port]
            full_solves = 0
            trials = 0
            accepted_residual = 0.0
            accepted_rounds = 0

            while accepted_rounds < probe_rounds:
                h_vec = u._draw_h(h_ranges, rng)
                initial_guess = np.zeros(K.shape[0], dtype=np.float64)

                if local_basis.shape[1]:
                    coefficients = projection.solve(h_vec, shift)
                    initial_guess = local_basis @ coefficients
                    residual_vector = (
                        K @ initial_guess + shift * (C @ initial_guess) - g
                    )
                    for h, H in zip(h_vec, boundary_terms, strict=True):
                        residual_vector += h * (H @ initial_guess)
                    residual = np.linalg.norm(residual_vector) / g_norm
                    validation_count += 1
                    trials += 1
                    if residual <= tolerance:
                        accepted_residual = max(accepted_residual, float(residual))
                        accepted_rounds += 1
                        continue

                if len(snapshots) >= max_order:
                    raise RuntimeError("FANTASTIC extraction reached max_order")

                A = full_operator(h_vec, shift)
                response = np.asarray(
                    u.spd_solve(A, g, x0=initial_guess, rtol=tolerance / 10.0),
                    dtype=np.float64,
                ).ravel()
                snapshots.append(response)
                full_solves += 1

                block = u.orthonormalize_block(local_basis, response[:, None])
                if block.shape[1]:
                    projection.append(local_basis, block)
                    local_basis = np.column_stack((local_basis, block))
                accepted_residual = 0.0
                accepted_rounds = 0

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

            local_bases[port] = local_basis

    snapshot_matrix = np.column_stack(snapshots)
    basis, singular_values = u._snapshot_svd_basis(snapshot_matrix, tolerance)
    svd_order = basis.shape[1]

    # The h-free BCI operator has the uniform-temperature null mode;
    # preserve it exactly.
    constant = np.ones((K.shape[0], 1), dtype=np.float64)
    constant /= np.linalg.norm(constant)
    constant = u.orthonormalize_block(basis, constant)
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
        "processed_candidate_count": len(snapshots),
        "validation_count": int(validation_count),
        "pre_svd_order": len(snapshots),
        "svd_kept_order": int(svd_order),
        "basis_order": int(basis.shape[1]),
        "maximum_order": int(max_order),
        "orthogonality_error": orthogonality_error,
        "max_accepted_residual": float(max_accepted_residual),
        "history": history,
        "seconds": time.perf_counter() - started,
        "singular_value_ratios": (singular_values / singular_values[0]).tolist(),
    }
