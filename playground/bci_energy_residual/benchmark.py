#!/usr/bin/env python3
"""Compare stock Extended FANTASTIC with a certified energy-residual test.

Production code is not modified. The baseline always calls the repository
build_parametric_basis function. The candidate changes only Algorithm 1's
probe acceptance criterion from a Euclidean residual to

    rho_A = r.T A^{-1} r / (b.T A^{-1} b) <= epsilon,

and decides that criterion with rigorous bounds rather than a full A^{-1}r
solve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from energy_residual import (  # noqa: E402
    certify_energy_ratio,
    residual_and_base_energy,
)

from metahotspot.macromodel import utils as u  # noqa: E402


DEFAULT_TOLERANCE = 1.0e-3
DEFAULT_PROBE_ROUNDS = 10


def load_problem(data_dir: Path):
    K = sp.load_npz(data_dir / "K.npz").tocsc()
    C = sp.load_npz(data_dir / "C.npz").tocsc()
    H = []
    j = 0
    while (data_dir / f"H{j}.npz").exists():
        H.append(sp.load_npz(data_dir / f"H{j}.npz").tocsc())
        j += 1
    data = np.load(data_dir / "data.npz")
    G = np.asarray(data["G"], dtype=np.float64)
    ranges = np.asarray(data["ranges"], dtype=np.float64)
    power = np.asarray(data["power"], dtype=np.float64)
    operators = u.normalized_operators(K, C, np.zeros(K.shape[0]))
    return operators, G, H, ranges, power


def full_operator(K, C, H, h_vec, shift):
    A = K + float(shift) * C
    for h, term in zip(h_vec, H):
        A = A + float(h) * term
    return A.tocsc()


def direct_solve(A, b, x0=None):
    del x0
    return np.asarray(spla.spsolve(A.tocsc(), b), dtype=np.float64).ravel()


def production_solve(A, b, x0=None):
    return u.spd_solve(A, b, x0=x0, rtol=u.ENRICH_RTOL)


def finalize_basis(K, snapshots, tolerance):
    snapshot_matrix = np.column_stack(snapshots)
    basis, singular_values = u._snapshot_svd_basis(snapshot_matrix, tolerance)
    svd_order = int(basis.shape[1])
    constant = np.ones((K.shape[0], 1))
    constant /= np.linalg.norm(constant)
    constant = u.orthonormalize_block(basis, constant)
    if constant.shape[1]:
        basis = np.column_stack((basis, constant))
    return np.ascontiguousarray(basis), singular_values, svd_order


def exact_inverse_quadratic(A, residual):
    correction = direct_solve(A, residual)
    q = max(float(np.dot(residual, correction)), 0.0)
    return q, correction


def extract_variant(
    operators,
    G,
    H,
    h_ranges,
    *,
    mode,
    tolerance,
    probe_rounds,
    seed,
    solver,
    max_order=2048,
):
    """Reproduce the stock loop while changing only the probe predicate."""
    started = time.perf_counter()
    K = operators.K.tocsc()
    C = operators.C.tocsc()
    c_diag = np.asarray(C.diagonal(), dtype=np.float64)
    if np.any(c_diag <= 0.0):
        raise ValueError("experiment requires positive diagonal C")
    rng = np.random.default_rng(seed)

    snapshots = []
    plans = []
    history = []
    validation_count = 0
    full_solve_count = 0
    exact_inverse_quadratic_solves = 0
    certificate_iterations = 0
    zero_step_accepts = 0
    certificate_accepts = 0
    certificate_rejects = 0
    certificate_seconds = 0.0
    full_solve_seconds = 0.0

    for port in range(G.shape[1]):
        g = G[:, port].copy()
        g_norm = float(np.linalg.norm(g))
        local_basis = np.empty((K.shape[0], 0), dtype=np.float64)
        projected = None
        lambda_min, lambda_max = u.port_eigenvalue_bounds(K, C, g)
        kappa = lambda_max / lambda_min
        shift_count = u.mpmm_elliptic_shift_count(
            tolerance, lambda_min, lambda_max
        )
        shifts = u.mpmm_elliptic_shifts(shift_count, lambda_max, kappa)
        plans.append(
            {
                "port": int(port),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "shift_count": int(shift_count),
                "shifts_per_s": shifts.tolist(),
            }
        )

        for shift in shifts:
            accepted_rounds = 0
            per_shift_full = 0
            per_shift_validations = 0
            per_shift_cert_iters = 0

            while accepted_rounds < probe_rounds:
                h_vec = u._draw_h(h_ranges, rng)
                initial_guess = np.zeros(K.shape[0])
                A = None
                oracle_response = None

                if projected is not None:
                    estimate = u._local_response(
                        projected, local_basis, h_vec, shift
                    )
                    A = full_operator(K, C, H, h_vec, shift)
                    residual = np.asarray(g - A @ estimate).ravel()
                    validation_count += 1
                    per_shift_validations += 1

                    if mode == "euclidean":
                        accept = (
                            float(np.linalg.norm(residual) / g_norm)
                            <= tolerance
                        )
                        initial_guess = estimate
                    elif mode == "energy_oracle":
                        residual2, base = residual_and_base_energy(
                            A, g, estimate
                        )
                        np.testing.assert_allclose(
                            residual2, residual, rtol=0.0, atol=1.0e-12
                        )
                        q, correction = exact_inverse_quadratic(A, residual)
                        exact_inverse_quadratic_solves += 1
                        ratio = q / (base + q) if q else 0.0
                        accept = ratio <= tolerance
                        oracle_response = estimate + correction
                        initial_guess = oracle_response
                    elif mode == "energy_certified":
                        t_cert = time.perf_counter()
                        cert = certify_energy_ratio(
                            A,
                            c_diag,
                            g,
                            estimate,
                            float(shift),
                            tolerance,
                            max_iterations=K.shape[0],
                        )
                        certificate_seconds += (
                            time.perf_counter() - t_cert
                        )
                        certificate_iterations += cert.iterations
                        per_shift_cert_iters += cert.iterations
                        if cert.decision == "undecided":
                            raise RuntimeError(
                                "certificate did not decide within N steps"
                            )
                        accept = cert.decision == "accept"
                        if accept:
                            certificate_accepts += 1
                            zero_step_accepts += int(cert.iterations == 0)
                        else:
                            certificate_rejects += 1
                        initial_guess = estimate + cert.correction
                    else:
                        raise ValueError(mode)

                    if accept:
                        accepted_rounds += 1
                        continue

                if len(snapshots) >= max_order:
                    raise RuntimeError("FANTASTIC extraction reached max_order")
                if A is None:
                    A = full_operator(K, C, H, h_vec, shift)

                if oracle_response is not None:
                    response = oracle_response
                else:
                    t_solve = time.perf_counter()
                    response = np.asarray(
                        solver(A, g, x0=initial_guess), dtype=np.float64
                    ).ravel()
                    full_solve_seconds += time.perf_counter() - t_solve

                snapshots.append(response)
                full_solve_count += 1
                per_shift_full += 1

                block = u.orthonormalize_block(
                    local_basis, response[:, None]
                )
                if block.shape[1]:
                    local_basis = np.column_stack((local_basis, block))
                projected = u._local_projection(
                    K, C, H, g, local_basis
                )
                accepted_rounds = 0

            history.append(
                {
                    "port": int(port),
                    "shift": float(shift),
                    "full_solves": int(per_shift_full),
                    "validations": int(per_shift_validations),
                    "certificate_iterations": int(
                        per_shift_cert_iters
                    ),
                }
            )

    basis, singular_values, svd_order = finalize_basis(
        K, snapshots, tolerance
    )
    return basis, {
        "mode": mode,
        "tolerance": float(tolerance),
        "probe_rounds": int(probe_rounds),
        "seed": int(seed),
        "full_solve_count": int(full_solve_count),
        "validation_count": int(validation_count),
        "exact_inverse_quadratic_solves": int(
            exact_inverse_quadratic_solves
        ),
        "certificate_iterations": int(certificate_iterations),
        "certificate_iterations_per_validation": (
            float(certificate_iterations / validation_count)
            if validation_count
            else 0.0
        ),
        "zero_step_accepts": int(zero_step_accepts),
        "certificate_accepts": int(certificate_accepts),
        "certificate_rejects": int(certificate_rejects),
        "pre_svd_order": int(len(snapshots)),
        "svd_kept_order": int(svd_order),
        "basis_order": int(basis.shape[1]),
        "seconds": float(time.perf_counter() - started),
        "certificate_seconds": float(certificate_seconds),
        "full_solve_seconds": float(full_solve_seconds),
        "per_port_plans": plans,
        "history": history,
        "singular_value_ratios": (
            singular_values / singular_values[0]
        ).tolist(),
    }


def projected_response(K, C, H, g, basis, h_vec, shift):
    Ar = basis.T @ (K @ basis)
    Ar = Ar + float(shift) * (basis.T @ (C @ basis))
    for h, term in zip(h_vec, H):
        Ar = Ar + float(h) * (basis.T @ (term @ basis))
    br = basis.T @ g
    y = scipy.linalg.solve(
        Ar, br, assume_a="pos", check_finite=False
    )
    return basis @ y


def energy_holdout(
    K,
    C,
    H,
    G,
    h_ranges,
    basis,
    plans,
    *,
    seed,
    samples_per_shift,
    tolerance,
    use_direct,
):
    rng = np.random.default_rng(seed)
    ratios = []
    euclidean = []
    solve = direct_solve if use_direct else production_solve

    for plan in plans:
        port = int(plan["port"])
        g = G[:, port]
        gnorm = float(np.linalg.norm(g))
        for shift in plan["shifts_per_s"]:
            for _ in range(samples_per_shift):
                h_vec = u._draw_h(h_ranges, rng)
                A = full_operator(K, C, H, h_vec, shift)
                estimate = projected_response(
                    K, C, H, g, basis, h_vec, shift
                )
                residual, base = residual_and_base_energy(
                    A, g, estimate
                )
                correction = solve(A, residual, x0=None)
                q = max(float(np.dot(residual, correction)), 0.0)
                ratios.append(q / (base + q) if q else 0.0)
                euclidean.append(
                    float(np.linalg.norm(residual) / gnorm)
                )

    ratios = np.asarray(ratios)
    euclidean = np.asarray(euclidean)
    return {
        "count": int(ratios.size),
        "max_energy_ratio": float(np.max(ratios)),
        "p95_energy_ratio": float(np.quantile(ratios, 0.95)),
        "energy_violations": int(np.sum(ratios > tolerance)),
        "max_euclidean_residual": float(np.max(euclidean)),
    }


def steady_holdout(
    K,
    H,
    G,
    power,
    h_ranges,
    basis,
    *,
    seed,
    count,
    use_direct,
):
    rng = np.random.default_rng(seed)
    rhs = G @ power
    solve = direct_solve if use_direct else production_solve
    Kb = basis.T @ (K @ basis)
    Hb = [basis.T @ (term @ basis) for term in H]
    br = basis.T @ rhs
    errors = []

    for _ in range(count):
        h_vec = u._draw_h(h_ranges, rng)
        A = K.copy()
        Ar = Kb.copy()
        for h, term, term_r in zip(h_vec, H, Hb):
            A = A + float(h) * term
            Ar = Ar + float(h) * term_r
        ref = solve(A.tocsc(), rhs, x0=None)
        y = scipy.linalg.solve(
            Ar, br, assume_a="pos", check_finite=False
        )
        approx = basis @ y
        scale = max(
            float(np.max(np.abs(ref))),
            np.finfo(float).tiny,
        )
        errors.append(
            float(np.max(np.abs(approx - ref)) / scale)
        )

    return {
        "count": int(count),
        "max_relative_fullfield_error": float(np.max(errors)),
        "mean_relative_fullfield_error": float(np.mean(errors)),
    }


def stock_baseline(
    operators,
    G,
    H,
    h_ranges,
    tolerance,
    probe_rounds,
    seed,
):
    t0 = time.perf_counter()
    basis, summary = u.build_parametric_basis(
        operators,
        G,
        H,
        h_ranges,
        tolerance=tolerance,
        max_order=2048,
        probe_rounds=probe_rounds,
        seed=seed,
    )
    result = dict(summary)
    result["wall_seconds"] = float(time.perf_counter() - t0)
    result["full_solve_count"] = int(
        summary["processed_candidate_count"]
    )
    return basis, result


def principal_subspace_cosine(A, B):
    s = scipy.linalg.svdvals(A.T @ B)
    return float(np.min(s)), float(np.max(s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--mesh", type=float, required=True)
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE
    )
    parser.add_argument(
        "--probe-rounds", type=int, default=DEFAULT_PROBE_ROUNDS
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    operators, G, H, h_ranges, power = load_problem(args.data)
    K = operators.K.tocsc()
    C = operators.C.tocsc()

    report = {
        "mesh_mm": args.mesh,
        "n": int(K.shape[0]),
        "seed": args.seed,
        "tolerance": args.tolerance,
        "probe_rounds": args.probe_rounds,
        "criterion": (
            "r^T A^{-1} r / (b^T A^{-1} b) <= tolerance"
        ),
    }

    stock_basis, stock = stock_baseline(
        operators,
        G,
        H,
        h_ranges,
        args.tolerance,
        args.probe_rounds,
        args.seed,
    )
    report["stock_euclidean"] = stock

    cert_basis, cert = extract_variant(
        operators,
        G,
        H,
        h_ranges,
        mode="energy_certified",
        tolerance=args.tolerance,
        probe_rounds=args.probe_rounds,
        seed=args.seed,
        solver=production_solve,
    )
    report["energy_certified"] = cert

    small = args.mesh >= 4.999
    samples_per_shift = 3 if small else 1
    steady_count = 12 if small else 6

    report["stock_energy_holdout"] = energy_holdout(
        K,
        C,
        H,
        G,
        h_ranges,
        stock_basis,
        stock["per_port_plans"],
        seed=args.seed + 991,
        samples_per_shift=samples_per_shift,
        tolerance=args.tolerance,
        use_direct=small,
    )
    report["certified_energy_holdout"] = energy_holdout(
        K,
        C,
        H,
        G,
        h_ranges,
        cert_basis,
        cert["per_port_plans"],
        seed=args.seed + 991,
        samples_per_shift=samples_per_shift,
        tolerance=args.tolerance,
        use_direct=small,
    )
    report["stock_steady_holdout"] = steady_holdout(
        K,
        H,
        G,
        power,
        h_ranges,
        stock_basis,
        seed=args.seed + 1771,
        count=steady_count,
        use_direct=small,
    )
    report["certified_steady_holdout"] = steady_holdout(
        K,
        H,
        G,
        power,
        h_ranges,
        cert_basis,
        seed=args.seed + 1771,
        count=steady_count,
        use_direct=small,
    )

    if small:
        oracle_basis, oracle = extract_variant(
            operators,
            G,
            H,
            h_ranges,
            mode="energy_oracle",
            tolerance=args.tolerance,
            probe_rounds=args.probe_rounds,
            seed=args.seed,
            solver=direct_solve,
        )
        cert_direct_basis, cert_direct = extract_variant(
            operators,
            G,
            H,
            h_ranges,
            mode="energy_certified",
            tolerance=args.tolerance,
            probe_rounds=args.probe_rounds,
            seed=args.seed,
            solver=direct_solve,
        )
        min_cos, max_cos = principal_subspace_cosine(
            oracle_basis, cert_direct_basis
        )
        report["energy_oracle_direct"] = oracle
        report["energy_certified_direct"] = cert_direct
        report["oracle_certificate_agreement"] = {
            "same_full_solve_count": (
                oracle["full_solve_count"]
                == cert_direct["full_solve_count"]
            ),
            "same_pre_svd_order": (
                oracle["pre_svd_order"]
                == cert_direct["pre_svd_order"]
            ),
            "same_basis_order": (
                oracle["basis_order"]
                == cert_direct["basis_order"]
            ),
            "minimum_principal_cosine": min_cos,
            "maximum_principal_cosine": max_cos,
        }

    report["relative_to_stock"] = {
        "full_solve_ratio": (
            cert["full_solve_count"]
            / stock["full_solve_count"]
        ),
        "basis_order_ratio": (
            cert["basis_order"]
            / stock["basis_order"]
        ),
        "wall_time_ratio": (
            cert["seconds"] / stock["wall_seconds"]
        ),
    }

    out = args.output / "result.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
