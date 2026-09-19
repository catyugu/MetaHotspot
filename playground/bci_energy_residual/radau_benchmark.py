#!/usr/bin/env python3
"""Matched stock-FANTASTIC versus Gauss-Radau energy certificate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from benchmark import (  # noqa: E402
    direct_solve,
    energy_holdout,
    exact_inverse_quadratic,
    finalize_basis,
    full_operator,
    load_problem,
    principal_subspace_cosine,
    production_solve,
    steady_holdout,
    stock_baseline,
)
from energy_radau import certify_energy_ratio_radau  # noqa: E402
from energy_residual import residual_and_base_energy  # noqa: E402
from metahotspot.macromodel import utils as u  # noqa: E402


def extract_radau(
    operators,
    G,
    H,
    h_ranges,
    *,
    tolerance,
    probe_rounds,
    seed,
    solver,
    max_order=2048,
):
    started = time.perf_counter()
    K = operators.K.tocsc()
    C = operators.C.tocsc()
    c_diag = np.asarray(C.diagonal(), dtype=np.float64)
    h_diagonals = [
        np.asarray(term.diagonal(), dtype=np.float64) for term in H
    ]
    rng = np.random.default_rng(seed)

    snapshots = []
    plans = []
    validation_count = 0
    full_solve_count = 0
    certificate_iterations = 0
    zero_step_accepts = 0
    certificate_accepts = 0
    certificate_rejects = 0
    certificate_seconds = 0.0
    full_solve_seconds = 0.0

    for port in range(G.shape[1]):
        g = G[:, port].copy()
        local_basis = np.empty((K.shape[0], 0), dtype=np.float64)
        projected = None

        lambda_min, lambda_max = u.port_eigenvalue_bounds(K, C, g)
        shifts = u.mpmm_elliptic_shifts(
            u.mpmm_elliptic_shift_count(
                tolerance, lambda_min, lambda_max
            ),
            lambda_max,
            lambda_max / lambda_min,
        )
        plans.append(
            {
                "port": int(port),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "shift_count": int(len(shifts)),
                "shifts_per_s": shifts.tolist(),
            }
        )

        for shift in shifts:
            accepted_rounds = 0
            while accepted_rounds < probe_rounds:
                h_vec = u._draw_h(h_ranges, rng)
                initial_guess = np.zeros(K.shape[0])
                A = None

                if projected is not None:
                    estimate = u._local_response(
                        projected, local_basis, h_vec, shift
                    )
                    A = full_operator(K, C, H, h_vec, shift)
                    anchor_diag = float(shift) * c_diag.copy()
                    for h, h_diag in zip(h_vec, h_diagonals):
                        anchor_diag += float(h) * h_diag

                    t_cert = time.perf_counter()
                    cert = certify_energy_ratio_radau(
                        A,
                        anchor_diag,
                        g,
                        estimate,
                        tolerance,
                        max_iterations=K.shape[0],
                    )
                    certificate_seconds += (
                        time.perf_counter() - t_cert
                    )
                    validation_count += 1
                    certificate_iterations += cert.iterations
                    if cert.decision == "undecided":
                        raise RuntimeError(
                            "Gauss-Radau certificate did not decide"
                        )
                    if cert.decision == "accept":
                        certificate_accepts += 1
                        zero_step_accepts += int(cert.iterations == 0)
                        accepted_rounds += 1
                        continue

                    certificate_rejects += 1
                    initial_guess = estimate + cert.correction

                if len(snapshots) >= max_order:
                    raise RuntimeError(
                        "FANTASTIC extraction reached max_order"
                    )
                if A is None:
                    A = full_operator(K, C, H, h_vec, shift)

                t_solve = time.perf_counter()
                response = np.asarray(
                    solver(A, g, x0=initial_guess),
                    dtype=np.float64,
                ).ravel()
                full_solve_seconds += time.perf_counter() - t_solve

                snapshots.append(response)
                full_solve_count += 1

                block = u.orthonormalize_block(
                    local_basis, response[:, None]
                )
                if block.shape[1]:
                    local_basis = np.column_stack(
                        (local_basis, block)
                    )
                projected = u._local_projection(
                    K, C, H, g, local_basis
                )
                accepted_rounds = 0

    basis, singular_values, svd_order = finalize_basis(
        K, snapshots, tolerance
    )
    return basis, {
        "mode": "energy_radau",
        "tolerance": float(tolerance),
        "probe_rounds": int(probe_rounds),
        "seed": int(seed),
        "full_solve_count": int(full_solve_count),
        "validation_count": int(validation_count),
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
        "singular_value_ratios": (
            singular_values / singular_values[0]
        ).tolist(),
    }


def extract_oracle(
    operators,
    G,
    H,
    h_ranges,
    *,
    tolerance,
    probe_rounds,
    seed,
    max_order=2048,
):
    """Exact inverse-quadratic oracle using direct sparse solves."""
    started = time.perf_counter()
    K = operators.K.tocsc()
    C = operators.C.tocsc()
    rng = np.random.default_rng(seed)
    snapshots = []
    plans = []
    validation_count = 0
    full_solve_count = 0

    for port in range(G.shape[1]):
        g = G[:, port].copy()
        local_basis = np.empty((K.shape[0], 0), dtype=np.float64)
        projected = None
        lambda_min, lambda_max = u.port_eigenvalue_bounds(K, C, g)
        shifts = u.mpmm_elliptic_shifts(
            u.mpmm_elliptic_shift_count(
                tolerance, lambda_min, lambda_max
            ),
            lambda_max,
            lambda_max / lambda_min,
        )
        plans.append(
            {
                "port": int(port),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "shift_count": int(len(shifts)),
                "shifts_per_s": shifts.tolist(),
            }
        )

        for shift in shifts:
            accepted_rounds = 0
            while accepted_rounds < probe_rounds:
                h_vec = u._draw_h(h_ranges, rng)
                A = full_operator(K, C, H, h_vec, shift)
                if projected is not None:
                    estimate = u._local_response(
                        projected, local_basis, h_vec, shift
                    )
                    residual, base = residual_and_base_energy(
                        A, g, estimate
                    )
                    q, correction = exact_inverse_quadratic(
                        A, residual
                    )
                    validation_count += 1
                    ratio = q / (base + q) if q else 0.0
                    if ratio <= tolerance:
                        accepted_rounds += 1
                        continue
                    response = estimate + correction
                else:
                    response = direct_solve(A, g)

                if len(snapshots) >= max_order:
                    raise RuntimeError(
                        "FANTASTIC extraction reached max_order"
                    )
                snapshots.append(response)
                full_solve_count += 1
                block = u.orthonormalize_block(
                    local_basis, response[:, None]
                )
                if block.shape[1]:
                    local_basis = np.column_stack(
                        (local_basis, block)
                    )
                projected = u._local_projection(
                    K, C, H, g, local_basis
                )
                accepted_rounds = 0

    basis, singular_values, svd_order = finalize_basis(
        K, snapshots, tolerance
    )
    return basis, {
        "mode": "energy_oracle",
        "full_solve_count": int(full_solve_count),
        "validation_count": int(validation_count),
        "pre_svd_order": int(len(snapshots)),
        "svd_kept_order": int(svd_order),
        "basis_order": int(basis.shape[1]),
        "seconds": float(time.perf_counter() - started),
        "per_port_plans": plans,
        "singular_value_ratios": (
            singular_values / singular_values[0]
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--probe-rounds", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    operators, G, H, h_ranges, power = load_problem(args.data)
    K, C = operators.K.tocsc(), operators.C.tocsc()

    stock_basis, stock = stock_baseline(
        operators,
        G,
        H,
        h_ranges,
        args.tolerance,
        args.probe_rounds,
        args.seed,
    )
    radau_basis, radau = extract_radau(
        operators,
        G,
        H,
        h_ranges,
        tolerance=args.tolerance,
        probe_rounds=args.probe_rounds,
        seed=args.seed,
        solver=production_solve,
    )

    small = args.mesh >= 4.999
    samples_per_shift = 3 if small else 1
    steady_count = 12 if small else 6

    report = {
        "mesh_mm": args.mesh,
        "n": int(K.shape[0]),
        "seed": args.seed,
        "tolerance": args.tolerance,
        "stock_euclidean": stock,
        "energy_radau": radau,
        "stock_energy_holdout": energy_holdout(
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
        ),
        "radau_energy_holdout": energy_holdout(
            K,
            C,
            H,
            G,
            h_ranges,
            radau_basis,
            radau["per_port_plans"],
            seed=args.seed + 991,
            samples_per_shift=samples_per_shift,
            tolerance=args.tolerance,
            use_direct=small,
        ),
        "stock_steady_holdout": steady_holdout(
            K,
            H,
            G,
            power,
            h_ranges,
            stock_basis,
            seed=args.seed + 1771,
            count=steady_count,
            use_direct=small,
        ),
        "radau_steady_holdout": steady_holdout(
            K,
            H,
            G,
            power,
            h_ranges,
            radau_basis,
            seed=args.seed + 1771,
            count=steady_count,
            use_direct=small,
        ),
    }

    report["relative_to_stock"] = {
        "full_solve_ratio": (
            radau["full_solve_count"]
            / stock["full_solve_count"]
        ),
        "basis_order_ratio": (
            radau["basis_order"] / stock["basis_order"]
        ),
        "wall_time_ratio": (
            radau["seconds"] / stock["wall_seconds"]
        ),
    }

    if small:
        oracle_basis, oracle = extract_oracle(
            operators,
            G,
            H,
            h_ranges,
            tolerance=args.tolerance,
            probe_rounds=args.probe_rounds,
            seed=args.seed,
        )
        radau_direct_basis, radau_direct = extract_radau(
            operators,
            G,
            H,
            h_ranges,
            tolerance=args.tolerance,
            probe_rounds=args.probe_rounds,
            seed=args.seed,
            solver=direct_solve,
        )
        min_cos, max_cos = principal_subspace_cosine(
            oracle_basis, radau_direct_basis
        )
        report["energy_oracle_direct"] = oracle
        report["energy_radau_direct"] = radau_direct
        report["oracle_radau_agreement"] = {
            "same_full_solve_count": (
                oracle["full_solve_count"]
                == radau_direct["full_solve_count"]
            ),
            "same_pre_svd_order": (
                oracle["pre_svd_order"]
                == radau_direct["pre_svd_order"]
            ),
            "same_basis_order": (
                oracle["basis_order"]
                == radau_direct["basis_order"]
            ),
            "minimum_principal_cosine": min_cos,
            "maximum_principal_cosine": max_cos,
        }

    (args.output / "result.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
