#!/usr/bin/env python3
"""Faithful reproduction of the BCI-FANTASTIC parametric MOR algorithm.

Reproduces Codecasa et al., "Matrix Reduction Tool for Creating Boundary
Condition Independent Dynamic Compact Thermal Models", THERMINIC 2015 — the
BCI extension of FANTASTIC (THERMINIC 2014).  Ambient coupling is modeled by
explicit boundary ports on the macro top face.  A boundary port k carries the
material half-cell conductance g_k (the DtN interface conductance) plus the
free heat-exchange coefficient h through the series network
    cell -- g_k -- h*A_k -- ambient ,
which is exactly the native convection discretization of the same face
(src/solver/assembler.cpp, ThirdType: face_k*h*A/(face_k + h*dx)).  The
boundary ports are then eliminated by a Schur complement, so the reduced
model contains no fixed h:  its operators are h-independent and h enters only
through a precomputed diagonal closure.  The model therefore generalizes to
arbitrary heat-exchange coefficients — the defining BCI property, validated
here on holdout h values that were never trained.

The reduced interior basis V is built by the paper's Algorithm 1 (block form):
  * frequency shifts are the elliptic-optimal real shifts of FANTASTIC
    eqs 4-5 (scipy.special.ellipk / ellipj), in decreasing order;
  * parameter samples h are drawn at random over the admissible range
    (not greedily — the paper argues greedy stagnates);
  * for each (h, shift) the residual of the current reduced model is
    measured; when it exceeds the tolerance, interior directions are
    augmented and the compact model rebuilt.

Structured after tangential_rational_krylov_reduction.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import ellipj, ellipk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macromodel.experiment_setup import (  # noqa: E402
    BaseConfig,
    Face,
    Operators,
    PortMap,
    Study,
    accuracy_summary,
    build_model,
    coordinate_map,
    format_accuracy,
    full_face_patches,
    full_reference,
    normalized_operators,
    patch_areas,
    port_patches,
    project_exact_ports,
    recover_temperature,
    solve_reduced,
)

REPORT = Path("results/bci_fantastic_results.json")
H_RANGE = (1.0, 1.0e5)
TRAIN_H = tuple(np.geomspace(H_RANGE[0], H_RANGE[1], 6))
HOLDOUT_H = (5.0, 200.0, 5000.0, 80000.0)
RESIDUAL_TOLERANCE = 5.0e-3
RANDOM_PARAMETER_SAMPLES = 8
MAX_SVD_ORDER = 2048
MAX_SHIFTS = 6
RANDOM_SEED = 20260805
QUICK_OVERRIDES = {
    "substrate_cells": 2,
    "bump_cells": 1,
    "die_cells": 1,
    "tim_cells": 1,
    "spreader_cells": 2,
    "cold_plate_cells": 2,
    "max_xy_cell_mm": 4.0,
    "bump_rows": 8,
    "bump_columns": 8,
    "speedup_target": 1.0,
    "compression_target": 1.5,
}


def symmetric_dense(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def eigenpairs_descending(matrix):
    values, vectors = scipy.linalg.eigh(symmetric_dense(matrix), check_finite=False)
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0.0), vectors[:, order]


def elliptic_frequency_shifts(
    lam_min: float,
    lam_max: float,
    error_target: float,
    max_shifts: int,
) -> np.ndarray:
    """Real positive shifts optimal for moment matching (FANTASTIC eqs 4-5).

    The number m is the smallest integer with
        4 * exp(m^2 pi^2 / log(4 kappa)) <= error_target,  kappa = lam_max/lam_min,
    and the shifts are  omega_j = sqrt(lam_max) * dn((2j-1) K(k')/(2m), k')
    with dn the Jacobi elliptic function and k' = sqrt(1 - 1/kappa^2).  The
    shifts are returned in decreasing order so the earliest solves are the
    cheapest (paper: warm start from increasingly accurate estimates).
    """
    kappa = lam_max / max(lam_min, np.finfo(float).tiny)
    kappa = min(max(kappa, 1.0 + 1.0e-12), 1.0e12)
    squared = 1.0 - 1.0 / (kappa * kappa)
    squared = min(max(squared, 1.0e-15), 1.0 - 1.0e-15)
    elliptic_k = ellipk(squared)
    count = int(
        math.ceil(
            math.sqrt(
                max(math.log(4.0 * kappa), 1.0e-9)
                * math.log(4.0 / max(error_target, 1.0e-15))
            )
            / math.pi
        )
    )
    count = min(max(count, 1), max_shifts)
    arguments = (2 * np.arange(1, count + 1) - 1) * elliptic_k / (2 * count)
    _, _, dn, _ = ellipj(arguments, squared)
    shifts = math.sqrt(lam_max) * dn
    shifts = np.sort(shifts[shifts > 0.0])[::-1]
    return shifts


def orthonormalize_block(basis, vectors):
    block = np.asarray(vectors, dtype=np.float64).copy()
    if not block.size:
        return np.empty((block.shape[0], 0), dtype=np.float64)
    for _ in range(2):
        if basis.shape[1]:
            block -= basis @ (basis.T @ block)
    q, r, _ = scipy.linalg.qr(block, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((block.shape[0], 0), dtype=np.float64)
    keep = diagonal > np.finfo(float).eps * max(block.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, keep])


def reduced_response(basis, A, B_dense):
    if not basis.shape[1]:
        return np.empty((0, B_dense.shape[1]), dtype=np.float64)
    reduced_A = symmetric_dense(basis.T @ (A @ basis))
    factor = scipy.linalg.cho_factor(
        reduced_A, lower=True, overwrite_a=False, check_finite=False
    )
    return scipy.linalg.cho_solve(
        factor,
        -(basis.T @ B_dense),
        overwrite_b=False,
        check_finite=False,
    )


def response_error(response, basis, reduced, A, reference):
    error_response = response - basis @ reduced if basis.shape[1] else response
    error_gram = symmetric_dense(error_response.T @ (A @ error_response))
    values, tangents = eigenpairs_descending(error_gram)
    score = math.sqrt(float(values[0]) / reference)
    return error_response, values, tangents, score


def eigen_bounds(K_ii, C_ii) -> tuple[float, float]:
    """Extremal eigenvalue estimates of the internal (K, C) pencil."""
    if spla.norm(C_ii) == 0.0:
        return 1.0e-3, 1.0e6
    rayleigh_max = spla.eigsh(C_ii, k=1, which="LM", return_eigenvectors=False)[0]
    if not np.isfinite(rayleigh_max) or rayleigh_max <= 0.0:
        return 1.0e-3, 1.0e6
    lam_min = spla.eigsh(
        K_ii, k=1, M=C_ii, sigma=0.0, which="LM", return_eigenvectors=False
    )[0]
    lam_max = spla.eigsh(K_ii, k=1, M=C_ii, which="LM", return_eigenvectors=False)[0]
    return max(float(lam_min), 1.0e-12), max(float(lam_max), float(rayleigh_max))


def extract_boundary_coupling(merged: Operators, p: int, q: int):
    """Per boundary port: coupled internal cell and conductance g.

    The DtN assembly adds four terms per exposed face (src/macromodel/
    modal_port.cpp):  +g on (port,port), -g on (port,cell), -g on (cell,port),
    +g on (cell,cell).  The coupled cell is the one with a negative entry on
    the boundary-port row.  Cell indices are returned in the internal block
    coordinate frame (shifted by p + q).
    """
    rows = merged.K[p : p + q, :].tocsr()
    cells = np.empty(q, dtype=np.int64)
    conductance = np.empty(q, dtype=np.float64)
    for k in range(q):
        row = rows[k]
        negative = [col for col in row.indices if row[0, col] < 0.0]
        if len(negative) != 1:
            raise RuntimeError("boundary port must couple to exactly one cell")
        cell = negative[0]
        cells[k] = cell - p - q
        conductance[k] = -row[0, cell]
    return cells, conductance


def closure_diagonal(h: float, boundary_cells, boundary_g, boundary_areas, n_cell):
    """Diagonal correction K_ii(h) = K_ii + diag(closure) after elimination.

    Each boundary port k couples cell c through conductance g_k; attaching the
    ambient heat exchange h*A_k at the port and eliminating the port adds
        closure_c = g_k * h * A_k / (g_k + h * A_k)
    to the cell diagonal — exactly the native convection coefficient
    face_k*h*A/(face_k + h*dx) since g_k = face_k*A/dx.  The closure is h-
    dependent but the operators are not: h enters only through this diagonal.
    """
    closure = np.zeros(n_cell)
    for cell, g, area in zip(boundary_cells, boundary_g, boundary_areas):
        closure[cell] += g * h * area / (g + h * area)
    return closure


def verify_closure(
    cfg: BaseConfig,
    macro,
    interface,
    core: Operators,
    boundary_cells,
    boundary_g,
    boundary_areas,
    h: float,
) -> float:
    """Regression guard: the boundary-port closure reproduces native convection.

    Compares the core-based operator with the h*closure diagonal added
    (the exact form used online) against a native-convection macro built at
    the same h, under one unit interface-port forcing.  Returns the max
    internal-cell temperature difference.
    """
    from scipy.sparse.linalg import spsolve

    native = build_model(
        cfg, Study.STEADY, detail=False, macro=True, convection_h=h
    ).compile()
    pm_native = PortMap(native, interface)
    K_native, _, f_native = pm_native.assemble()
    p = len(interface)
    n_cell = macro.cell_count

    closure = closure_diagonal(h, boundary_cells, boundary_g, boundary_areas, n_cell)
    K_eff = core.K.copy().tolil()
    for cell in range(n_cell):
        K_eff[p + cell, p + cell] += closure[cell]
    K_eff = K_eff.tocsc()
    f_eff = np.asarray(core.f, dtype=np.float64).copy()
    f_eff[p:] += closure * cfg.ambient_K

    rhs_eff = f_eff.copy()
    rhs_eff[0] += 1.0
    rhs_native = np.asarray(f_native).ravel().copy()
    rhs_native[0] += 1.0
    x_eff = spsolve(K_eff, rhs_eff)
    x_native = spsolve(K_native, rhs_native)
    return float(np.abs(x_eff[p:] - x_native[p:]).max())


def build_bci_basis(
    cfg,
    core: Operators,
    p: int,
    boundary_cells,
    boundary_g,
    boundary_areas,
    *,
    h_samples,
    residual_tolerance,
    max_order,
):
    """Paper Algorithm 1: random h + elliptic shifts + residual enrichment.

    The internal block (interface ports stripped) is reduced.  For each
    (h, shift) the full response to interface-port forcing is solved; the
    residual of the current reduced model against it is measured; directions
    above tolerance are inserted into the basis immediately and the compact
    model rebuilt.  Returns the orthonormal interior basis V (n_cell x n_hat).
    """
    started = time.perf_counter()
    n_cell = core.K.shape[0] - p
    K_ii = core.K[p:, p:].tocsc()
    C_ii = core.C[p:, p:].tocsc()
    K_ip = core.K[p:, :p].tocsc()
    C_ip = core.C[p:, :p].tocsc()
    if spla.norm(C_ip) > 1.0e-14 * max(spla.norm(C_ii), 1.0):
        raise RuntimeError("interface ports unexpectedly carry capacitance")

    lam_min, lam_max = eigen_bounds(K_ii, C_ii)
    shifts = elliptic_frequency_shifts(lam_min, lam_max, residual_tolerance, MAX_SHIFTS)

    basis = np.empty((n_cell, 0), dtype=np.float64)
    order_limit = min(max_order, n_cell)
    history = []
    worst_score = 0.0
    converged = True

    for h_value in h_samples:
        closure = closure_diagonal(
            h_value, boundary_cells, boundary_g, boundary_areas, n_cell
        )
        closure_operator = sp.diags(closure, format="csc")
        for shift in shifts:
            A = (shift * C_ii + K_ii + closure_operator).tocsc()
            A = (0.5 * (A + A.T)).tocsc()
            B = (K_ip + shift * C_ip).tocsc()
            B_dense = B.toarray()

            response = np.asarray(spla.splu(A).solve(-B_dense))
            response_gram = symmetric_dense(-response.T @ B_dense)
            response_values, _ = eigenpairs_descending(response_gram)
            reference = max(float(response_values[0]), np.finfo(float).tiny)

            order_before = basis.shape[1]
            reduced = reduced_response(basis, A, B_dense)
            error_response, error_values, tangents, score_before = response_error(
                response, basis, reduced, A, reference
            )
            requested = int(
                np.count_nonzero(error_values > residual_tolerance**2 * reference)
            )
            available = order_limit - basis.shape[1]
            count = min(requested, available)

            added = 0
            if count:
                block = orthonormalize_block(
                    basis, error_response @ tangents[:, :count]
                )
                if not block.shape[1]:
                    raise RuntimeError("BCI enrichment stalled")
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
                    response, basis, reduced, A, reference
                )

            worst_score = max(worst_score, score_after)
            history.append(
                {
                    "h_W_m2K": float(h_value),
                    "shift_per_s": float(shift),
                    "order_before": int(order_before),
                    "order_after": int(basis.shape[1]),
                    "relative_response_error_before": float(score_before),
                    "relative_response_error_after": float(score_after),
                    "requested_directions": int(requested),
                    "added_directions": int(added),
                }
            )

            if requested > available or score_after > residual_tolerance:
                converged = False
                break
        if not converged:
            break

    if basis.shape[1]:
        orthogonality = basis.T @ basis
        orthogonality -= np.eye(basis.shape[1])
        orthogonality_error = float(np.max(np.abs(orthogonality)))
    else:
        orthogonality_error = 0.0
    if orthogonality_error > 1.0e-10:
        raise RuntimeError("BCI basis lost orthogonality")

    return basis, {
        "lambda_min_per_s": lam_min,
        "lambda_max_per_s": lam_max,
        "elliptic_shift_count": int(shifts.size),
        "elliptic_shifts_per_s": shifts.tolist(),
        "parameter_samples_W_m2K": h_samples.tolist(),
        "candidate_count": len(history),
        "basis_order": int(basis.shape[1]),
        "maximum_order": int(order_limit),
        "orthogonality_error": orthogonality_error,
        "relative_response_error": float(worst_score),
        "residual_tolerance": residual_tolerance,
        "converged": bool(converged and len(history) == len(h_samples) * shifts.size),
        "history": history,
        "seconds": time.perf_counter() - started,
    }


def run_experiment(cfg, boundaries, strict):
    offline_started = time.perf_counter()
    full_layout = build_model(cfg, Study.STEADY, detail=True, macro=True).compile()
    detail_steady = build_model(cfg, Study.STEADY, detail=True, macro=False).compile()
    detail_transient = build_model(
        cfg, Study.TRANSIENT, detail=True, macro=False
    ).compile()

    detail_patches = port_patches(cfg, Face.ZP, cfg.detail_height_mm * 1.0e-3)
    detail_ports_steady = PortMap(detail_steady, detail_patches)
    detail_ports_transient = PortMap(detail_transient, detail_patches)

    # Offline: insulated macro + merged PortMap (interface ZM + full-top ZP).
    extraction_started = time.perf_counter()
    macro = build_model(cfg, Study.STEADY, detail=False, macro=True).compile()
    interface = port_patches(cfg, Face.ZM, 0.0)
    boundary = full_face_patches(cfg, Face.ZP, cfg.macro_height_mm * 1.0e-3)
    boundary_areas = patch_areas(cfg, boundary)
    pm_merged = PortMap(macro, interface + boundary)
    merged = normalized_operators(*pm_merged.assemble())
    boundary_cells, boundary_g = extract_boundary_coupling(
        merged, len(interface), len(boundary)
    )

    pm_core = PortMap(macro, interface)
    core = normalized_operators(*pm_core.assemble())
    ports = pm_core.port_count
    if ports != cfg.ports:
        raise RuntimeError("configured interface port count is inconsistent")
    extraction_s = time.perf_counter() - extraction_started

    # Regression guard: the closure must reproduce native convection.
    closure_error = verify_closure(
        cfg,
        macro,
        interface,
        core,
        boundary_cells,
        boundary_g,
        boundary_areas,
        cfg.affine_anchor_h,
    )
    if closure_error > 1.0e-8 * max(cfg.ambient_K, 1.0):
        raise RuntimeError(
            "boundary-port closure fails to reproduce native convection: "
            f"{closure_error:.3e}"
        )

    detail_to_full = coordinate_map(detail_steady, full_layout, 0, "detail/full")
    if not np.array_equal(
        detail_to_full,
        coordinate_map(detail_transient, full_layout, 0, "transient/full"),
    ):
        raise RuntimeError("steady and transient detail orderings differ")
    macro_to_full = coordinate_map(macro, full_layout, cfg.detail_nz, "macro/full")
    combined = np.r_[detail_to_full, macro_to_full]
    if (
        combined.size != full_layout.cell_count
        or np.unique(combined).size != combined.size
    ):
        raise RuntimeError("detail and macro maps do not partition the full model")

    rng = np.random.default_rng(RANDOM_SEED)
    # Geometric training grid plus log-uniform random samples over [1, 1e5].
    # Linear uniform sampling over a 5-decade range would concentrate on the
    # high end; log-uniform spreads the enrichment across all decades.
    h_samples = np.unique(
        np.r_[
            np.asarray(boundaries, dtype=np.float64),
            10.0
            ** rng.uniform(
                math.log10(H_RANGE[0]), math.log10(H_RANGE[1]), RANDOM_PARAMETER_SAMPLES
            ),
        ]
    )
    basis, basis_summary = build_bci_basis(
        cfg,
        core,
        ports,
        boundary_cells,
        boundary_g,
        boundary_areas,
        h_samples=h_samples,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_order=MAX_SVD_ORDER,
    )
    if not basis_summary["converged"]:
        raise RuntimeError(
            "BCI extraction did not converge: "
            f"order={basis_summary['basis_order']}, "
            f"worst relative response error="
            f"{basis_summary['relative_response_error']:.3e}, "
            f"target={basis_summary['residual_tolerance']:.3e}"
        )

    full_macro_order = ports + core.K.shape[0] - ports
    reduced_macro_order = ports + basis.shape[1]
    compression = full_macro_order / reduced_macro_order
    print(
        f"Grid {cfg.nx}x{cfg.nx}x{cfg.nz}; exact ports={ports}; "
        f"boundary ports={len(boundary)}; "
        f"macro states {full_macro_order:,}->{reduced_macro_order:,} "
        f"({compression:.2f}x); "
        f"BCI residual={basis_summary['relative_response_error']:.3e}"
    )

    def online_operators(h_value):
        """Assemble and project the BCI operator at an arbitrary h."""
        n_cell = core.K.shape[0] - ports
        closure = closure_diagonal(
            h_value, boundary_cells, boundary_g, boundary_areas, n_cell
        )
        K_h = core.K.copy().tolil()
        for cell in range(n_cell):
            K_h[ports + cell, ports + cell] += closure[cell]
        f_h = np.asarray(core.f, dtype=np.float64).copy()
        f_h[ports:] += closure * cfg.ambient_K
        return project_exact_ports(
            normalized_operators(K_h, core.C, f_h), ports, basis, cfg.ambient_K
        )

    results = []
    detail_count = detail_steady.cell_count
    test_h = np.asarray(HOLDOUT_H, dtype=np.float64)
    for convection_h in test_h:
        (
            reference_steady,
            reference_times,
            reference_history,
            full_compile_s,
            full_steady_s,
            full_transient_s,
            full_order,
        ) = full_reference(cfg, convection_h)

        started = time.perf_counter()
        reduced = online_operators(convection_h)
        assembly_s = time.perf_counter() - started
        initial = np.r_[
            np.full(detail_count + ports, cfg.ambient_K),
            np.zeros(basis.shape[1]),
        ]
        steady_state, reduced_steady_s = solve_reduced(
            detail_steady,
            detail_ports_steady,
            reduced,
            initial,
            cfg,
            False,
        )
        times, transient_states, reduced_transient_s = solve_reduced(
            detail_transient,
            detail_ports_transient,
            reduced,
            initial,
            cfg,
            True,
        )
        if times.shape != reference_times.shape or not np.allclose(
            times, reference_times, atol=1.0e-12, rtol=0.0
        ):
            raise RuntimeError("full and reduced output times differ")

        recovered_steady = recover_temperature(
            steady_state,
            full_count=full_layout.cell_count,
            detail_map=detail_to_full,
            macro_map=macro_to_full,
            detail_count=detail_count,
            ports=ports,
            basis=basis,
            ambient_K=cfg.ambient_K,
        )[0]
        recovered_history = recover_temperature(
            transient_states,
            full_count=full_layout.cell_count,
            detail_map=detail_to_full,
            macro_map=macro_to_full,
            detail_count=detail_count,
            ports=ports,
            basis=basis,
            ambient_K=cfg.ambient_K,
        )
        accuracy = accuracy_summary(
            reference_steady,
            recovered_steady,
            reference_history,
            recovered_history,
            cfg.ambient_K,
        )
        speedup = full_transient_s / max(reduced_transient_s, np.finfo(float).tiny)
        speedup_passed = not strict or speedup >= cfg.speedup_target
        result = {
            "h_W_m2K": convection_h,
            **accuracy,
            "full_compile_s": full_compile_s,
            "full_steady_solve_s": full_steady_s,
            "reduced_steady_solve_s": reduced_steady_s,
            "full_transient_solve_s": full_transient_s,
            "reduced_transient_solve_s": reduced_transient_s,
            "online_reduced_assembly_s": assembly_s,
            "transient_speedup": speedup,
            "full_order": full_order,
            "reduced_online_order": detail_count + reduced.K.shape[0],
            "speedup_passed": speedup_passed,
            "passed": accuracy["accuracy_passed"] and speedup_passed,
        }
        results.append(result)
        print(
            f"h={convection_h:g} W/(m^2 K): {format_accuracy(accuracy)}; "
            f"full/ROM={full_transient_s:.3f}/{reduced_transient_s:.3f}s, "
            f"speedup={speedup:.2f}x "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

    offline_s = time.perf_counter() - offline_started
    compression_passed = compression >= cfg.compression_target
    if not results:
        raise RuntimeError("no holdout h evaluated")
    return {
        "schema_version": 25,
        "method": (
            "faithful BCI-FANTASTIC parametric MOR (boundary ports + random "
            "residual enrichment) BCI-ROM"
        ),
        "configuration": cfg.report_dict(),
        "bci_structure": {
            "interface_ports": ports,
            "boundary_ports": int(boundary_areas.size),
            "boundary_g_W_K": boundary_g.tolist(),
            "boundary_areas_m2": boundary_areas.tolist(),
            "closure_vs_native_max_cell_error_K": closure_error,
            "h_range_W_m2K": list(H_RANGE),
        },
        "reduction": {
            "full_macro_order": full_macro_order,
            "reduced_macro_order": reduced_macro_order,
            "internal_full_order": core.K.shape[0] - ports,
            "internal_reduced_order": basis.shape[1],
            "compression_ratio": compression,
            "compression_target": cfg.compression_target,
            "compression_passed": compression_passed,
            "trained_h_W_m2K": [
                float(h) for h in np.asarray(boundaries, dtype=np.float64)
            ],
            "bci": basis_summary,
        },
        "timing": {
            "macro_extraction_s": extraction_s,
            "offline_s": offline_s,
        },
        "bci_holdout": results,
        "passed": bool(
            compression_passed and all(result["passed"] for result in results)
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="small smoke experiment")
    mode.add_argument("--strict", action="store_true", help="full benchmark gates")
    args = parser.parse_args(argv)

    cfg = replace(BaseConfig(), **QUICK_OVERRIDES) if args.quick else BaseConfig()
    report = run_experiment(cfg, TRAIN_H, args.strict)
    report["mode"] = "quick" if args.quick else "strict"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {REPORT}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
