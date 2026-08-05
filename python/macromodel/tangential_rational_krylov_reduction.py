#!/usr/bin/env python3
"""Run the tangential rational Krylov thermal macromodel experiment.

Builds a certified interior basis by residual-driven enrichment over the
exact boundary-port closure (g*h*A/(g+h*A)) at several heat-exchange
coefficients, then validates the reduced model on the same range.  The
operators contain no fixed h — h enters only through the precomputed diagonal
closure, so the model is boundary-condition independent.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macromodel.experiment_setup import (  # noqa: E402
    BaseConfig,
    Face,
    PortMap,
    Study,
    accuracy_summary,
    build_model,
    closure_diagonal,
    coordinate_map,
    extract_boundary_groups,
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

REPORT = Path("results/bci_rom_parametric_krylov_results.json")
H_RANGE = (1.0, 1.0e5)
BOUNDARIES = tuple(np.geomspace(H_RANGE[0], H_RANGE[1], 5))
FREQUENCY_SAMPLES = 5
RESIDUAL_TOLERANCE = 5.0e-3
MAX_ORDER = 1024
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


def training_points(cfg, h_range, boundaries, frequency_samples):
    """Return h samples (h_range geometric grid + boundaries) and BDF1 shifts."""
    h_values = np.unique(
        np.r_[np.asarray(boundaries, dtype=np.float64), np.geomspace(*h_range, 8)]
    )
    positive_count = max(1, frequency_samples - 1)
    positive_shifts = np.geomspace(
        1.0 / cfg.duration_s,
        2.0 / cfg.dt_s,
        positive_count,
    )
    shifts = np.unique(np.r_[0.0, positive_shifts])
    return np.asarray(h_values, dtype=np.float64), shifts


def internal_blocks(operators, ports):
    return (
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, ports:].tocsc(),
        operators.K[ports:, :ports].tocsc(),
        operators.C[ports:, :ports].tocsc(),
    )


def orthonormalize_block(basis, vectors):
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
    error_response = response - basis @ reduced if basis.shape[1] else response
    error_gram = symmetric_dense(error_response.T @ (A @ error_response))
    values, tangents = eigenpairs_descending(error_gram)
    score = math.sqrt(float(values[0]) / reference)
    return error_response, values, tangents, score


def build_krylov_basis(
    cfg,
    core,
    ports,
    boundary_cells,
    boundary_g,
    boundary_areas,
    *,
    h_range,
    boundaries,
    frequency_samples,
    residual_tolerance,
    max_order,
):
    """Build a certified basis with one streamed residual solve per candidate.

    Each candidate operator is ``K_ii + shift*C_ii + diag(closure(h))`` where
    closure is the exact boundary-port saturation term.  Residual directions
    above the tolerance are inserted immediately, so a candidate never needs
    to be rescanned and no full-state response is retained.
    """
    started = time.perf_counter()
    K0, C0, B0, D0 = internal_blocks(core, ports)
    h_values, shifts = training_points(cfg, h_range, boundaries, frequency_samples)
    raw_points = [(float(h), float(shift)) for h in h_values for shift in shifts]

    internal_order = K0.shape[0]
    order_limit = min(max_order, internal_order)
    basis = np.empty((internal_order, 0), dtype=np.float64)
    history = []
    worst_score = 0.0
    converged = True

    for h_value, shift in raw_points:
        closure = closure_diagonal(
            h_value, boundary_cells, boundary_g, boundary_areas, internal_order
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
                "relative_response_error_before": float(score_before),
                "relative_response_error_after": float(score_after),
                "h_W_m2K": h_value,
                "shift_per_s": shift,
                "requested_directions": requested,
                "added_directions": int(added),
            }
        )

        if requested > available or score_after > residual_tolerance:
            converged = False
            break

    if basis.shape[1]:
        orthogonality = basis.T @ basis
        orthogonality -= np.eye(basis.shape[1])
        orthogonality_error = float(np.max(np.abs(orthogonality)))
    else:
        orthogonality_error = 0.0
    if orthogonality_error > 1.0e-10:
        raise RuntimeError("rational Krylov basis lost orthogonality")

    return basis, {
        "parameter_samples_W_m2K": h_values.tolist(),
        "frequency_shifts_per_s": shifts.tolist(),
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


def verify_ambient_balance(operators, ports, reduced_order, ambient_K, label):
    state = np.r_[np.full(ports, ambient_K), np.zeros(reduced_order)]
    defect = np.asarray(operators.K @ state - operators.f).ravel()
    scale = max(
        np.linalg.norm(operators.K @ state),
        np.linalg.norm(operators.f),
        1.0,
    )
    if np.linalg.norm(defect) > 1.0e-10 * scale:
        raise RuntimeError(f"{label} reduced operator violates ambient balance")


def run_experiment(cfg, boundaries, strict, krylov_options):
    offline_started = time.perf_counter()
    full_layout = build_model(cfg, Study.STEADY, detail=True, macro=True).compile()
    detail_steady = build_model(
        cfg,
        Study.STEADY,
        detail=True,
        macro=False,
    ).compile()
    detail_transient = build_model(
        cfg,
        Study.TRANSIENT,
        detail=True,
        macro=False,
    ).compile()

    detail_patches = port_patches(cfg, Face.ZP, cfg.detail_height_mm * 1.0e-3)
    detail_ports_steady = PortMap(detail_steady, detail_patches)
    detail_ports_transient = PortMap(detail_transient, detail_patches)

    extraction_started = time.perf_counter()
    macro = build_model(cfg, Study.STEADY, detail=False, macro=True).compile()
    interface = port_patches(cfg, Face.ZM, 0.0)
    boundary = full_face_patches(cfg, Face.ZP, cfg.macro_height_mm * 1.0e-3)
    boundary_areas = patch_areas(cfg, boundary)
    pm_merged = PortMap(macro, interface + boundary)
    merged = normalized_operators(*pm_merged.assemble())
    boundary_cells, boundary_g = extract_boundary_groups(
        merged, len(interface), [len(boundary)]
    )[0]

    pm_core = PortMap(macro, interface)
    core = normalized_operators(*pm_core.assemble())
    ports = pm_core.port_count
    if ports != cfg.ports:
        raise RuntimeError("configured interface port count is inconsistent")
    extraction_s = time.perf_counter() - extraction_started

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

    basis, basis_summary = build_krylov_basis(
        cfg,
        core,
        ports,
        boundary_cells,
        boundary_g,
        boundary_areas,
        h_range=H_RANGE,
        boundaries=boundaries,
        frequency_samples=FREQUENCY_SAMPLES,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_order=MAX_ORDER,
    )
    if not basis_summary["converged"]:
        raise RuntimeError(
            "Krylov extraction did not converge: "
            f"order={basis_summary['basis_order']}, "
            f"worst relative response error="
            f"{basis_summary['relative_response_error']:.3e}, "
            f"target={basis_summary['residual_tolerance']:.3e}"
        )

    reduced_core = project_exact_ports(core, ports, basis, cfg.ambient_K)
    verify_ambient_balance(
        reduced_core,
        ports,
        basis.shape[1],
        cfg.ambient_K,
        "base",
    )
    offline_s = time.perf_counter() - offline_started

    full_macro_order = core.K.shape[0]
    reduced_macro_order = ports + basis.shape[1]
    compression = full_macro_order / reduced_macro_order
    print(
        f"Grid {cfg.nx}x{cfg.nx}x{cfg.nz}; exact ports={ports}; "
        f"macro states {full_macro_order:,}->{reduced_macro_order:,} "
        f"({compression:.2f}x); "
        f"Krylov residual={basis_summary['relative_response_error']:.3e}"
    )

    def online_operators(convection_h):
        n_cell = core.K.shape[0] - ports
        closure = closure_diagonal(
            convection_h, boundary_cells, boundary_g, boundary_areas, n_cell
        )
        K = core.K.copy().tolil()
        for cell in range(n_cell):
            K[ports + cell, ports + cell] += closure[cell]
        f = np.asarray(core.f, dtype=np.float64).copy()
        f[ports:] += closure * cfg.ambient_K
        return project_exact_ports(
            normalized_operators(K, core.C, f), ports, basis, cfg.ambient_K
        )

    results = []
    detail_count = detail_steady.cell_count
    initial = np.r_[
        np.full(detail_count + ports, cfg.ambient_K),
        np.zeros(basis.shape[1]),
    ]
    for convection_h in boundaries:
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
            times,
            reference_times,
            atol=1.0e-12,
            rtol=0.0,
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
        speedup = full_transient_s / max(
            reduced_transient_s,
            np.finfo(float).tiny,
        )
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

    compression_passed = compression >= cfg.compression_target
    return {
        "method": (
            "exact-closure boundary-port tangential rational Krylov BCI-ROM "
            "(no affine linearization)"
        ),
        "configuration": cfg.report_dict(),
        "reduction": {
            "full_macro_order": full_macro_order,
            "reduced_macro_order": reduced_macro_order,
            "internal_full_order": basis.shape[0],
            "internal_reduced_order": basis.shape[1],
            "compression_ratio": compression,
            "compression_target": cfg.compression_target,
            "compression_passed": compression_passed,
            "temperature_coordinates": (
                "absolute port temperature and internal rise above ambient"
            ),
            "krylov": basis_summary,
        },
        "timing": {
            "macro_extraction_s": extraction_s,
            "offline_s": offline_s,
        },
        "boundary_reuse": results,
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

    print("=" * 96)
    print("Transient BCI-ROM extraction - tangential rational Krylov")
    print("=" * 96)
    print(
        f"Grid target: max XY cell={cfg.max_xy_cell_mm:g} mm, "
        f"vertical cells={cfg.nz}"
    )

    report = run_experiment(cfg, BOUNDARIES, args.strict, {})
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
