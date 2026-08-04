#!/usr/bin/env python3
"""Run the tangential rational Krylov thermal macromodel experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macromodel.experiment_setup import (  # noqa: E402
    BOUNDARIES,
    BaseConfig,
    Face,
    Operators,
    PortMap,
    Study,
    accuracy_summary,
    affine_operators,
    build_model,
    coordinate_map,
    format_accuracy,
    full_reference,
    normalized_operators,
    port_patches,
    project_exact_ports,
    recover_temperature,
    solve_reduced,
    validate_affine_macro,
)

REPORT = Path("results/bci_rom_parametric_krylov_results.json")
PARAMETER_SAMPLES = 3
FREQUENCY_SAMPLES = 5
RESIDUAL_TOLERANCE = 5.0e-3
SEED_DIRECTIONS = 1
BLOCK_SIZE = 16
MAX_ORDER = 1024
MAX_REFINEMENTS = 32
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


def training_points(cfg, boundaries, parameter_samples, frequency_samples):
    h_values = np.geomspace(min(boundaries), max(boundaries), parameter_samples)
    h_values = np.unique(np.r_[h_values, cfg.affine_anchor_h])
    low = 1.0 / cfg.duration_s
    high = 2.0 / cfg.dt_s
    interior_count = max(0, frequency_samples - 2)
    interior = (
        np.geomspace(low, high, interior_count + 2)[1:-1]
        if interior_count
        else np.empty(0)
    )
    shifts = np.unique(np.r_[0.0, low, interior, high])
    return h_values, shifts


def internal_blocks(operators, ports):
    return (
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, ports:].tocsc(),
        operators.K[ports:, :ports].tocsc(),
        operators.C[ports:, :ports].tocsc(),
    )


def orthonormalize_block(basis, vectors):
    block = np.asarray(vectors, dtype=np.float64).copy()
    for _ in range(2):
        if basis.shape[1]:
            block -= basis @ (basis.T @ block)
    q, r, _ = scipy.linalg.qr(block, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((block.shape[0], 0), dtype=np.float64)
    keep = diagonal > np.finfo(float).eps * max(block.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, keep])


def training_response(K0, C0, B0, D0, K1, C1, B1, D1, mu, h_value, shift):
    """Factor a training operator once and retain only its port-space Gramian."""
    A = (K0 + mu * K1 + shift * (C0 + mu * C1)).tocsc()
    A = (0.5 * (A + A.T)).tocsc()
    B = (B0 + mu * B1 + shift * (D0 + mu * D1)).tocsc()
    response = np.asarray(spla.splu(A).solve(-B.toarray()))
    gram = symmetric_dense(response.T @ (A @ response))
    values, tangents = eigenpairs_descending(gram)
    return (
        {
            "mu": mu,
            "h_W_m2K": h_value,
            "shift_per_s": shift,
            "A": A,
            "B": B,
            "gram": gram,
            "reference": max(float(values[0]), np.finfo(float).tiny),
        },
        response,
        values,
        tangents,
    )


def build_krylov_basis(
    cfg,
    boundaries,
    base,
    delta,
    ports,
    *,
    parameter_samples,
    frequency_samples,
    residual_tolerance,
    seed_directions,
    block_size,
    max_order,
    max_refinements,
):
    """Build a memory-bounded tangential basis by streaming training points."""
    started = time.perf_counter()
    K0, C0, B0, D0 = internal_blocks(base, ports)
    K1, C1, B1, D1 = internal_blocks(delta, ports)
    h_values, shifts = training_points(
        cfg, boundaries, parameter_samples, frequency_samples
    )
    raw_points = [
        (float(h / cfg.affine_anchor_h), float(h), float(shift))
        for h in h_values
        for shift in shifts
    ]

    internal_order = K0.shape[0]
    order_limit = min(max_order, internal_order)
    basis = np.empty((internal_order, 0), dtype=np.float64)
    history = []
    points = []

    # Stream the large full-state response through seeding. Only the small
    # port-space Gramian remains resident for later greedy scans.
    for mu, h_value, shift in raw_points:
        point, response, values, tangents = training_response(
            K0,
            C0,
            B0,
            D0,
            K1,
            C1,
            B1,
            D1,
            mu,
            h_value,
            shift,
        )
        points.append(point)
        count = min(
            seed_directions,
            np.count_nonzero(values > 0.0),
            max(0, order_limit - basis.shape[1]),
        )
        if count:
            block = orthonormalize_block(basis, response @ tangents[:, :count])
            basis = np.column_stack((basis, block[:, : order_limit - basis.shape[1]]))

    converged = False
    worst_score = math.inf
    for refinement in range(max_refinements + 1):
        worst = None
        for point in points:
            A = point["A"]
            B = point["B"]
            reference = point["reference"]

            if basis.shape[1]:
                reduced_A = symmetric_dense(basis.T @ (A @ basis))
                reduced_B = basis.T @ B
                reduced_response = scipy.linalg.solve(
                    reduced_A,
                    -reduced_B,
                    assume_a="sym",
                    check_finite=False,
                )
                error_gram = point["gram"] - symmetric_dense(
                    reduced_response.T @ (reduced_A @ reduced_response)
                )
            else:
                reduced_response = np.empty((0, B.shape[1]), dtype=np.float64)
                error_gram = point["gram"]

            error_values, tangents = eigenpairs_descending(error_gram)
            score = math.sqrt(float(error_values[0]) / reference)
            if worst is None or score > worst["score"]:
                worst = {
                    "score": score,
                    "point": point,
                    "reference": reference,
                    "reduced_response": reduced_response,
                    "error_values": error_values,
                    "tangents": tangents,
                }

        worst_score = float(worst["score"])
        entry = {
            "order": int(basis.shape[1]),
            "relative_response_error": worst_score,
            "h_W_m2K": worst["point"]["h_W_m2K"],
            "shift_per_s": worst["point"]["shift_per_s"],
            "added_directions": 0,
        }
        history.append(entry)
        if worst_score <= residual_tolerance:
            converged = True
            break
        if refinement == max_refinements or basis.shape[1] >= order_limit:
            break

        relative_directions = np.sqrt(worst["error_values"] / worst["reference"])
        requested = max(
            1, int(np.count_nonzero(relative_directions > residual_tolerance))
        )
        count = min(requested, block_size, order_limit - basis.shape[1])
        tangents = worst["tangents"][:, :count]
        point = worst["point"]
        response = np.asarray(spla.splu(point["A"]).solve(-(point["B"] @ tangents)))
        error_response = response - basis @ (worst["reduced_response"] @ tangents)
        block = orthonormalize_block(basis, error_response)
        if not block.shape[1]:
            raise RuntimeError("rational Krylov enrichment stalled")
        block = block[:, : order_limit - basis.shape[1]]
        basis = np.column_stack((basis, block))
        entry["added_directions"] = int(block.shape[1])

    orthogonality_error = np.linalg.norm(
        basis.T @ basis - np.eye(basis.shape[1]), ord=2
    )
    if orthogonality_error > 1.0e-10:
        raise RuntimeError("rational Krylov basis lost orthogonality")

    return basis, {
        "parameter_samples_W_m2K": h_values.tolist(),
        "frequency_shifts_per_s": shifts.tolist(),
        "candidate_count": len(points),
        "seed_directions_per_candidate": seed_directions,
        "block_size": block_size,
        "basis_order": int(basis.shape[1]),
        "maximum_order": int(order_limit),
        "orthogonality_error": float(orthogonality_error),
        "relative_response_error": worst_score,
        "residual_tolerance": residual_tolerance,
        "converged": converged,
        "history": history,
        "seconds": time.perf_counter() - started,
        "memory_strategy": "cache port Gramians; solve full-state responses only once per candidate",
    }


def verify_ambient_balance(operators, ports, reduced_order, ambient_K, label):
    state = np.r_[np.full(ports, ambient_K), np.zeros(reduced_order)]
    defect = np.asarray(operators.K @ state - operators.f).ravel()
    scale = max(np.linalg.norm(operators.K @ state), np.linalg.norm(operators.f), 1.0)
    if np.linalg.norm(defect) > 1.0e-10 * scale:
        raise RuntimeError(f"{label} reduced operator violates ambient balance")


def run_experiment(cfg, boundaries, strict, krylov_options):
    offline_started = time.perf_counter()
    with ExitStack() as stack:
        full_layout = build_model(cfg, Study.STEADY, detail=True, macro=True).compile()
        stack.callback(full_layout.close)
        detail_steady = build_model(
            cfg, Study.STEADY, detail=True, macro=False
        ).compile()
        stack.callback(detail_steady.close)
        detail_transient = build_model(
            cfg, Study.TRANSIENT, detail=True, macro=False
        ).compile()
        stack.callback(detail_transient.close)

        detail_patches = port_patches(cfg, Face.ZP, cfg.detail_height_mm * 1.0e-3)
        detail_ports_steady = PortMap(detail_steady, detail_patches)
        stack.callback(detail_ports_steady.close)
        detail_ports_transient = PortMap(detail_transient, detail_patches)
        stack.callback(detail_ports_transient.close)

        extraction_started = time.perf_counter()
        macro = build_model(cfg, Study.STEADY, detail=False, macro=True).compile()
        stack.callback(macro.close)
        macro_ports = PortMap(macro, port_patches(cfg, Face.ZM, 0.0))
        stack.callback(macro_ports.close)
        base = normalized_operators(*macro_ports.assemble())

        anchor = build_model(
            cfg,
            Study.STEADY,
            detail=False,
            macro=True,
            convection_h=cfg.affine_anchor_h,
        ).compile()
        stack.callback(anchor.close)
        anchor_ports = PortMap(anchor, port_patches(cfg, Face.ZM, 0.0))
        stack.callback(anchor_ports.close)
        anchor_operators = normalized_operators(*anchor_ports.assemble())
        if anchor_operators.K.shape != base.K.shape:
            raise RuntimeError("convection changed macro state ordering")
        delta = normalized_operators(
            anchor_operators.K - base.K,
            anchor_operators.C - base.C,
            np.asarray(anchor_operators.f) - base.f,
        )
        validate_affine_macro(base, delta, cfg.ambient_K)
        extraction_s = time.perf_counter() - extraction_started

        ports = macro_ports.port_count
        if ports != cfg.ports:
            raise RuntimeError("configured interface port count is inconsistent")

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
            boundaries,
            base,
            delta,
            ports,
            **krylov_options,
        )
        if not basis_summary["converged"]:
            raise RuntimeError(
                "Krylov extraction did not converge: "
                f"order={basis_summary['basis_order']}, "
                f"worst relative response error={basis_summary['relative_response_error']:.3e}, "
                f"target={basis_summary['residual_tolerance']:.3e}"
            )

        projection_started = time.perf_counter()
        reduced_base = project_exact_ports(base, ports, basis, cfg.ambient_K)
        reduced_delta = project_exact_ports(delta, ports, basis, cfg.ambient_K)
        projection_s = time.perf_counter() - projection_started
        verify_ambient_balance(
            reduced_base, ports, basis.shape[1], cfg.ambient_K, "base"
        )
        verify_ambient_balance(
            reduced_delta, ports, basis.shape[1], cfg.ambient_K, "convection increment"
        )
        offline_s = time.perf_counter() - offline_started

        full_macro_order = ports + basis.shape[0]
        reduced_macro_order = ports + basis.shape[1]
        compression = full_macro_order / reduced_macro_order
        print(
            f"Grid {cfg.nx}x{cfg.nx}x{cfg.nz}; exact ports={ports}; "
            f"macro states {full_macro_order:,}->{reduced_macro_order:,} ({compression:.2f}x); "
            f"Krylov residual={basis_summary['relative_response_error']:.3e}"
        )

        results = []
        detail_count = detail_steady.cell_count
        initial = np.r_[
            np.full(detail_count + ports, cfg.ambient_K), np.zeros(basis.shape[1])
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
            reduced = affine_operators(
                reduced_base, reduced_delta, convection_h / cfg.affine_anchor_h
            )
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
                f"speedup={speedup:.2f}x {'PASS' if result['passed'] else 'FAIL'}"
            )

        compression_passed = compression >= cfg.compression_target
        return {
            "schema_version": 22,
            "method": "exact-port affine-parametric tangential rational Krylov BCI-ROM",
            "configuration": cfg.report_dict(),
            "reduction": {
                "full_macro_order": full_macro_order,
                "reduced_macro_order": reduced_macro_order,
                "internal_full_order": basis.shape[0],
                "internal_reduced_order": basis.shape[1],
                "compression_ratio": compression,
                "compression_target": cfg.compression_target,
                "compression_passed": compression_passed,
                "temperature_coordinates": "absolute port temperature and internal rise above ambient",
                "krylov": basis_summary,
            },
            "timing": {
                "macro_extraction_s": extraction_s,
                "projection_s": projection_s,
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
    krylov_options = {
        "parameter_samples": 2 if args.quick else PARAMETER_SAMPLES,
        "frequency_samples": 4 if args.quick else FREQUENCY_SAMPLES,
        "residual_tolerance": 1.0e-2 if args.quick else RESIDUAL_TOLERANCE,
        "seed_directions": SEED_DIRECTIONS,
        "block_size": 24 if args.quick else BLOCK_SIZE,
        "max_order": 384 if args.quick else MAX_ORDER,
        "max_refinements": 16 if args.quick else MAX_REFINEMENTS,
    }

    print("=" * 96)
    print("Transient BCI-ROM extraction - tangential rational Krylov")
    print("=" * 96)
    print(
        f"Grid target: max XY cell={cfg.max_xy_cell_mm:g} mm, vertical cells={cfg.nz}"
    )

    report = run_experiment(cfg, BOUNDARIES, args.strict, krylov_options)
    report["mode"] = "quick" if args.quick else "strict"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Report: {REPORT}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
