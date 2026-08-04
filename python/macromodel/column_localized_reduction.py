#!/usr/bin/env python3
"""Run the column-localized thermal macromodel experiment."""

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
    grid_cells,
    normalized_operators,
    port_patches,
    project_exact_ports,
    recover_temperature,
    solve_reduced,
    validate_affine_macro,
)

REPORT = Path("results/bci_rom_uniform_convection_results.json")
LOCAL_DYNAMIC_MODES = 2
BDF1_SHIFTS = (1.0, 2.0)
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


def column_basis(compiled, cfg, base, delta, ports, dynamic_modes, shift_multipliers):
    """Build one independent orthonormal basis block per lateral column."""
    started = time.perf_counter()
    K_ip = base.K[ports:, :ports].tocsc()
    K_ii = base.K[ports:, ports:].tocsc()
    C_ip = base.C[ports:, :ports].tocsc()
    C_ii = base.C[ports:, ports:].tocsc()
    dK_ip = delta.K[ports:, :ports].tocsc()
    dK_ii = delta.K[ports:, ports:].tocsc()

    port_lookup = {
        (int(ix), int(iy)): port
        for port, (ix, iy) in enumerate(
            (ix, iy) for ix in cfg.port_indices for iy in cfg.port_indices
        )
    }
    grid = grid_cells(compiled)
    rows, columns, values, orders = [], [], [], []
    offset = 0
    seen_ports = 0

    for ix in range(compiled.nx):
        for iy in range(compiled.ny):
            cells = grid[ix, iy]
            cells = cells[cells >= 0].astype(np.int64)
            if not cells.size:
                continue

            k = K_ii[cells][:, cells].toarray()
            c = C_ii[cells][:, cells].toarray()
            candidates = [np.ones(cells.size)]

            mode_count = min(dynamic_modes, cells.size)
            if mode_count:
                eigenvalues, modes = scipy.linalg.eigh(
                    k,
                    c,
                    subset_by_index=(0, mode_count - 1),
                    check_finite=False,
                )
                candidates.extend(modes[:, eigenvalues <= math.pi / cfg.dt_s].T)

            port = port_lookup.get((ix, iy))
            if port is not None:
                seen_ports += 1
                b = K_ip[cells, port].toarray().ravel()
                cp = C_ip[cells, port].toarray().ravel()
                static = scipy.linalg.solve(k, -b, assume_a="sym", check_finite=False)
                candidates.append(static)

                sensitivity_rhs = (
                    dK_ii[cells][:, cells] @ static
                    + dK_ip[cells, port].toarray().ravel()
                )
                if np.linalg.norm(sensitivity_rhs) > 1.0e-14 * max(
                    np.linalg.norm(b), 1.0
                ):
                    candidates.append(
                        scipy.linalg.solve(
                            k, -sensitivity_rhs, assume_a="sym", check_finite=False
                        )
                    )

                for multiplier in shift_multipliers:
                    shift = multiplier / cfg.dt_s
                    response = scipy.linalg.solve(
                        k + shift * c,
                        -(b + shift * cp),
                        assume_a="sym",
                        check_finite=False,
                    )
                    candidates.append(response - static)

            matrix = np.column_stack(candidates)
            q, r, _ = scipy.linalg.qr(
                matrix, mode="economic", pivoting=True, check_finite=False
            )
            diagonal = np.abs(np.diag(r))
            keep = diagonal > np.finfo(float).eps * max(matrix.shape) * diagonal[0]
            local = np.ascontiguousarray(q[:, keep])
            orders.append(local.shape[1])

            for local_row, cell in enumerate(cells):
                nonzero = np.flatnonzero(np.abs(local[local_row]) > 1.0e-14)
                rows.extend([int(cell)] * nonzero.size)
                columns.extend((offset + nonzero).tolist())
                values.extend(local[local_row, nonzero].tolist())
            offset += local.shape[1]

    if seen_ports != ports:
        raise RuntimeError("interface-port/column mapping is inconsistent")

    basis = sp.csc_matrix((values, (rows, columns)), shape=(K_ii.shape[0], offset))
    ones = np.ones(basis.shape[0])
    if np.linalg.norm(basis @ (basis.T @ ones) - ones) > 1.0e-10 * math.sqrt(ones.size):
        raise RuntimeError("column basis does not preserve uniform temperature")
    if spla.norm(basis.T @ basis - sp.eye(basis.shape[1], format="csc")) > 1.0e-10:
        raise RuntimeError("column basis lost orthogonality")

    initial_internal = np.asarray(
        basis.T @ np.full(basis.shape[0], cfg.ambient_K)
    ).ravel()
    return basis, np.asarray(orders), initial_internal, time.perf_counter() - started


def run_experiment(cfg, boundaries, strict, dynamic_modes, shift_multipliers):
    offline_started = time.perf_counter()
    full_layout = build_model(cfg, Study.STEADY, detail=True, macro=True).compile()
    detail_steady = build_model(cfg, Study.STEADY, detail=True, macro=False).compile()
    detail_transient = build_model(
        cfg, Study.TRANSIENT, detail=True, macro=False
    ).compile()

    detail_patches = port_patches(cfg, Face.ZP, cfg.detail_height_mm * 1.0e-3)
    detail_ports_steady = PortMap(detail_steady, detail_patches)
    detail_ports_transient = PortMap(detail_transient, detail_patches)

    extraction_started = time.perf_counter()
    macro = build_model(cfg, Study.STEADY, detail=False, macro=True).compile()
    macro_ports = PortMap(macro, port_patches(cfg, Face.ZM, 0.0))
    base = normalized_operators(*macro_ports.assemble())

    anchor = build_model(
        cfg,
        Study.STEADY,
        detail=False,
        macro=True,
        convection_h=cfg.affine_anchor_h,
    ).compile()
    anchor_ports = PortMap(anchor, port_patches(cfg, Face.ZM, 0.0))
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

    basis, orders, initial_internal, basis_s = column_basis(
        macro,
        cfg,
        base,
        delta,
        ports,
        dynamic_modes,
        shift_multipliers,
    )
    projection_started = time.perf_counter()
    reduced_base = project_exact_ports(base, ports, basis)
    reduced_delta = project_exact_ports(delta, ports, basis)
    projection_s = time.perf_counter() - projection_started
    offline_s = time.perf_counter() - offline_started

    full_macro_order = ports + basis.shape[0]
    reduced_macro_order = ports + basis.shape[1]
    compression = full_macro_order / reduced_macro_order
    print(
        f"Grid {cfg.nx}x{cfg.nx}x{cfg.nz}; exact ports={ports}; "
        f"macro states {full_macro_order:,}->{reduced_macro_order:,} ({compression:.2f}x)"
    )

    results = []
    detail_count = detail_steady.cell_count
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
        initial = np.r_[
            np.full(detail_count + ports, cfg.ambient_K),
            initial_internal,
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
            ambient_K=None,
        )[0]
        recovered_history = recover_temperature(
            transient_states,
            full_count=full_layout.cell_count,
            detail_map=detail_to_full,
            macro_map=macro_to_full,
            detail_count=detail_count,
            ports=ports,
            basis=basis,
            ambient_K=None,
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
        "method": "exact-port affine-convection column-local Galerkin BCI-ROM",
        "configuration": cfg.report_dict(),
        "reduction": {
            "dynamic_modes_per_column": dynamic_modes,
            "bdf1_shift_multipliers": list(shift_multipliers),
            "full_macro_order": full_macro_order,
            "reduced_macro_order": reduced_macro_order,
            "compression_ratio": compression,
            "compression_target": cfg.compression_target,
            "compression_passed": compression_passed,
            "column_count": int(orders.size),
            "local_order_min": int(orders.min()),
            "local_order_mean": float(orders.mean()),
            "local_order_max": int(orders.max()),
            "basis_nnz": basis.nnz,
        },
        "timing": {
            "macro_extraction_s": extraction_s,
            "basis_extraction_s": basis_s,
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
    print("=" * 96)
    print("Transient BCI-ROM extraction - column-localized reduction")
    print("=" * 96)
    print(
        f"Grid target: max XY cell={cfg.max_xy_cell_mm:g} mm, vertical cells={cfg.nz}"
    )

    report = run_experiment(
        cfg, BOUNDARIES, args.strict, LOCAL_DYNAMIC_MODES, BDF1_SHIFTS
    )
    report["mode"] = "quick" if args.quick else "strict"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Report: {REPORT}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
