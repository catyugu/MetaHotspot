#!/usr/bin/env python3
"""Run the column-localized thermal macromodel experiment.

This script is *model-agnostic*: it obtains its model from the
:mod:`affine_parametric_models` factory (``create``) and drives it through the
abstract :class:`AffineParametricModel` contract.  It never names a concrete
model or a config field, so it runs unchanged against any registered
implementation — ``--model chiplet_stack`` (default) or ``--model toy_1d``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metahotspot.compiled import Operators  # noqa: E402
from affine_parametric_models import create  # noqa: E402
from utils import (  # noqa: E402
    accuracy_summary,
    closure_diagonal,
    format_accuracy,
    normalized_operators,
    project_exact_ports,
)

REPORT = Path("results/bci_rom_uniform_convection_results.json")
H_RANGE = (1.0, 1.0e5)
BOUNDARIES = tuple(np.geomspace(H_RANGE[0], H_RANGE[1], 5))
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
}


def column_basis(
    model,
    core,
    ports,
    boundary_cells,
    boundary_g,
    boundary_areas,
    dynamic_modes,
    shift_multipliers,
):
    """Build one independent orthonormal basis block per lateral column.

    The h-dependence enters through the exact boundary-port closure diagonal
    ``closure(h)`` (no affine linearization).  Each column block contains the
    constant, local dynamic modes, the static port response, the closure-
    driven static response (captures the convection saturation), and the BDF1
    shift responses.
    """
    started = time.perf_counter()
    K_ip = core.K[ports:, :ports].tocsc()
    K_ii = core.K[ports:, ports:].tocsc()
    C_ip = core.C[ports:, :ports].tocsc()
    C_ii = core.C[ports:, ports:].tocsc()

    port_lookup = model.port_lookup
    grid = model.macro_grid
    dt = model.dt
    rows, columns, values, orders = [], [], [], []
    offset = 0
    seen_ports = 0

    # Closure at a representative high h: saturating bound of the convection.
    closure_hi = closure_diagonal(
        H_RANGE[1], boundary_cells, boundary_g, boundary_areas, K_ii.shape[0]
    )
    # Closure at a representative low h: near-insulated correction.
    closure_lo = closure_diagonal(
        H_RANGE[0], boundary_cells, boundary_g, boundary_areas, K_ii.shape[0]
    )
    closure_delta = closure_hi - closure_lo

    for ix in range(model.macro_nx):
        for iy in range(model.macro_ny):
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
                candidates.extend(modes[:, eigenvalues <= math.pi / dt].T)

            port = port_lookup.get((ix, iy))
            if port is not None:
                seen_ports += 1
                b = K_ip[cells, port].toarray().ravel()
                cp = C_ip[cells, port].toarray().ravel()
                static = scipy.linalg.solve(k, -b, assume_a="sym", check_finite=False)
                candidates.append(static)

                # Closure-driven sensitivity: response to the h-range delta on
                # the cells touched by convection (the boundary ports' cells).
                sensitivity_rhs = closure_delta[cells] * static
                if np.linalg.norm(sensitivity_rhs) > 1.0e-14 * max(
                    np.linalg.norm(b), 1.0
                ):
                    candidates.append(
                        scipy.linalg.solve(
                            k, -sensitivity_rhs, assume_a="sym", check_finite=False
                        )
                    )

                for multiplier in shift_multipliers:
                    shift = multiplier / dt
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
        basis.T @ np.full(basis.shape[0], model.ambient_K)
    ).ravel()
    return basis, np.asarray(orders), initial_internal, time.perf_counter() - started


def run_experiment(model, boundaries, strict, dynamic_modes, shift_multipliers):
    ambient_K = model.ambient_K
    offline_started = time.perf_counter()

    core = model.core_operators()
    ports = model.port_count
    groups = model.boundary_groups()
    boundary_cells, boundary_g = groups[0].cells, groups[0].g
    boundary_areas = groups[0].areas

    basis, orders, initial_internal, basis_s = column_basis(
        model,
        core,
        ports,
        boundary_cells,
        boundary_g,
        boundary_areas,
        dynamic_modes,
        shift_multipliers,
    )
    offline_s = time.perf_counter() - offline_started

    full_macro_order = ports + basis.shape[0]
    reduced_macro_order = ports + basis.shape[1]
    compression = full_macro_order / reduced_macro_order
    cfg = model.report_dict()
    print(
        f"Grid {cfg['nx']}x{cfg['nx']}x{cfg['nz']}; exact ports={ports}; "
        f"macro states {full_macro_order:,}->{reduced_macro_order:,} ({compression:.2f}x)"
    )

    results = []
    detail_count = model.detail_cell_count
    layout = model.state_layout(basis.shape[1])
    for convection_h in boundaries:
        reference = model.full_reference((convection_h,))

        started = time.perf_counter()
        n_cell = core.K.shape[0] - ports
        closure = closure_diagonal(
            convection_h, boundary_cells, boundary_g, boundary_areas, n_cell
        )
        K = core.K.copy().tolil()
        for cell in range(n_cell):
            K[ports + cell, ports + cell] += closure[cell]
        f = np.asarray(core.f, dtype=np.float64).copy()
        f[ports:] += closure * ambient_K
        reduced = project_exact_ports(normalized_operators(K, core.C, f), ports, basis)
        assembly_s = time.perf_counter() - started
        initial = np.r_[
            np.full(layout.detail_count + layout.port_count, ambient_K),
            initial_internal,
        ]
        steady_state, reduced_steady_s = model.solve_reduced(reduced, initial, False)
        times, transient_states, reduced_transient_s = model.solve_reduced(
            reduced, initial, True
        )
        if times.shape != reference.times.shape or not np.allclose(
            times, reference.times, atol=1.0e-12, rtol=0.0
        ):
            raise RuntimeError("full and reduced output times differ")

        recovered_steady = model.recover_temperature(
            steady_state,
            basis=basis,
            ports=ports,
            ambient_K=None,
        )[0]
        recovered_history = model.recover_temperature(
            transient_states,
            basis=basis,
            ports=ports,
            ambient_K=None,
        )
        accuracy = accuracy_summary(
            reference.steady_temperature,
            recovered_steady,
            reference.history,
            recovered_history,
            ambient_K,
        )
        speedup = reference.transient_s / max(reduced_transient_s, np.finfo(float).tiny)
        result = {
            "h_W_m2K": convection_h,
            **accuracy,
            "full_compile_s": reference.compile_s,
            "full_steady_solve_s": reference.steady_s,
            "reduced_steady_solve_s": reduced_steady_s,
            "full_transient_solve_s": reference.transient_s,
            "reduced_transient_solve_s": reduced_transient_s,
            "online_reduced_assembly_s": assembly_s,
            "transient_speedup": speedup,
            "full_order": reference.full_order,
            "reduced_online_order": detail_count + reduced.K.shape[0],
            "passed": accuracy["accuracy_passed"],
        }
        results.append(result)
        print(
            f"h={convection_h:g} W/(m^2 K): {format_accuracy(accuracy)}; "
            f"full/ROM={reference.transient_s:.3f}/{reduced_transient_s:.3f}s, "
            f"speedup={speedup:.2f}x {'PASS' if result['passed'] else 'FAIL'}"
        )

    return {
        "method": (
            "exact-closure boundary-port column-local Galerkin BCI-ROM "
            "(no affine linearization)"
        ),
        "configuration": cfg,
        "reduction": {
            "dynamic_modes_per_column": dynamic_modes,
            "bdf1_shift_multipliers": list(shift_multipliers),
            "full_macro_order": full_macro_order,
            "reduced_macro_order": reduced_macro_order,
            "compression_ratio": compression,
            "column_count": int(orders.size),
            "local_order_min": int(orders.min()),
            "local_order_mean": float(orders.mean()),
            "local_order_max": int(orders.max()),
            "basis_nnz": basis.nnz,
        },
        "timing": {
            "macro_extraction_s": 0.0,
            "basis_extraction_s": basis_s,
            "offline_s": offline_s,
        },
        "boundary_reuse": results,
        "passed": bool(all(result["passed"] for result in results)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="small smoke experiment")
    mode.add_argument("--strict", action="store_true", help="full benchmark gates")
    parser.add_argument(
        "--model",
        default="chiplet_stack",
        help="registered affine parametric model name (default: chiplet_stack)",
    )
    args = parser.parse_args(argv)

    model = create(args.model, overrides=QUICK_OVERRIDES if args.quick else None)
    print("=" * 96)
    print("Transient BCI-ROM extraction - column-localized reduction")
    print("=" * 96)
    cfg = model.report_dict()
    print(
        f"Grid target: {cfg.get('nx', 'n/a')}x{cfg.get('nx', 'n/a')}"
        f"x{cfg.get('nz', 'n/a')}"
    )

    report = run_experiment(
        model, BOUNDARIES, args.strict, LOCAL_DYNAMIC_MODES, BDF1_SHIFTS
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
