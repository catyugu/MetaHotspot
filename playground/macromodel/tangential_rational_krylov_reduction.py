#!/usr/bin/env python3
"""Run the tangential rational Krylov thermal macromodel experiment.

Builds a certified interior basis by residual-driven enrichment over the
exact boundary-port closure (g*h*A/(g+h*A)) at several heat-exchange
coefficients, then validates the reduced model on the same range.  The
operators contain no fixed h — h enters only through the precomputed diagonal
closure, so the model is boundary-condition independent.

This script is *model-agnostic*: it obtains its model from the
:mod:`affine_parametric_models` factory (``create``) and drives it through the
abstract :class:`AffineParametricModel` contract.  It never names a concrete
model or a config field, and it is *parameter-count-agnostic*: the number of
affine parameters is whatever ``boundary_groups()`` reports (one heat-exchange
coefficient per group), the holdout sweep derives from the model's own group
ranges, and the shared :func:`build_parametric_basis` enrichment is driven
unchanged — ``--model chiplet_stack`` (1 group) and ``--model bci_pkg`` (2
groups) both run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metahotspot.compiled import Operators  # noqa: E402
from affine_parametric_models import create  # noqa: E402
from utils import (  # noqa: E402
    accuracy_summary,
    build_parametric_basis,
    format_accuracy,
    project_exact_ports,
    project_closure_group,
)

REPORT = Path("results/bci_rom_parametric_krylov_results.json")
RESIDUAL_TOLERANCE = 5.0e-3
MAX_ORDER = 2048
SWEEP_COUNT = 5  # holdout points per sweep axis


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


def run_experiment(model):
    ambient_K = model.ambient_K
    offline_started = time.perf_counter()

    core = model.core_operators()
    ports = model.port_count
    groups = model.boundary_groups()
    boundary_groups = [(g.cells, g.g) for g in groups]
    group_areas = [g.areas for g in groups]
    h_ranges = np.asarray([g.h_range for g in groups], dtype=np.float64)
    # The model decides how its parameter space is laid out (one h-vector per
    # boundary group); the experiment just iterates what it provides.
    points = model.parameter_points(count=SWEEP_COUNT)

    detail_count = model.detail_cell_count

    basis, basis_summary = build_parametric_basis(
        core,
        ports,
        boundary_groups,
        group_areas,
        h_ranges=h_ranges,
        boundaries=points,
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

    reduced_core = project_exact_ports(core, ports, basis, ambient_K)
    verify_ambient_balance(
        reduced_core,
        ports,
        basis.shape[1],
        ambient_K,
        "base",
    )
    offline_s = time.perf_counter() - offline_started

    full_macro_order = core.K.shape[0]
    reduced_macro_order = ports + basis.shape[1]
    compression = full_macro_order / reduced_macro_order
    cfg = model.report_dict()
    print(
        f"Grid {cfg['nx']}x{cfg['nx']}x{cfg['nz']}; exact ports={ports}; "
        f"macro states {full_macro_order:,}->{reduced_macro_order:,} "
        f"({compression:.2f}x); "
        f"Krylov residual={basis_summary['relative_response_error']:.3e}"
    )

    n_modes = basis.shape[1]
    n_cell = basis.shape[0]
    proj_closure = [
        project_closure_group(g.cells, g.g, g.areas, n_cell, basis) for g in groups
    ]

    def online_operators(h_vec):
        delta = sum(cm(h).toarray() for cm, h in zip(proj_closure, h_vec))
        D = sp.bmat(
            (
                (sp.csc_matrix((ports, ports)), sp.csc_matrix((ports, n_modes))),
                (
                    sp.csc_matrix((n_modes, ports)),
                    sp.csc_matrix(delta),
                ),
            ),
            format="csc",
        )
        return Operators(
            (reduced_core.K + D).tocsc(),
            reduced_core.C,
            reduced_core.f,
        )

    results = []
    initial = model.initial_state(basis.shape[1])
    for h_vec in points:
        reference = model.full_reference(h_vec)

        started = time.perf_counter()
        reduced = online_operators(h_vec)
        assembly_s = time.perf_counter() - started
        steady_state, reduced_steady_s = model.solve_reduced(reduced, initial, False)
        times, transient_states, reduced_transient_s = model.solve_reduced(
            reduced, initial, True
        )
        if times.shape != reference.times.shape or not np.allclose(
            times,
            reference.times,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise RuntimeError("full and reduced output times differ")

        recovered_steady = model.recover_temperature(
            steady_state,
            basis=basis,
            ports=ports,
            ambient_K=ambient_K,
        )[0]
        recovered_history = model.recover_temperature(
            transient_states,
            basis=basis,
            ports=ports,
            ambient_K=ambient_K,
        )
        accuracy = accuracy_summary(
            reference.steady_temperature,
            recovered_steady,
            reference.history,
            recovered_history,
            ambient_K,
        )
        speedup = reference.transient_s / max(
            reduced_transient_s,
            np.finfo(float).tiny,
        )
        result = {
            "h_W_m2K": h_vec[0],
            "h_vec": list(h_vec),
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
            f"h={h_vec[0]:g} W/(m^2 K): {format_accuracy(accuracy)}; "
            f"full/ROM={reference.transient_s:.3f}/{reduced_transient_s:.3f}s, "
            f"speedup={speedup:.2f}x "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

    return {
        "method": (
            "exact-closure boundary-port tangential rational Krylov BCI-ROM "
            "(no affine linearization)"
        ),
        "configuration": model.report_dict(),
        "reduction": {
            "full_macro_order": full_macro_order,
            "reduced_macro_order": reduced_macro_order,
            "internal_full_order": basis.shape[0],
            "internal_reduced_order": basis.shape[1],
            "compression_ratio": compression,
            "temperature_coordinates": (
                "absolute port temperature and internal rise above ambient"
            ),
            "krylov": basis_summary,
        },
        "timing": {
            "offline_s": offline_s,
        },
        "boundary_reuse": results,
        "passed": bool(all(result["passed"] for result in results)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--model",
        default="chiplet_stack",
        help="registered affine parametric model name (default: chiplet_stack)",
    )
    args = parser.parse_args(argv)

    model = create(args.model, quick=args.quick)
    report = run_experiment(model)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {REPORT}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
