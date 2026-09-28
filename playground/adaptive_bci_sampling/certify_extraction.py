#!/usr/bin/env python3
"""Certified deterministic BCI extraction on Case 1, against the stock loop.

Runs the deterministic design of :mod:`deterministic_design`, the stock random
Extended-FANTASTIC extractor, the whole-box certificate of
:mod:`certified_box` for both, and the same full-order validation for both.
Generated JSON belongs outside the repository.

Two different metrics are reported and they must not be confused:

* ``worst_relative_port_defect`` (``sweep``) is the certified whole-box
  fixed-shift port defect of the delivered basis (``[PORT-FIXED-S]``), an upper
  bound over every HTC vector of the box and every input combination;
* the ``exact`` block is the same quantity evaluated by the Woodbury exact map,
  used as an oracle to check tightness.

The sampled entrywise port step-response comparison (``[PORT-STEP]``, measured
and not certified) lives in the separate, optional ``validate_vendor_step.py``.

    PYTHONPATH=python python playground/adaptive_bci_sampling/certify_extraction.py 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la

from sparse_solve import AmgSolver

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

from certified_box import BoxCertificate, logarithmic_edges  # noqa: E402
from exact_error import AffineErrorMap  # noqa: E402
from deterministic_design import (  # noqa: E402
    frequency_plan,
    build_basis,
    certified_greedy_points,
    full_operator,
)
from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    build_parametric_basis,
)


def audit_certificate(certificate, kernel, terms, source, basis, cells, order, samples, seed=20260926):
    """Check the certified cell bound against full-order solves inside cells.

    Every cell of a uniform log partition is sampled at ``samples**d`` random
    interior points and the *exact* relative port defect
    ``lambda_max(Y - Y_V, Y)`` is computed from a full-order transfer solve.  The
    audit fails loudly if the measured defect exceeds the cell bound; otherwise it
    reports the worst ratio and the largest values.
    """
    ranges = certificate.ranges
    edges = [
        logarithmic_edges(low, high, count)
        for (low, high), count in zip(ranges, [cells] * ranges.shape[0])
    ]
    rng = np.random.default_rng(seed)
    worst_ratio = 0.0
    largest_measured = 0.0
    largest_certified = 0.0
    sampled = 0
    for choice in itertools.product(*[range(cells)] * ranges.shape[0]):
        low = np.array([edges[axis][index] for axis, index in enumerate(choice)])
        high = np.array([edges[axis][index + 1] for axis, index in enumerate(choice)])
        gram = certificate.anchor_gram(
            certificate.block_lower(certificate.block_index(low))
        )
        weight, _fallback = certificate.cell_weight(high, "reduced")
        bound = certificate.cell_bound(gram, low, high, weight, order)
        measured = 0.0
        for _ in range(samples ** ranges.shape[0]):
            parameter = np.exp(rng.uniform(np.log(low), np.log(high)))
            operator = full_operator(kernel, terms, parameter)
            factor = AmgSolver(operator)
            transfer = np.ascontiguousarray(source.T @ factor.solve(source))
            reduced = basis.T @ (operator @ basis)
            coefficients = la.solve(
                reduced, basis.T @ source, assume_a="pos", check_finite=False
            )
            residual = source - operator @ (basis @ coefficients)
            defect = np.ascontiguousarray(residual.T @ factor.solve(residual))
            measured = max(measured, float(np.max(la.eigvalsh(
                0.5 * (defect + defect.T), 0.5 * (transfer + transfer.T),
                check_finite=False,
            ))))
            sampled += 1
        certified = float(bound)
        if certified < measured - 1e-12:
            raise RuntimeError(f"certificate violated on cell {choice}")
        largest_measured = max(largest_measured, measured)
        largest_certified = max(largest_certified, certified)
        if measured > 0.0:
            worst_ratio = max(worst_ratio, certified / measured)
    return {
        "cells": int(cells ** ranges.shape[0]),
        "order": int(order),
        "sampled_points": int(sampled),
        "violations": 0,
        "worst_bound_over_measured": worst_ratio,
        "largest_measured": largest_measured,
        "largest_certified": largest_certified,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_mm", nargs="?", type=float, default=5.0)
    parser.add_argument("--greedy-tolerance", type=float, default=1e-5)
    parser.add_argument("--greedy-maximum", type=int, default=8)
    parser.add_argument("--greedy-grid", type=int, default=41)
    parser.add_argument("--cutoff", type=float, default=1e-3)
    parser.add_argument("--steady-cells", type=int, default=32)
    parser.add_argument("--steady-order", type=int, default=2)
    parser.add_argument("--certificate-blocks", type=int, default=4)
    parser.add_argument("--exact-grid", type=int, default=21,
                        help="cells per axis of the exact error grid")
    parser.add_argument("--exact-cells", type=int, default=8,
                        help="cells per axis of the rigorous exact bracket")
    parser.add_argument("--skip-stock", action="store_true")
    parser.add_argument("--audit-cells", type=int, default=0,
                        help="cells per axis of the full-order audit partition")
    parser.add_argument("--audit-samples", type=int, default=2,
                        help="samples per axis inside each audited cell")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = Case1Model(Case1Config(max_xy_cell_mm=args.mesh_mm, max_z_cell_mm=args.mesh_mm))
    kernel = model.core.K.tocsc()
    mass = model.core.C.tocsc()
    source = np.asarray(model.source_shape, dtype=np.float64)
    terms = [term.tocsc() for term in model.boundary_terms]
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    report = {"mesh_mm": args.mesh_mm, "cells": int(kernel.shape[0]),
              "cutoff": args.cutoff, "h_ranges": ranges.tolist()}

    plan = frequency_plan(kernel, mass, source, args.cutoff)
    report["frequency_plan"] = {k: plan[k] for k in ("lower", "upper", "count")}
    print(f"frequency plan: count={plan['count']} "
          f"lambda=[{plan['lower']:.6g}, {plan['upper']:.6g}]", flush=True)

    started = time.perf_counter()
    response_cache = {}
    points, selection_certificate, selection = certified_greedy_points(
        kernel, terms, source, ranges, None,
        tolerance=args.greedy_tolerance, maximum_points=args.greedy_maximum,
        grid=args.greedy_grid, progress=True,
        cache=response_cache,
    )
    print(f"design points={len(points)} selection_certificate={selection_certificate:.3e} "
          f"t={time.perf_counter()-started:.1f}s", flush=True)
    design_basis, design_snapshots, design_info = build_basis(
        kernel, mass, terms, source, points, plan=plan,
        tolerance=args.cutoff, include_dc=True,
        cache=response_cache,
    )
    print(f"design operators={design_info['operators']} "
          f"shifts={len(plan['shifts']) + 1} points={len(points)}", flush=True)
    design_info["selection_certificate"] = selection_certificate
    design_info["selection"] = selection
    design_info["selection_factorizations"] = int(selection["factorizations"])
    design_info["selection_rhs_solves"] = int(selection["fresh_rhs_solves"])
    design_info["total_factorizations"] = int(
        selection["factorizations"] + design_info["factorizations"]
    )
    report["design"] = design_info
    print(f"design solves={design_info['full_rhs_solves']} "
          f"factorizations={design_info['total_factorizations']} "
          f"(selection {design_info['selection_factorizations']}) "
          f"order={design_info['basis_order']} "
          f"t={design_info['seconds']:.1f}s", flush=True)

    bases = {"deterministic": design_basis}
    if not args.skip_stock:
        for stock_seed in (20260805, 7):
            clock = time.perf_counter()
            stock, stats = build_parametric_basis(
                model.core, source, terms, ranges,
                tolerance=args.cutoff, max_order=4096, probe_rounds=10,
                seed=stock_seed,
            )
            clock = time.perf_counter() - clock
            label = f"stock_{stock_seed}"
            bases[label] = stock
            report[label] = {
                "full_rhs_solves": int(stats["pre_svd_order"]),
                "basis_order": int(stock.shape[1]),
                "candidate_count": int(stats["candidate_count"]),
                "validation_count": int(stats["validation_count"]),
                "per_port_plans": stats["per_port_plans"],
                "seconds": clock,
            }
            print(f"{label} solves={stats['pre_svd_order']} "
                  f"candidates={stats['candidate_count']} order={stock.shape[1]} "
                  f"t={clock:.1f}s", flush=True)

    certificates = {}
    for name, basis in bases.items():
        certificate = BoxCertificate(
            kernel, terms, source, ranges, basis, blocks=(args.certificate_blocks,) * 2
        )
        steady = certificate.sweep(args.steady_cells, order=args.steady_order)
        print(f"steady[{name}]: cells={steady['cells']} "
              f"port_defect={steady['worst_relative_port_defect']:.3e} "
              f"denominators={steady['denominator_points']} t={steady['seconds']:.1f}s", flush=True)
        certificates[name] = {"steady": steady}
    report["certificates"] = certificates

    exact = {}
    for name, basis in bases.items():
        mapping = AffineErrorMap(kernel, terms, source, ranges, basis)
        grid = mapping.worst_port_defect_on_grid(args.exact_grid)
        bracket = mapping.sweep(args.exact_cells)
        exact[name] = {"grid": grid, "bracket": bracket}
        print(f"exact[{name}]: grid={grid['worst_relative_port_defect']:.3e} at "
              f"{grid['location']} cell_bound={bracket['cell_bound']:.3e} "
              f"prep={bracket['preparation_seconds']:.1f}s "
              f"t={bracket['seconds']:.1f}s", flush=True)
    report["exact"] = exact

    if args.audit_cells:
        audit_certificate_for = BoxCertificate(
            kernel, terms, source, ranges, bases["deterministic"],
            blocks=(args.certificate_blocks,) * 2,
        )
        audit = audit_certificate(
            audit_certificate_for, kernel, terms, source, bases["deterministic"],
            args.audit_cells, args.steady_order, max(args.audit_samples, 2),
        )
        report["audit"] = audit
        print(f"audit: {audit}", flush=True)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("design",) if k in report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
