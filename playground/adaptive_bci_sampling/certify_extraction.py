#!/usr/bin/env python3
"""Certified deterministic BCI extraction on Case 1, against the stock loop.

Runs the deterministic design of :mod:`deterministic_design`, the stock random
Extended-FANTASTIC extractor, the whole-box certificate of
:mod:`certified_box` for both, and the same full-order validation for both.
Generated JSON belongs outside the repository.

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
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

from certified_box import BoxCertificate, logarithmic_edges  # noqa: E402
from deterministic_design import build_basis, certified_greedy_points, full_operator  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import build_parametric_basis  # noqa: E402


def step_transfer(kernel, mass, source, *, dt, duration):
    """BDF1 response to independent unit power steps, zero initial rise."""
    steps = round(duration / dt)
    factor = spla.splu(sp.csc_matrix(kernel + mass / dt).tocsc())
    state = np.zeros_like(source)
    outputs = np.zeros((steps + 1, source.shape[1], source.shape[1]))
    for index in range(1, outputs.shape[0]):
        state = factor.solve(source + mass @ state / dt)
        outputs[index] = source.T @ state
    return outputs


def reduced_step(reduced_kernel, reduced_mass, projected_source, *, dt, duration):
    steps = round(duration / dt)
    matrix = reduced_kernel + reduced_mass / dt
    factor = la.cho_factor(matrix, check_finite=False)
    state = np.zeros_like(projected_source)
    outputs = np.zeros((steps + 1, projected_source.shape[1], projected_source.shape[1]))
    for index in range(1, outputs.shape[0]):
        state = la.cho_solve(
            factor, projected_source + reduced_mass @ state / dt, check_finite=False
        )
        outputs[index] = projected_source.T @ state
    return outputs


def validate(kernel, mass, source, terms, bases, parameters, *, dt, duration):
    """Full-order validation of every basis on the same parameter set."""
    records = {name: {} for name in bases}
    tiny = np.finfo(float).tiny
    for parameter in parameters:
        operator = full_operator(kernel, terms, parameter)
        factor = spla.splu(operator.tocsc())
        exact_steady = np.ascontiguousarray(source.T @ factor.solve(source))
        exact_step = step_transfer(operator, mass, source, dt=dt, duration=duration)
        for name, basis in bases.items():
            reduced_kernel = basis.T @ (kernel @ basis)
            reduced_mass = basis.T @ (mass @ basis)
            projected_source = basis.T @ source
            reduced_operator = basis.T @ (operator @ basis)
            coefficients = la.solve(
                reduced_operator, projected_source, assume_a="pos", check_finite=False
            )
            approximate_steady = np.ascontiguousarray(projected_source.T @ coefficients)
            approximate_step = reduced_step(
                reduced_operator, reduced_mass, projected_source, dt=dt, duration=duration
            )
            records[name]["worst_step_entry"] = max(
                records[name].get("worst_step_entry", 0.0),
                float(np.max(np.abs(approximate_step - exact_step) / np.maximum(np.abs(exact_steady), tiny))),
            )
            records[name]["worst_steady_entry"] = max(
                records[name].get("worst_steady_entry", 0.0),
                float(np.max(np.abs(approximate_steady - exact_steady) / np.maximum(np.abs(exact_steady), tiny))),
            )
    return records


def validation_parameters(ranges, *, corners=True, correlated=12, random=24, seed=20260926):
    ranges = np.asarray(ranges, dtype=np.float64)
    blocks = []
    if corners:
        blocks.append(np.asarray(list(itertools.product(*ranges))))
    rng = np.random.default_rng(seed)
    paired = np.exp(
        rng.uniform(np.log(ranges[:, 0]), np.log(ranges[:, 1]), size=(correlated, ranges.shape[0]))
    )
    blocks.append(paired)
    blocks.append(
        np.exp(rng.uniform(np.log(ranges[:, 0]), np.log(ranges[:, 1]), size=(random, ranges.shape[0])))
    )
    return np.vstack(blocks)


def audit_certificate(certificate, kernel, terms, source, basis, cells, order, samples, seed=20260926):
    """Check the certified cell bound against full-order solves inside cells.

    Every cell of a uniform log partition is sampled at ``samples**d`` random
    interior points.  The audit fails loudly if any measured error exceeds the
    cell bound; otherwise it reports the worst ratio and the largest values.
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
        bound = certificate.cell_bound(gram, low, high, order)
        measured = 0.0
        for _ in range(samples ** ranges.shape[0]):
            parameter = np.exp(rng.uniform(np.log(low), np.log(high)))
            operator = full_operator(kernel, terms, parameter)
            factor = spla.splu(operator.tocsc())
            exact = np.ascontiguousarray(source.T @ factor.solve(source))
            reduced = basis.T @ (operator @ basis)
            coefficients = la.solve(
                reduced, basis.T @ source, assume_a="pos", check_finite=False
            )
            approximate = np.ascontiguousarray(source.T @ (basis @ coefficients))
            measured = max(measured, float(np.max(np.abs(approximate - exact))))
            sampled += 1
        certified = float(np.max(bound))
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
    parser.add_argument("--dt", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=2000.0)
    parser.add_argument("--greedy-tolerance", type=float, default=1e-5)
    parser.add_argument("--greedy-maximum", type=int, default=8)
    parser.add_argument("--greedy-grid", type=int, default=41)
    parser.add_argument("--cutoff", type=float, default=1e-3)
    parser.add_argument("--steady-cells", type=int, default=32)
    parser.add_argument("--steady-order", type=int, default=2)
    parser.add_argument("--certificate-blocks", type=int, default=4)
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
    power = np.asarray(model.nominal_power(), dtype=np.float64)
    report = {"mesh_mm": args.mesh_mm, "cells": int(kernel.shape[0]), "dt": args.dt,
              "duration": args.duration, "cutoff": args.cutoff,
              "h_ranges": ranges.tolist()}

    started = time.perf_counter()
    response_cache = {}
    points, selection_certificate, selection = certified_greedy_points(
        kernel, terms, source, ranges, None,
        tolerance=args.greedy_tolerance, maximum_points=args.greedy_maximum,
        grid=args.greedy_grid, power=power, metric="entrywise", progress=True,
        cache=response_cache,
    )
    print(f"design points={len(points)} selection_certificate={selection_certificate:.3e} "
          f"t={time.perf_counter()-started:.1f}s", flush=True)
    design_basis, design_snapshots, design_info = build_basis(
        kernel, mass, terms, source, points,
        tolerance=args.cutoff, low_shift_threshold=1.0 / args.dt, include_dc=True,
        cache=response_cache,
    )
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
                tolerance=args.cutoff, max_order=4096, probe_rounds=10, seed=stock_seed,
            )
            clock = time.perf_counter() - clock
            bases[f"stock_{stock_seed}"] = stock
            report[f"stock_{stock_seed}"] = {
                "full_rhs_solves": int(stats["pre_svd_order"]),
                "basis_order": int(stock.shape[1]),
                "candidate_count": int(stats["candidate_count"]),
                "validation_count": int(stats["validation_count"]),
                "seconds": clock,
            }
            print(f"stock_{stock_seed} solves={stats['pre_svd_order']} "
                  f"candidates={stats['candidate_count']} order={stock.shape[1]} "
                  f"t={clock:.1f}s", flush=True)

    certificates = {}
    for name, basis in bases.items():
        certificate = BoxCertificate(
            kernel, terms, source, ranges, basis, blocks=(args.certificate_blocks,) * 2
        )
        steady = certificate.sweep(args.steady_cells, order=args.steady_order)
        print(f"steady[{name}]: cells={steady['cells']} bound={steady['steady_relative_bound']:.3e} "
              f"diagonal={steady['steady_diagonal_bound']:.3e} "
              f"denominators={steady['denominator_points']} t={steady['seconds']:.1f}s", flush=True)
        certificates[name] = {"steady": steady}
    report["certificates"] = certificates

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

    parameters = validation_parameters(ranges)
    clock = time.perf_counter()
    validation = validate(
        kernel, mass, source, terms, bases, parameters, dt=args.dt, duration=args.duration
    )
    report["validation"] = {"parameters": len(parameters), "worst": validation,
                            "seconds": time.perf_counter() - clock}
    for name in bases:
        print(f"validation[{name}]: worst_step={validation[name]['worst_step_entry']:.3e} "
              f"worst_steady={validation[name]['worst_steady_entry']:.3e}", flush=True)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("design",) if k in report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
