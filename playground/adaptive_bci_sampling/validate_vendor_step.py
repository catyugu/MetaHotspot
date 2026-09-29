#!/usr/bin/env python3
"""Optional external diagnostic: sampled port step-response comparison.

This is **not** an extraction certificate and it is not part of the acceptance
path.  The strict statement about a delivered basis is the whole-box
``[PORT-FIXED-S]`` port-defect certificate produced by
``certify_extraction.py``.  What this script adds is a vendor-inspired external
sanity check: the same monitor-point step response that the vendor validation
protocol compares, at a few time steps, on a sampled parameter set.

It deliberately reports a sampled, entrywise, steady-normalized quantity
(``[PORT-STEP]``, measured, never certified) and must not be quoted as a
guarantee.  The normalization follows the vendor convention: every entry is
divided by the *exact steady* transfer of the same parameter, not by the value
at the current time.

    PYTHONPATH=python python playground/adaptive_bci_sampling/validate_vendor_step.py 5
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

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

from sparse_solve import AmgSolver  # noqa: E402
from deterministic_design import (  # noqa: E402
    frequency_plan,
    build_basis,
    certified_greedy_points,
    full_operator,
)
from metahotspot.macromodel.utils import build_parametric_basis  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402


def step_transfer(kernel, mass, source, *, dt, duration):
    """Sampled port step response: BDF1 unit power steps, zero initial rise.

    Only ``source.T @ state`` is recorded, so the observable is the port
    transfer ``[PORT-STEP]``; the state trajectory itself is never compared.
    """
    steps = round(duration / dt)
    factor = AmgSolver(kernel + mass / dt)
    state = np.zeros_like(source)
    outputs = np.zeros((steps + 1, source.shape[1], source.shape[1]))
    for index in range(1, outputs.shape[0]):
        state = factor.solve(source + mass @ state / dt)
        outputs[index] = source.T @ state
    return outputs


def reduced_step(reduced_kernel, reduced_mass, projected_source, *, dt, duration):
    steps = round(duration / dt)
    factor = la.cho_factor(reduced_kernel + reduced_mass / dt, check_finite=False)
    state = np.zeros_like(projected_source)
    outputs = np.zeros((steps + 1, projected_source.shape[1], projected_source.shape[1]))
    for index in range(1, outputs.shape[0]):
        state = la.cho_solve(
            factor, projected_source + reduced_mass @ state / dt, check_finite=False
        )
        outputs[index] = projected_source.T @ state
    return outputs


def validate(kernel, mass, source, terms, bases, parameters, *, dt, duration):
    """Sampled entrywise step/steady comparison of every basis, one parameter set."""
    records = {name: {} for name in bases}
    tiny = np.finfo(float).tiny
    for parameter in parameters:
        operator = full_operator(kernel, terms, parameter)
        factor = AmgSolver(operator)
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
                float(np.max(np.abs(approximate_step - exact_step)
                             / np.maximum(np.abs(exact_steady), tiny))),
            )
            records[name]["worst_steady_entry"] = max(
                records[name].get("worst_steady_entry", 0.0),
                float(np.max(np.abs(approximate_steady - exact_steady)
                             / np.maximum(np.abs(exact_steady), tiny))),
            )
    return records


def validation_parameters(ranges, *, corners=True, correlated=12, random=24, seed=20260926):
    ranges = np.asarray(ranges, dtype=np.float64)
    blocks = []
    if corners:
        blocks.append(np.asarray(list(itertools.product(*ranges))))
    rng = np.random.default_rng(seed)
    blocks.append(
        np.exp(rng.uniform(np.log(ranges[:, 0]), np.log(ranges[:, 1]),
                           size=(correlated, ranges.shape[0])))
    )
    blocks.append(
        np.exp(rng.uniform(np.log(ranges[:, 0]), np.log(ranges[:, 1]),
                           size=(random, ranges.shape[0])))
    )
    return np.vstack(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_mm", nargs="?", type=float, default=5.0)
    parser.add_argument("--dt", type=float, nargs="+", default=[5.0, 50.0, 500.0])
    parser.add_argument("--duration", type=float, default=2000.0)
    parser.add_argument("--greedy-tolerance", type=float, default=1e-5)
    parser.add_argument("--greedy-maximum", type=int, default=3)
    parser.add_argument("--greedy-grid", type=int, default=41)
    parser.add_argument("--cutoff", type=float, default=1e-3)
    parser.add_argument("--stock-seeds", type=int, nargs="*", default=[20260805, 7])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = Case1Model(Case1Config(max_xy_cell_mm=args.mesh_mm, max_z_cell_mm=args.mesh_mm))
    kernel = model.core.K.tocsc()
    mass = model.core.C.tocsc()
    source = np.asarray(model.source_shape, dtype=np.float64)
    terms = [term.tocsc() for term in model.boundary_terms]
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)

    plan = frequency_plan(kernel, mass, source, args.cutoff)
    response_cache = {}
    started = time.perf_counter()
    points, selection_certificate, _selection = certified_greedy_points(
        kernel, terms, source, ranges, None,
        tolerance=args.greedy_tolerance, maximum_points=args.greedy_maximum,
        grid=args.greedy_grid, progress=True, cache=response_cache,
    )
    design_basis, _snapshots, _info = build_basis(
        kernel, mass, terms, source, points, plan=plan,
        tolerance=args.cutoff, include_dc=True, cache=response_cache,
    )
    print(f"design points={len(points)} selection_certificate={selection_certificate:.3e} "
          f"order={design_basis.shape[1]} t={time.perf_counter() - started:.1f}s", flush=True)

    bases = {"deterministic": design_basis}
    for seed in args.stock_seeds:
        stock, _stats = build_parametric_basis(
            model.core, source, terms, ranges,
            tolerance=args.cutoff, max_order=4096, probe_rounds=10, seed=seed,
        )
        bases[f"stock_{seed}"] = stock

    parameters = validation_parameters(ranges)
    report = {"mesh_mm": args.mesh_mm, "parameters": int(len(parameters)),
              "dt": args.dt, "duration": args.duration,
              "metric": "measured sampled entrywise port step error, steady-normalized",
              "scope": "PORT-STEP", "worst": {}}
    for dt in args.dt:
        report["worst"][str(dt)] = validate(
            kernel, mass, source, terms, bases, parameters, dt=dt, duration=args.duration
        )
        for name in bases:
            print(f"validation[{name}][dt={dt:g}]: "
                  f"worst_step={report['worst'][str(dt)][name]['worst_step_entry']:.3e} "
                  f"worst_steady={report['worst'][str(dt)][name]['worst_steady_entry']:.3e}",
                  flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
