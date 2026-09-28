#!/usr/bin/env python3
"""Falsification of the matrix-valued cell certificate on Case 1.

After the cheap ``C``-metric scalar was rejected, the surviving statement is the
matrix one: for every HTC vector ``h`` in the box and every input combination
``w``, the delivered basis must reproduce the exact collocated port transfer,
i.e.

    E(h) = R(h)^T A(h)^-1 R(h),   Y(h) = G^T A(h)^-1 G,
    sup_h lambda_max(E(h), Y(h)) = sup_h sup_w w^T (Y(h) - Y_V(h)) w / w^T Y(h) w

must be small.  No square root is taken anywhere: this is the port quantity
itself.  The equal ``A``-energy state error is a lemma-level reinterpretation of
the same number and is not reported.  On one HTC cell ``Q`` with lower corner ``a`` and upper corner
``b`` the certificate claims, from ``H_i >= 0`` plus Galerkin optimality plus
matrix convexity of the Bernstein enclosure,

    sup_{h in Q} lambda_max(E(h), Y(h))
        <= max_nu lambda_max(Xi_nu^T S_a Xi_nu, Y(b)),

with ``S_a`` the Riesz Gram of the residual span at the anchor ``a`` and
``Xi_nu`` the tensor Bernstein coefficients of one Taylor trial residual.

This bench checks that claim against the exact map and reports the three things
that decide whether the certificate can drive a greedy:

1. violations: a cell bound below the exact worst value of the same cell;
2. refinement monotonicity: the bound must fall as the partition is refined;
3. effectivity on the cells that dominate the exact maximum, against the
   pre-registered gate "after refinement, the dominating cells are within 10x".

    PYTHONPATH=python python bench_matrix_cell_certificate.py 5 --samples 7
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

from certified_box import (  # noqa: E402
    BoxCertificate,
    generalized_max,
    logarithmic_edges,
    sparse_factor,
)
from deterministic_design import (  # noqa: E402
    build_basis,
    certified_greedy_points,
    frequency_plan,
)
from metahotspot.macromodel.utils import build_parametric_basis  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402


def build_model(mesh):
    model = Case1Model(Case1Config(max_xy_cell_mm=mesh, max_z_cell_mm=mesh))
    return (
        model,
        model.core.K.tocsc(),
        model.core.C.tocsc(),
        [term.tocsc() for term in model.boundary_terms],
        np.asarray(model.h_ranges(), dtype=np.float64),
        np.asarray(model.source_shape, dtype=np.float64),
    )


def operator_at(kernel, mass, terms, point, shift):
    operator = kernel + shift * mass
    for value, term in zip(point, terms):
        operator = operator + float(value) * term
    return operator.tocsc()


def exact_pair(operator, basis, source):
    """Exact ``(E(h), Y(h))`` of one operator, one factorization each."""
    factor = sparse_factor(operator)
    projected = np.asarray(basis.T @ (operator @ basis))
    reduced = np.linalg.solve(projected, np.asarray(basis.T @ source))
    residual = np.asarray(source) - np.asarray(operator @ (basis @ reduced))
    error = np.asarray(residual.T @ factor.solve(residual))
    transfer = np.asarray(np.asarray(source).T @ factor.solve(source))
    return (
        0.5 * (error + error.T),
        0.5 * (transfer + transfer.T),
    )


def cell_samples(low, high, samples):
    """Log-spaced interior samples plus both corners of one cell."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    fractions = np.linspace(0.0, 1.0, samples + 2)
    axes = [
        np.exp(np.log(low[axis]) + fractions * (np.log(high[axis]) - np.log(low[axis])))
        for axis in range(low.size)
    ]
    return [np.array(point) for point in itertools.product(*axes)]


def grid_points(ranges, cells_per_axis):
    edges = [
        logarithmic_edges(low, high, cells_per_axis)
        for low, high in np.asarray(ranges)
    ]
    axes = [np.unique(np.concatenate([edges[axis], [float(ranges[axis][1])]]))
            for axis in range(len(ranges))]
    return [np.array(point) for point in itertools.product(*axes)]


def exact_field(kernel, mass, terms, source, basis, points, shift):
    """Exact ``lambda_max(E(h), Y(h))`` on a list of HTC vectors."""
    values = []
    for point in points:
        operator = operator_at(kernel, mass, terms, point, shift)
        error, transfer = exact_pair(operator, basis, source)
        values.append(float(generalized_max(error, transfer)))
    return np.asarray(values)


def covering_cell(point, ranges, cells_per_axis):
    """Index of the uniform log cell that contains ``point``."""
    index = []
    for axis, (low, high) in enumerate(np.asarray(ranges)):
        fraction = (np.log(point[axis]) - np.log(low)) / (np.log(high) - np.log(low))
        index.append(min(int(fraction * cells_per_axis), cells_per_axis - 1))
    return tuple(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_mm", nargs="?", type=float, default=5.0)
    parser.add_argument("--basis", choices=("design", "stock"), default="design")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--cutoff", type=float, default=1e-3)
    parser.add_argument("--shift-index", type=int, default=None,
                        help="plan shift to certify; omit for the steady operator")
    parser.add_argument("--cells", default="2,4,8",
                        help="comma separated cells per axis to compare")
    parser.add_argument("--samples", type=int, default=7,
                        help="interior samples per axis inside every cell")
    parser.add_argument("--reference-cells", type=int, default=24,
                        help="cells per axis of the box-wide exact reference")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--orders", default="2",
                        help="comma separated Taylor jet orders of the bound")
    parser.add_argument("--trials", default="taylor",
                        help="comma separated polynomial trials, e.g. taylor,chebyshev")
    parser.add_argument("--denominators", default="reduced",
                        help="comma separated cell denominators: reduced,exact")
    parser.add_argument("--anchors", default="both",
                        help="anchor rules to evaluate: block, local or both")
    parser.add_argument("--plan", choices=("box", "legacy"), default="box",
                        help="frequency plan provenance; box is the delivered route")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model, kernel, mass, terms, ranges, source = build_model(args.mesh_mm)
    plan = frequency_plan(kernel, mass, terms, source, ranges, args.cutoff, args.plan)
    shift = 0.0 if args.shift_index is None else float(plan["shifts"][args.shift_index])
    report = {
        "mesh_mm": args.mesh_mm,
        "cells": int(kernel.shape[0]),
        "basis": args.basis,
        "seed": args.seed,
        "shift": shift,
        "orders": [int(value) for value in args.orders.split(",")],
        "trials": [value.strip() for value in args.trials.split(",")],
        "denominators": [value.strip() for value in args.denominators.split(",")],
        "samples": args.samples,
        "plan": {"kind": plan["kind"], "lower": plan["lower"],
                 "upper": plan["upper"], "count": int(plan["count"])},
    }
    print(f"mesh={args.mesh_mm} dofs={kernel.shape[0]} shift={shift:.6g} "
          f"plan={plan['kind']} shifts={plan['count']} "
          f"lambda=[{plan['lower']:.6g}, {plan['upper']:.6g}]", flush=True)

    clock = time.perf_counter()
    if args.basis == "design":
        power = np.asarray(model.nominal_power(), dtype=np.float64)
        points, selection_certificate, selection = certified_greedy_points(
            kernel, terms, source, ranges, None,
            tolerance=1e-5, maximum_points=3, grid=41, power=power,
            metric="entrywise", progress=False, cache={},
        )
        basis, _, info = build_basis(
            kernel, mass, terms, source, points, plan=plan,
            tolerance=args.cutoff, include_dc=True, cache={},
        )
        report["design"] = {"points": len(points),
                            "selection_certificate": selection_certificate,
                            "order": int(basis.shape[1]),
                            "full_rhs_solves": int(info["full_rhs_solves"])}
    else:
        basis, stats = build_parametric_basis(
            model.core, source, terms, ranges, tolerance=args.cutoff,
            max_order=4096, probe_rounds=10, seed=args.seed,
        )
        report["stock"] = {"order": int(basis.shape[1]),
                           "full_rhs_solves": int(stats["pre_svd_order"])}
    print(f"basis={args.basis} order={basis.shape[1]} t={time.perf_counter()-clock:.1f}s",
          flush=True)

    certificate = BoxCertificate(
        kernel, terms, source, ranges, basis, shift=shift, mass=mass,
        blocks=(args.blocks,) * 2,
    )
    print(f"residual span columns={certificate.span_columns}", flush=True)

    # ------------------------------------------------ box-wide exact reference
    clock = time.perf_counter()
    reference_grid = grid_points(ranges, args.reference_cells)
    reference_values = exact_field(
        kernel, mass, terms, source, basis, reference_grid, shift
    )
    reference_max = float(np.max(reference_values))
    report["exact_reference"] = {
        "cells_per_axis": args.reference_cells,
        "points": len(reference_grid),
        "worst": reference_max,
        "worst_point": [float(value) for value in
                        reference_grid[int(np.argmax(reference_values))]],
        "seconds": time.perf_counter() - clock,
    }
    print(f"exact box reference: {reference_max:.6e} on {len(reference_grid)} points "
          f"t={report['exact_reference']['seconds']:.1f}s", flush=True)

    # ----------------------------------------------------------- partitions
    partitions = []
    trial_sets = [(kind, kind) for kind in report["trials"]]
    if len(report["trials"]) > 1:
        trial_sets.append(("+".join(report["trials"]), tuple(report["trials"])))
    for cells_per_axis in [int(value) for value in args.cells.split(",")]:
        clock = time.perf_counter()
        edges = [
            logarithmic_edges(low, high, cells_per_axis) for low, high in ranges
        ]
        records = []
        gram_cache = {}
        anchor_rules = (
            (("local", True), ("block", False))
            if args.anchors == "both"
            else ((("local", True),) if args.anchors == "local" else (("block", False),))
        )
        for name, local in anchor_rules:
            for choice in itertools.product(
                *[range(cells_per_axis)] * len(ranges)
            ):
                low = np.array([edges[axis][index] for axis, index in enumerate(choice)])
                high = np.array([edges[axis][index + 1] for axis, index in enumerate(choice)])
                if local:
                    gram = certificate.anchor_gram(low)
                else:
                    block = certificate.block_index(low)
                    gram = gram_cache.get(block)
                    if gram is None:
                        gram = certificate.anchor_gram(certificate.block_lower(block))
                        gram_cache[block] = gram
                for denominator in report["denominators"]:
                    weight, fallback = certificate.cell_weight(high, denominator)
                    for order in report["orders"]:
                        for label, kinds in trial_sets:
                            records.append({
                                "cell": choice,
                                "low": low.tolist(),
                                "high": high.tolist(),
                                "anchor_kind": name,
                                "denominator": denominator,
                                "denominator_fallback": fallback,
                                "order": order,
                                "trial": label,
                                "bound": certificate.cell_matrix_bound(
                                    gram, low, high, weight, order, kinds
                                ),
                            })
        # exact worst inside every cell, sampled
        exact_cells = {}
        for choice in itertools.product(*[range(cells_per_axis)] * len(ranges)):
            low = np.array([edges[axis][index] for axis, index in enumerate(choice)])
            high = np.array([edges[axis][index + 1] for axis, index in enumerate(choice)])
            samples = cell_samples(low, high, args.samples)
            values = exact_field(kernel, mass, terms, source, basis, samples, shift)
            exact_cells[choice] = float(np.max(values))
        exact_worst = max(exact_cells.values())
        for record in records:
            record["exact_cell"] = exact_cells[record["cell"]]
        grouped = {}
        for record in records:
            grouped.setdefault(
                (record["order"], record["trial"], record["denominator"],
                 record["anchor_kind"]),
                [],
            ).append(record)
        rows = []
        for (order, trial, denominator, kind), group in sorted(grouped.items()):
            violations = [
                r for r in group if r["bound"] < r["exact_cell"] * (1.0 - 1e-9)
            ]
            dominating = [r for r in group if r["exact_cell"] >= 0.9 * exact_worst]
            ratios = sorted(
                (r["bound"] / r["exact_cell"] for r in group if r["exact_cell"] > 0.0),
                reverse=True,
            )
            worst_cell = max(group, key=lambda r: r["bound"])
            rows.append({
                "order": order,
                "trial": trial,
                "denominator": denominator,
                "anchor_kind": kind,
                "denominator_fallbacks": sum(
                    r["denominator_fallback"] for r in group
                ),
                "worst_bound": worst_cell["bound"],
                "worst_cell": worst_cell["cell"],
                "worst_cell_low": worst_cell["low"],
                "worst_cell_exact": worst_cell["exact_cell"],
                "worst_cell_effectivity": (
                    worst_cell["bound"] / worst_cell["exact_cell"]
                    if worst_cell["exact_cell"] > 0.0 else 0.0
                ),
                "worst_cell_dominates": bool(
                    worst_cell["exact_cell"] >= 0.9 * exact_worst
                ),
                "violations": len(violations),
                "violation_examples": violations[:2],
                "dominating_cells": len(dominating),
                "dominating_worst_bound": max(
                    (r["bound"] for r in dominating), default=0.0
                ),
                "dominating_worst_effectivity": max(
                    (r["bound"] / r["exact_cell"] for r in dominating), default=0.0
                ),
                "effectivity_median": float(np.median(ratios)) if ratios else 0.0,
                "effectivity_worst": ratios[0] if ratios else 0.0,
            })
        partitions.append({
            "cells_per_axis": cells_per_axis,
            "cells": cells_per_axis ** len(ranges),
            "exact_worst_sampled": exact_worst,
            "box_reference": reference_max,
            "rows": rows,
            "seconds": time.perf_counter() - clock,
        })
        for row in rows:
            print(f"cells/axis={cells_per_axis:3d} order={row['order']} "
                  f"trial={row['trial']:9s} den={row['denominator']:7s} "
                  f"anchor={row['anchor_kind']:5s} bound={row['worst_bound']:.6e} "
                  f"exact={exact_worst:.6e} U*/L*={row['worst_bound']/exact_worst:.3f} "
                  f"Ucell/Lcell={row['worst_cell_effectivity']:.3f} "
                  f"Ucell/L*={row['worst_cell_exact']/exact_worst:.3f} "
                  f"dom={int(row['worst_cell_dominates'])} "
                  f"violations={row['violations']} "
                  f"dom_eff={row['dominating_worst_effectivity']:.3f} "
                  f"fb={row['denominator_fallbacks']}", flush=True)
        print(f"   partition t={partitions[-1]['seconds']:.1f}s", flush=True)

    report["partitions"] = partitions

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}", flush=True)
    return report


if __name__ == "__main__":
    main()
