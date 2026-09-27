#!/usr/bin/env python3
"""Certified branch and bound over the HTC box, without enrichment.

The cell certificate gives, for one cell ``Q`` with anchor ``a`` and reduced
denominator ``Y_V(b)``,

    sup_{h in Q} lambda_max(E(h), Y(h)) <= U(Q),

while the exact map gives the true value ``L(h)`` at a single parameter.  A
greedy needs both: an enrichment solve may only be spent where a *witness*
``L(h)`` is above tolerance, while ``U`` decides which cells are unresolved.
This bench runs the refinement loop that keeps the two apart, cheapest first:

* p-refinement inside the leaf (successive trial orders) - no new Gram matrix;
* h-refinement by log-bisecting the widest axis of the leaf that carries the
  largest bound;
* its own anchor only for a leaf that is still unresolved after p-refinement,
  because the enclosing block anchor is valid but loose.

Per threshold it reports the number of leaves, Gram matrices and exact witness
solves needed, and checks that the final accepted cell really contains the box
maximum.  The thresholds are demonstration levels for the refinement behaviour,
not claimed tolerances: the certificate is an upper bound, so no threshold below
the true box maximum can ever be reached.

    PYTHONPATH=python python bench_box_branch_and_bound.py 5 --thresholds 1e-2,1e-3,3e-4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

import bench_matrix_cell_certificate as bench  # noqa: E402
from certified_box import BoxCertificate  # noqa: E402
from deterministic_design import (  # noqa: E402
    build_basis,
    certified_greedy_points,
    frequency_plan,
)
from metahotspot.macromodel.utils import build_parametric_basis  # noqa: E402


def contains(cell, point):
    """True when ``point`` lies in the closed box ``cell = [low, high]``."""
    low, high = (np.asarray(edge, dtype=np.float64) for edge in cell)
    point = np.asarray(point, dtype=np.float64)
    return bool(np.all(point >= low * (1.0 - 1e-12)) and np.all(point <= high * (1.0 + 1e-12)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_mm", type=float, help="mesh size in mm")
    parser.add_argument("--cutoff", type=float, default=1e-3)
    parser.add_argument("--basis", choices=("design", "stock"), default="design")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--blocks", type=int, default=4, help="blocks per axis")
    parser.add_argument("--initial-cells", type=int, default=2)
    parser.add_argument("--orders", default="2,3,4,5")
    parser.add_argument("--trial", default="taylor")
    parser.add_argument("--oracle", action="store_true",
                        help="also compute the exact witness per leaf (full-order solves)")
    parser.add_argument("--upgrade-anchor", action="store_true",
                        help="let an unresolved leaf build its own anchor Gram")
    parser.add_argument("--samples", type=int, default=2,
                        help="log samples per axis inside a witness grid")
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--thresholds", default="1e-2,1e-3,3e-4")
    parser.add_argument("--shifts", default="none",
                        help="none (s=0), all plan shifts, or comma separated indices")
    parser.add_argument("--plan", choices=("box", "legacy"), default="box",
                        help="frequency plan provenance; box is the delivered route")
    parser.add_argument("--span", choices=("shift", "common"), default="shift",
                        help="residual span: per shift, or one shift-free span for all")
    parser.add_argument("--floor-probe", action="store_true",
                        help="evaluate the anchor floor of the worst unresolved leaf")
    parser.add_argument("--reference-cells", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model, kernel, mass, terms, ranges, source = bench.build_model(args.mesh_mm)
    plan = frequency_plan(kernel, mass, terms, source, ranges, args.cutoff, args.plan)
    print(f"mesh={args.mesh_mm} dofs={kernel.shape[0]} shift=0 "
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
        basis_info = {"points": len(points),
                      "selection_certificate": selection_certificate,
                      "full_rhs_solves": int(info["full_rhs_solves"])}
    else:
        basis, stats = build_parametric_basis(
            model.core, source, terms, ranges, tolerance=args.cutoff,
            max_order=4096, probe_rounds=10, seed=args.seed,
        )
        basis_info = {"full_rhs_solves": int(stats["pre_svd_order"])}
    basis_info["order"] = int(basis.shape[1])
    print(f"basis={args.basis} order={basis.shape[1]} t={time.perf_counter()-clock:.1f}s",
          flush=True)

    if args.shifts == "none":
        shift_choices = [(None, 0.0)]
    elif args.shifts == "all":
        shift_choices = list(enumerate(float(value) for value in plan["shifts"]))
    else:
        shift_choices = [
            (int(value), float(plan["shifts"][int(value)]))
            for value in args.shifts.split(",")
        ]

    reference_grid = bench.grid_points(ranges, args.reference_cells)
    reference_values = bench.exact_field(
        kernel, mass, terms, source, basis, reference_grid, 0.0
    )
    reference_max = float(np.max(reference_values))
    reference_point = reference_grid[int(np.argmax(reference_values))]
    print(f"exact box reference at s=0: {reference_max:.6e} on {len(reference_grid)} "
          f"points at {np.array2string(reference_point, precision=4)}", flush=True)

    report = {
        "mesh_mm": args.mesh_mm,
        "cell_count": int(kernel.shape[0]),
        "basis": args.basis,
        "basis_info": basis_info,
        "blocks": args.blocks,
        "initial_cells": args.initial_cells,
        "orders": [int(value) for value in args.orders.split(",")],
        "trial": args.trial,
        "samples": args.samples,
        "span": args.span,
        "plan": {"kind": plan["kind"], "lower": plan["lower"],
                 "upper": plan["upper"], "count": int(plan["count"])},
        "exact_reference": {"worst": reference_max,
                            "worst_point": [float(value) for value in reference_point],
                            "cells_per_axis": args.reference_cells,
                            "points": len(reference_grid)},
        "runs": [],
    }
    shared_grams = {} if args.span == "common" else None
    for shift_index, shift in shift_choices:
        certificate = BoxCertificate(
            kernel, terms, source, ranges, basis, shift=shift, mass=mass,
            blocks=(args.blocks,) * 2, span=args.span,
            gram_cache=shared_grams,
        )
        print(f"shift_index={shift_index} shift={shift:.6e} span={args.span}", flush=True)
        for text in args.thresholds.split(","):
            threshold = float(text)
            clock = time.perf_counter()
            result = certificate.branch_and_bound(
                threshold,
                initial_cells=args.initial_cells,
                orders=[int(value) for value in args.orders.split(",")],
                trial=args.trial,
                samples=args.samples,
                max_cells=args.max_cells,
                max_rounds=args.max_rounds,
                oracle=args.oracle,
                upgrade_anchor=args.upgrade_anchor,
            )
            result["threshold"] = threshold
            result["shift"] = shift
            result["shift_index"] = shift_index
            result["wall_seconds"] = time.perf_counter() - clock
            result["accepted"] = bool(result["accepted"])
            result["reference"] = reference_max if shift == 0.0 else None
            if args.floor_probe:
                other_span = "shift" if args.span == "common" else "common"
                counterpart = BoxCertificate(
                    kernel, terms, source, ranges, basis, shift=shift, mass=mass,
                    blocks=(args.blocks,) * 2, span=other_span,
                )
                low, high = (np.array(value, dtype=float) for value in result["worst_cell"])
                points = [0.5 * (low + high)]
                points.extend(
                    np.array([low[axis] if (bit >> axis) & 1 == 0 else high[axis]
                              for axis in range(low.size)])
                    for bit in range(2 ** low.size)
                )
                probe = {}
                for name, candidate in ((args.span, certificate), (other_span, counterpart)):
                    block = candidate.block_index(low)
                    gram = candidate.anchor_gram(candidate.block_lower(block))
                    samples_at = [candidate.point_floor(gram, point) for point in points]
                    probe[name] = {"center": samples_at[0],
                                   "min": float(min(samples_at)),
                                   "max": float(max(samples_at))}
                result["floor"] = probe
                print("  floor probe on the worst leaf "
                      f"{[f'{value:.3e}' for value in low]}..{[f'{value:.3e}' for value in high]}: "
                      + " ".join(f"F_B[{name}]={probe[name]['min']:.4e}..{probe[name]['max']:.4e} "
                                 f"(center {probe[name]['center']:.4e})" for name in probe),
                      flush=True)
            if shift == 0.0:
                result["contains_reference_worst"] = contains(
                    result["worst_cell"], reference_point
                )
                result["bound_over_exact"] = result["bound"] / reference_max
            report["runs"].append(result)
            witness = result["witness"]
            ratio = result.get("bound_over_exact")
            print(f"shift={shift:.4e} threshold={threshold:.3e} "
                  f"accepted={result['accepted']} "
                  f"splits={result['splits']} leaves={result['leaves']} "
                  f"created={result['leaves_created']} measures={result['measurements']} "
                  f"grams={result['gram_matrices']} witnesses={result['witness_solves']} "
                  f"bound={result['bound']:.4e} "
                  f"witness={'n/a' if witness is None else f'{witness:.4e}'} "
                  f"ratio={'n/a' if ratio is None else f'{ratio:.3f}'} "
                  f"t={result['wall_seconds']:.1f}s", flush=True)
            for entry in result["history"]:
                seen = entry["worst_witness"]
                print(f"    step={entry['round']:3d} {entry['action']:6s} "
                      f"leaves={entry['leaves']:3d} anchors={entry['anchors']:3d} "
                      f"order={entry['worst_order']} "
                      f"worst_bound={entry['worst_bound']:.4e} "
                      f"worst_witness={'n/a' if seen is None else f'{seen:.4e}'}",
                      flush=True)

    if args.output:
        report["shared_anchor_grams"] = None if shared_grams is None else len(shared_grams)
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {args.output}", flush=True)
        if shared_grams is not None:
            print(f"shared anchor Grams across the plan: {len(shared_grams)}", flush=True)


if __name__ == "__main__":
    main()
