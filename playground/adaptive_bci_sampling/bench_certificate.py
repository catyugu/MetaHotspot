#!/usr/bin/env python3
"""Executable evidence for the fixed-shift port certificate, two modes.

Both modes study the same machinery on the same model; they differ only in the
question they answer, which is why they live in one script:

``--mode tightness``
    Is the matrix cell certificate correct and tight?  On one fixed real shift it
    compares the certified bound with the exact port defect inside every cell of a
    uniform log partition, over both anchor rules, several trial orders and both
    denominators, and reports the three things that decide whether the certificate
    can drive a greedy:

    1. violations: a cell bound below the exact worst value of the same cell;
    2. refinement monotonicity: the bound must fall as the partition is refined;
    3. effectivity on the cells that dominate the exact maximum, against the
       pre-registered gate "after refinement, the dominating cells are within 10x".

``--mode bandb``
    Can the continuous box statement be decided without enrichment, and does the
    refinement loop beat a uniform partition?  The loop refines cheapest first
    (p-refinement inside a leaf, then log-bisecting the widest axis of the leaf
    carrying the largest bound), keeping the certificate ``U`` and the exact
    witness ``L`` apart: an enrichment solve may only be spent where a witness is
    above tolerance, while ``U`` decides which cells are unresolved.  The
    thresholds are demonstration levels, not adopted tolerances.

Every quantity is the relative all-input collocated port defect
``lambda_max(E(h), Y(h))`` at one fixed shift - ``[PORT-FIXED-S]``.  No square root
is taken, and the equal A-energy state error is not reported.

    PYTHONPATH=python python bench_certificate.py tightness 5 --samples 7
    PYTHONPATH=python python bench_certificate.py bandb 5 --thresholds 1e-2,1e-3,3e-4
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

from sparse_solve import AmgSolver  # noqa: E402
from certified_box import (  # noqa: E402
    BoxCertificate,
    generalized_max,
    logarithmic_edges,
)
from deterministic_design import (  # noqa: E402
    build_basis,
    certified_greedy_points,
    frequency_plan,
)
from metahotspot.macromodel.utils import build_parametric_basis  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402


# --------------------------------------------------------------- shared helpers


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
    """Exact ``(E(h), Y(h))`` of one operator."""
    factor = AmgSolver(operator)
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


def build_delivered_basis(arguments, model, kernel, mass, terms, ranges, source, plan):
    """The design basis, or the stock extractor's, with its solve count."""
    clock = time.perf_counter()
    if arguments.basis == "design":
        points, selection_certificate, _selection = certified_greedy_points(
            kernel, terms, source, ranges, None,
            tolerance=1e-5, maximum_points=3, grid=41,
            progress=False, cache={},
        )
        basis, _, info = build_basis(
            kernel, mass, terms, source, points, plan=plan,
            tolerance=arguments.cutoff, include_dc=True, cache={},
        )
        record = {"points": len(points),
                  "selection_certificate": selection_certificate,
                  "order": int(basis.shape[1]),
                  "full_rhs_solves": int(info["full_rhs_solves"])}
    else:
        basis, stats = build_parametric_basis(
            model.core, source, terms, ranges, tolerance=arguments.cutoff,
            max_order=4096, probe_rounds=10, seed=arguments.seed,
        )
        record = {"order": int(basis.shape[1]),
                  "full_rhs_solves": int(stats["pre_svd_order"])}
    print(f"basis={arguments.basis} order={basis.shape[1]} "
          f"t={time.perf_counter() - clock:.1f}s", flush=True)
    return basis, record


def add_common_arguments(parser):
    parser.add_argument("mesh_mm", nargs="?", type=float, default=5.0)
    parser.add_argument("--basis", choices=("design", "stock"), default="design")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--cutoff", type=float, default=1e-3)
    parser.add_argument("--blocks", type=int, default=4, help="blocks per axis")
    parser.add_argument("--reference-cells", type=int, default=24,
                        help="cells per axis of the box-wide exact reference")
    parser.add_argument("--output", type=Path)
    return parser


# ------------------------------------------------------------------ mode: tightness


def run_tightness(arguments):
    model, kernel, mass, terms, ranges, source = build_model(arguments.mesh_mm)
    plan = frequency_plan(kernel, mass, source, arguments.cutoff)
    shift = 0.0 if arguments.shift_index is None else float(plan["shifts"][arguments.shift_index])
    report = {
        "mode": "tightness",
        "mesh_mm": arguments.mesh_mm,
        "cells": int(kernel.shape[0]),
        "basis": arguments.basis,
        "seed": arguments.seed,
        "shift": shift,
        "orders": [int(value) for value in arguments.orders.split(",")],
        "trials": [value.strip() for value in arguments.trials.split(",")],
        "denominators": [value.strip() for value in arguments.denominators.split(",")],
        "samples": arguments.samples,
        "plan": {"lower": plan["lower"],
                 "upper": plan["upper"], "count": int(plan["count"])},
    }
    print(f"mesh={arguments.mesh_mm} dofs={kernel.shape[0]} shift={shift:.6g} "
          f"plan={plan['kind']} shifts={plan['count']} "
          f"lambda=[{plan['lower']:.6g}, {plan['upper']:.6g}]", flush=True)

    basis, basis_record = build_delivered_basis(
        arguments, model, kernel, mass, terms, ranges, source, plan
    )
    report[arguments.basis] = basis_record

    certificate = BoxCertificate(
        kernel, terms, source, ranges, basis, shift=shift, mass=mass,
        blocks=(arguments.blocks,) * 2,
    )
    print(f"residual span columns={certificate.span_columns}", flush=True)

    clock = time.perf_counter()
    reference_grid = grid_points(ranges, arguments.reference_cells)
    reference_values = exact_field(
        kernel, mass, terms, source, basis, reference_grid, shift
    )
    reference_max = float(np.max(reference_values))
    report["exact_reference"] = {
        "cells_per_axis": arguments.reference_cells,
        "points": len(reference_grid),
        "worst": reference_max,
        "worst_point": [float(value) for value in
                        reference_grid[int(np.argmax(reference_values))]],
        "seconds": time.perf_counter() - clock,
    }
    print(f"exact box reference: {reference_max:.6e} on {len(reference_grid)} points "
          f"t={report['exact_reference']['seconds']:.1f}s", flush=True)

    partitions = []
    trial_sets = [(kind, kind) for kind in report["trials"]]
    if len(report["trials"]) > 1:
        trial_sets.append(("+".join(report["trials"]), tuple(report["trials"])))
    for cells_per_axis in [int(value) for value in arguments.cells.split(",")]:
        clock = time.perf_counter()
        edges = [logarithmic_edges(low, high, cells_per_axis) for low, high in ranges]
        records = []
        gram_cache = {}
        anchor_rules = (
            (("local", True), ("block", False))
            if arguments.anchors == "both"
            else ((("local", True),) if arguments.anchors == "local" else (("block", False),))
        )
        for name, local in anchor_rules:
            for choice in itertools.product(*[range(cells_per_axis)] * len(ranges)):
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
                                "bound": certificate.cell_bound(
                                    gram, low, high, weight, order, kinds
                                ),
                            })
        exact_cells = {}
        for choice in itertools.product(*[range(cells_per_axis)] * len(ranges)):
            low = np.array([edges[axis][index] for axis, index in enumerate(choice)])
            high = np.array([edges[axis][index + 1] for axis, index in enumerate(choice)])
            samples = cell_samples(low, high, arguments.samples)
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
            violations = [r for r in group if r["bound"] < r["exact_cell"] * (1.0 - 1e-9)]
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
                "denominator_fallbacks": sum(r["denominator_fallback"] for r in group),
                "worst_bound": worst_cell["bound"],
                "worst_cell": worst_cell["cell"],
                "worst_cell_low": worst_cell["low"],
                "worst_cell_exact": worst_cell["exact_cell"],
                "worst_cell_effectivity": (
                    worst_cell["bound"] / worst_cell["exact_cell"]
                    if worst_cell["exact_cell"] > 0.0 else 0.0
                ),
                "worst_cell_dominates": bool(worst_cell["exact_cell"] >= 0.9 * exact_worst),
                "violations": len(violations),
                "violation_examples": violations[:2],
                "dominating_cells": len(dominating),
                "dominating_worst_bound": max((r["bound"] for r in dominating), default=0.0),
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
    return report


# --------------------------------------------------------------------- mode: bandb


def contains(cell, point):
    """True when ``point`` lies in the closed box ``cell = [low, high]``."""
    low, high = (np.asarray(edge, dtype=np.float64) for edge in cell)
    point = np.asarray(point, dtype=np.float64)
    return bool(np.all(point >= low * (1.0 - 1e-12)) and np.all(point <= high * (1.0 + 1e-12)))


def run_bandb(arguments):
    model, kernel, mass, terms, ranges, source = build_model(arguments.mesh_mm)
    plan = frequency_plan(kernel, mass, source, arguments.cutoff)
    print(f"mesh={arguments.mesh_mm} dofs={kernel.shape[0]} shift=0 "
          f"shifts={plan['count']} "
          f"lambda=[{plan['lower']:.6g}, {plan['upper']:.6g}]", flush=True)

    basis, basis_record = build_delivered_basis(
        arguments, model, kernel, mass, terms, ranges, source, plan
    )

    if arguments.shifts == "none":
        shift_choices = [(None, 0.0)]
    elif arguments.shifts == "all":
        shift_choices = list(enumerate(float(value) for value in plan["shifts"]))
    else:
        shift_choices = [
            (int(value), float(plan["shifts"][int(value)]))
            for value in arguments.shifts.split(",")
        ]

    reference_grid = grid_points(ranges, arguments.reference_cells)
    reference_values = exact_field(kernel, mass, terms, source, basis, reference_grid, 0.0)
    reference_max = float(np.max(reference_values))
    reference_point = reference_grid[int(np.argmax(reference_values))]
    print(f"exact box reference at s=0: {reference_max:.6e} on {len(reference_grid)} "
          f"points at {np.array2string(reference_point, precision=4)}", flush=True)

    report = {
        "mode": "bandb",
        "mesh_mm": arguments.mesh_mm,
        "cell_count": int(kernel.shape[0]),
        "basis": arguments.basis,
        "basis_info": basis_record,
        "blocks": arguments.blocks,
        "initial_cells": arguments.initial_cells,
        "orders": [int(value) for value in arguments.orders.split(",")],
        "trial": arguments.trial,
        "samples": arguments.samples,
        "span": arguments.span,
        "plan": {"lower": plan["lower"],
                 "upper": plan["upper"], "count": int(plan["count"])},
        "exact_reference": {"worst": reference_max,
                            "worst_point": [float(value) for value in reference_point],
                            "cells_per_axis": arguments.reference_cells,
                            "points": len(reference_grid)},
        "runs": [],
    }
    shared_grams = {} if arguments.span == "common" else None
    for shift_index, shift in shift_choices:
        certificate = BoxCertificate(
            kernel, terms, source, ranges, basis, shift=shift, mass=mass,
            blocks=(arguments.blocks,) * 2, span=arguments.span,
            gram_cache=shared_grams,
        )
        print(f"shift_index={shift_index} shift={shift:.6e} span={arguments.span}", flush=True)
        for text in arguments.thresholds.split(","):
            threshold = float(text)
            clock = time.perf_counter()
            result = certificate.branch_and_bound(
                threshold,
                initial_cells=arguments.initial_cells,
                orders=[int(value) for value in arguments.orders.split(",")],
                trial=arguments.trial,
                samples=arguments.samples,
                max_cells=arguments.max_cells,
                max_rounds=arguments.max_rounds,
                oracle=arguments.oracle,
                upgrade_anchor=arguments.upgrade_anchor,
            )
            result["threshold"] = threshold
            result["shift"] = shift
            result["shift_index"] = shift_index
            result["wall_seconds"] = time.perf_counter() - clock
            result["accepted"] = bool(result["accepted"])
            result["reference"] = reference_max if shift == 0.0 else None
            if arguments.floor_probe:
                other_span = "shift" if arguments.span == "common" else "common"
                counterpart = BoxCertificate(
                    kernel, terms, source, ranges, basis, shift=shift, mass=mass,
                    blocks=(arguments.blocks,) * 2, span=other_span,
                )
                low, high = (np.array(value, dtype=float) for value in result["worst_cell"])
                points = [0.5 * (low + high)]
                points.extend(
                    np.array([low[axis] if (bit >> axis) & 1 == 0 else high[axis]
                              for axis in range(low.size)])
                    for bit in range(2 ** low.size)
                )
                probe = {}
                for name, candidate in ((arguments.span, certificate), (other_span, counterpart)):
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
                result["contains_reference_worst"] = contains(result["worst_cell"], reference_point)
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
                      f"witness={'n/a' if seen is None else f'{seen:.4e}'}", flush=True)

    report["shared_anchor_grams"] = None if shared_grams is None else len(shared_grams)
    if shared_grams is not None:
        print(f"shared anchor Grams across the plan: {len(shared_grams)}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tightness", "bandb"))
    add_common_arguments(parser)
    parser.add_argument("--shift-index", type=int, default=None,
                        help="tightness: plan shift to certify; omit for the steady operator")
    parser.add_argument("--cells", default="2,4,8",
                        help="tightness: comma separated cells per axis to compare")
    parser.add_argument("--samples", type=int, default=7,
                        help="interior samples per axis inside a cell")
    parser.add_argument("--orders", default=None,
                        help="comma separated Taylor jet orders of the bound")
    parser.add_argument("--trials", default="taylor",
                        help="tightness: comma separated polynomial trials")
    parser.add_argument("--denominators", default="reduced",
                        help="tightness: comma separated cell denominators: reduced,exact")
    parser.add_argument("--anchors", default="both",
                        help="tightness: anchor rules to evaluate: block, local or both")
    parser.add_argument("--initial-cells", type=int, default=2,
                        help="bandb: cells per axis of the initial partition")
    parser.add_argument("--trial", default="taylor", help="bandb: polynomial trial")
    parser.add_argument("--oracle", action="store_true",
                        help="bandb: also compute the exact witness per leaf")
    parser.add_argument("--upgrade-anchor", action="store_true",
                        help="bandb: let an unresolved leaf build its own anchor Gram")
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--thresholds", default="1e-2,1e-3,3e-4")
    parser.add_argument("--shifts", default="none",
                        help="bandb: none (s=0), all plan shifts, or comma separated indices")
    parser.add_argument("--span", choices=("shift", "common"), default="shift",
                        help="bandb: residual span, per shift or one shift-free span for all")
    parser.add_argument("--floor-probe", action="store_true",
                        help="bandb: evaluate the anchor floor of the worst unresolved leaf")
    arguments = parser.parse_args()

    if arguments.mode == "tightness":
        arguments.orders = arguments.orders or "2"
        report = run_tightness(arguments)
    else:
        arguments.orders = arguments.orders or "2,3,4,5"
        report = run_bandb(arguments)

    if arguments.output:
        arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
