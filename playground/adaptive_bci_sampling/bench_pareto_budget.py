#!/usr/bin/env python3
"""Extraction budget, snapshot compression and the certified matching defect.

The dynamic bridge is not yet closed, so no vendor-derived stopping threshold
exists; what can be stated now is a Pareto table.  For every combination of

    (parameter points requested, SVD cutoff)

the delivered basis ``V`` of :func:`deterministic_design.build_basis` is rebuilt
from the *same* cached snapshots - lowering the cutoff therefore costs no further
full-order solve - and certified on the delivered, box-corrected frequency plan
shift by shift over the whole box:

    delta_cert(V) = sqrt( max_j max_Q U_{Q,j}(V) ),      sqrt(L*) = sampled exact.

Counting convention, shared with the rest of the repository: one *operator block* is
one factorization of ``A(h, s)``, and every block serves one right-hand side per
port, so

    N_RHS = number_of_ports * (selection_blocks + build_blocks).

Only ``N_RHS`` may be compared with the stock extractor's RHS counter.  The
selection's residual certificate and the box certificate itself are certification
overhead: reported separately, never added to ``N_RHS``.  ``tau_moment`` stays
symbolic throughout; there is no threshold in this script.

    PYTHONPATH=python python playground/adaptive_bci_sampling/bench_pareto_budget.py 5 \\
        --points 2,3,8 --cutoffs 1e-3,3e-4,1e-4 --witness --output <path>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

from certified_box import BoxCertificate  # noqa: E402
from deterministic_design import box_frequency_plan, build_basis, certified_greedy_points  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import build_parametric_basis  # noqa: E402


def witness_max(certificate, ranges, cells):
    """Largest exact matching defect over a lattice of the box (research mode)."""
    edges = [np.linspace(low, high, cells + 1) for low, high in ranges]
    worst = 0.0
    for first in edges[0]:
        for second in edges[1]:
            point = np.array([first, second], dtype=np.float64)
            value, _ = certificate.defect(point)
            worst = max(worst, float(value))
    return float(worst)


def certify(kernel, terms, source, ranges, basis, shifts, arguments, mass):
    """The box certificate over every shift of the plan, plus the sampled witness."""
    per_shift = {}
    worst_bound = 0.0
    worst_witness = 0.0
    anchors = 0
    started = time.perf_counter()
    for shift in shifts:
        certificate = BoxCertificate(
            kernel, terms, source, ranges, basis,
            shift=shift, mass=mass, blocks=(arguments.blocks,) * 2,
        )
        sweep = certificate.sweep_matrix(arguments.cells, order=arguments.order,
                                         trial=arguments.trial)
        bound = float(np.sqrt(sweep["worst_relative_energy_bound"]))
        entry = {"bound": bound, "anchors": int(sweep["anchors"]),
                 "cells": int(sweep["cells"]), "seconds": float(sweep["seconds"])}
        if arguments.witness:
            entry["witness"] = float(np.sqrt(witness_max(certificate, ranges,
                                                         arguments.witness_cells)))
            worst_witness = max(worst_witness, entry["witness"])
        per_shift[f"{shift:.6e}"] = entry
        worst_bound = max(worst_bound, bound)
        anchors += int(sweep["anchors"])
    return {"per_shift": per_shift, "delta_cert": worst_bound,
            "witness": worst_witness if arguments.witness else None,
            "effectivity": worst_bound / worst_witness if worst_witness else None,
            "certificate_anchors": int(anchors),
            "certificate_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mesh_mm", type=float)
    parser.add_argument("--points", default="2,3,8",
                        help="requested greedy point counts; the last should saturate")
    parser.add_argument("--cutoffs", default="1e-3,3e-4,1e-4,3e-5,1e-5")
    parser.add_argument("--greedy-tolerance", type=float, default=1e-5)
    parser.add_argument("--greedy-grid", type=int, default=41)
    parser.add_argument("--cells", type=int, default=16, help="certificate cells per axis")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--trial", default="taylor")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--witness-cells", type=int, default=2)
    parser.add_argument("--witness", action="store_true", help="enable the exact witness")
    parser.add_argument("--stock-seeds", default="")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    model = Case1Model(Case1Config(max_xy_cell_mm=arguments.mesh_mm,
                                   max_z_cell_mm=arguments.mesh_mm))
    kernel = model.core.K.tocsc()
    mass = model.core.C.tocsc()
    source = np.asarray(model.source_shape, dtype=np.float64)
    terms = [term.tocsc() for term in model.boundary_terms]
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    power = np.asarray(model.nominal_power(), dtype=np.float64)
    ports = int(source.shape[1])

    plan = box_frequency_plan(kernel, mass, terms, ranges, 1e-3)
    shifts = [float(value) for value in plan["shifts"]] + [0.0]
    print(f"mesh={arguments.mesh_mm}mm cells={kernel.shape[0]} plan={plan['kind']} "
          f"shifts={len(shifts)} ports={ports} "
          f"kappa={plan['upper'] / plan['lower']:.4e}", flush=True)

    cache: dict = {}
    report = {"mesh_mm": arguments.mesh_mm, "cells": int(kernel.shape[0]), "ports": ports,
              "plan": {k: plan[k] for k in ("kind", "lower", "upper", "count")},
              "shifts": shifts, "runs": [], "stock": []}
    cumulative_blocks = 0
    cumulative_rhs = 0

    for count in [int(value) for value in arguments.points.split(",")]:
        selection_started = time.perf_counter()
        points, selection_certificate, selection = certified_greedy_points(
            kernel, terms, source, ranges, None,
            tolerance=arguments.greedy_tolerance, maximum_points=count,
            grid=arguments.greedy_grid, power=power, metric="entrywise", cache=cache,
        )
        selection_seconds = time.perf_counter() - selection_started
        selection_blocks = int(selection["factorizations"])
        selection_rhs = int(selection["fresh_rhs_solves"])
        cumulative_blocks += selection_blocks
        cumulative_rhs += selection_rhs
        print(f"points_requested={count} points_selected={len(points)} "
              f"selection_blocks={selection_blocks} selection_rhs={selection_rhs} "
              f"selection_certificate={selection_certificate:.3e} "
              f"t={selection_seconds:.1f}s", flush=True)
        for cutoff in [float(value) for value in arguments.cutoffs.split(",")]:
            started = time.perf_counter()
            basis, _, info = build_basis(
                kernel, mass, terms, source, points, plan=plan,
                tolerance=cutoff, include_dc=True, cache=cache,
            )
            build_blocks = int(info["factorizations"])
            build_rhs = build_blocks * ports
            cumulative_blocks += build_blocks
            cumulative_rhs += build_rhs
            row = {
                "points_requested": int(count),
                "points_selected": int(len(points)),
                "cutoff": float(cutoff),
                "selection_blocks": selection_blocks,
                "selection_rhs_solves": selection_rhs,
                "build_blocks": build_blocks,
                "build_rhs_solves": build_rhs,
                "cumulative_blocks": cumulative_blocks,
                "cumulative_rhs_solves": cumulative_rhs,
                "snapshot_columns": int(info["full_rhs_solves"]),
                "rom_order": int(basis.shape[1]),
                "svd_kept_order": int(info["svd_kept_order"]),
                "selection_certificate": float(selection_certificate),
            }
            row.update(certify(kernel, terms, source, ranges, basis, shifts,
                               arguments, mass))
            row["seconds"] = time.perf_counter() - started
            report["runs"].append(row)
            print(f"points={len(points)} cutoff={cutoff:.1e} "
                  f"fresh_blocks={selection_blocks + build_blocks} "
                  f"fresh_rhs={selection_rhs + build_rhs} "
                  f"cumulative_rhs={cumulative_rhs} order={row['rom_order']} "
                  f"delta_cert={row['delta_cert']:.6e} witness={row['witness']} "
                  f"effectivity={row['effectivity']} "
                  f"cert_seconds={row['certificate_seconds']:.1f}", flush=True)

    for seed in [int(value) for value in arguments.stock_seeds.split(",") if value]:
        started = time.perf_counter()
        basis, stats = build_parametric_basis(
            model.core, source, terms, ranges,
            tolerance=1e-3, max_order=4096, probe_rounds=10, seed=seed,
        )
        stats["seconds"] = time.perf_counter() - started
        row = {"seed": int(seed), "rom_order": int(basis.shape[1]),
               "rhs_solves": int(stats["pre_svd_order"]),
               "candidate_count": int(stats["candidate_count"]),
               "shift_count": int(stats["frequency_plan"]["shift_count"]),
               "plan_interval": [float(stats["frequency_plan"]["lambda_min"]),
                                 float(stats["frequency_plan"]["lambda_max"])]}
        row.update(certify(kernel, terms, source, ranges, basis, shifts,
                           arguments, mass))
        report["stock"].append(row)
        print(f"stock seed={seed} rhs_solves={row['rhs_solves']} order={row['rom_order']} "
              f"delta_cert={row['delta_cert']:.6e} witness={row['witness']} "
              f"effectivity={row['effectivity']}", flush=True)

    if arguments.output:
        arguments.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
