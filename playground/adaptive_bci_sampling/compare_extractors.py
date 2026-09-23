#!/usr/bin/env python3
"""Compare random and closed-form HTC choices in the full extraction loop.

Unlike ``compare_sampling.py``, this script retains the production MPMM shift
plan.  The only changed variable is the choice of the affine HTC vector:

* ``random`` is the current residual-driven Extended FANTASTIC extractor;
* ``padua-log`` solves every source/shift pair at the same explicit Padua set.

Both paths use the production normalized-snapshot closing SVD and preserve the
uniform-temperature mode.  Accuracy is reported only at the junctions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "bci_rom_testcase1"))

from compare_sampling import (  # noqa: E402
    map_box,
    padua_points,
    reduced_junction,
    response,
    smolyak_points,
    snapshot_basis,
    validation_points,
)
from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    build_parametric_basis,
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
    port_eigenvalue_bounds,
    spd_solve,
)


def deterministic_basis(
    model,
    unit_points: np.ndarray,
    extraction_tolerance: float,
    closing_tolerance: float,
):
    K = model.core.K.tocsc()
    C = model.core.C.tocsc()
    G = np.asarray(model.source_shape, dtype=np.float64)
    terms = [term.tocsc() for term in model.boundary_terms]
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    samples = map_box(unit_points, ranges, logarithmic=True)
    snapshots = []
    shift_counts = []

    started = time.perf_counter()
    for port in range(G.shape[1]):
        g = G[:, port]
        lambda_min, lambda_max = port_eigenvalue_bounds(K, C, g)
        count = mpmm_elliptic_shift_count(
            extraction_tolerance, lambda_min, lambda_max
        )
        shifts = mpmm_elliptic_shifts(count, lambda_max, lambda_max / lambda_min)
        shift_counts.append(count)
        for shift in shifts:
            for h in samples:
                A = K + float(shift) * C
                for value, term in zip(h, terms):
                    A = A + float(value) * term
                snapshots.append(
                    np.asarray(spd_solve(A.tocsc(), g), dtype=np.float64).ravel()
                )
    basis = snapshot_basis(
        [snapshot[:, None] for snapshot in snapshots], tolerance=closing_tolerance
    )
    return basis, {
        "parameter_points": int(samples.shape[0]),
        "pre_svd_order": len(snapshots),
        "basis_order": int(basis.shape[1]),
        "shift_counts": shift_counts,
        "seconds": time.perf_counter() - started,
    }


def worst_junction_error(K, terms, G, power, basis, holdout, exact_junctions):
    worst = -1.0
    worst_h = None
    for h, exact in zip(holdout, exact_junctions):
        approximate = reduced_junction(K, terms, G, power, basis, h)
        relative = np.max(
            np.abs(approximate - exact)
            / np.maximum(np.abs(exact), np.finfo(float).tiny)
        )
        if relative > worst:
            worst = float(relative)
            worst_h = tuple(float(value) for value in h)
    return worst, worst_h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_mm", nargs="?", type=float, default=2.5)
    parser.add_argument("--degrees", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--smolyak-levels", nargs="*", type=int, default=[])
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument(
        "--closing-tolerance",
        type=float,
        default=None,
        help="deterministic closing-SVD cutoff (default: --tolerance)",
    )
    parser.add_argument("--probe-rounds", type=int, default=10)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--validation-grid", type=int, default=21)
    parser.add_argument("--random-holdout", type=int, default=128)
    args = parser.parse_args()

    model = Case1Model(
        Case1Config(max_xy_cell_mm=args.mesh_mm, max_z_cell_mm=args.mesh_mm)
    )
    K = model.core.K.tocsc()
    terms = [term.tocsc() for term in model.boundary_terms]
    G = np.asarray(model.source_shape, dtype=np.float64)
    power = np.asarray(model.nominal_power(), dtype=np.float64)
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    holdout = validation_points(ranges, args.validation_grid, args.random_holdout)

    print(
        f"mesh={args.mesh_mm:g} mm cells={K.shape[0]} sources={G.shape[1]} "
        f"groups={ranges.shape[0]} holdout={holdout.shape[0]}"
    )
    exact_junctions = []
    started = time.perf_counter()
    for index, h in enumerate(holdout, start=1):
        exact_junctions.append(G.T @ (response(K, terms, G, h) @ power))
        if index % 100 == 0:
            print(f"  exact holdout {index}/{holdout.shape[0]}", flush=True)
    exact_junctions = np.asarray(exact_junctions)
    print(f"exact holdout seconds={time.perf_counter() - started:.2f}")

    for degree in args.degrees:
        closing_tolerance = (
            args.tolerance
            if args.closing_tolerance is None
            else args.closing_tolerance
        )
        basis, summary = deterministic_basis(
            model, padua_points(degree), args.tolerance, closing_tolerance
        )
        error, at = worst_junction_error(
            K, terms, G, power, basis, holdout, exact_junctions
        )
        print(
            f"padua-log degree={degree} closing={closing_tolerance:g} "
            f"points={summary['parameter_points']} "
            f"snapshots={summary['pre_svd_order']} order={summary['basis_order']} "
            f"extract={summary['seconds']:.2f}s worst={100.0 * error:.5f}% at {at}",
            flush=True,
        )

    for level in args.smolyak_levels:
        basis, summary = deterministic_basis(
            model, smolyak_points(level), args.tolerance, closing_tolerance
        )
        error, at = worst_junction_error(
            K, terms, G, power, basis, holdout, exact_junctions
        )
        print(
            f"smolyak-log level={level} closing={closing_tolerance:g} "
            f"points={summary['parameter_points']} "
            f"snapshots={summary['pre_svd_order']} order={summary['basis_order']} "
            f"extract={summary['seconds']:.2f}s worst={100.0 * error:.5f}% at {at}",
            flush=True,
        )

    random_results = []
    for seed_offset in range(args.random_seeds):
        seed = 20260805 + seed_offset
        started = time.perf_counter()
        basis, summary = build_parametric_basis(
            model.core,
            G,
            terms,
            ranges,
            tolerance=args.tolerance,
            max_order=1024,
            probe_rounds=args.probe_rounds,
            seed=seed,
        )
        elapsed = time.perf_counter() - started
        error, at = worst_junction_error(
            K, terms, G, power, basis, holdout, exact_junctions
        )
        random_results.append((error, summary["pre_svd_order"], basis.shape[1]))
        print(
            f"random seed={seed} snapshots={summary['pre_svd_order']} "
            f"candidates={summary['candidate_count']} order={basis.shape[1]} "
            f"extract={elapsed:.2f}s worst={100.0 * error:.5f}% at {at}",
            flush=True,
        )
    if random_results:
        errors = np.asarray([item[0] for item in random_results])
        solves = [item[1] for item in random_results]
        orders = [item[2] for item in random_results]
        print(
            f"random summary seeds={len(random_results)} "
            f"snapshots={min(solves)}..{max(solves)} order={min(orders)}..{max(orders)} "
            f"worst min/median/max={100.0 * errors.min():.5f}%/"
            f"{100.0 * np.median(errors):.5f}%/{100.0 * errors.max():.5f}%"
        )


if __name__ == "__main__":
    main()
