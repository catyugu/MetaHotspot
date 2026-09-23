#!/usr/bin/env python3
"""Compare deterministic HTC sampling with the reproduced FANTASTIC loop.

Both extractors use the same Case 1 operators, four individual heat ports,
per-port elliptic frequency shifts, tolerance, and normalized snapshot SVD.
The holdout tests the full four-by-four junction step-response matrix and the
nominal-power temperature history against the same full-order BDF1 solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "bci_rom_testcase1")]

from adaptive_zolotarev import adaptive_basis  # noqa: E402
from certified_greedy import prepare_residual_certificate  # noqa: E402
from compare_zolotarev import logarithmic_tensor_grid, tensor_product  # noqa: E402
from compare_zolotarev import certified_greedy_design  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    _snapshot_svd_basis,
    build_parametric_basis,
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
    orthonormalize_block,
    port_eigenvalue_bounds,
    spd_solve,
)
from zolotarev import coordinate_spectral_enclosures, zolotarev_rule  # noqa: E402


def step_transfer(K, C, G, *, dt: float, duration: float) -> np.ndarray:
    """BDF1 response to four independent unit power steps, zero initial rise."""
    if dt <= 0 or duration <= 0 or abs(duration / dt - round(duration / dt)) > 1e-10:
        raise ValueError("duration must be a positive whole number of positive steps")
    K = sp.csc_matrix(K)
    C = sp.csc_matrix(C)
    G = np.asarray(G, dtype=np.float64)
    factor = spla.splu(K + C / dt)
    state = np.zeros_like(G)
    outputs = np.zeros((round(duration / dt) + 1, G.shape[1], G.shape[1]))
    for i in range(1, outputs.shape[0]):
        state = factor.solve(G + C @ state / dt)
        outputs[i] = G.T @ state
    return outputs


def evaluate_step_case(K, C, G, power, bases, *, dt: float, duration: float):
    """Use each junction's same-parameter steady rise as the denominator."""
    G = np.asarray(G, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    exact_steady = G.T @ spla.splu(sp.csc_matrix(K)).solve(G)
    exact_step = step_transfer(K, C, G, dt=dt, duration=duration)
    tiny = np.finfo(float).tiny
    results = []
    for basis in bases:
        projected_source = basis.T @ G
        reduced_K = basis.T @ (K @ basis)
        reduced_C = basis.T @ (C @ basis)
        steady_coefficients = scipy.linalg.solve(
            reduced_K, projected_source, assume_a="pos", check_finite=False
        )
        rom_steady = projected_source.T @ steady_coefficients
        cholesky = scipy.linalg.cho_factor(
            reduced_K + reduced_C / dt, check_finite=False
        )
        state = np.zeros_like(projected_source)
        rom_step = np.zeros_like(exact_step)
        for i in range(1, rom_step.shape[0]):
            state = scipy.linalg.cho_solve(
                cholesky, projected_source + reduced_C @ state / dt,
                check_finite=False,
            )
            rom_step[i] = projected_source.T @ state
        relative_nominal = (
            np.abs((rom_step - exact_step) @ power)
            / np.maximum(np.abs(exact_steady @ power), tiny)
        )
        relative_entrywise = (
            np.abs(rom_step - exact_step)
            / np.maximum(np.abs(exact_steady), tiny)
        )
        relative_steady_nominal = (
            np.abs((rom_steady - exact_steady) @ power)
            / np.maximum(np.abs(exact_steady @ power), tiny)
        )
        relative_steady_entrywise = (
            np.abs(rom_steady - exact_steady)
            / np.maximum(np.abs(exact_steady), tiny)
        )
        results.append({
            "worst_nominal_step": float(np.max(relative_nominal)),
            "worst_entrywise_step": float(np.max(relative_entrywise)),
            "worst_nominal_steady": float(np.max(relative_steady_nominal)),
            "worst_entrywise_steady": float(np.max(relative_steady_entrywise)),
            "worst_nominal_step_at_s": float(
                np.unravel_index(np.argmax(relative_nominal), relative_nominal.shape)[0]
                * dt
            ),
            "step_transfer": rom_step,
        })
    return {
        "reference_steady": exact_steady,
        "reference_step": exact_step,
        "roms": results,
    }


def selected_frequency_basis(
    model, points, tolerance, closing_tolerance=None, additional_cutoffs=()
):
    """Use a chosen parameter set with the production four-port shift and SVD plan."""
    K = model.core.K.tocsc()
    C = model.core.C.tocsc()
    G = np.asarray(model.source_shape, dtype=np.float64)
    terms = [term.tocsc() for term in model.boundary_terms]
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(terms) or not len(points):
        raise ValueError("points must be a nonempty (n_points, n_groups) array")
    started = time.perf_counter()
    plans = []
    snapshots = []
    for port in range(G.shape[1]):
        g = G[:, port]
        low, high = port_eigenvalue_bounds(K, C, g)
        count = mpmm_elliptic_shift_count(tolerance, low, high)
        shifts = mpmm_elliptic_shifts(count, high, high / low)
        plans.append(count)
        for shift in shifts:
            shifted = K + float(shift) * C
            for point in points:
                A = shifted.copy()
                for value, term in zip(point, terms):
                    A = A + float(value) * term
                snapshots.append(spd_solve(A.tocsc(), g))
    solve_seconds = time.perf_counter() - started
    started_svd = time.perf_counter()
    cutoff = tolerance if closing_tolerance is None else closing_tolerance
    snapshot_matrix = np.column_stack(snapshots)
    basis, singular_values = _snapshot_svd_basis(snapshot_matrix, cutoff)
    additional_bases = {}
    for extra_cutoff in additional_cutoffs:
        extra_basis, _ = _snapshot_svd_basis(snapshot_matrix, extra_cutoff)
        additional_bases[extra_cutoff] = extra_basis
    constant = np.ones((K.shape[0], 1), dtype=np.float64)
    constant /= np.linalg.norm(constant)
    addition = orthonormalize_block(basis, constant)
    if addition.shape[1]:
        basis = np.column_stack((basis, addition))
    for extra_cutoff, extra_basis in additional_bases.items():
        extra_constant = orthonormalize_block(extra_basis, constant)
        if extra_constant.shape[1]:
            additional_bases[extra_cutoff] = np.column_stack(
                (extra_basis, extra_constant)
            )
    svd_seconds = time.perf_counter() - started_svd
    return basis, {
        "parameter_points": int(points.shape[0]),
        "full_rhs_solves": len(snapshots),
        "shift_counts": plans,
        "svd_kept_order": int(np.count_nonzero(
            singular_values >= cutoff * singular_values[0]
        )),
        "basis_order": int(basis.shape[1]),
        "frequency_plan_and_solves_s": solve_seconds,
        "svd_s": svd_seconds,
        "seconds_without_spectrum": solve_seconds + svd_seconds,
        "additional_bases": additional_bases,
    }


def fixed_tensor_basis(model, axes, tolerance, closing_tolerance=None):
    """Replace only the random HTC loop; retain the production frequency/SVD."""
    return selected_frequency_basis(
        model, tensor_product(axes), tolerance, closing_tolerance
    )


def output_selected_frequency_basis(
    model, spectra, candidates, tolerance, maximum_points, closing_tolerance=None,
    additional_cutoffs=(),
):
    """Choose HTC with steady transfer bounds; build the same dynamic ROM."""
    K = model.core.K.tocsc()
    G = np.asarray(model.source_shape, dtype=np.float64)
    terms = [term.tocsc() for term in model.boundary_terms]
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    started = time.perf_counter()
    selected = certified_greedy_design(
        K, terms, G, model.nominal_power(), ranges, spectra, candidates,
        tolerance, maximum_points, {}, "entrywise",
    )
    selection_seconds = time.perf_counter() - started
    basis, summary = selected_frequency_basis(
        model, selected.points, tolerance, closing_tolerance, additional_cutoffs
    )
    # The greedy certificate applies to its raw steady span, so check the
    # *actual compressed dynamic ROM* separately before reporting a bound.
    started = time.perf_counter()
    certificate = prepare_residual_certificate(K, terms, G, ranges, basis)
    final_scores = np.array([
        np.max(certificate.evaluate_entrywise(point)[2]) for point in candidates
    ])
    audit_seconds = time.perf_counter() - started
    summary.update({
        "selected_effective_points": selected.points.tolist(),
        "selection_points": len(selected.points),
        "selection_steady_certificate": selected.selection_certificate,
        "final_steady_grid_certificate": float(np.max(final_scores)),
        "final_steady_unresolved_candidates": int(
            np.count_nonzero(final_scores > tolerance)
        ),
        "steady_selection_rhs_solves": len(selected.points) * G.shape[1],
        "selection_s": selection_seconds,
        "post_svd_steady_audit_s": audit_seconds,
        "seconds_without_spectrum": (
            selection_seconds + summary["seconds_without_spectrum"] + audit_seconds
        ),
    })
    return basis, summary


def holdout_parameters(model, random_count: int) -> list[dict]:
    physical_ranges = np.asarray(model.config.h_ranges, dtype=np.float64)
    corners = tensor_product([physical_ranges[i] for i in range(2)])
    physical = [("corner", row) for row in corners]
    physical += [("reproduction", np.array([50.0, 1000.0]))]
    physical += [("center", np.sqrt(np.prod(physical_ranges, axis=1)))]
    generator = np.random.default_rng(20260924)
    for row in generator.uniform(
        np.log(physical_ranges[:, 0]), np.log(physical_ranges[:, 1]),
        size=(random_count, 2),
    ):
        physical.append(("holdout", np.exp(row)))
    return [{
        "label": label,
        "physical_h": [float(value) for value in row],
        "effective_p": [float(value) for value in model.physical_to_effective(row)],
    } for label, row in physical]


def compare(model, tolerances, counts, seeds, random_count, dt, duration,
            fixed_closing_tolerance=None, skip_validation=False,
            greedy_seed_counts=(), greedy_grid=9, max_extra_per_shift=12,
            greedy_closing_tolerance=None, output_greedy_grid=0,
            output_max_points=12, output_closing_tolerance=None):
    K = model.core.K.tocsc()
    C = model.core.C.tocsc()
    G = np.asarray(model.source_shape, dtype=np.float64)
    terms = [term.tocsc() for term in model.boundary_terms]
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    started = time.perf_counter()
    spectra = coordinate_spectral_enclosures(K, terms, ranges)
    spectrum_seconds = time.perf_counter() - started
    print(f"spectral preparation={spectrum_seconds:.3f}s", flush=True)

    designs = []
    bases = []
    for tolerance in tolerances:
        for seed in seeds:
            started = time.perf_counter()
            basis, summary = build_parametric_basis(
                model.core, G, terms, ranges, tolerance=tolerance,
                max_order=4096, probe_rounds=10, seed=seed,
            )
            elapsed = time.perf_counter() - started
            record = {
                "name": f"reproduction-seed-{seed}",
                "tolerance": tolerance,
                "extraction_s": elapsed,
                "parameter_points": None,
                "full_rhs_solves": summary["pre_svd_order"],
                "random_residual_probes": summary["validation_count"],
                "shift_counts": [p["shift_count"] for p in summary["per_port_plans"]],
                "basis_order": int(basis.shape[1]),
                "svd_kept_order": summary["svd_kept_order"],
                "max_accepted_residual": summary["max_accepted_residual"],
            }
            designs.append(record)
            bases.append(basis)
            print(f"extracted {record['name']} tol={tolerance:g} "
                  f"order={basis.shape[1]} rhs={record['full_rhs_solves']} "
                  f"time={elapsed:.3f}s", flush=True)
        for count in counts:
            axes = [zolotarev_rule(
                (spectrum.lower, spectrum.upper), tuple(ranges[i]), count
            ).parameter_nodes for i, spectrum in enumerate(spectra)]
            basis, summary = fixed_tensor_basis(
                model, axes, tolerance, fixed_closing_tolerance
            )
            record = {
                "name": f"zolotarev-{count}x{count}",
                "tolerance": tolerance,
                "closing_tolerance": (
                    tolerance if fixed_closing_tolerance is None
                    else fixed_closing_tolerance
                ),
                "extraction_s": summary["seconds_without_spectrum"] + spectrum_seconds,
                "shared_spectrum_s": spectrum_seconds,
                "frequency_plan_and_solves_s": summary["frequency_plan_and_solves_s"],
                "svd_s": summary["svd_s"],
                **{key: summary[key] for key in (
                    "parameter_points", "full_rhs_solves", "shift_counts",
                    "basis_order", "svd_kept_order",
                )},
            }
            designs.append(record)
            bases.append(basis)
            print(f"extracted {record['name']} tol={tolerance:g} "
                  f"order={basis.shape[1]} rhs={record['full_rhs_solves']} "
                  f"total={record['extraction_s']:.3f}s", flush=True)

        for count in greedy_seed_counts:
            axes = [zolotarev_rule(
                (spectrum.lower, spectrum.upper), tuple(ranges[i]), count
            ).parameter_nodes for i, spectrum in enumerate(spectra)]
            seed_points = tensor_product(axes)
            candidates = logarithmic_tensor_grid(ranges, greedy_grid)
            basis, summary = adaptive_basis(
                model, seed_points, candidates, tolerance, max_extra_per_shift,
                closing=greedy_closing_tolerance,
            )
            record = {
                "name": f"zolo-{count}x{count}-greedy-grid-{greedy_grid}",
                "tolerance": tolerance,
                "closing_tolerance": (
                    tolerance if greedy_closing_tolerance is None
                    else greedy_closing_tolerance
                ),
                "extraction_s": summary["seconds_without_spectrum"] + spectrum_seconds,
                "shared_spectrum_s": spectrum_seconds,
                "frequency_plan_and_solves_s": summary["frequency_plan_and_solves_s"],
                "svd_s": summary["svd_s"],
                "parameter_points": None,
                **{key: summary[key] for key in (
                    "full_rhs_solves", "shift_counts", "basis_order",
                    "svd_kept_order", "max_pre_svd_grid_residual",
                    "unresolved_shifts", "extra_solves", "candidate_evaluations",
                    "history",
                )},
            }
            designs.append(record)
            bases.append(basis)
            print(f"extracted {record['name']} tol={tolerance:g} "
                  f"order={basis.shape[1]} rhs={record['full_rhs_solves']} "
                  f"extra={summary['extra_solves']} "
                  f"unresolved={summary['unresolved_shifts']} "
                  f"pre-SVD-residual={summary['max_pre_svd_grid_residual']:.3e} "
                  f"total={record['extraction_s']:.3f}s", flush=True)

        if output_greedy_grid:
            candidates = logarithmic_tensor_grid(ranges, output_greedy_grid)
            basis, summary = output_selected_frequency_basis(
                model, spectra, candidates, tolerance, output_max_points,
                output_closing_tolerance,
            )
            record = {
                "name": f"zolo-steady-output-greedy-grid-{output_greedy_grid}",
                "tolerance": tolerance,
                "closing_tolerance": (
                    tolerance if output_closing_tolerance is None
                    else output_closing_tolerance
                ),
                "extraction_s": summary["seconds_without_spectrum"] + spectrum_seconds,
                "shared_spectrum_s": spectrum_seconds,
                "full_rhs_solves": (
                    summary["full_rhs_solves"] + summary["steady_selection_rhs_solves"]
                ),
                "dynamic_rhs_solves": summary["full_rhs_solves"],
                "steady_selection_rhs_solves": summary["steady_selection_rhs_solves"],
                **{key: summary[key] for key in (
                    "parameter_points", "selection_points", "selected_effective_points",
                    "selection_steady_certificate", "final_steady_grid_certificate",
                    "final_steady_unresolved_candidates", "shift_counts",
                    "basis_order", "svd_kept_order", "selection_s",
                    "frequency_plan_and_solves_s", "svd_s", "post_svd_steady_audit_s",
                )},
            }
            designs.append(record)
            bases.append(basis)
            print(f"extracted {record['name']} tol={tolerance:g} "
                  f"order={basis.shape[1]} rhs={record['full_rhs_solves']} "
                  f"steady-cert={record['final_steady_grid_certificate']:.3e} "
                  f"total={record['extraction_s']:.3f}s", flush=True)

    if skip_validation:
        return {
            "mesh_mm": model.config.max_xy_cell_mm,
            "cells": int(K.shape[0]),
            "effective_ranges": ranges.tolist(),
            "spectral_preparation_s": spectrum_seconds,
            "validation_skipped": True,
            "designs": designs,
        }

    holdout = holdout_parameters(model, random_count)
    worst_keys = (
        "worst_nominal_step", "worst_entrywise_step",
        "worst_nominal_steady", "worst_entrywise_steady",
    )
    for record in designs:
        record["worst"] = {key: {"error": -1.0} for key in worst_keys}
    validation_started = time.perf_counter()
    for index, point in enumerate(holdout, 1):
        A = K.copy()
        for value, term in zip(point["effective_p"], terms):
            A = A + value * term
        evaluated = evaluate_step_case(
            A.tocsc(), C, G, model.nominal_power(), bases,
            dt=dt, duration=duration,
        )
        for record, result in zip(designs, evaluated["roms"]):
            for key in worst_keys:
                if result[key] > record["worst"][key]["error"]:
                    record["worst"][key] = {
                        "error": result[key],
                        "physical_h": point["physical_h"],
                        "label": point["label"],
                        **({"time_s": result["worst_nominal_step_at_s"]}
                           if key == "worst_nominal_step" else {}),
                    }
        print(f"validated {index}/{len(holdout)} {point['label']} "
              f"h={point['physical_h']}", flush=True)
    return {
        "mesh_mm": model.config.max_xy_cell_mm,
        "cells": int(K.shape[0]),
        "dt_s": dt,
        "duration_s": duration,
        "effective_ranges": ranges.tolist(),
        "spectrum": [{"lower": s.lower, "upper": s.upper}
                     for s in spectra],
        "spectral_preparation_s": spectrum_seconds,
        "holdout": holdout,
        "validation_s": time.perf_counter() - validation_started,
        "designs": designs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_mm", type=float, nargs="?", default=2.5)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--counts", type=int, nargs="*", default=[2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="*", default=[20260805, 7])
    parser.add_argument("--fixed-closing-tolerance", type=float, default=None)
    parser.add_argument("--greedy-seed-counts", type=int, nargs="*", default=[])
    parser.add_argument("--greedy-grid", type=int, default=9)
    parser.add_argument("--max-extra-per-shift", type=int, default=12)
    parser.add_argument("--greedy-closing-tolerance", type=float, default=None)
    parser.add_argument("--output-greedy-grid", type=int, default=0)
    parser.add_argument("--output-max-points", type=int, default=12)
    parser.add_argument("--output-closing-tolerance", type=float, default=None)
    parser.add_argument("--random-holdout", type=int, default=6)
    parser.add_argument("--dt", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=2000.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    model = Case1Model(Case1Config(
        max_xy_cell_mm=args.mesh_mm, max_z_cell_mm=args.mesh_mm,
        duration_s=args.duration, dt_s=args.dt,
    ))
    report = compare(
        model, args.tolerances, args.counts, args.seeds,
        args.random_holdout, args.dt, args.duration,
        fixed_closing_tolerance=args.fixed_closing_tolerance,
        skip_validation=args.skip_validation,
        greedy_seed_counts=args.greedy_seed_counts,
        greedy_grid=args.greedy_grid,
        max_extra_per_shift=args.max_extra_per_shift,
        greedy_closing_tolerance=args.greedy_closing_tolerance,
        output_greedy_grid=args.output_greedy_grid,
        output_max_points=args.output_max_points,
        output_closing_tolerance=args.output_closing_tolerance,
    )
    for item in report["designs"]:
        line = (f"{item['name']} tol={item['tolerance']:g} "
                f"extract={item['extraction_s']:.3f}s "
                f"rhs={item['full_rhs_solves']} order={item['basis_order']}")
        if "worst" in item:
            errors = item["worst"]
            line += (f" step={errors['worst_nominal_step']['error']:.6e} "
                     f"entrywise={errors['worst_entrywise_step']['error']:.6e}")
        print(line)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
