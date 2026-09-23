#!/usr/bin/env python3
"""Measure finite-grid BDF1 output bound tightness with nested SVD spaces.

Run: python playground/adaptive_bci_sampling/probe_dual_space.py 2.5
Full reference solves are validation cost, excluded from extraction figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "bci_rom_testcase1")]

from certified_greedy import prepare_residual_certificate  # noqa: E402
from compare_transient import (  # noqa: E402
    holdout_parameters, output_selected_frequency_basis, selected_frequency_basis,
    step_transfer,
)
from compare_zolotarev import logarithmic_tensor_grid, tensor_product  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402
from transient_dual_certificate import prepare_step_certificate  # noqa: E402
from zolotarev import coordinate_spectral_enclosures, zolotarev_rule  # noqa: E402


def run(mesh_mm: float, tolerance=1e-3, design="output", dt=50.0,
        duration=2000.0, medium_factor=10.0, grid_size=9):
    model = Case1Model(Case1Config(
        max_xy_cell_mm=mesh_mm, max_z_cell_mm=mesh_mm,
        dt_s=dt, duration_s=duration,
    ))
    K = model.core.K.tocsc()
    C = model.core.C.tocsc()
    H = [term.tocsc() for term in model.boundary_terms]
    G = np.asarray(model.source_shape, dtype=np.float64)
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    started = time.perf_counter()
    spectra = coordinate_spectral_enclosures(K, H, ranges)
    spectrum_s = time.perf_counter() - started
    points = logarithmic_tensor_grid(ranges, grid_size)
    cutoffs = (tolerance / medium_factor, tolerance / 100)
    if design == "output":
        V, extraction = output_selected_frequency_basis(
            model, spectra, points, tolerance, 12, additional_cutoffs=cutoffs
        )
        selection_rhs = extraction["steady_selection_rhs_solves"]
    else:
        if design == "fixed2":
            axes = [zolotarev_rule(
                (spectrum.lower, spectrum.upper), tuple(ranges[i]), 2
            ).parameter_nodes for i, spectrum in enumerate(spectra)]
            selected = tensor_product(axes)
        elif design == "mixed3":
            seed = np.array([zolotarev_rule(
                (spectrum.lower, spectrum.upper), tuple(ranges[i]), 1
            ).parameter_nodes[0] for i, spectrum in enumerate(spectra)])
            selected = np.vstack((
                seed, [ranges[0, 0], ranges[1, 1]],
                [ranges[0, 1], ranges[1, 0]],
            ))
        else:
            raise ValueError(f"unknown design: {design}")
        V, extraction = selected_frequency_basis(
            model, selected, tolerance, additional_cutoffs=cutoffs
        )
        extraction["selection_points"] = len(selected)
        extraction["selected_effective_points"] = selected.tolist()
        selection_rhs = 0
    medium = extraction["additional_bases"][cutoffs[0]]
    high = extraction["additional_bases"][cutoffs[1]]
    primal_order, medium_order, high_order = (
        basis.shape[1] for basis in (V, medium, high)
    )
    witnesses = [
        (f"{primal_order}-to-{primal_order}", V, V),
        (f"{primal_order}-to-{medium_order}", V, medium),
        (f"{primal_order}-to-{high_order}", V, high),
        (f"{medium_order}-to-{high_order}", medium, high),
        (f"{high_order}-to-{high_order}", high, high),
    ]
    minimum = K.copy()
    for value, term in zip(ranges[:, 0], H):
        minimum = minimum + float(value) * term
    factor = spla.splu(minimum.tocsc())
    started = time.perf_counter()
    residual_cache = {}
    certs = [(name, prepare_step_certificate(
        K, C, H, G, ranges, primal, dual,
        dt=dt, minimum_factor=factor, residual_cache=residual_cache
    )) for name, primal, dual in witnesses]
    steady_cache = {}
    for _name, primal, _dual in witnesses:
        if id(primal) not in steady_cache:
            steady_cache[id(primal)] = prepare_residual_certificate(
            K, H, G, ranges, primal, minimum_factor=factor
            )
    steady = {
        name: steady_cache[id(primal)]
        for name, primal, _dual in witnesses
    }
    preparation_s = time.perf_counter() - started
    print(
        f"mesh={mesh_mm:g} design={design} cells={K.shape[0]} rhs="
        f"{extraction['full_rhs_solves'] + selection_rhs} "
        f"orders={[(primal.shape[1], dual.shape[1]) for _, primal, dual in witnesses]} "
        f"spectra={spectrum_s:.2f}s extraction={extraction['seconds_without_spectrum']:.2f}s "
        f"certificate-setup={preparation_s:.2f}s", flush=True
    )
    report = {
        "mesh_mm": mesh_mm,
        "design": design,
        "tolerance": tolerance,
        "medium_factor": medium_factor,
        "grid_size": grid_size,
        "cells": K.shape[0],
        "spectrum_s": spectrum_s,
        "extraction_s": spectrum_s + extraction["seconds_without_spectrum"],
        "certificate_preparation_s": preparation_s,
        "selection_points": extraction["selection_points"],
        "full_rhs_solves": (
            extraction["full_rhs_solves"] + selection_rhs
        ),
        "selected_effective_points": extraction["selected_effective_points"],
        "primal_order": V.shape[1],
        "holdout": [],
        "witnesses": {},
    }
    for name, primal, dual in witnesses:
        report["witnesses"][name] = {
            "primal_order": primal.shape[1], "dual_order": dual.shape[1],
            "worst_true": 0.0,
            "worst_bound_exact_denominator": 0.0,
            "worst_bound_certified_denominator": 0.0,
            "worst_correction_exact_denominator": 0.0,
            "violations": 0, "online_s": 0.0,
            "candidate_grid_max": 0.0,
            "candidate_grid_max_at": None,
        }
    validation_started = time.perf_counter()
    steps = round(duration / dt)
    for physical in holdout_parameters(model, 6):
        p = physical["effective_p"]
        A = K.copy()
        for value, term in zip(p, H):
            A = A + float(value) * term
        reference = step_transfer(A, C, G, dt=dt, duration=duration)
        exact_steady = G.T @ spla.splu(A.tocsc()).solve(G)
        sample = {"physical_h": physical["physical_h"], "witnesses": {}}
        for name, cert in certs:
            estimated_steady, absolute_steady, _ = steady[name].evaluate_entrywise(p)
            certified_steady_lower = np.abs(estimated_steady) - absolute_steady
            started = time.perf_counter()
            rom, correction, bound = cert.evaluate(p, steps=steps)
            evaluation_s = time.perf_counter() - started
            true = float(np.max(
                np.abs(reference - rom) / np.abs(exact_steady)[None, :, :]
            ))
            relative = bound / np.abs(exact_steady)[None, :, :]
            exact_normalized = float(np.max(relative))
            if np.any(certified_steady_lower <= 0):
                certified_normalized = float("inf")
            else:
                certified_normalized = float(np.max(
                    bound / certified_steady_lower[None, :, :]
                ))
            correction_normalized = float(np.max(
                np.abs(correction) / np.abs(exact_steady)[None, :, :]
            ))
            violations = int(np.count_nonzero(
                np.abs(reference - rom) > bound * (1 + 1e-7) + 1e-8
            ))
            item = sample["witnesses"][name] = {
                "true": true,
                "bound_exact_denominator": exact_normalized,
                "bound_certified_denominator": certified_normalized,
                "correction_exact_denominator": correction_normalized,
                "violations": violations,
            }
            overall = report["witnesses"][name]
            overall["worst_true"] = max(overall["worst_true"], true)
            for field in ("bound_exact_denominator", "bound_certified_denominator",
                          "correction_exact_denominator"):
                overall["worst_" + field] = max(
                    overall["worst_" + field], item[field]
                )
            overall["violations"] += violations
            overall["online_s"] += evaluation_s
        report["holdout"].append(sample)
        print(f"validated {len(report['holdout'])}/12 h={physical['physical_h']}",
              flush=True)
    report["validation_s"] = time.perf_counter() - validation_started
    started = time.perf_counter()
    for point in points:
        for name, cert in certs:
            estimated, absolute_steady, _ = steady[name].evaluate_entrywise(point)
            lower = np.abs(estimated) - absolute_steady
            _rom, _correction, bound = cert.evaluate(point, steps=steps)
            score = (float("inf") if np.any(lower <= 0) else float(
                np.max(bound / lower[None, :, :])
            ))
            entry = report["witnesses"][name]
            if score > entry["candidate_grid_max"]:
                entry["candidate_grid_max"] = score
                entry["candidate_grid_max_at"] = point.tolist()
    report["candidate_grid_scoring_s"] = time.perf_counter() - started
    for name, item in report["witnesses"].items():
        print(f"{name}: true={item['worst_true']:.3e} "
              f"bound/exact-den={item['worst_bound_exact_denominator']:.3e} "
              f"bound/cert-den={item['worst_bound_certified_denominator']:.3e} "
              f"correction={item['worst_correction_exact_denominator']:.3e} "
              f"grid={item['candidate_grid_max']:.3e} "
              f"violations={item['violations']} online={item['online_s']:.2f}s",
              flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_mm", type=float, nargs="?", default=2.5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--design", choices=("output", "fixed2", "mixed3"),
                        default="output")
    parser.add_argument("--medium-factor", type=float, default=10.0)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    outcome = run(args.mesh_mm, tolerance=args.tolerance, design=args.design,
                  medium_factor=args.medium_factor, grid_size=args.grid_size)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(outcome, indent=2))
        print(f"results written to {args.output}")
