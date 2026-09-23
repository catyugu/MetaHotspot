#!/usr/bin/env python3
"""Probe transient output-error greedy sampling at an unchanged SVD cutoff.

The score is the certified 40-step, 4x4 junction-transfer error divided by
a certified same-parameter steady transfer magnitude. All stages use the
production frequency shift count and the same closing SVD tolerance.
Selection cost includes every repeated extraction in this diagnostic driver.
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

from certified_greedy import prepare_residual_certificate, select_worst_certificate  # noqa: E402
from compare_transient import (  # noqa: E402
    evaluate_step_case, holdout_parameters, selected_frequency_basis,
)
from compare_zolotarev import logarithmic_tensor_grid  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402
from transient_dual_certificate import prepare_step_certificate  # noqa: E402
from zolotarev import coordinate_spectral_enclosures, zolotarev_rule  # noqa: E402


def run(mesh_mm: float, *, tolerance=1e-3, maximum_points=4,
        grid_size=9, dt=50.0, duration=2000.0):
    model = Case1Model(Case1Config(
        max_xy_cell_mm=mesh_mm, max_z_cell_mm=mesh_mm,
        dt_s=dt, duration_s=duration,
    ))
    K = model.core.K.tocsc()
    C = model.core.C.tocsc()
    terms = [term.tocsc() for term in model.boundary_terms]
    G = np.asarray(model.source_shape, dtype=np.float64)
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    t0 = time.perf_counter()
    spectra = coordinate_spectral_enclosures(K, terms, ranges)
    initial = np.asarray([
        zolotarev_rule((s.lower, s.upper), tuple(ranges[i]), 1)
        .parameter_nodes[0] for i, s in enumerate(spectra)
    ])
    print(f"spectral setup {time.perf_counter() - t0:.2f}s, seed {initial}",
          flush=True)
    candidates = logarithmic_tensor_grid(ranges, grid_size)
    minimum = K.copy()
    for value, term in zip(ranges[:, 0], terms):
        minimum = minimum + float(value) * term
    factor = spla.splu(minimum.tocsc())
    setup_s = time.perf_counter() - t0

    chosen = [initial]
    stages = []
    started = time.perf_counter()
    for stage in range(1, maximum_points + 1):
        V, extraction = selected_frequency_basis(
            model, np.asarray(chosen), tolerance, closing_tolerance=tolerance
        )
        prep_started = time.perf_counter()
        transient = prepare_step_certificate(
            K, C, terms, G, ranges, V, V, dt=dt, minimum_factor=factor
        )
        steady = prepare_residual_certificate(
            K, terms, G, ranges, V, minimum_factor=factor
        )
        prep_seconds = time.perf_counter() - prep_started
        score_started = time.perf_counter()
        scores = []
        absolute = []
        for p in candidates:
            estimated, steady_error, _ = steady.evaluate_entrywise(p)
            lower = np.abs(estimated) - steady_error
            _rom, _correction, bound = transient.evaluate(
                p, steps=round(duration / dt)
            )
            if np.any(lower <= 0):
                score = float("inf")
            else:
                score = float(np.max(bound / lower[None, :, :]))
            scores.append(score)
            absolute.append(float(np.max(bound)))
        available = [i for i, p in enumerate(candidates) if all(
            not np.allclose(p, existing, rtol=1e-10, atol=1e-10)
            for existing in chosen
        )]
        winner = available[select_worst_certificate(
            np.asarray(scores)[available], np.asarray(absolute)[available]
        )] if available else None
        item = {
            "points": [p.tolist() for p in chosen],
            "rom_order": V.shape[1],
            "dynamic_rhs": extraction["full_rhs_solves"],
            "extraction_s": extraction["seconds_without_spectrum"],
            "certificate_prep_s": prep_seconds,
            "candidate_scoring_s": time.perf_counter() - score_started,
            "grid_max_bound": max(scores),
            "next_point": candidates[winner].tolist() if winner is not None else None,
            "next_bound": scores[winner] if winner is not None else None,
        }
        stages.append((item, V))
        print(f"stage {stage}: order={V.shape[1]} rhs={item['dynamic_rhs']} "
              f"bound={max(scores):.3e} next={item['next_point']} "
              f"next_bound={item['next_bound']:.3e} "
              f"extract={item['extraction_s']:.2f}s "
              f"prep={prep_seconds:.2f}s "
              f"score={item['candidate_scoring_s']:.2f}s", flush=True)
        if winner is None:
            break
        chosen.append(candidates[winner])
    selection_and_extraction_s = time.perf_counter() - started

    worst = [0.0] * len(stages)
    validation_started = time.perf_counter()
    for p in holdout_parameters(model, 6):
        operator = K.copy()
        for value, term in zip(p["effective_p"], terms):
            operator = operator + float(value) * term
        errors = evaluate_step_case(
            operator, C, G, np.ones(G.shape[1]),
            [V for _item, V in stages], dt=dt, duration=duration,
        )
        worst = [max(value, rom["worst_entrywise_step"])
                 for value, rom in zip(worst, errors["roms"])]
    for (item, _basis), error in zip(stages, worst):
        item["holdout_worst_step"] = error
        print(f"{len(item['points'])} points: actual worst={error:.3e}", flush=True)
    return {
        "mesh_mm": mesh_mm, "tolerance": tolerance,
        "svd_cutoff": tolerance, "grid_size": grid_size,
        "spectral_and_factor_setup_s": setup_s,
        "selection_and_extraction_s": selection_and_extraction_s,
        "validation_s": time.perf_counter() - validation_started,
        "stages": [item for item, _basis in stages],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_mm", type=float, nargs="?", default=2.5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--maximum-points", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.mesh_mm, tolerance=args.tolerance,
                 maximum_points=args.maximum_points, grid_size=args.grid_size)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
