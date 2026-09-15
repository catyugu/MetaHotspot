#!/usr/bin/env python3
"""Focused diagnostics for the whole-PoP contour-element boundary reduction.

This script is intentionally an experiment-only check. It reuses an already
extracted whole-PoP basis, compares the continuous contour Gram used by the
current experiment with the explicit discrete Eq. (16)/(17) environment map,
and isolates the error introduced by each surface and polynomial degree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import pop
from mor_common import contour_trace_coefficients
from metahotspot.macromodel.utils import spd_solve


def contour_gram(port, basis: np.ndarray, degree: int, *, discrete: bool):
    elements, coefficients, relative = contour_trace_coefficients(port, basis, degree)
    psi_avg = elements.averages(port.rects)
    psi_gram = psi_avg.T @ (port.area[:, None] * psi_avg)
    if discrete:
        gram = coefficients.T @ psi_gram @ coefficients
    else:
        gram = coefficients.T @ coefficients
    return np.asarray(gram), float(relative), np.asarray(psi_gram)


def reduced_metrics(domain, basis, K0_hat, F_hat, grams, case, reference):
    metrics, _ = pop.evaluate_reduced_case(
        domain,
        basis,
        K0_hat,
        F_hat,
        grams,
        case,
        reference,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nxy", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    args = parser.parse_args()

    basis = np.load(args.out / "basis_whole_pop.npy")
    domain = pop.build_whole_pop(args.nxy, args.nz)
    if basis.shape[0] != domain.n_cells:
        raise RuntimeError(
            f"basis/domain mismatch: {basis.shape[0]} rows vs {domain.n_cells} cells"
        )

    ports, terms, _, _ = pop.boundary_data(domain)
    K0_hat = np.asarray(basis.T @ (domain.K @ basis), dtype=np.float64)
    F_hat = np.asarray(basis.T @ domain.source_matrix, dtype=np.float64)
    full_grams, current_contour, current_diag = pop._boundary_grams(ports, basis)

    discrete_contour = {}
    surface_checks = {}
    for side, port in ports.items():
        degree = pop.FACE_DEGREES[side]
        continuous, relative, psi_gram = contour_gram(
            port, basis, degree, discrete=False
        )
        discrete, _, _ = contour_gram(port, basis, degree, discrete=True)
        discrete_contour[side] = discrete

        full = np.asarray(full_grams[side])
        residual = 0.5 * ((full - continuous) + (full - continuous).T)
        residual_min_eig = float(np.linalg.eigvalsh(residual).min())
        psi_eigs = np.linalg.eigvalsh(0.5 * (psi_gram + psi_gram.T))
        denom = max(float(np.linalg.norm(full, ord="fro")), 1.0e-300)
        surface_checks[side] = {
            **current_diag[side],
            "projection_error_recomputed_pct": 100.0 * relative,
            "continuous_vs_full_gram_fro_pct": float(
                100.0 * np.linalg.norm(continuous - full, ord="fro") / denom
            ),
            "discrete_vs_full_gram_fro_pct": float(
                100.0 * np.linalg.norm(discrete - full, ord="fro") / denom
            ),
            "continuous_vs_discrete_gram_fro_pct_of_full": float(
                100.0 * np.linalg.norm(continuous - discrete, ord="fro") / denom
            ),
            "full_minus_continuous_min_eig": residual_min_eig,
            "psi_discrete_gram_eig_min": float(psi_eigs.min()),
            "psi_discrete_gram_eig_max": float(psi_eigs.max()),
            "psi_discrete_gram_max_abs_I": float(
                np.max(np.abs(psi_gram - np.eye(psi_gram.shape[0])))
            ),
        }

    cases = {}
    references = {}
    for name in ("paper_fig2", "range_high"):
        case = pop.CASES[name]
        reference = spd_solve(pop._full_operator(domain, terms, case), domain.rhs)
        references[name] = reference
        cases[name] = {
            "full_trace": reduced_metrics(
                domain, basis, K0_hat, F_hat, full_grams, case, reference
            ),
            "current_contour_continuous": reduced_metrics(
                domain, basis, K0_hat, F_hat, current_contour, case, reference
            ),
            "contour_via_discrete_eq16_17": reduced_metrics(
                domain, basis, K0_hat, F_hat, discrete_contour, case, reference
            ),
            "one_surface_contour": {},
        }
        for side in pop.FACE_DEGREES:
            hybrid = dict(full_grams)
            hybrid[side] = current_contour[side]
            cases[name]["one_surface_contour"][side] = reduced_metrics(
                domain, basis, K0_hat, F_hat, hybrid, case, reference
            )

    range_high = pop.CASES["range_high"]
    ref_high = references["range_high"]
    sweeps = {}

    def run_surface_sweep(label: str, targets: tuple[str, ...], degrees: tuple[int, ...]):
        rows = []
        for degree in degrees:
            grams = dict(full_grams)
            projection = {}
            for side in targets:
                gram, relative, _ = contour_gram(
                    ports[side], basis, degree, discrete=False
                )
                grams[side] = gram
                projection[side] = 100.0 * relative
            rows.append(
                {
                    "degree": degree,
                    "projection_error_pct": projection,
                    **reduced_metrics(
                        domain, basis, K0_hat, F_hat, grams, range_high, ref_high
                    ),
                }
            )
        sweeps[label] = rows

    run_surface_sweep("sides_only", pop.SIDE_FACES, (2, 4, 6, 8, 10))
    run_surface_sweep("bottom_only", (pop.BOTTOM_FACE,), (4, 6, 8, 10, 12, 16, 20, 24))
    run_surface_sweep("top_only", (pop.TOP_FACE,), (10, 12, 14, 16, 20, 24))

    payload = {
        "basis_shape": list(basis.shape),
        "surface_checks": surface_checks,
        "case_checks": cases,
        "degree_sweeps_range_high_one_group_at_a_time": sweeps,
    }
    path = args.out / "contour_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("contour diagnostic summary")
    for side, row in surface_checks.items():
        print(
            f"  {side}: projection={row['projection_error_pct']:.3f}% "
            f"PsiGram=[{row['psi_discrete_gram_eig_min']:.6f}, "
            f"{row['psi_discrete_gram_eig_max']:.6f}] "
            f"continuous-vs-discrete={row['continuous_vs_discrete_gram_fro_pct_of_full']:.3f}%"
        )
    for name, row in cases.items():
        print(
            f"  {name}: full={row['full_trace']['global_inf_pct']:.4f}% "
            f"contour={row['current_contour_continuous']['global_inf_pct']:.4f}% "
            f"Eq16/17-discrete={row['contour_via_discrete_eq16_17']['global_inf_pct']:.4f}%"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
