#!/usr/bin/env python3
"""Pick HTC corners from the seed's output-weighted tangent error geometry.

At the seed, four steady full-order source solves define X0. For each output
pair and parameter displacement d, the raw seed snapshot space has relative
steady transfer error bounded by d.T @ Q[i,j] @ d. Its 2x2 PSD matrix uses
K(p_min)^-1 and the positive transfer at the maximum HTC corner. This
requires a nonnegative source and an SPD M-matrix thermal operator. The
maximum of a convex quadratic over a box occurs at a vertex. Rank the four
vertices and take the Zolotarev seed plus the largest vertices.

The quadratic bound guides sampling only: it does not bound the final SVD
ROM's transient transfer error. The final compression cutoff is unchanged.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "bci_rom_testcase1")]

from compare_transient import (  # noqa: E402
    evaluate_step_case, holdout_parameters, selected_frequency_basis,
)
from model_case1 import Case1Config, Case1Model  # noqa: E402
from zolotarev import coordinate_spectral_enclosures, zolotarev_rule  # noqa: E402


def seed_corner_rule(K, terms, source, ranges, seed):
    """Return corners in descending diagonal output-majorant order.

    Geometry has shape (n_sources, n_sources, n_parameters, n_parameters).
    Its quadratic bounds all relative steady transfer entries of a raw
    Galerkin snapshot space containing all source responses at the seed.
    """
    K = sp.csc_matrix(K)
    terms = [sp.csc_matrix(t) for t in terms]
    G = np.asarray(source, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    seed = np.asarray(seed, dtype=float)
    off_diagonal = K.tocoo()
    if np.any(off_diagonal.data[
        off_diagonal.row != off_diagonal.col
    ] > 0) or np.any(G < 0):
        raise ValueError('entrywise transfer monotonicity needs an M-matrix '
                         'and nonnegative sources')
    for term in terms:
        entries = term.tocoo()
        if np.any(entries.row != entries.col) or np.any(entries.data < 0):
            raise ValueError('each boundary term must be nonnegative diagonal')
    minimum, maximum, seed_operator = K.copy(), K.copy(), K.copy()
    for t, extent, value in zip(terms, ranges, seed):
        minimum = minimum + float(extent[0]) * t
        maximum = maximum + float(extent[1]) * t
        seed_operator = seed_operator + float(value) * t
    X0 = spla.splu(seed_operator.tocsc()).solve(G)
    lower = G.T @ spla.splu(maximum.tocsc()).solve(G)
    if np.any(lower <= 0):
        raise ValueError('steady source transfer has no positive lower bound')
    actions = [t @ X0 for t in terms]
    factor = spla.splu(minimum.tocsc())
    riesz = factor.solve(np.column_stack(actions))
    riesz = np.split(riesz, len(actions), axis=1)
    q = np.empty((G.shape[1], len(terms), len(terms)))
    for a, left in enumerate(actions):
        for b, right in enumerate(riesz):
            q[:, a, b] = np.einsum('ni,ni->i', left, right)
    q = 0.5 * (q + q.swapaxes(1, 2))
    geometry = (q[:, None, :, :] + q[None, :, :, :]) / (
        2.0 * lower[:, :, None, None]
    )
    corners = np.asarray(list(product(*ranges)), dtype=float)
    displacement = corners - seed
    scores = np.max(np.einsum(
        'ka,ijab,kb->kij', displacement, geometry, displacement
    ), axis=(1, 2))
    ordering = np.argsort(-scores, kind='stable')
    return corners[ordering], scores[ordering], geometry


def run(mesh_mm, *, tolerance=1e-3, minimum_points=2, maximum_points=4,
        dt=50.0, duration=2000.0, random_holdout=6):
    if not 2 <= minimum_points <= maximum_points <= 5:
        raise ValueError('require 2 <= minimum_points <= maximum_points <= 5')
    model = Case1Model(Case1Config(
        max_xy_cell_mm=mesh_mm, max_z_cell_mm=mesh_mm,
        dt_s=dt, duration_s=duration,
    ))
    K = model.core.K.tocsc()
    C = model.core.C.tocsc()
    terms = [t.tocsc() for t in model.boundary_terms]
    G = np.asarray(model.source_shape, dtype=float)
    ranges = np.asarray(model.h_ranges(), dtype=float)
    start = time.perf_counter()
    spectra = coordinate_spectral_enclosures(K, terms, ranges)
    seed = np.asarray([
        zolotarev_rule((s.lower, s.upper), tuple(ranges[i]), 1)
        .parameter_nodes[0] for i, s in enumerate(spectra)
    ])
    corners, scores, _geometry = seed_corner_rule(
        K, terms, G, ranges, seed
    )
    selection_s = time.perf_counter() - start
    print(f'seed {seed.tolist()} selected vertices {corners.tolist()} '
          f'scores {scores.tolist()} setup={selection_s:.2f}s', flush=True)

    results, bases = [], []
    for count in range(minimum_points, maximum_points + 1):
        chosen = np.vstack((seed, corners[:count - 1]))
        V, info = selected_frequency_basis(model, chosen, tolerance)
        bases.append(V)
        results.append({
            'points': chosen.tolist(),
            'frequency_tolerance': tolerance,
            'svd_cutoff': tolerance,
            'dynamic_rhs': info['full_rhs_solves'],
            'steady_selection_rhs': 2 * G.shape[1],
            'riesz_selection_rhs': len(terms) * G.shape[1],
            'rom_order': int(V.shape[1]),
            'final_extraction_s': info['seconds_without_spectrum'],
        })
        print(f'{count} points: rhs={info["full_rhs_solves"]} '
              f'order={V.shape[1]} extract={info["seconds_without_spectrum"]:.2f}s',
              flush=True)
    holdout_start = time.perf_counter()
    worst = [0.0] * len(bases)
    for p in holdout_parameters(model, random_holdout):
        operator = K.copy()
        for value, term in zip(p['effective_p'], terms):
            operator = operator + float(value) * term
        cases = evaluate_step_case(
            operator, C, G, np.ones(G.shape[1]), bases,
            dt=dt, duration=duration,
        )
        worst = [max(value, rom['worst_entrywise_step'])
                 for value, rom in zip(worst, cases['roms'])]
    for result, actual in zip(results, worst):
        result['holdout_worst_step'] = actual
        print(f'{len(result["points"])} points: worst={actual:.3e}', flush=True)
    return {
        'mesh_mm': mesh_mm, 'tolerance': tolerance,
        'seed': seed.tolist(), 'ordered_corners': corners.tolist(),
        'normalized_corner_scores': scores.tolist(),
        'selection_s': selection_s,
        'final_design_s': selection_s + results[-1]['final_extraction_s'],
        'holdout_points': 6 + random_holdout,
        'validation_s': time.perf_counter() - holdout_start,
        'results': results,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_mm', type=float, nargs='?', default=2.5)
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--minimum-points', type=int, default=2)
    parser.add_argument('--maximum-points', type=int, default=4)
    parser.add_argument('--random-holdout', type=int, default=6)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run(args.mesh_mm, tolerance=args.tolerance,
                 minimum_points=args.minimum_points,
                 maximum_points=args.maximum_points,
                 random_holdout=args.random_holdout)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
