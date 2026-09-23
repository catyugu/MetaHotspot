#!/usr/bin/env python3
"""Test two conditional Zolotarev nodes on the exposed HTC edge.

The tangent majorant chooses the edge, while the one-parameter rational
resolvent problem chooses two interior samples on that fixed edge. This is
an experimental sampling rule, not a transient certificate after SVD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from compare_transient import evaluate_step_case, holdout_parameters, selected_frequency_basis
from model_case1 import Case1Config, Case1Model
from probe_tangent_corners import seed_corner_rule
from zolotarev import coordinate_spectral_enclosures, zolotarev_rule


def conditional_edge_rule(K, terms, source, ranges):
    """Degree-one product seed and degree-two rule on the worst-ranked edge."""
    spectra = coordinate_spectral_enclosures(K, terms, ranges)
    seed = np.asarray([
        zolotarev_rule((s.lower, s.upper), tuple(ranges[i]), 1)
        .parameter_nodes[0] for i, s in enumerate(spectra)
    ])
    corners, scores, _ = seed_corner_rule(K, terms, source, ranges, seed)
    if not np.isclose(corners[0, 0], corners[1, 0]):
        raise ValueError('top two tangent corners are not on one HTC edge')
    fixed = corners[0, 0]
    conditional_ranges = np.array(ranges, copy=True)
    conditional_ranges[0] = fixed
    conditional_spectrum = coordinate_spectral_enclosures(
        K, terms, conditional_ranges
    )[1]
    rule = zolotarev_rule(
        (conditional_spectrum.lower, conditional_spectrum.upper),
        tuple(ranges[1]), 2,
    )
    points = np.vstack((seed, np.column_stack((
        np.full(2, fixed), rule.parameter_nodes,
    ))))
    return points, {
        'seed': seed.tolist(), 'top_corners': corners[:2].tolist(),
        'corner_scores': scores[:2].tolist(),
        'conditional_spectrum': [conditional_spectrum.lower, conditional_spectrum.upper],
        'conditional_zolotarev_bound': rule.error_bound,
    }


def run(mesh_mm=2.5, *, tolerance=1e-3, random_holdout=64,
        dt=50.0, duration=2000.0):
    model = Case1Model(Case1Config(
        max_xy_cell_mm=mesh_mm, max_z_cell_mm=mesh_mm,
        dt_s=dt, duration_s=duration,
    ))
    K, C = model.core.K.tocsc(), model.core.C.tocsc()
    G = np.asarray(model.source_shape, dtype=float)
    terms = [t.tocsc() for t in model.boundary_terms]
    ranges = np.asarray(model.h_ranges(), dtype=float)
    started = time.perf_counter()
    points, metadata = conditional_edge_rule(K, terms, G, ranges)
    preparation_s = time.perf_counter() - started
    corner_points = np.vstack((points[0], metadata['top_corners']))
    designs = [('conditional-Zolotarev', points), ('tangent-corners', corner_points)]
    bases, records = [], []
    for label, chosen in designs:
        basis, info = selected_frequency_basis(model, chosen, tolerance)
        bases.append(basis)
        records.append({
            'name': label, 'points': chosen.tolist(),
            'dynamic_rhs': info['full_rhs_solves'], 'rom_order': int(basis.shape[1]),
            'dynamic_extraction_s': info['seconds_without_spectrum'],
            'full_design_s': preparation_s + info['seconds_without_spectrum'],
            'worst_step': 0.0, 'worst_steady': 0.0,
        })
        print(f'{label}: {chosen.tolist()} rhs={info["full_rhs_solves"]} '
              f'order={basis.shape[1]} extract={info["seconds_without_spectrum"]:.2f}s',
              flush=True)
    for parameter in holdout_parameters(model, random_holdout):
        operator = K.copy()
        for value, term in zip(parameter['effective_p'], terms):
            operator = operator + float(value) * term
        result = evaluate_step_case(
            operator, C, G, np.ones(G.shape[1]), bases,
            dt=dt, duration=duration,
        )
        for record, rom in zip(records, result['roms']):
            for key, measured in [('worst_step', 'worst_entrywise_step'),
                                  ('worst_steady', 'worst_entrywise_steady')]:
                if rom[measured] > record[key]:
                    record[key] = rom[measured]
                    record[key + '_at'] = parameter
    for record in records:
        print(f'{record["name"]}: worst step={record["worst_step"]:.4e} '
              f'worst steady={record["worst_steady"]:.4e}', flush=True)
    return {
        'mesh_mm': mesh_mm, 'tolerance': tolerance,
        'closing_svd_cutoff': tolerance, 'holdout_points': random_holdout + 6,
        'selection_preparation_s': preparation_s,
        'conditional_rule': metadata, 'results': records,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_mm', nargs='?', type=float, default=2.5)
    parser.add_argument('--random-holdout', type=int, default=64)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.mesh_mm, random_holdout=args.random_holdout)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
