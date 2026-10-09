#!/usr/bin/env python3
"""Reproducible diagnostic sweeps; settings are audit knobs, not a new extractor."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

HERE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'python'), str(HERE.parent/'bci_rom_testcase1')]
from rational_dynamic import AffineSystem, cell_certificate, exact_impulse_gram, shift_sequence
from metahotspot._compiled_data import Operators
from metahotspot.macromodel.utils import build_parametric_basis
from metahotspot.macromodel import utils as stock_utils


def acceptance_threshold(tolerance, basis_stage):
    return tolerance*(2. if basis_stage == 'final' else 1.)


def synthetic():
    """A 32-cell conservative conduction chain, two sources, two Robin groups."""
    n = 32
    edge = np.geomspace(.1, 10., n-1)
    K = sp.diags([np.r_[edge, 0]+np.r_[0, edge], -edge, -edge], [0, 1, -1]).tocsc()
    C = sp.diags(np.geomspace(.2, 2., n)).tocsc()
    G = np.zeros((n, 2))
    G[8:12, 0], G[21:25, 1] = .25, .25
    H = [sp.diags(np.eye(n)[i]).tocsc() for i in [0, n-1]]
    ranges = np.array([[1., 4.], [1., 4.]])
    return K, C, G, H, ranges, {'model': 'conservative_chain', 'native_validated': False}


from case1_system import case1_reconstruction


def measure(system, points):
    values = []
    for h in points:
        J, Q = exact_impulse_gram(system.operator(h), system.C, system.G, system.V)
        error = float(np.sqrt(max(0., la.eigvalsh(J, Q)[-1])))
        values.append({'h': np.asarray(h).tolist(), 'error': error})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['chain', 'case1', 'native-case1'], default='chain')
    parser.add_argument('--mesh-mm', type=float, default=10.)
    parser.add_argument('--terms', type=int, default=64)
    parser.add_argument('--basis-stage', choices=['final', 'raw'], default='final',
                        help='raw diagnoses the closing SVD; it is a distinct fixed basis')
    parser.add_argument('--degrees', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--widths', nargs='+', type=float, default=[1., .2, .02, .002])
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--cover-cells', type=int, default=0,
                        help='uniform cells per axis, optional full-domain diagnostic')
    parser.add_argument('--cover-degree', type=int, default=6)
    parser.add_argument('--propagation', choices=['independent', 'coupled', 'energy', 'matrix'], default='independent')
    parser.add_argument('--envelope-depth', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    accept_tolerance = acceptance_threshold(args.tolerance, args.basis_stage)
    if args.model == 'chain':
        K, C, G, H, ranges, metadata = synthetic()
    elif args.model == 'case1':
        K, C, G, H, ranges, metadata = case1_reconstruction(args.mesh_mm)
    else:
        from model_case1 import Case1Model, Case1Config
        model = Case1Model(Case1Config(max_xy_cell_mm=args.mesh_mm, max_z_cell_mm=args.mesh_mm))
        K, C, G, H, ranges = model.core.K, model.core.C, model.source_shape, model.boundary_terms, model.h_ranges()
        metadata = {'model': 'native Case1', 'native_validated': True, 'mesh_mm': args.mesh_mm}
    started = time.perf_counter()
    raw = []
    original_svd = stock_utils._snapshot_svd_basis
    def capture(snapshots, tolerance):
        raw.append(np.array(snapshots, copy=True))
        return original_svd(snapshots, tolerance)
    with patch.object(stock_utils, '_snapshot_svd_basis', capture):
        V, stats = build_parametric_basis(Operators(K, C, np.zeros(K.shape[0])), G, H,
                                           ranges, tolerance=args.tolerance, seed=20260805)
    if args.basis_stage == 'raw':
        snapshots = raw[0]
        snapshots /= la.norm(snapshots, axis=0)
        snapshots = np.column_stack([snapshots, np.ones(K.shape[0])/np.sqrt(K.shape[0])])
        U, s, _ = la.svd(snapshots, full_matrices=False)
        V = U[:, s > np.finfo(float).eps*max(snapshots.shape)*s[0]]
    system = AffineSystem(K, C, G, H, V)
    center = np.sqrt(ranges[:, 0]*ranges[:, 1])
    shifts = shift_sequence(system, center, args.terms)
    report = {'metadata': metadata, 'n': K.shape[0], 'basis_order': V.shape[1],
              'basis_stage': args.basis_stage,
              'basis_sha256': hashlib.sha256(V.tobytes()).hexdigest(),
              'stock_settings': stats, 'tolerance': args.tolerance,
              'extraction_tolerance': args.tolerance, 'acceptance_tolerance': accept_tolerance,
              'ranges': ranges.tolist(), 'center': center.tolist(),
              'shifts': shifts.tolist(), 'experiments': [],
              'floating_point_certified': False,
              'full_domain_certified': False,
              'reference_method': 'dense eigenmode exact-time integration; subtractive Gram',
              'solver': 'stock AMG-CG extraction; sparse LU trial/Riesz prototype'}
    for fraction in args.widths:
        low = center-fraction*(center-ranges[:, 0])
        high = center+fraction*(ranges[:, 1]-center)
        rng = np.random.default_rng(20261007)
        points = [low, high, center, *rng.uniform(low, high, (4, len(low)))]
        references = measure(system, points)
        for degree in args.degrees:
            t0 = time.perf_counter()
            cert = cell_certificate(system, low, high, shifts, degree, propagation_model=args.propagation, envelope_depth=args.envelope_depth)
            maximum = max(x['error'] for x in references)
            if maximum > cert['bound']*(1+1e-7)+1e-9:
                raise RuntimeError('independent reference exceeds analytic construction')
            record = {'width_fraction': fraction, 'certificate': cert,
                      'sampled_error': maximum, 'references': references,
                      'effectivity_sampled': cert['bound']/maximum if maximum else None,
                      'accepted_analytic': bool(cert['bound'] <= accept_tolerance),
                      'seconds': time.perf_counter()-t0}
            report['experiments'].append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(f"width={fraction:g} p={degree} bound={cert['bound']:.6g} "
                  f"sample={maximum:.6g} main={cert['main']:.3g} "
                  f"prop={cert['propagation']:.3g} tail={cert['full_tail']+cert['reduced_tail']:.3g} "
                  f"RHS={sum(v for key,v in cert['counts'].items() if key.endswith('_rhs'))}", flush=True)
    if args.cover_cells:
        from itertools import product
        edges = [np.linspace(a, b, args.cover_cells+1) for a, b in ranges]
        cover, t0 = [], time.perf_counter()
        for index in product(range(args.cover_cells), repeat=len(ranges)):
            low = np.array([edges[j][i] for j, i in enumerate(index)])
            high = np.array([edges[j][i+1] for j, i in enumerate(index)])
            cert = cell_certificate(system, low, high, shifts, args.cover_degree, propagation_model=args.propagation, envelope_depth=args.envelope_depth)
            reference = measure(system, [(low+high)/2])[0]
            if reference['error'] > cert['bound']*(1+1e-7)+1e-9:
                raise RuntimeError('cover reference exceeds analytic construction')
            cover.append({'index': list(index), 'certificate': cert, 'reference': reference,
                          'accepted_analytic': bool(cert['bound'] <= accept_tolerance)})
        report['cover'] = {'cells_per_axis': args.cover_cells, 'degree': args.cover_degree,
                           'cells': cover, 'seconds': time.perf_counter()-t0,
                           'max_bound': max(r['certificate']['bound'] for r in cover),
                           'all_accepted_analytic': all(r['accepted_analytic'] for r in cover),
                           'total_rhs': sum(v for r in cover for key,v in r['certificate']['counts'].items()
                                            if key.endswith('_rhs'))}
        report['full_domain_analytic_accepted'] = report['cover']['all_accepted_analytic']
        print(f"cover {len(cover)} cells: bound={report['cover']['max_bound']:.6g} "
              f"all_accepted={report['cover']['all_accepted_analytic']} "
              f"RHS={report['cover']['total_rhs']}", flush=True)
    report['seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
