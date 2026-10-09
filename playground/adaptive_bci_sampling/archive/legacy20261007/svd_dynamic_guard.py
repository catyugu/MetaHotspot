"""Research-only SVD truncation guarded by all-input dynamic certificates.

The nested triangle theorem is standard, not claimed novel. The new signed
Poisson/Loewner residual majorant is used to validate the final SVD subspace.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
from unittest.mock import patch
import numpy as np
import scipy.linalg as la
from run_rational_dynamic import (case1_reconstruction, synthetic, stock_utils,
    build_parametric_basis, Operators, measure)
from rational_dynamic import AffineSystem, cell_certificate, shift_sequence, exact_impulse_gram


def svd_candidate_coordinates(U, raw_basis, kept):
    """Represent the SVD directions and constant mode exactly in raw coordinates."""
    coordinates = raw_basis.T @ U[:, :kept]
    coordinates = la.qr(coordinates, mode='economic')[0]
    constant = raw_basis.T @ (np.ones(U.shape[0])/np.sqrt(U.shape[0]))
    for _ in range(2):
        constant -= coordinates @ (coordinates.T @ constant)
    if la.norm(constant) > np.finfo(float).eps*max(raw_basis.shape):
        coordinates = np.column_stack([coordinates, constant/la.norm(constant)])
    return coordinates


def parse_arguments(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['case1', 'native-case1', 'chain'], default='case1')
    p.add_argument('--seed', type=int, default=20260805)
    p.add_argument('--mesh-mm', type=float, default=10.)
    p.add_argument('--fraction', type=float, default=.0002)
    p.add_argument('--terms', type=int, default=80)
    p.add_argument('--degree', type=int, default=2)
    p.add_argument('--tolerance', type=float, default=.001)
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args(argv)


def main():
    args = parse_arguments()
    if args.model == 'native-case1':
        from model_case1 import Case1Model, Case1Config
        model = Case1Model(Case1Config(max_xy_cell_mm=args.mesh_mm, max_z_cell_mm=args.mesh_mm))
        K, C, G, H, ranges = model.core.K, model.core.C, model.source_shape, model.boundary_terms, model.h_ranges()
        metadata = {'model': 'native Case1', 'native_validated': True, 'mesh_mm': args.mesh_mm}
    else:
        K, C, G, H, ranges, metadata = case1_reconstruction(args.mesh_mm) if args.model == 'case1' else synthetic()
    snapshots = []
    stock_svd = stock_utils._snapshot_svd_basis
    def capture(A, tolerance):
        snapshots.append(np.array(A, copy=True))
        return stock_svd(A, tolerance)
    with patch.object(stock_utils, '_snapshot_svd_basis', capture):
        stock_V, stats = build_parametric_basis(Operators(K, C, np.zeros(K.shape[0])), G, H,
            ranges, tolerance=args.tolerance, seed=args.seed)
    S = snapshots[0]/la.norm(snapshots[0], axis=0)
    U, singular_values, _ = la.svd(S, full_matrices=False)
    augmented = np.column_stack([S, np.ones(K.shape[0])/np.sqrt(K.shape[0])])
    W, s, _ = la.svd(augmented, full_matrices=False)
    W = W[:, s > np.finfo(float).eps*max(augmented.shape)*s[0]]
    raw = AffineSystem(K, C, G, H, W)
    center = np.sqrt(ranges[:, 0]*ranges[:, 1])
    low = center-args.fraction*(center-ranges[:, 0])
    high = center+args.fraction*(ranges[:, 1]-center)
    shifts = shift_sequence(raw, center, args.terms)
    started = time.perf_counter()
    raw_certificate = cell_certificate(raw, low, high, shifts, args.degree, propagation_model='matrix')
    report = {'metadata': metadata, 'seed': args.seed, 'extraction_tolerance': args.tolerance,
        'final_acceptance_tolerance': 2*args.tolerance, 'raw_order': W.shape[1],
        'raw_basis_sha256': hashlib.sha256(W.tobytes()).hexdigest(),
        'stock_final_order': stock_V.shape[1], 'stock_extraction': stats,
        'low': low.tolist(), 'high': high.tolist(), 'raw_certificate': raw_certificate,
        'selection_history': [], 'floating_point_certified': False,
        'scope': 'one continuous parameter box; fixed stock snapshot span and SVD directions',
        'raw_model_counts': {}, 'reference_method': 'independent exact-time eigenmode Gram'}
    def checkpoint():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    if raw_certificate['bound'] > args.tolerance:
        report['accepted_analytic'] = False
        report['reason'] = 'raw continuous certificate exceeds its tau budget'
        checkpoint()
        return
    small_K, small_C, small_G = W.T @ (K @ W), W.T @ (C @ W), W.T @ G
    small_H = [W.T @ (term @ W) for term in H]
    Kcenter = small_K+sum(value*term for value, term in zip(center, small_H))
    stock_kept = int((singular_values >= args.tolerance*singular_values[0]).sum())
    rng = np.random.default_rng(20261007)
    points = [low, high, center, *rng.uniform(low, high, (4, len(low)))]
    for kept in range(stock_kept, U.shape[1]+1):
        T = svd_candidate_coordinates(U, W, kept)
        J, Q = exact_impulse_gram(Kcenter, small_C, small_G, T)
        center_error = float(np.sqrt(max(0., la.eigvalsh(J, Q)[-1])))
        entry = {'svd_kept': kept, 'final_order': T.shape[1],
                 'small_center_error': center_error,
                 'singular_value_ratio_at_cutoff': float(singular_values[kept-1]/singular_values[0])}
        report['selection_history'].append(entry)
        # Rejection only. No monotonicity assumption, no grid-based acceptance.
        if center_error > args.tolerance:
            entry['point_rejected'] = True
            checkpoint()
            continue
        small = AffineSystem(small_K, small_C, small_G, small_H, T)
        cert = cell_certificate(small, low, high, shifts, args.degree,
                                propagation_model='matrix', spectral_contraction='dense')
        entry['continuous_certificate'] = cert
        for key, value in cert['counts'].items():
            report['raw_model_counts'][key] = report['raw_model_counts'].get(key, 0)+value
        if cert['bound'] > args.tolerance:
            entry['continuous_rejected'] = True
            checkpoint()
            continue
        V = W @ T
        final = AffineSystem(K, C, G, H, V)
        direct_cert = cell_certificate(final, low, high, shifts, args.degree,
                                       propagation_model='matrix')
        references = measure(final, points)
        nested_bound = raw_certificate['bound']+cert['bound']
        report.update({'accepted_analytic': bool(nested_bound <= 2*args.tolerance),
            'selected_svd_kept': kept, 'final_order': V.shape[1],
            'selected_basis_sha256': hashlib.sha256(V.tobytes()).hexdigest(),
            'nested_bound': nested_bound, 'direct_final_certificate': direct_cert,
            'independent_final_references': references,
            'sampled_final_error': max(r['error'] for r in references),
            'stock_references': measure(AffineSystem(K, C, G, H, stock_V), points),
            'seconds': time.perf_counter()-started,
            'rank_selection_eigen_checks': len(report['selection_history'])})
        np.savez(args.output.with_suffix('.npz'), final_basis=V, raw_basis=W, coordinates=T,
                 singular_values=singular_values, K=K.toarray(), C=C.toarray(), G=G,
                 H=np.asarray([term.toarray() for term in H]), low=low, high=high, shifts=shifts,
                 ranges=ranges, tolerance=args.tolerance, seed=args.seed)
        checkpoint()
        print(f"stock={stock_V.shape[1]} guarded={V.shape[1]} nested={nested_bound:.9g} "
              f"direct={direct_cert['bound']:.9g} sample={report['sampled_final_error']:.9g}", flush=True)
        return
    report['accepted_analytic'] = False
    report['reason'] = 'no continuous compression certificate passed'
    checkpoint()


if __name__ == '__main__':
    main()
