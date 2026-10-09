"""Reproduce native matrix comparison and direct final-SVD mesh audit.

Direct final certification is a separate sufficient acceptance route from the
split-tau nested triangle. The extraction tolerance and final 2*tau stay fixed.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import scipy.linalg as la
from run_rational_dynamic import (case1_reconstruction, stock_utils,
    build_parametric_basis, Operators, measure)
from rational_dynamic import AffineSystem, cell_certificate, shift_sequence
from svd_dynamic_guard import svd_candidate_coordinates


def compare_native(mesh):
    from model_case1 import Case1Model, Case1Config
    K, C, G, H, ranges, _ = case1_reconstruction(mesh)
    model = Case1Model(Case1Config(max_xy_cell_mm=mesh, max_z_cell_mm=mesh))
    pairs = [('K', K, model.core.K), ('C', C, model.core.C),
             ('G', G, model.source_shape), ('ranges', ranges, model.h_ranges())]
    pairs += [(f'H{i}', a, b) for i, (a, b) in enumerate(zip(H, model.boundary_terms))]
    row = {'mesh_mm': mesh, 'n': K.shape[0], 'differences': {}}
    for name, a, b in pairs:
        a = a.toarray() if hasattr(a, 'toarray') else a
        b = b.toarray() if hasattr(b, 'toarray') else b
        row['differences'][name] = {'relative_frobenius': float(la.norm(a-b)/la.norm(b)),
                                   'max_abs': float(np.max(abs(a-b)))}
        assert row['differences'][name]['relative_frobenius'] < 1e-12
    return row


def direct_audit(mesh, output):
    K, C, G, H, ranges, _ = case1_reconstruction(mesh)
    snapshots, original = [], stock_utils._snapshot_svd_basis
    def capture(A, tolerance):
        snapshots.append(A.copy())
        return original(A, tolerance)
    with patch.object(stock_utils, '_snapshot_svd_basis', capture):
        stock, stats = build_parametric_basis(Operators(K, C, np.zeros(K.shape[0])),
                     G, H, ranges, tolerance=.001, seed=20260805)
    S = snapshots[0]/la.norm(snapshots[0], axis=0)
    U, s, _ = la.svd(S, full_matrices=False)
    A = np.column_stack([S, np.ones(K.shape[0])/np.sqrt(K.shape[0])])
    W, ws, _ = la.svd(A, full_matrices=False)
    W = W[:, ws > np.finfo(float).eps*max(A.shape)*ws[0]]
    center = np.sqrt(ranges[:, 0]*ranges[:, 1])
    low = center-.0002*(center-ranges[:, 0])
    high = center+.0002*(ranges[:, 1]-center)
    shifts = shift_sequence(AffineSystem(K, C, G, H, W), center, 80)
    report = {'n': K.shape[0], 'mesh_mm': mesh, 'raw_order': W.shape[1],
        'stock_order': stock.shape[1], 'extraction': stats,
        'extraction_tolerance': .001, 'final_threshold': .002,
        'experiments': [], 'floating_point_certified': False,
        'scope': 'direct final-basis certification; bypasses sufficient split-tau budget without modifying tolerance'}
    for kept in [100, 104, 108, 110, 112]:
        if kept > U.shape[1]:
            continue
        T = svd_candidate_coordinates(U, W, kept)
        V = W @ T
        system = AffineSystem(K, C, G, H, V)
        cert = cell_certificate(system, low, high, shifts, 2, propagation_model='matrix')
        refs = measure(system, [low, center, high])
        row = {'svd_kept': kept, 'order': V.shape[1], 'certificate': cert,
               'references': refs, 'accepted_analytic': bool(cert['bound'] <= .002)}
        report['experiments'].append(row)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(kept, V.shape[1], cert['bound'], max(r['error'] for r in refs), flush=True)
        if row['accepted_analytic']:
            np.savez(output.with_suffix('.npz'), final_basis=V, raw_basis=W, coordinates=T,
                singular_values=s, K=K.toarray(), C=C.toarray(), G=G,
                H=np.array([term.toarray() for term in H]), ranges=ranges, tolerance=.001,
                seed=20260805, low=low, high=high, shifts=shifts)
            break
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = [compare_native(mesh) for mesh in [10., 8.]]
    (args.output_dir/'native_matrix_comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    direct_audit(8., args.output_dir/'guard_mesh8_direct.json')


if __name__ == '__main__':
    main()
