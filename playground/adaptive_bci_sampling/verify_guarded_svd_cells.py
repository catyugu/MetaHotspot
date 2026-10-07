"""Independent full-model validation of a saved, fixed guarded SVD basis."""
from pathlib import Path
import argparse
import json
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.integrate import quad_vec
from rational_dynamic import AffineSystem, cell_certificate, exact_impulse_gram
from run_rational_dynamic import case1_reconstruction, measure


def load_saved_system(pack):
    """Validate the saved model itself, including its extraction tolerance."""
    K, C = sp.csc_matrix(pack['K']), sp.csc_matrix(pack['C'])
    H = [sp.csc_matrix(term) for term in pack['H']]
    system = AffineSystem(K, C, pack['G'], H, pack['final_basis'])
    # Compatibility with the first 10-mm experiment, before metadata was saved.
    ranges = pack['ranges'] if 'ranges' in pack else case1_reconstruction(10.)[4]
    threshold = 2*float(pack['tolerance']) if 'tolerance' in pack else .002
    return system, ranges, threshold


def positive_time_reference(K, C, G, V):
    K, C = K.toarray(), C.toarray()
    rates, U = la.eigh(K, C)
    rr, Ur = la.eigh(V.T @ K @ V, V.T @ C @ V)
    W = V @ Ur
    b, br = U.T @ G, W.T @ G
    def integrand(t):
        full = U @ (np.exp(-rates[:, None]*t)*b)
        error = full-W @ (np.exp(-rr[:, None]*t)*br)
        return np.stack([error.T @ C @ error, full.T @ C @ full])
    positive, estimated_error = quad_vec(integrand, 0., np.inf, epsabs=1e-9, epsrel=1e-9)
    J, Q = exact_impulse_gram(K, C, G, V)
    reference = float(np.sqrt(la.eigvalsh(positive[0], positive[1])[-1]))
    algebraic = float(np.sqrt(la.eigvalsh(J, Q)[-1]))
    if abs(reference-algebraic) > 1e-8*algebraic+1e-10:
        raise RuntimeError('independent positive time quadrature differs from eigenmode Gram')
    return {'positive_time_error': reference, 'algebraic_error': algebraic,
            'estimated_quadrature_error': float(estimated_error),
            'positive_J': positive[0].tolist(), 'positive_Q': positive[1].tolist(),
            'method': 'positive integrand quad_vec over [0,infinity); a reference, not a certificate'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--basis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    pack = np.load(args.basis)
    system, ranges, threshold = load_saved_system(pack)
    C, G = system.C, system.G
    center = np.sqrt(ranges[:, 0]*ranges[:, 1])
    report = {'basis_order': system.V.shape[1], 'threshold': threshold,
              'fixed_basis_source': args.basis.name, 'metadata': {'model': 'saved matrices'},
              'experiments': [], 'floating_point_certified': False}
    for fraction, degree in [(0., 0), (.0002, 2), (.002, 2), (.0022, 2), (.0024, 2), (.003, 2), (.002, 3), (.02, 3)]:
        low = center-fraction*(center-ranges[:, 0])
        high = center+fraction*(ranges[:, 1]-center)
        cert = cell_certificate(system, low, high, pack['shifts'], degree, propagation_model='matrix')
        refs = measure(system, [low, high, center,
            *np.random.default_rng(20261007).uniform(low, high, (4, 2))])
        report['experiments'].append({'fraction': fraction, 'certificate': cert, 'references': refs,
            'sampled_error': max(r['error'] for r in refs), 'accepted_analytic': bool(cert['bound'] <= threshold)})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(f"fraction={fraction} p={degree} bound={cert['bound']:.9g} "
              f"sample={report['experiments'][-1]['sampled_error']:.9g}", flush=True)
    report['independent_positive_time_reference'] = positive_time_reference(system.operator(center), C, G, system.V)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(report['independent_positive_time_reference']['positive_time_error'], flush=True)


if __name__ == '__main__':
    main()
