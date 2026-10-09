"""Seeded stress experiments with independent forced-recursion and time references.

These numerical checks challenge the theorem's implementation; they do not
replace the continuous-spectrum/continuous-parameter proof or interval arithmetic.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.linalg as la
from matrix_innovation import innovation_gram, spectral_majorant, joint_control_norm
from rational_dynamic import exact_impulse_gram
from verify_guarded_svd_cells import load_saved_system, positive_time_reference


def stress(seed=20261007, cases=100):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(cases):
        n, N, m = 7, int(rng.integers(4, 25)), 3
        U, Z = [la.qr(rng.normal(size=(n, n)))[0] for _ in range(2)]
        C = Z @ np.diag(np.geomspace(.2, 5., n)) @ Z.T
        Ka = U @ np.diag(np.geomspace(.01, 20., n)) @ U.T
        B = rng.normal(size=(n, n))
        K = Ka+.02*B @ B.T
        shifts = np.exp(rng.uniform(np.log(.001), np.log(100.), N))
        F = rng.normal(size=(1, N, n, m))
        actions = la.solve(Ka, F[0].transpose(1, 0, 2).reshape(n, -1), assume_a='pos')
        actions = actions.reshape(n, N, m).transpose(1, 0, 2)[None]
        rates = la.eigvalsh(K, C)
        M, meta = spectral_majorant(shifts, (rates[0], rates[-1]))
        bound = joint_control_norm(F, actions, 1, np.eye(m), M)
        q, energy = np.zeros((n, m)), np.zeros((m, m))
        for sigma, f in zip(shifts, F[0]):
            x = np.sqrt(2*sigma)*la.solve(K+sigma*C, q+f, assume_a='pos')
            energy += x.T @ C @ x
            q -= np.sqrt(2*sigma)*C @ x
        energy += .5*q.T @ la.solve(K, q, assume_a='pos')
        actual = np.sqrt(la.eigvalsh(energy)[-1])
        probes = np.r_[rates, np.geomspace(rates[0], rates[-1], 101)]
        slack = min(la.eigvalsh(M-innovation_gram(rate, shifts))[0] for rate in probes)
        # A dense, nonorthogonal congruence checks coordinate invariance.
        P = la.qr(rng.normal(size=(n, n)))[0] @ np.diag(np.geomspace(.5, 2., n))
        Ft = np.einsum('ab,tbc->tac', P.T, F[0])[None]
        At = np.einsum('ab,tbc->tac', la.inv(P), actions[0])[None]
        changed = joint_control_norm(Ft, At, 1, np.eye(m), M)
        assert actual <= bound*(1+1e-10), (index, actual, bound)
        assert slack >= -1e-10*max(1., la.norm(M, 2)), (index, slack)
        assert abs(changed-bound) <= 1e-10*bound
        rows.append({'case': index, 'terms': N, 'spectral_interval': rates[[0,-1]].tolist(),
                     'bound': bound, 'actual': float(actual), 'ratio': float(bound/actual),
                     'minimum_probed_loewner_slack': float(slack),
                     'coordinate_relative_difference': float(abs(changed-bound)/bound),
                     'majorant': meta})
    return {'seed': seed, 'cases': cases, 'violations': 0, 'rows': rows,
            'floating_point_certified': False,
            'scope': 'dense SPD C and K; unsorted shifts; three input directions; exact terminal energy'}


def validate_pack(path):
    pack = np.load(path)
    system, ranges, threshold = load_saved_system(pack)
    center = np.sqrt(ranges[:, 0]*ranges[:, 1])
    rng = np.random.default_rng(20261008)
    fraction = .0022
    low = center-fraction*(center-ranges[:, 0])
    high = center+fraction*(ranges[:, 1]-center)
    points = [low, high, center, *rng.uniform(low, high, (32, len(low)))]
    references = []
    for h in points:
        J, Q = exact_impulse_gram(system.operator(h), system.C, system.G, system.V)
        values, inputs = la.eigh(J, Q)
        a = inputs[:, -1]
        references.append({'h': h.tolist(), 'error': float(np.sqrt(max(0., values[-1]))),
            'worst_input_Q_normalized': a.tolist(),
            'rayleigh_error': float(np.sqrt(max(0., a @ J @ a/(a @ Q @ a))))})
    positives = [dict(h=h.tolist(), **positive_time_reference(system.operator(h),
                    system.C, system.G, system.V)) for h in [low, center, high]]
    return {'basis': path.name, 'order': system.V.shape[1], 'threshold': threshold,
            'fraction': fraction, 'references': references, 'positive_time_references': positives,
            'sampled_max': max(r['error'] for r in references),
            'floating_point_certified': False,
            'scope': 'independent audit points and positive quadrature; acceptance uses separate continuous certificates'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--basis', nargs='*', type=Path, default=[])
    args = p.parse_args()
    report = {'stress': stress(), 'basis_audits': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in args.basis:
        audit = validate_pack(path)
        report['basis_audits'].append(audit)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(path.name, audit['sampled_max'], flush=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('stress passed:', report['stress']['cases'], flush=True)


if __name__ == '__main__':
    main()
