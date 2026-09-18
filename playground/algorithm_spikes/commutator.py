"""Controlled screen of filtered commutator directions versus ordered actions.

This is NOT BIRKA or a complete nice-selection implementation. The strongest
independent baseline here is a converged symmetric generalized-Lyapunov Gramian
subspace under a recorded conservative bilinear input scaling. All evaluations
use the original, unscaled time-varying family. Exact semigroup steps remove
time-discretization bias. No test trajectory participates in basis selection.
"""
from __future__ import annotations
import itertools
import time
import numpy as np
from scipy import linalg as la
from common import basis, extend, fingerprint, sym


def make_family(n: int, seed: int, kind: str, alpha: float):
    if kind not in ('low_rank', 'dense') or n < 8 or alpha < 0:
        raise ValueError('invalid family parameters')
    rng = np.random.default_rng(seed)
    d0 = np.geomspace(1., 80., n)
    d1 = rng.permutation(np.geomspace(.3, 25., n))
    d2 = rng.permutation(np.geomspace(.2, 30., n))
    if kind == 'low_rank':
        q = la.qr(rng.normal(size=(n, 2)), mode='economic')[0]
        skew = np.outer(q[:, 0], q[:, 1])-np.outer(q[:, 1], q[:, 0])
    else:
        skew = rng.normal(size=(n, n)); skew -= skew.T
        skew /= la.norm(skew, 2)
    rotation = la.expm(alpha*skew)
    ops = [np.diag(d0), np.diag(d1), sym((rotation*d2)@rotation.T)]
    # Same port directions across alpha and family kind (separate RNG).
    b = np.random.default_rng(seed+700).normal(size=(n, 2))
    b /= la.norm(b, axis=0)
    return {'operators': ops, 'B': b}


def projected_defect(a, b, v):
    qa = a@v-v@(v.T@a@v); qb = b@v-v@(v.T@b@v)
    ar, br = v.T@a@v, v.T@b@v
    comm_v = a@(b@v)-b@(a@v)
    lhs = v.T@comm_v-(ar@br-br@ar)
    rhs = qa.T@qb-qb.T@qa
    return lhs, rhs, comm_v-v@(v.T@comm_v)


def exact_step(a, x, force, dt):
    eig, u = la.eigh(a, check_finite=False)
    if eig[0] <= 0 or dt <= 0:
        raise ValueError('positive operator and timestep required')
    decay = np.exp(-dt*eig)
    phi = -np.expm1(-dt*eig)/eig
    return u@(decay*(u.T@x)+phi*(u.T@force))


def trajectory(ops, b, mus, inputs, dt):
    x = np.zeros(b.shape[0]); states = []
    for mu, control in zip(mus, inputs):
        a = ops[0]+mu[0]*ops[1]+mu[1]*ops[2]
        eig, u = la.eigh(a, check_finite=False)
        decay = np.exp(-dt*eig/3.)
        phi = -np.expm1(-dt*eig/3.)/eig
        force = u.T@(b@control)
        for _ in range(3):
            x = u@(decay*(u.T@x)+phi*force)
            states.append(x.copy())
    return np.asarray(states)


def homogeneous_terminal(ops, initial, mus, dt):
    x = initial.copy()
    for mu in mus:
        a = ops[0]+mu[0]*ops[1]+mu[1]*ops[2]
        eig, u = la.eigh(a, check_finite=False)
        x = u@(np.exp(-dt*eig)*(u.T@x))
    return x


def bilinear_gramian(a, ns, b):
    eig, u = la.eigh(a, check_finite=False)
    # Sufficient contraction in Frobenius norm; avoids an undefined H2 metric.
    norm_sq = sum(la.norm(n, 2)**2 for n in ns)
    gamma = min(1., np.sqrt(.6*2.*eig[0]/max(norm_sq, 1e-30)))
    denom = eig[:, None]+eig[None, :]
    def solve(q):
        return u@((u.T@q@u)/denom)@u.T
    bb = b@b.T; p = solve(bb)
    for iteration in range(1, 301):
        nxt = solve(bb+gamma**2*sum(n@p@n for n in ns))
        p = sym(nxt)
        res = -a@p-p@a+gamma**2*sum(n@p@n for n in ns)+bb
        residual = float(la.norm(res)/max(la.norm(bb), 1e-30))
        if residual < 1e-11:
            break
    else:
        raise RuntimeError('generalized Gramian did not converge')
    return p, {'gamma': float(gamma), 'iterations': iteration, 'relative_residual': residual}


def prepare(family, rank, seed):
    ops, b = family['operators'], family['B']; n, m = b.shape
    shifts = [.1, 1., 10., 100.]
    environments = [(0.,0.), (1.,0.), (0.,1.), (1.,1.), (.5,.5)]
    begin = time.perf_counter()
    snapshots = np.column_stack([
        la.solve(ops[0]+mu[0]*ops[1]+mu[1]*ops[2]+s*np.eye(n), b, assume_a='pos')
        for mu in environments for s in shifts
    ])
    snapshots /= la.norm(snapshots, axis=0)
    base = basis(snapshots, 6)
    base_seconds = time.perf_counter()-begin
    star = ops[0]+.5*(ops[1]+ops[2])
    begin = time.perf_counter()
    words = []; comm = []; anti = []
    for s in shifts:
        chol = la.cho_factor(star+s*np.eye(n), check_finite=False)
        rb = la.cho_solve(chol, b, check_finite=False)
        for i, j in itertools.combinations(range(3), 2):
            u = la.cho_solve(chol, ops[i]@(ops[j]@rb), check_finite=False)
            v = la.cho_solve(chol, ops[j]@(ops[i]@rb), check_finite=False)
            scale = np.maximum(np.maximum(la.norm(u, axis=0), la.norm(v, axis=0)), 1e-30)
            u, v = u/scale, v/scale
            words.extend([u, v]); comm.append(u-v); anti.append(u+v)
    word_seconds = time.perf_counter()-begin
    pools = {'filtered_commutator': np.column_stack(comm),
             'filtered_ordered': np.column_stack(words),
             'filtered_symmetric': np.column_stack(anti)}
    prepared = {}; costs = {}
    # All three are explicitly charged the SAME underlying ordered-action bank.
    for name, candidates in pools.items():
        start = time.perf_counter()
        if la.norm(candidates) < 1e-10:
            candidates = snapshots
        prepared[name] = extend(base, candidates, rank, snapshots)
        costs[name] = {'offline_s': base_seconds+word_seconds+time.perf_counter()-start,
                       'column_solves': 40+8+48, 'operator_column_actions': 96, 'matrix_factorizations':24,
                       'training_kind': 'same_ordered_bank'}
    start = time.perf_counter()
    rng = np.random.default_rng(seed+101)
    extra = []
    for mu, s in zip(rng.uniform(0, 1, (28, 2)), np.geomspace(.06, 160., 28)):
        z = la.solve(ops[0]+mu[0]*ops[1]+mu[1]*ops[2]+s*np.eye(n), b, assume_a='pos')
        extra.append(z/la.norm(z, axis=0))
    prepared['frozen_more'] = extend(base, np.column_stack(extra), rank, snapshots)
    costs['frozen_more'] = {'offline_s': base_seconds+time.perf_counter()-start,
                            'column_solves': 40+56, 'operator_column_actions': 0, 'matrix_factorizations':48,
                            'training_kind': 'same_column_solve_budget'}
    start = time.perf_counter()
    p, ginfo = bilinear_gramian(star, ops[1:], b)
    prepared['bilinear_gramian'] = la.eigh(p, check_finite=False)[1][:, -rank:]
    costs['bilinear_gramian'] = {'offline_s': time.perf_counter()-start,
                                'column_solves': 0, 'operator_column_actions': 0, 'matrix_factorizations':0,
                                'training_kind': 'dense_generalized_lyapunov', **ginfo}
    comm_s = la.svdvals(pools['filtered_commutator'])
    info = {'bank_commutator_norm': float(la.norm(pools['filtered_commutator'])),
            'bank_commutator_rank_1e6': int(np.sum(comm_s > max(comm_s[0]*1e-6, 1e-10))),
            'base_seconds': base_seconds, 'base_rank': base.shape[1]}
    return prepared, costs, info


def run(seed, smoke=False):
    n = 24 if smoke else 48
    kinds = ['low_rank'] if smoke else ['low_rank', 'dense']
    alphas = [0.8] if smoke else [0., .08, .8]
    ranks = [10] if smoke else [10, 14]
    rows = []; audits = []
    rng = np.random.default_rng(seed+4001)
    # Same test paths across operator families; never used by prepare().
    mus = rng.uniform(0., 1., (24, 2)); controls = rng.uniform(0., 1., (24, 2))
    mus[::4] = [1.,0.]; mus[2::4] = [0.,1.]
    dt_values = [.04] if smoke else [.005, .05, .5]
    paths = [(mus, controls), (mus[::-1], controls[::-1])]
    params = rng.uniform(0., 1., (18, 2)); shifts = np.geomspace(.07, 140., 18)
    for kind, alpha in itertools.product(kinds, alphas):
        fam = make_family(n, seed, kind, alpha); ops, b = fam['operators'], fam['B']
        refs = {(p, dt): trajectory(ops, b, path[0], path[1], dt)
                for p, path in enumerate(paths) for dt in dt_values}
        frozen = [la.solve(ops[0]+mu[0]*ops[1]+mu[1]*ops[2]+s*np.eye(n), b, assume_a='pos')
                  for mu, s in zip(params, shifts)]
        for rank in ranks:
            models, costs, info = prepare(fam, rank, seed)
            audits.append({'kind':kind, 'alpha':alpha, 'rank':rank, **info, 'costs':costs})
            for method, v in models.items():
                projected = [sym(v.T@a@v) for a in ops]; br = v.T@b
                frozen_error = []
                for mu, s, ref in zip(params, shifts, frozen):
                    xr = v@la.solve(projected[0]+mu[0]*projected[1]+mu[1]*projected[2]+s*np.eye(v.shape[1]), br, assume_a='pos')
                    frozen_error.append(la.norm(xr-ref)/la.norm(ref))
                for dt in dt_values:
                    errors = []; outputs = []; maxerr = []; online = 0.
                    reduced_paths = []
                    for p, path in enumerate(paths):
                        start = time.perf_counter()
                        small = trajectory(projected, br, path[0], path[1], dt)
                        xr = small@v.T
                        online += time.perf_counter()-start
                        ref = refs[p, dt]; delta = xr-ref
                        errors.append(la.norm(delta)/la.norm(ref))
                        outputs.append(la.norm(delta@b)/max(la.norm(ref@b), 1e-30))
                        maxerr.append(np.max(la.norm(delta, axis=1))/np.max(la.norm(ref, axis=1)))
                        reduced_paths.append(xr)
                    order_ref = homogeneous_terminal(ops, b[:,0], mus, dt)-homogeneous_terminal(ops, b[:,0], mus[::-1], dt)
                    order_rom = v@(homogeneous_terminal(projected, br[:,0], mus, dt)-homogeneous_terminal(projected, br[:,0], mus[::-1], dt))
                    order_err = order_rom-order_ref
                    row = {'kind':kind, 'alpha':alpha, 'n':n, 'rank':v.shape[1], 'dt':dt,
                           'method':method, 'state_relative_l2':float(max(errors)),
                           'output_relative_l2':float(max(outputs)), 'peak_state_relative':float(max(maxerr)),
                           'frozen_relative_max':float(max(frozen_error)),
                           'order_effect_absolute':float(la.norm(order_ref)),
                           'order_effect_error_absolute':float(la.norm(order_err)),
                           'online_two_paths_s':online, 'basis_bytes':v.nbytes, **costs[method]}
                    rows.append(row)
            print(f'commutator {kind=} {alpha=} {rank=} complete', flush=True)
    return rows, {'dimensionless_symmetric_matrices_not_native_fvm':True,
                  'test_path_sha256':fingerprint(mus,controls,params,shifts), 'basis_audits':audits,
                  'sequences':2, 'segments_per_sequence':24, 'samples_per_segment':3,
                  'no_BIRKA_claim':True, 'holdout_used_for_selection':False}
