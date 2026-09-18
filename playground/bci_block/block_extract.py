"""Block-response and energy-optimal inexact BCI extraction experiments.

The original FANTASTIC implementation is never patched. Shared projection and
closing response compression are imported unchanged from the audited comparison
branch. Kernels are pure NumPy/SciPy; only extract() requires repository PyAMG.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bci_comparison'))
from algorithms import SharedSpace, response_compress

METHODS = ('full_block', 'ritz_single', 'ritz_block')


def choose_rank(gains, fraction=.9, cap=4):
    """Choose a prefix of DESCENDING nonnegative gains, subject to a hard cap."""
    gains = np.asarray(gains, dtype=float)
    if not 0 < fraction <= 1 or cap < 1:
        raise ValueError('invalid retained-gain fraction or block cap')
    if not np.all(np.isfinite(gains)) or np.any(gains < 0):
        raise ValueError('gains must be finite and nonnegative')
    if not gains.size or gains.sum() == 0:
        return 0
    cutoff = 64 * np.finfo(float).eps * max(gains.size, 1) * gains[0]
    positive = int(np.count_nonzero(gains > cutoff))
    required = int(np.searchsorted(np.cumsum(gains), fraction*gains.sum())) + 1
    return min(required, cap, positive)


def residual_tangents(R, cap=4, fraction=.9):
    """Right singular directions of a column-normalized source residual block."""
    _, singular, vh = la.svd(R, full_matrices=False, check_finite=False)
    gains = singular**2
    k = choose_rank(gains, fraction, cap)
    return vh[:k].T.copy(), {'retained': k, 'gains': gains.tolist()}


def ritz_corrections(A, R, Z, V, AV, Ar, cap=4, fraction=.9):
    """Best rank-k energy correction WITHIN span((I-P_A) Z).

    Usually Z=M R with an SPD approximate inverse M. After A-orthogonal
    deflation against V, whiten Z in the A metric and choose left singular
    vectors of W.T R. The sum of retained squared singular values equals the
    decrease in summed exact A-error squared for Galerkin solutions at THIS
    parameter/shift. This is a local identity, not a global error certificate.
    """
    Z = np.array(Z, dtype=float, copy=True)
    AZ = A @ Z
    if V.shape[1]:
        coeff = la.solve(Ar, V.T @ AZ, assume_a='pos', check_finite=False)
        Z -= V @ coeff
        AZ -= AV @ coeff
    gram = Z.T @ AZ
    gram = (gram + gram.T) * .5
    vals, U = la.eigh(gram, check_finite=False)
    scale = max(float(np.max(np.abs(vals))), np.finfo(float).tiny)
    cutoff = 64*np.finfo(float).eps*max(Z.shape)*scale
    if vals[0] < -cutoff:
        raise ValueError('trial correction has indefinite energy')
    keep = vals > cutoff
    if not np.any(keep):
        return Z[:, :0], {'retained': 0, 'trial_rank': 0, 'retained_gain': 0., 'gains': []}
    W = Z @ (U[:, keep] / np.sqrt(vals[keep]))
    lhs, singular, _ = la.svd(W.T @ R, full_matrices=False, check_finite=False)
    gains = singular**2
    k = choose_rank(gains, fraction, cap)
    return np.ascontiguousarray(W @ lhs[:, :k]), {
        'retained': k, 'trial_rank': int(np.count_nonzero(keep)),
        'retained_gain': float(gains[:k].sum()), 'gains': gains.tolist()}


def extract(core, G, boundary_terms, ranges, *, tolerance, seed, method,
            probe_rounds=10, max_order=2048, fraction=.9, block_cap=4):
    """Two block constructions with identical BCI input/output contracts.

    full_block fully solves a few residual-SVD input combinations at each failed
    sample. ritz_block never claims a converged snapshot: it adds energy-optimal
    directions from one preconditioner application per source. ritz_single uses
    the same calculation but only its first direction. All rejected samples are
    rechecked with the enlarged Galerkin space; no current residual is silently
    accepted. Per-source relative Euclidean acceptance matches the baseline
    criterion but is not a continuous-domain or temperature-error guarantee.
    """
    if method not in METHODS:
        raise ValueError(f'unknown method: {method}')
    from metahotspot.macromodel import utils as stock
    start = time.perf_counter()
    G = np.asarray(G, dtype=float)
    norms = la.norm(G, axis=0)
    if np.any(norms == 0):
        raise ValueError('zero source')
    bounds = [stock.port_eigenvalue_bounds(core.K, core.C, G[:, j]) for j in range(G.shape[1])]
    lo = min(x[0] for x in bounds); hi = max(x[1] for x in bounds)
    count = stock.mpmm_elliptic_shift_count(tolerance, lo, hi)
    shifts = stock.mpmm_elliptic_shifts(count, hi, hi/lo)
    planning = time.perf_counter()-start
    Gn = G/norms
    ops = [core.K, core.C, *boundary_terms]
    space = SharedSpace(ops, Gn)
    rng = np.random.default_rng(seed)
    points = []; history = []
    checks = full_solves = cycles = preconditioners = 0
    times = dict(preconditioner=0., residual=0., enrichment=0., append=0.)
    for shift in shifts:
        accepted = 0
        while accepted < probe_rounds:
            h = stock._draw_h(ranges, rng)
            coeff = (1., float(shift), *h)
            points.append(coeff)
            M = None; A = None
            while True:
                t = time.perf_counter()
                Y, R = space.response(coeff)
                checks += 1
                score = float(np.max(la.norm(R, axis=0)))
                times['residual'] += time.perf_counter()-t
                if score <= tolerance:
                    accepted += 1
                    break
                accepted = 0
                if space.V.shape[1] >= max_order or checks > 10000:
                    raise RuntimeError('declared extraction budget reached')
                if M is None:
                    t = time.perf_counter()
                    A = sum(c*op for c, op in zip(coeff, ops)).tocsc()
                    M = stock._rs_preconditioner(A.tocsr())
                    preconditioners += 1
                    times['preconditioner'] += time.perf_counter()-t
                t = time.perf_counter()
                cap = min(block_cap, max_order-space.V.shape[1])
                if method == 'full_block':
                    D, info = residual_tangents(R, cap, fraction)
                    cols = []
                    for d in D.T:
                        inner = [0]
                        def increment(_x):
                            inner[0] += 1
                        x, flag = sla.cg(A, Gn@d, x0=space.V@(Y@d), M=M,
                                         rtol=stock.ENRICH_RTOL, atol=0.,
                                         maxiter=2000, callback=increment)
                        if flag != 0:
                            raise RuntimeError(f'block full solve failed: {flag}')
                        cols.append(x); full_solves += 1; cycles += inner[0]
                    X = np.column_stack(cols)
                else:
                    Z = np.column_stack([M@R[:, j] for j in range(R.shape[1])])
                    cycles += R.shape[1]
                    Ar = sum(c*op for c, op in zip(coeff, space.small))
                    AV = sum(c*im for c, im in zip(coeff, space.images))
                    X, info = ritz_corrections(A, R, Z, space.V, AV, Ar,
                                              1 if method == 'ritz_single' else cap, fraction)
                times['enrichment'] += time.perf_counter()-t
                before = space.V.shape[1]
                t = time.perf_counter(); added = space.append(X)
                times['append'] += time.perf_counter()-t
                if added == 0:
                    raise RuntimeError('no independent correction before acceptance')
                history.append({'shift': float(shift), 'effective_h': list(h),
                                'residual_before': score, 'order_before': before,
                                'added': added, **info})
    t = time.perf_counter()
    V, comp = response_compress(space.V, space.small, G, points, tolerance)
    compression = time.perf_counter()-t
    return V, {'method': method, 'seconds': time.perf_counter()-start,
               'planning_seconds': planning, 'compression_seconds': compression,
               'stage_seconds': times, 'bounds': bounds, 'shifts_per_s': shifts.tolist(),
               'full_solves': full_solves, 'preconditioners': preconditioners,
               'preconditioner_column_applications': cycles, 'checks': checks,
               'history': history, 'basis_bytes': V.nbytes,
               'peak_explicit_space_bytes': space.V.nbytes + sum(x.nbytes for x in space.images),
               'orthogonality_error': float(la.norm(V.T@V-np.eye(V.shape[1]), ord=2)), **comp}
