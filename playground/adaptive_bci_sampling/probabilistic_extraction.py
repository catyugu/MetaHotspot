"""Source-separated pool greedy and independent residual-operator witnesses.

Research prototype, NOT the production extractor. See PROBABILISTIC_PROOF.md.
Full-field solve counts include witnesses; holdout risk is not uniform safety.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.stats import beta as beta_distribution

from case1_system import case1_reconstruction
from numerics import Solver, operator, decay_lower, c_basis, sym, relative, counts
from metahotspot.macromodel import utils
from metahotspot._compiled_data import Operators


def parameters(ranges, rng, size):
    return np.exp(np.log(ranges[:, 0]) + rng.random((size, len(ranges))) *
                  np.log(ranges[:, 1] / ranges[:, 0]))


def corners(ranges):
    import itertools
    return np.asarray(list(itertools.product(*ranges)))


class Responses:
    """Exact affine residual coordinates, with no full-dimensional online work."""
    def __init__(self, K, C, H, g, V, ranges, shifts):
        self.V = V
        self.r = V.shape[1]
        self.hs = np.sqrt(ranges[:, 0] * ranges[:, 1])
        self.ss = max(float(np.max(shifts)), 1.)
        self.blocks = [K @ V] + [scale * (J @ V) for J, scale in zip(H, self.hs)] + [self.ss * (C @ V)]
        self.reduced = np.asarray([sym(V.T @ B) for B in self.blocks])
        self.g = V.T @ g
        self.R = np.column_stack([g] + [-B for B in self.blocks])
        # Stable QR rather than a squared Gram near residual cancellation.
        _, self.T = la.qr(self.R, mode='economic', check_finite=False)
        self.gnorm = la.norm(g)

    def coefficients(self, h, s):
        h = np.atleast_2d(h)
        s = np.broadcast_to(s, len(h))
        weights = np.column_stack([np.ones(len(h)), h / self.hs, s / self.ss])
        Ar = np.einsum('bi,ijk->bjk', weights, self.reduced)
        y = np.linalg.solve(Ar, np.broadcast_to(self.g[None, :, None], (len(h), self.r, 1)))[:, :, 0]
        coeff = np.column_stack([np.ones(len(h))] + [weights[:, j, None] * y for j in range(weights.shape[1])])
        energy = np.einsum('i,bi->b', self.g, y)
        return coeff, y, energy

    def residual(self, h, s):
        coeff, _, _ = self.coefficients(h, s)
        return la.norm(self.T @ coeff.T, axis=0) / self.gnorm


def stock(K, C, G, H, ranges, seed):
    """Invoke actual utils.py; capture its pre-SVD snapshots without changing decisions."""
    original_svd = utils._snapshot_svd_basis
    original_solve = utils.spd_solve
    capture = {}
    nsolve = 0

    def svd(S, tolerance):
        capture['snapshots'] = S.copy()
        return original_svd(S, tolerance)

    def solve(*args, **kwargs):
        nonlocal nsolve
        nsolve += 1
        return original_solve(*args, **kwargs)

    utils._snapshot_svd_basis = svd
    utils.spd_solve = solve
    try:
        post, info = utils.build_parametric_basis(Operators(K, C, G.sum(axis=1)), G, H, ranges, seed=seed)
    finally:
        utils._snapshot_svd_basis = original_svd
        utils.spd_solve = original_solve
    V = c_basis(np.column_stack([capture['snapshots'], np.ones(len(G))]), C.diagonal())
    info.update(full_rhs=nsolve, pre_svd_independent_order=V.shape[1], post_svd_order=post.shape[1])
    return V, post, info


def pool_greedy(K, C, G, H, ranges, plans, seed, pool_size, tolerance, mode='shared'):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    hp = np.vstack([corners(ranges), parameters(ranges, rng, pool_size)])
    locals_ = []
    cost = []
    histories = []
    if mode == 'shared':
        shifts = np.unique(np.r_[0., *[p['shifts_per_s'] for p in plans]])
        hs = np.repeat(hp, len(shifts), axis=0)
        ss = np.tile(shifts, len(hp))
        V = np.empty((len(G), 0))
        history = []
        for iteration in range(256):
            if V.shape[1]:
                reduced = np.asarray([sym(V.T @ (K @ V))] +
                                     [sym(V.T @ (J @ V)) for J in H] + [sym(V.T @ (C @ V))])
                R = np.column_stack([G, -(K @ V)] + [-(J @ V) for J in H] + [-(C @ V)])
                # Training only: Gram acceleration. Final witnesses use stable QR.
                gram = sym(R.T @ R)
                weights = np.column_stack([np.ones(len(hs)), hs, ss])
                Ar = np.einsum('bi,ijk->bjk', weights, reduced)
                Y = np.linalg.solve(Ar, np.broadcast_to(V.T @ G, (len(hs), V.shape[1], G.shape[1])))
                coeff = np.concatenate([np.broadcast_to(np.eye(G.shape[1]), (len(hs), G.shape[1], G.shape[1]))] +
                                       [weights[:, i, None, None] * Y for i in range(len(reduced))], axis=1)
                values = np.einsum('bip,ij,bjp->bp', coeff, gram, coeff, optimize=True)
                residual = np.sqrt(np.maximum(values, 0.)) / la.norm(G, axis=0)[None, :]
                index, port = np.unravel_index(np.argmax(residual), residual.shape)
                if residual[index, port] <= tolerance:
                    # Resolve any squared-Gram cancellation before accepting training.
                    exact = np.empty_like(residual)
                    for b in range(0, len(hs), 128):
                        field = np.einsum('ij,bjp->bip', R, coeff[b:b+128], optimize=True)
                        exact[b:b+128] = la.norm(field, axis=1) / la.norm(G, axis=0)[None, :]
                    index, port = np.unravel_index(np.argmax(exact), exact.shape)
                    residual = exact
                    if residual[index, port] <= tolerance:
                        break
            else:
                index, port = len(shifts)//2, 0
                residual = np.ones((len(ss), G.shape[1]))
            solver = Solver(operator(K, H, hs[index]) + ss[index] * C, rtol=1e-9)
            x = solver.solve(G[:, port])
            cost.append(solver.counts())
            block = utils.orthonormalize_block(V, x[:, None])
            if not block.shape[1]:
                raise RuntimeError('shared pool greedy stagnated')
            V = np.column_stack([V, block])
            history.append(dict(port=int(port), h=hs[index].tolist(), shift=float(ss[index]), residual=float(residual[index, port])))
        else:
            raise RuntimeError('shared pool greedy reached cap')
        shared = c_basis(np.column_stack([V, np.ones(len(G))]), C.diagonal())
        return shared, [shared] * G.shape[1], dict(full_rhs=len(cost), costs=counts(cost),
            seconds=time.perf_counter()-start, pool_size=len(hp), order=shared.shape[1],
            mode=mode, history=history, per_source_pool_max=residual.max(axis=0).tolist())
    for j, plan in enumerate(plans):
        shifts = np.r_[0., plan['shifts_per_s']]
        hs = np.repeat(hp, len(shifts), axis=0)
        ss = np.tile(shifts, len(hp))
        V = np.empty((len(G), 0))
        g = G[:, j]
        history = []
        for iteration in range(256):
            if V.shape[1]:
                model = Responses(K, C, H, g, V, ranges, shifts)
                residual = model.residual(hs, ss)
                index = int(np.argmax(residual))
                if residual[index] <= tolerance:
                    break
            else:
                index = len(shifts) // 2
                residual = np.ones(len(ss))
            solver = Solver(operator(K, H, hs[index]) + ss[index] * C, rtol=1e-9)
            x = solver.solve(g)
            cost.append(solver.counts())
            # Match the stock per-port Euclidean Galerkin space.
            block = utils.orthonormalize_block(V, x[:, None])
            if not block.shape[1]:
                raise RuntimeError('pool greedy stagnated before residual acceptance')
            V = np.column_stack([V, block])
            history.append(dict(iteration=iteration, h=hs[index].tolist(), shift=float(ss[index]), residual=float(residual[index])))
        else:
            raise RuntimeError('pool greedy reached its order cap')
        locals_.append(V)
        histories.append(dict(port=j, order=V.shape[1], pool_max_residual=float(residual.max()), history=history))
    shared = c_basis(np.column_stack(locals_ + [np.ones((len(G), 1))]), C.diagonal())
    return shared, locals_, dict(full_rhs=len(cost), costs=counts(cost), seconds=time.perf_counter()-start,
                                pool_size=len(hp), histories=histories, order=shared.shape[1])


class Witness:
    """Dimension-free Gaussian upper bound for a fixed residual operator.

    Q W coefficient metric is fit before any witness Gaussian is drawn.
    CG defects are bounded by a verified Collatz lower bound alpha.
    """
    def __init__(self, K, C, H, g, V, ranges, fit_h, correction=None):
        self.model = Responses(K, C, H, g, V, ranges, [0.])
        self.c = C.diagonal()
        self.correction = correction
        R = self.model.R
        if correction is not None and correction.shape[1]:
            self.correction_blocks = np.asarray([sym(correction.T @ (K @ correction))] +
                [scale * sym(correction.T @ (J @ correction)) for scale, J in zip(self.model.hs, H)])
            self.correction_rhs = correction.T @ self.model.R
            R = np.column_stack([R, -(K @ correction)] +
                                [-scale * (J @ correction) for scale, J in zip(self.model.hs, H)])
        sc = np.sqrt(self.c)
        self.Q, self.T = la.qr(R / sc[:, None], mode='economic', check_finite=False)
        coeff, energy = self.coefficients(fit_h)
        Z = (self.T @ coeff.T) / np.sqrt(energy)[None, :]
        covariance = sym(Z @ Z.T / Z.shape[1])
        eigenvalues, U = la.eigh(covariance)
        floor = max(float(eigenvalues[-1]) * 1e-8, 1e-28)
        self.W = (U * np.sqrt(np.maximum(eigenvalues, floor))) @ U.T
        self.Winv = (U / np.sqrt(np.maximum(eigenvalues, floor))) @ U.T
        self.info = dict(port_order=V.shape[1], coefficient_dimension=self.W.shape[0], covariance_floor=floor)

    def coefficients(self, h):
        h = np.atleast_2d(h)
        coeff, _, energy = self.model.coefficients(h, 0.)
        if self.correction is not None and self.correction.shape[1]:
            weights = np.column_stack([np.ones(len(h)), h / self.model.hs])
            Mh = np.einsum('bi,ijk->bjk', weights, self.correction_blocks)
            rhs = self.correction_rhs @ coeff.T
            z = np.linalg.solve(Mh, rhs.T[:, :, None])[:, :, 0]
            coeff = np.column_stack([coeff] + [weights[:, i, None] * z for i in range(weights.shape[1])])
        return coeff, energy

    def certify(self, solver, alpha, rng, probes, delta, E):
        self.E = E
        sc = np.sqrt(self.c)
        self.projection = E.T @ (sc[:, None] * self.Q)
        gaussian = rng.standard_normal((self.W.shape[0], probes))
        B = sc[:, None] * (self.Q @ (self.W @ gaussian))
        if E.shape[1]:
            B -= (solver.A @ E) @ (E.T @ B)
        X = solver.solve(B)
        defects = B - solver.A @ X
        # ||A^{-1/2} B|| <= ||A^{1/2} X|| + ||defect||_{C^-1}/sqrt(alpha).
        xn = np.sqrt(np.maximum(np.sum(X * (solver.A @ X), axis=0), 0.))
        correction = la.norm(defects / sc[:, None], axis=0) / np.sqrt(alpha)
        self.factor = np.sqrt(2 / np.pi) * delta ** (-1 / probes)
        self.L = self.factor * float(np.max(xn + correction))
        self.metric = np.vstack([self.projection, self.L * self.Winv])
        self.info.update(probes=probes,
                         delta=delta, factor=self.factor, operator_bound=self.L,
                         maximum_CG_correction=float(correction.max()), deflation_order=E.shape[1])

    def absolute(self, h):
        coeff, _ = self.coefficients(h)
        return la.norm(self.metric @ (self.T @ coeff.T), axis=0)

    def components(self, h):
        coeff, _ = self.coefficients(h)
        Z = self.T @ coeff.T
        return (self.projection @ Z).T, self.L * la.norm(self.Winv @ Z, axis=0)

    def oracle(self, solver):
        B = np.sqrt(self.c)[:, None] * (self.Q @ self.W)
        if self.E.shape[1]:
            B -= (solver.A @ self.E) @ (self.E.T @ B)
        X = solver.solve(B)
        exactnorm = np.sqrt(max(0., la.eigvalsh(sym(B.T @ X))[-1]))
        return dict(operator_norm=exactnorm, bound=self.L, effectivity=self.L / exactnorm,
                    bound_holds=bool(self.L >= exactnorm))


def shared_energy(K, H, G, V, hs):
    blocks = np.asarray([sym(V.T @ (K @ V))] + [sym(V.T @ (J @ V)) for J in H])
    A = np.einsum('bi,ijk->bjk', np.column_stack([np.ones(len(hs)), hs]), blocks)
    F = V.T @ G
    Y = np.linalg.solve(A, np.broadcast_to(F, (len(hs), *F.shape)))
    return np.einsum('ij,bjk->bik', F.T, Y)


def joint_certify(witnesses, solver, alpha, rng, probes, delta, E):
    """One block operator, separately fitted source metrics; no source dilution."""
    sc = np.sqrt(witnesses[0].c)
    B = sum((sc[:, None] * (w.Q @ (w.W @ rng.standard_normal((w.W.shape[0], probes))))
             for w in witnesses), np.zeros((len(sc), probes)))
    if E.shape[1]:
        B -= (solver.A @ E) @ (E.T @ B)
    X = solver.solve(B)
    defects = B - solver.A @ X
    correction = la.norm(defects / sc[:, None], axis=0) / np.sqrt(alpha)
    norms = np.sqrt(np.maximum(np.sum(X * (solver.A @ X), axis=0), 0.)) + correction
    factor = np.sqrt(2 / np.pi) * delta ** (-1 / probes)
    L = factor * float(norms.max())
    for w in witnesses:
        w.E = E
        w.projection = E.T @ (sc[:, None] * w.Q)
        w.factor, w.L = factor, L
        w.metric = np.vstack([w.projection, L * w.Winv])
        w.info.update(probes=probes, delta=delta, factor=factor, operator_bound=L,
                      maximum_CG_correction=float(correction.max()), deflation_order=E.shape[1], shared_witness=True)


def evaluate_bound(K, H, G, V, witnesses, hs, batch=128):
    out = []
    for start in range(0, len(hs), batch):
        h = hs[start:start+batch]
        parts = [w.components(h) for w in witnesses]
        projections = np.stack([x[0] for x in parts], axis=2)
        tails = np.column_stack([x[1] for x in parts])
        Q = shared_energy(K, H, G, V, h)
        for P, b, q in zip(projections, tails, Q):
            # Retain known cross-source correlations; bound only the unknown tails.
            tail_factor = 1 if witnesses[0].info.get('shared_witness') else len(witnesses)
            envelope = sym(P.T @ P) + tail_factor * np.diag(b*b)
            lam = float(la.eigvalsh(envelope, sym(q))[-1])
            out.append(np.sqrt(max(0., lam) / (1 + max(0., lam))))
        if witnesses[0].correction is not None and witnesses[0].correction.shape[1]:
            w0 = witnesses[0]
            weights = np.column_stack([np.ones(len(h)), h / w0.model.hs])
            Mh = np.einsum('bi,ijk->bjk', weights, w0.correction_blocks)
            rhs = np.stack([w.correction_rhs @ w.model.coefficients(h, 0.)[0].T for w in witnesses], axis=2).transpose(1, 0, 2)
            Z = np.linalg.solve(Mh, rhs)
            exact_correction = np.einsum('bki,bkj->bij', rhs, Z)
            replacement = []
            for D, b, q in zip(exact_correction, tails, Q):
                lam = float(la.eigvalsh(sym(D) + tail_factor*np.diag(b*b), sym(q))[-1])
                replacement.append(np.sqrt(max(0., lam)/(1+max(0., lam))))
            out[-len(h):] = replacement
    return np.asarray(out)


def cover_steady(K, H, G, V, witnesses, ranges, tolerance, max_cells):
    """Reduced interval Neumann cover; no FOM solve and no HTC sampling claim."""
    start = time.perf_counter()
    Kr = sym(V.T @ (K @ V))
    Hr = [sym(V.T @ (J @ V)) for J in H]
    F = V.T @ G
    queue = [ranges.copy()]
    accepted = []
    evaluated = 0
    worst = 0.
    while queue and evaluated < max_cells:
        cell = queue.pop()
        center = np.sqrt(cell[:, 0] * cell[:, 1])
        radii = np.maximum(center-cell[:, 0], cell[:, 1]-center)
        Ac = Kr + sum((h*J for h, J in zip(center, Hr)), np.zeros_like(Kr))
        eigenvalues, U = la.eigh(sym(Ac))
        invsqrt = (U / np.sqrt(eigenvalues)) @ U.T
        sqrtA = (U * np.sqrt(eigenvalues)) @ U.T
        yc = la.solve(Ac, F, assume_a='pos')
        rho = sum(radius * la.norm(invsqrt @ J @ invsqrt, 2) for radius, J in zip(radii, Hr))
        bound = float('inf')
        if rho < 1:
            source_bounds = []
            for j, w in enumerate(witnesses):
                coeff, _, _ = w.model.coefficients(center, 0.)
                zc = w.metric @ (w.T @ coeff[0])
                r = V.shape[1]
                tk = -w.T[:, 1:1+r].copy()
                for i, h in enumerate(center):
                    tk -= h / w.model.hs[i] * w.T[:, 1+(i+1)*r:1+(i+2)*r]
                derivative = 0.
                for i, (radius, J) in enumerate(zip(radii, Hr)):
                    th = w.T[:, 1+(i+1)*r:1+(i+2)*r] / w.model.hs[i]
                    D = tk @ la.solve(Ac, J, assume_a='pos') + th
                    derivative += radius * la.norm(w.metric @ D @ invsqrt, 2)
                coeff_upper = la.norm(zc) + derivative * la.norm(sqrtA @ yc[:, j]) / (1-rho)
                source_bounds.append(coeff_upper)
            Qupper = shared_energy(K, H, G, V, cell[:, 1][None, :])[0]
            lam = float(la.eigvalsh(len(witnesses)*np.diag(np.square(source_bounds)), sym(Qupper))[-1])
            bound = np.sqrt(max(0., lam)/(1+max(0., lam)))
        evaluated += 1
        if bound <= tolerance:
            accepted.append(dict(cell=cell.tolist(), bound=float(bound), rho=float(rho)))
            worst = max(worst, bound)
        else:
            index = int(np.argmax(np.log(cell[:, 1]/cell[:, 0])))
            cut = center[index]
            lo, hi = cell.copy(), cell.copy()
            lo[index, 1] = cut
            hi[index, 0] = cut
            queue.extend([hi, lo])
    return dict(passed=not queue, evaluated_cells=evaluated, accepted_cells=len(accepted),
                remaining_cells=len(queue), worst_accepted_bound=worst, seconds=time.perf_counter()-start,
                scope='simultaneous steady continuous-box guarantee on the witness event; exact-arithmetic theorem',
                leaves=accepted)


def audit_steady(K, C, H, G, ranges, bases, rng, samples):
    hs = np.vstack([corners(ranges), parameters(ranges, rng, samples)])
    rows = []
    costs = []
    for h in hs:
        A = operator(K, H, h)
        solver = Solver(A)
        X = solver.solve(G)
        costs.append(solver.counts())
        row = dict(h=h.tolist())
        for name, V in bases.items():
            Y = la.solve(sym(V.T @ (A @ V)), V.T @ G, assume_a='pos')
            row[name] = relative(X - V @ Y, X, A)
        rows.append(row)
    return dict(rows=rows, costs=counts(costs))


def run(a):
    started = time.perf_counter()
    K, C, G, H, ranges, meta = case1_reconstruction(a.mesh_mm)
    stock_pre, stock_post, baseline = stock(K, C, G, H, ranges, a.seed)
    print(json.dumps(dict(stage='stock_complete', n=len(G), rhs=baseline['full_rhs'], seconds=baseline['seconds'])), flush=True)
    V, locals_, greedy = pool_greedy(K, C, G, H, ranges, baseline['per_port_plans'], a.seed+1, a.pool, a.training_tolerance, a.pool_mode)
    print(json.dumps(dict(stage='pool_complete', n=len(G), rhs=greedy['full_rhs'], seconds=greedy['seconds'], order=V.shape[1])), flush=True)
    rng = np.random.default_rng(a.seed+2)
    fit_h = np.vstack([corners(ranges), parameters(ranges, rng, a.fit)])
    witness_solver = Solver(operator(K, H, ranges[:, 0]))
    alpha = decay_lower(witness_solver.A, C.diagonal(), witness_solver)
    witnesses = [Witness(K, C, H, G[:, j], V, ranges, fit_h) for j in range(G.shape[1])]
    E = np.empty((len(G), 0))
    if a.deflate:
        # Fit a shared inverse range before drawing independent tail witnesses.
        B = sum((np.sqrt(C.diagonal())[:, None] * (w.Q @ (w.W @ rng.standard_normal((w.W.shape[0], a.deflate))))
                 for w in witnesses), np.zeros((len(G), a.deflate)))
        X = witness_solver.solve(B)
        values, U = la.eigh(sym(X.T @ (witness_solver.A @ X)))
        keep = values > values[-1] * 1e-12
        E = (X @ U[:, keep]) / np.sqrt(values[keep])[None, :]
    if a.correction and E.shape[1]:
        witnesses = [Witness(K, C, H, G[:, j], V, ranges, fit_h, correction=E) for j in range(G.shape[1])]
        # Actual K(h)-Galerkin error correction is accounted separately.
        E = np.empty((len(G), 0))
    if a.witness_mode == 'joint':
        joint_certify(witnesses, witness_solver, alpha, rng, a.probes, a.delta/2, E)
    else:
        for w in witnesses:
            w.certify(witness_solver, alpha, rng, a.probes, a.delta/(2*G.shape[1]), E)
    witness_cost = witness_solver.counts().copy()
    oracle = [w.oracle(witness_solver) for w in witnesses] if len(G) <= 2000 else None
    nvalidation = math.ceil(math.log(a.delta/2) / math.log1p(-a.risk))
    # Frozen candidate/metric/witnesses: never use this stream for training.
    validation_h = parameters(ranges, np.random.default_rng(a.seed+3), nvalidation)
    bt = time.perf_counter()
    upper = evaluate_bound(K, H, G, V, witnesses, validation_h)
    failures = int(np.sum(upper > a.tolerance))
    risk_upper = (float(beta_distribution.ppf(1-a.delta/2, failures+1, nvalidation-failures))
                  if failures < nvalidation else 1.)
    validation = dict(samples=nvalidation, failures=failures, tolerance=a.tolerance, risk_requested=a.risk,
                      risk_upper=risk_upper, confidence_delta=a.delta, zero_failure_accept=failures == 0,
                      bound_max=float(upper.max()), bound_median=float(np.median(upper)), seconds=time.perf_counter()-bt,
                      claim='continuous log-uniform HTC violation risk; NOT simultaneous safety of entire box')
    cover = cover_steady(K, H, G, V, witnesses, ranges, a.tolerance, a.cover_cells) if a.cover_cells else None
    audit = audit_steady(K, C, H, G, ranges, dict(stock_pre=stock_pre, stock_post=stock_post, pool_pre=V),
                         np.random.default_rng(a.seed+4), a.audit)
    out = dict(n=len(G), metadata=meta, seed=a.seed, ranges=ranges.tolist(), stock=baseline, pool=greedy,
               witness=dict(alpha=alpha, costs=witness_cost, per_source=[w.info for w in witnesses], small_oracle=oracle),
               validation=validation, continuous_steady_cover=cover, steady_audit=audit, seconds=time.perf_counter()-started,
               total_pool_and_certificate_rhs=greedy['full_rhs']+witness_cost['rhs'],
               scope='pre-SVD steady K-energy. No all-time transient acceptance. Ordinary floating point; no outward rounding.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+'\n')
    np.savez_compressed(a.output.with_suffix('.npz'), V=V, G=G, ranges=ranges,
                        stock_pre=stock_pre, stock_post=stock_post, alpha=alpha)
    print(json.dumps(dict(stage='complete', n=len(G), stock_rhs=baseline['full_rhs'],
                         pool_rhs=greedy['full_rhs'], certificate_rhs=witness_cost['rhs'], validation=validation,
                         audit_max={name:max(row[name] for row in audit['rows']) for name in ['stock_pre','stock_post','pool_pre']},
                         seconds=out['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mesh-mm', type=float, default=10)
    p.add_argument('--seed', type=int, default=20261009)
    p.add_argument('--pool', type=int, default=64)
    p.add_argument('--pool-mode', choices=['shared', 'local'], default='shared')
    p.add_argument('--fit', type=int, default=128)
    p.add_argument('--probes', type=int, default=16)
    p.add_argument('--deflate', type=int, default=0)
    p.add_argument('--correction', action='store_true')
    p.add_argument('--witness-mode', choices=['joint', 'separate'], default='joint')
    p.add_argument('--risk', type=float, default=.01)
    p.add_argument('--delta', type=float, default=1e-6)
    p.add_argument('--tolerance', type=float, default=.001)
    p.add_argument('--training-tolerance', type=float, default=.001)
    p.add_argument('--audit', type=int, default=8)
    p.add_argument('--cover-cells', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not (0 < args.risk < 1 and 0 < args.delta < 1 and args.probes > 0 and args.tolerance > 0 and args.training_tolerance > 0):
        p.error('risk and delta must be in (0,1); probes and tolerances must be positive')
    if args.correction and args.cover_cells:
        p.error('the corrected-residual variant has no implemented continuous-box cover')
    run(args)
