"""Rational-time/Bernstein continuous-box impulse-error research prototype.

The theorem is exact-arithmetic. This implementation does not use interval
arithmetic: `floating_point_certified` is always False. It certifies a fixed
Galerkin basis, not a new extraction method. See RATIONAL_DYNAMIC_PROOF.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import comb

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from matrix_innovation import spectral_majorant, joint_control_norm


@dataclass
class AffineSystem:
    K: object
    C: object
    G: np.ndarray
    H: list
    V: np.ndarray
    counts: dict = field(default_factory=dict)

    def operator(self, h):
        if len(h) != len(self.H):
            raise ValueError('parameter dimension mismatch')
        result = self.K.copy()
        for value, term in zip(h, self.H):
            result = result + value * term
        return result


def _dense(A):
    return A.toarray() if sp.issparse(A) else np.asarray(A)


class _Solve:
    def __init__(self, A, counts=None, phase='trial'):
        self.counts, self.phase = counts, phase
        if sp.issparse(A):
            self.solve = sla.splu(A.tocsc()).solve
        else:
            factor = la.cho_factor(np.asarray(A), check_finite=False)
            self.solve = lambda b: la.cho_solve(factor, b, check_finite=False)
        if counts is not None:
            key = phase + '_factorizations'
            counts[key] = counts.get(key, 0) + 1

    def __call__(self, b):
        if self.counts is not None:
            key = self.phase + '_rhs'
            self.counts[key] = self.counts.get(key, 0) + (b.shape[1] if b.ndim > 1 else 1)
        return self.solve(b)


def coefficients(K, C, G, shifts, counts=None):
    """Physical-state coefficients; dual residual q, not C-whitened vectors.

    q0=G; xk=sqrt(2*sigma)*(K+sigma*C)^-1 qk;
    q{k+1}=qk-sqrt(2*sigma)*C*xk. A sign-alternated TM convention is used.
    """
    q = np.array(G, dtype=float, copy=True)
    xs = []
    factors = {}
    for sigma in shifts:
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('all temporal shifts must be positive and finite')
        if sigma not in factors:
            factors[sigma] = _Solve(K + sigma*C, counts)
        x = np.sqrt(2*sigma) * factors[sigma](q)
        xs.append(x)
        q -= np.sqrt(2*sigma)*(C @ x)
    return np.asarray(xs), q


def shift_sequence(system, center, count):
    """Diagnostic scheduling only; validity does not require certified Ritz values.

    A log cycle spans nominal reduced low rates and a full operator row-sum
    upper estimate. Its size is tied to that span, not fitted to error results.
    The actual computed remainder is always included in the bound.
    """
    K = system.operator(center)
    V, C = system.V, system.C
    rates = la.eigvalsh(V.T @ (K @ V), V.T @ (C @ V))
    low = max(float(rates[0])/10., np.finfo(float).tiny)
    cmin = float(np.min(C.diagonal())) if sp.issparse(C) else float(la.eigvalsh(C)[0])
    upper = float(np.max(np.asarray(abs(K).sum(axis=1)))) / cmin
    high = max(low, upper)
    cycle_size = max(1, int(np.ceil(np.log(high/low)/np.log(3)))+1)
    cycle = np.geomspace(low, high, cycle_size)
    return np.resize(cycle, count)


def exact_impulse_gram(K, C, G, V):
    """Independent dense full/reduced eigenmode integration, no time truncation.

    This subtractive reference can lose relative accuracy near machine-zero
    error. It is not the certificate and is not suitable for very large FOMs.
    """
    K, C = _dense(K), _dense(C)
    rates, U = la.eigh(K, C, check_finite=False)
    rr, Ur = la.eigh(V.T @ K @ V, V.T @ C @ V, check_finite=False)
    W = V @ Ur
    b, br = U.T @ G, W.T @ G
    Q = b.T @ (b/(2*rates[:, None]))
    Qr = br.T @ (br/(2*rr[:, None]))
    cross = b.T @ ((U.T @ C @ W)/(rates[:, None]+rr)) @ br
    J = Q + Qr - cross - cross.T
    return (J+J.T)/2, (Q+Q.T)/2


def _axis_transform(a, matrix, axis):
    transformed = np.tensordot(matrix, a, axes=(1, axis))
    return np.moveaxis(transformed, 0, axis)


def polynomial_controls(power, dimension):
    """Convert exact tensor power coefficients to tensor Bernstein controls."""
    result = np.asarray(power)
    for axis in range(dimension):
        degree = result.shape[axis]-1
        transform = np.zeros((degree+1, degree+1))
        for i in range(degree+1):
            for j in range(i+1):
                transform[i, j] = comb(i, j)/comb(degree, j)
        result = _axis_transform(result, transform, axis)
    return result


def _apply(A, p):
    """Apply a state matrix to a tensor of polynomial matrix coefficients."""
    shape = p.shape
    n, m = shape[-2:]
    flat = p.reshape(-1, n, m).transpose(1, 0, 2).reshape(n, -1)
    out = A @ flat
    nout = A.shape[0]
    return np.asarray(out).reshape(nout, -1, m).transpose(1, 0, 2).reshape((*shape[:-2], nout, m))


def _inverse_norm_controls(power, dimension, solver, return_actions=False):
    controls = polynomial_controls(power, dimension)
    shape, n, m = controls.shape[:-2], *controls.shape[-2:]
    flat = controls.reshape(-1, n, m)
    rhs = flat.transpose(1, 0, 2).reshape(n, -1)
    actions = solver(rhs).reshape(n, -1, m).transpose(1, 0, 2)
    grams = np.einsum('bni,bnj->bij', flat, actions)
    eigenvalues = np.linalg.eigvalsh((grams+grams.transpose(0, 2, 1))/2)
    norm = float(np.sqrt(max(0., eigenvalues[:, -1].max())))
    if return_actions:
        return norm, controls, actions.reshape(controls.shape)
    return norm


def _m_matrix_interval(Ka, Kb, C, solver):
    """Collatz lower bound from a positive test vector; row-sum upper bound.

    Valid for symmetric Z-matrix stiffness and positive diagonal C. The vector
    need not be an exact solution: the actual Ka*v is used in the quotient.
    Floating arithmetic is still not outward rounded.
    """
    Cd = np.asarray(C.diagonal())
    if sp.issparse(C):
        if (C-sp.diags(Cd)).nnz:
            return None
    elif not np.array_equal(C, np.diag(Cd)):
        return None
    if np.any(Cd <= 0):
        return None
    for K in [Ka, Kb]:
        off = K-sp.diags(K.diagonal()) if sp.issparse(K) else K-np.diag(K.diagonal())
        if sp.issparse(off):
            if off.nnz and off.data.max() > 0:
                return None
        elif np.any(off > 0):
            return None
    v = solver(Cd[:, None])[:, 0]
    if np.any(v <= 0):
        return None
    alpha = float(np.min((Ka @ v)/(Cd*v)))
    beta = float(np.max(np.asarray(abs(Kb).sum(axis=1)).ravel()/Cd))
    return (alpha, beta) if 0 < alpha <= beta else None


def split_bernstein(controls, axis):
    """Exact de Casteljau subdivision at one half, along a tensor axis."""
    work = np.moveaxis(controls, axis, 0).copy()
    degree = work.shape[0]-1
    left, right = np.empty_like(work), np.empty_like(work)
    left[0], right[degree] = work[0], work[-1]
    for k in range(1, degree+1):
        work = (work[:-1]+work[1:])/2
        left[k], right[degree-k] = work[0], work[-1]
    return np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis)


def _refined_envelopes(data, low, high, depth):
    """Stream exact polynomial restrictions; retain the original inverse actions."""
    dimension = len(low)
    def visit(payload, a, b, step):
        if step == depth*dimension:
            yield payload, a, b
            return
        axis = step % dimension
        splits = [split_bernstein(item, axis) for item in payload]
        middle = (a[axis]+b[axis])/2
        left_b, right_a = b.copy(), a.copy()
        left_b[axis], right_a[axis] = middle, middle
        yield from visit(tuple(pair[0] for pair in splits), a, left_b, step+1)
        yield from visit(tuple(pair[1] for pair in splits), right_a, b, step+1)
    yield from visit(data, low, high, 0)


def _dual_control_sequence_norms(controls, actions, dimension, change):
    controls, actions = controls @ change, actions @ change
    n, m = controls.shape[-2:]
    terms = controls.shape[dimension]
    f = controls.reshape(-1, terms, n, m)
    a = actions.reshape(f.shape)
    gram = np.einsum('btni,btnj->btij', f, a)
    gram = (gram+gram.swapaxes(-1, -2))/2
    return np.sqrt(np.maximum(0., np.linalg.eigvalsh(gram)[..., -1].max(axis=0)))


def innovation_energy_bound(residual_norms, shifts, spectral_interval=None):
    """Bound the joint coefficient-defect / terminal-energy Hilbert norm.

    A residual injected at j has squared gain b_j=2*sigma/(lambda+sigma).
    Its cross Gram with injection k>j is
    -b_j*b_k*prod(t_{j+1:k})/2. Absolute spectral envelopes give a
    continuous-spectrum quadratic majorant, rather than N scalar state tubes.
    """
    eta, shifts = np.asarray(residual_norms), np.asarray(shifts)
    if eta.shape != shifts.shape or np.any(eta < 0):
        raise ValueError('one nonnegative residual norm per shift required')
    if spectral_interval is None:
        b, rho = np.full(len(shifts), 2.), np.ones(len(shifts))
    else:
        alpha, beta = spectral_interval
        b = 2*shifts/(alpha+shifts)
        rho = np.maximum(abs((alpha-shifts)/(alpha+shifts)),
                         abs((beta-shifts)/(beta+shifts)))
    square = float(np.dot(b, eta**2))
    for j in range(len(shifts)):
        contraction = 1.
        for k in range(j+1, len(shifts)):
            square += b[j]*b[k]*contraction*eta[j]*eta[k]
            contraction *= rho[k]
    # Both bounds are valid; avoid worsening the triangle inequality.
    return min(float(np.sqrt(max(0., square))), float(np.dot(np.sqrt(b), eta)))


def cell_certificate(system, low, high, shifts, degree=1, denominator='exact',
                     spectral_contraction=True, propagation_model='independent', envelope_depth=0):
    """Exact-arithmetic upper-bound construction on one affine parameter box.

    Arbitrary polynomial full/reduced coefficient trials are validated through
    their *exact polynomial residuals*. Nodal interpolation is only a way to
    build trials, never a certificate for values between the nodes.
    """
    low, high = np.asarray(low, float), np.asarray(high, float)
    shifts = np.asarray(shifts, float)
    d, n, m = len(low), system.G.shape[0], system.G.shape[1]
    if high.shape != low.shape or d != len(system.H) or np.any(high < low):
        raise ValueError('invalid cell')
    if degree < 0 or int(degree) != degree or not len(shifts):
        raise ValueError('nonnegative integer degree and nonempty shifts required')
    if np.any(shifts <= 0) or not np.all(np.isfinite(shifts)):
        raise ValueError('all temporal shifts must be positive and finite')
    if int(envelope_depth) != envelope_depth or envelope_depth < 0:
        raise ValueError('nonnegative integer envelope_depth required')
    envelope_depth = int(envelope_depth)
    if envelope_depth and propagation_model not in ['energy', 'matrix']:
        raise ValueError('envelope refinement requires energy or matrix propagation')
    degree = int(degree)
    V, C = system.V, system.C
    Ka, Kb = system.operator(low), system.operator(high)
    Kr_a, Kr_b, Cr = V.T @ (Ka @ V), V.T @ (Kb @ V), V.T @ (C @ V)
    Gr = V.T @ system.G
    counts = {}
    if denominator == 'exact':
        Qb = .5 * system.G.T @ _Solve(Kb, counts, 'denominator')(system.G)
    elif denominator == 'reduced':
        Qb = .5 * Gr.T @ la.solve(Kr_b, Gr, assume_a='pos')
    else:
        raise ValueError('denominator must be exact or reduced')
    eigenvalues = la.eigvalsh(Qb)
    if eigenvalues[0] <= np.finfo(float).eps * m * eigenvalues[-1]:
        raise ValueError('denominator lower bound numerically rank deficient; no regularization')
    root = la.cholesky(Qb, lower=True)
    normalize = la.solve_triangular(root.T, np.eye(m))
    G = system.G @ normalize
    Gr = V.T @ G

    nodes = np.array([.5]) if degree == 0 else (1-np.cos(np.pi*np.arange(degree+1)/degree))/2
    shape = (degree+1,)*d
    values = np.empty((*shape, len(shifts), n, m))
    reduced_values = np.empty((*shape, len(shifts), V.shape[1], m))
    for index in product(range(degree+1), repeat=d):
        h = low+(high-low)*nodes[list(index)]
        K = system.operator(h)
        values[index], _ = coefficients(K, C, G, shifts, counts)
        reduced_values[index], _ = coefficients(V.T @ (K @ V), Cr, Gr, shifts)
    inverse_vandermonde = la.inv(np.vander(nodes, degree+1, increasing=True))
    for axis in range(d):
        values = _axis_transform(values, inverse_vandermonde, axis)
        reduced_values = _axis_transform(reduced_values, inverse_vandermonde, axis)

    # Use degree p+1 on every parameter axis for all residual polynomials.
    rshape = (degree+2,)*d
    q = np.zeros((*rshape, n, m))
    qr = np.zeros((*rshape, V.shape[1], m))
    zero = (0,)*d
    q[zero], qr[zero] = G, Gr
    pad_slice = (slice(0, degree+1),)*d
    solve_a = _Solve(Ka, counts, 'riesz')
    solve_ra = _Solve(Kr_a)
    if spectral_contraction == 'dense':
        spectral_interval = (float(la.eigvalsh(_dense(Ka), _dense(C))[0]),
                             float(la.eigvalsh(_dense(Kb), _dense(C))[-1]))
        counts['dense_spectral_eigendecompositions'] = 2
    else:
        spectral_interval = _m_matrix_interval(Ka, Kb, C, solve_a) if spectral_contraction else None
    if propagation_model not in ['independent', 'coupled', 'energy', 'matrix']:
        raise ValueError('propagation_model must be independent, coupled, energy or matrix')
    coupling_gain = lift_gain = None
    if propagation_model == 'coupled':
        lift = (C @ V) @ la.inv(Cr)
        inverse_mass_root = la.solve_triangular(la.cholesky(Cr, lower=True).T, np.eye(Cr.shape[0]))
        coupling_gain = 0.
        for corner in product([0, 1], repeat=d):
            Kcorner = system.operator(np.where(corner, high, low))
            residual = (Kcorner @ V-lift @ (V.T @ (Kcorner @ V))) @ inverse_mass_root
            gram = residual.T @ solve_a(residual)
            coupling_gain = max(coupling_gain, np.sqrt(max(0., la.eigvalsh((gram+gram.T)/2)[-1])))
        gram = lift.T @ solve_a(lift)
        lift_gain = np.sqrt(max(0., la.eigvalsh((gram+gram.T)/2, la.inv(Kr_b))[-1]))
    matrix_majorant = matrix_metadata = None
    if propagation_model == 'matrix':
        matrix_majorant, matrix_metadata = spectral_majorant(shifts, spectral_interval)
    retain_actions = bool(envelope_depth or propagation_model == 'matrix')
    full_tube = reduced_tube = 0.
    coefficient_tubes = []
    residuals, reduced_residuals = [], []
    main_grams = None
    envelope_full, envelope_full_actions = [], []
    envelope_red, envelope_red_actions = [], []
    for k, sigma in enumerate(shifts):
        x = values[(slice(None),)*d+(k,)]
        y = reduced_values[(slice(None),)*d+(k,)]
        difference = x - _apply(V, y)
        diff_controls = polynomial_controls(difference, d).reshape(-1, n, m)
        Cdiff = _apply(C, diff_controls)
        grams = np.einsum('bni,bnj->bij', diff_controls, Cdiff)
        main_grams = grams if main_grams is None else main_grams+grams

        xp = np.zeros_like(q)
        yp = np.zeros_like(qr)
        xp[pad_slice], yp[pad_slice] = x, y
        rx = q - _apply(Ka+sigma*C, xp)/np.sqrt(2*sigma)
        ry = qr - _apply(Kr_a+sigma*Cr, yp)/np.sqrt(2*sigma)
        for axis, H in enumerate(system.H):
            src = [slice(None)]*d
            dst = [slice(None)]*d
            src[axis], dst[axis] = slice(0, -1), slice(1, None)
            rx[tuple(dst)] -= (high[axis]-low[axis])*_apply(H, xp[tuple(src)])/np.sqrt(2*sigma)
            ry[tuple(dst)] -= (high[axis]-low[axis])*_apply(V.T @ (H @ V), yp[tuple(src)])/np.sqrt(2*sigma)
        primary_residual = rx-_apply(lift, ry) if propagation_model == 'coupled' else rx
        if retain_actions:
            ef, fc, fa = _inverse_norm_controls(primary_residual, d, solve_a, True)
            er, rc, ra = _inverse_norm_controls(ry, d, solve_ra, True)
            envelope_full.append(fc)
            envelope_full_actions.append(fa)
            envelope_red.append(rc)
            envelope_red_actions.append(ra)
        else:
            ef = _inverse_norm_controls(primary_residual, d, solve_a)
            er = _inverse_norm_controls(ry, d, solve_ra)
        residuals.append(ef)
        reduced_residuals.append(er)
        rho, feedback, gain = 1., 2., 1/np.sqrt(2)
        if spectral_interval is not None:
            alpha, beta = spectral_interval
            rho = max(abs((alpha-sigma)/(alpha+sigma)), abs((beta-sigma)/(beta+sigma)))
            feedback = 2*sigma/(alpha+sigma)
            lam = np.clip(sigma, alpha, beta)
            gain = np.sqrt(2*sigma*lam)/(lam+sigma)
        forcing = ef
        if propagation_model == 'coupled':
            forcing += coupling_gain*gain*(reduced_tube+er)/np.sqrt(2*sigma)
            coefficient_tubes.append(gain*(full_tube+forcing))
        else:
            coefficient_tubes.append(gain*(full_tube+ef+reduced_tube+er))
        full_tube = rho*full_tube+feedback*forcing
        reduced_tube = rho*reduced_tube+feedback*er
        q -= np.sqrt(2*sigma)*_apply(C, xp)
        qr -= np.sqrt(2*sigma)*_apply(Cr, yp)

    main = float(np.sqrt(max(0., np.linalg.eigvalsh(main_grams)[:, -1].max())))
    propagation = float(la.norm(coefficient_tubes))
    if envelope_depth:
        ft, ftc, fta = _inverse_norm_controls(q, d, solve_a, True)
        rt, rtc, rta = _inverse_norm_controls(qr, d, solve_ra, True)
        full_tail_trial, reduced_tail_trial = ft/np.sqrt(2), rt/np.sqrt(2)
    else:
        full_tail_trial = _inverse_norm_controls(q, d, solve_a)/np.sqrt(2)
        reduced_tail_trial = _inverse_norm_controls(qr, d, solve_ra)/np.sqrt(2)
    full_tail_tube = full_tube+lift_gain*reduced_tube if propagation_model == 'coupled' else full_tube
    full_tail = full_tail_trial+full_tail_tube/np.sqrt(2)
    reduced_tail = reduced_tail_trial+reduced_tube/np.sqrt(2)
    bound = main+propagation+full_tail+reduced_tail
    independent_bound = bound if propagation_model != 'coupled' else None
    energy_bound = energy_defect = None
    selected_majorant = propagation_model
    if propagation_model in ['energy', 'matrix']:
        energy_defect = innovation_energy_bound(residuals, shifts, spectral_interval)
        energy_defect += innovation_energy_bound(reduced_residuals, shifts, spectral_interval)
        energy_bound = main+energy_defect+full_tail_trial+reduced_tail_trial
        if energy_bound <= bound:
            bound, propagation = energy_bound, energy_defect
            full_tail, reduced_tail = full_tail_trial, reduced_tail_trial
            selected_majorant = 'energy'
        else:
            selected_majorant = 'independent'
    matrix_bound = matrix_defect = None
    if propagation_model == 'matrix':
        fc = np.stack(envelope_full, axis=d)
        fa = np.stack(envelope_full_actions, axis=d)
        rc = np.stack(envelope_red, axis=d)
        ra = np.stack(envelope_red_actions, axis=d)
        matrix_defect = joint_control_norm(fc, fa, d, np.eye(m), matrix_majorant)
        matrix_defect += joint_control_norm(rc, ra, d, np.eye(m), matrix_majorant)
        matrix_bound = main+matrix_defect+full_tail_trial+reduced_tail_trial
        if matrix_bound <= bound:
            bound, propagation = matrix_bound, matrix_defect
            full_tail, reduced_tail = full_tail_trial, reduced_tail_trial
            selected_majorant = 'matrix'
    envelope_bound = None
    envelope_cells = 0
    envelope_worst_cell = None
    if envelope_depth:
        difference = values-_apply(V, reduced_values)
        payload = (polynomial_controls(difference, d),
                   np.stack(envelope_full, axis=d), np.stack(envelope_full_actions, axis=d),
                   np.stack(envelope_red, axis=d), np.stack(envelope_red_actions, axis=d),
                   np.expand_dims(ftc, d), np.expand_dims(fta, d),
                   np.expand_dims(rtc, d), np.expand_dims(rta, d))
        worst = None
        for restricted, a, b in _refined_envelopes(payload, low, high, envelope_depth):
            if np.array_equal(b, high):
                change = np.eye(m)
            else:
                Klocal = system.operator(b)
                if denominator == 'exact':
                    Dlocal = .5*system.G.T @ _Solve(Klocal, counts, 'envelope_denominator')(system.G)
                else:
                    Dlocal = .5*(V.T @ system.G).T @ la.solve(V.T @ (Klocal @ V), V.T @ system.G,
                                                            assume_a='pos')
                local_root = la.cholesky(Dlocal, lower=True)
                change = root.T @ la.solve_triangular(local_root.T, np.eye(m))
            dc, fc, fa, rc, ra, ftc_, fta_, rtc_, rta_ = restricted
            dc = dc @ change
            flat = dc.reshape(-1, len(shifts), n, m)
            gram = np.einsum('btni,btnj->bij', flat, _apply(C, flat))
            local_main = np.sqrt(max(0., np.linalg.eigvalsh(gram)[:, -1].max()))
            ef = _dual_control_sequence_norms(fc, fa, d, change)
            er = _dual_control_sequence_norms(rc, ra, d, change)
            local_defect = innovation_energy_bound(ef, shifts, spectral_interval)
            local_defect += innovation_energy_bound(er, shifts, spectral_interval)
            if propagation_model == 'matrix':
                signed_defect = joint_control_norm(fc, fa, d, change, matrix_majorant)
                signed_defect += joint_control_norm(rc, ra, d, change, matrix_majorant)
                local_defect = min(local_defect, signed_defect)
            local_ft = _dual_control_sequence_norms(ftc_, fta_, d, change)[0]/np.sqrt(2)
            local_rt = _dual_control_sequence_norms(rtc_, rta_, d, change)[0]/np.sqrt(2)
            local_bound = float(local_main+local_defect+local_ft+local_rt)
            envelope_cells += 1
            if worst is None or local_bound > worst[0]:
                worst = (local_bound, float(local_main), local_defect, local_ft, local_rt)
                envelope_worst_cell = {'low': a.tolist(), 'high': b.tolist()}
        envelope_bound = worst[0]
        if envelope_bound <= bound:
            bound, main, propagation, full_tail, reduced_tail = worst
            selected_majorant = 'refined_matrix' if propagation_model == 'matrix' else 'refined_energy'
            full_tail_trial, reduced_tail_trial = full_tail, reduced_tail
    if not np.isfinite(bound):
        raise FloatingPointError('nonfinite certificate')
    return {'bound': bound, 'main': main, 'propagation': propagation,
            'full_tail': full_tail, 'reduced_tail': reduced_tail,
            'full_tail_trial': full_tail_trial, 'reduced_tail_trial': reduced_tail_trial,
            'max_primary_residual': max(residuals),
            'max_full_residual': max(residuals) if propagation_model != 'coupled' else None,
            'max_reduced_residual': max(reduced_residuals),
            'degree': degree, 'temporal_terms': len(shifts),
            'low': low.tolist(), 'high': high.tolist(),
            'point_lower_bound': max(0., main-propagation)
                if np.array_equal(low, high) and denominator == 'exact' else None,
            'denominator': denominator, 'spectral_interval': spectral_interval,
            'propagation_model': propagation_model, 'selected_majorant': selected_majorant,
            'independent_bound': independent_bound, 'energy_bound': energy_bound,
            'energy_defect': energy_defect, 'matrix_bound': matrix_bound,
            'matrix_defect': matrix_defect, 'matrix_majorant': matrix_metadata,
            'envelope_depth': envelope_depth,
            'envelope_cells': envelope_cells, 'envelope_bound': envelope_bound,
            'envelope_worst_cell': envelope_worst_cell,
            'coupling_gain': coupling_gain,
            'lift_gain': lift_gain,
            'counts': counts, 'floating_point_certified': False}
