"""Signed innovation Gram and a Poisson-kernel continuous-spectrum envelope.

Exact-arithmetic theorem only. No entrywise absolute-value relaxation of the
residual time/spatial Gram. No claim of floating-point interval certification.
"""
import numpy as np
import scipy.linalg as la


def innovation_gram(rate, shifts):
    shifts = np.asarray(shifts, float)
    if rate <= 0 or np.any(shifts <= 0):
        raise ValueError('positive rate and shifts required')
    b = 2*shifts/(rate+shifts)
    t = (rate-shifts)/(rate+shifts)
    gram = np.diag(b)
    for j in range(len(shifts)):
        product = 1.
        for k in range(j+1, len(shifts)):
            gram[j, k] = gram[k, j] = -.5*b[j]*b[k]*product
            product *= t[k]
    return gram


def spectral_majorant(shifts, spectral_interval, ratio=2.):
    """M >= A(lambda) everywhere, via relative Poisson-kernel domination.

    Each geometric bin [a,b] uses r*A(sqrt(a*b)), r=sqrt(b/a).
    The common Loewner upper bound is built by positive-part updates.
    Reverse ordering is also valid; choose the smaller trace, not matrixwise min.
    With no spectral interval, tr(A)<=2*N gives the universal 2*N*I.
    """
    N = len(shifts)
    if spectral_interval is None:
        return 2*N*np.eye(N), {'construction': 'universal_trace', 'bins': 0}
    alpha, beta = spectral_interval
    if not 0 < alpha <= beta or ratio <= 1:
        raise ValueError('positive ordered interval and ratio>1 required')
    bins = max(1, int(np.ceil(np.log(beta/alpha)/np.log(ratio))))
    edges = np.geomspace(alpha, beta, bins+1)
    envelopes = [np.sqrt(b/a)*innovation_gram(np.sqrt(a*b), shifts)
                 for a, b in zip(edges[:-1], edges[1:])]
    def join(items):
        M = items[0].copy()
        for A in items[1:]:
            difference = (A-M+A.T-M.T)/2
            rates, U = la.eigh(difference, check_finite=False)
            M += (U*np.maximum(rates, 0.)) @ U.T
        return (M+M.T)/2
    forward, reverse = join(envelopes), join(envelopes[::-1])
    M = forward if np.trace(forward) <= np.trace(reverse) else reverse
    return M, {'construction': 'poisson_loewner', 'bins': bins,
               'bin_ratio': ratio, 'trace': float(np.trace(M)),
               'max_eigenvalue': float(la.eigvalsh(M)[-1])}


def joint_control_norm(controls, actions, dimension, change, majorant):
    """Convex PSD joint time/state/input Gram, maximized over Bernstein controls.

    actions must be exact K(a)^-1 controls (or the corresponding reduced inverse).
    Both tensors have parameter axes, then time, state, input axes.
    """
    n, m = controls.shape[-2:]
    N = controls.shape[dimension]
    f = (controls @ change).reshape(-1, N, n, m)
    a = (actions @ change).reshape(f.shape)
    weighted = np.einsum('jk,bknl->bjnl', majorant, a, optimize=True)
    gram = np.einsum('bjni,bjnl->bil', f, weighted, optimize=True)
    gram = (gram+gram.swapaxes(-1, -2))/2
    return float(np.sqrt(max(0., np.linalg.eigvalsh(gram)[:, -1].max())))
