"""Continuous-box / all-time static-lift research prototype, not unit tests.

The toy proof uses Fraction throughout (including SPD and acceptance checks).
Case1 uses AMG-CG with explicit solve defects, ordinary floats, not rounding
certification. See CONTINUOUS_LIFT_PROOF.md for hypotheses and scope.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
from fractions import Fraction as F
import itertools
import json
import math
from pathlib import Path
import numpy as np
import scipy.linalg as la
from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower, norm, f


def rational(a):
    return np.array([[F(int(x)) if isinstance(x, np.integer) else F(x)
                      for x in row] for row in a], dtype=object)


def rational_solve(a, b):
    n = len(a)
    m = np.column_stack((a.copy(), b.copy()))
    for j in range(n):
        pivot = next(i for i in range(j, n) if m[i, j])
        m[[j, pivot]] = m[[pivot, j]]
        m[j] /= m[j, j]
        for i in range(n):
            if i != j:
                m[i] -= m[i, j] * m[j]
    return m[:, n:]


def inverse(a):
    return rational_solve(a, rational(np.eye(len(a), dtype=int)))


def positive_pivots(a):
    """Exact symmetric LDL criterion; strictly positive definite only."""
    if np.any(a != a.T):
        raise ValueError('not symmetric')
    a = a.copy()
    pivots = []
    for j in range(len(a)):
        p = a[j, j]
        pivots.append(p)
        if p <= 0:
            return False, pivots
        for i in range(j+1, len(a)):
            for k in range(j+1, len(a)):
                a[i, k] -= a[i, j] * a[j, k] / p
    return True, pivots


def sqrt_upper(a, denominator=10**9):
    # Integer arithmetic: ceil(sqrt(a) * denominator), with no float decision.
    x = a.numerator * denominator**2
    k = math.isqrt(x // a.denominator)
    if k*k*a.denominator < x:
        k += 1
    return F(k, denominator)


def exact_toy(coupling):
    eps = F(coupling)
    k = rational([[2, F(-1,3), -eps, -2*eps],
                  [F(-1,3), 3, -eps, -eps],
                  [-eps, -eps, 4, F(-1,4)],
                  [-2*eps, -eps, F(-1,4), 5]])
    h = [rational(np.diag(v)) for v in ([1,0,2,0], [0,3,0,1])]
    v = rational([[1,0],[0,1],[0,0],[0,0]])
    eye = rational(np.eye(4, dtype=int))
    alpha = F(1)
    if any(k[i,j]>0 for i in range(4) for j in range(4) if i!=j) or any(sum(row)<=0 for row in k):
        raise ValueError('grounded thermal structure failed')
    spd, pivots = positive_pivots(k-alpha*eye)
    if not spd:
        raise ValueError('uniform coercivity proof failed')
    dlo = k@v-v@(v.T@k@v)
    ylo = rational_solve(k, dlo)
    tau = F(1,1000)
    vertices = []
    for p in itertools.product((0,1), repeat=2):
        kh = k + sum((p[i]*h[i] for i in range(2)), np.zeros_like(k))
        a = v.T @ kh @ v
        d = kh @ v - v @ a
        if np.any(d != dlo):
            raise ValueError('toy requires parameter-independent residual D')
        q = d.T @ ylo
        trace = sum(q[i,i] for i in range(2))
        ok, lp = positive_pivots(tau*tau*a-q)
        vertices.append(dict(parameter=p, dual_trace=trace,
                             steady_matrix_spd=ok, steady_ldl_pivots=lp))
    trace = max(x['dual_trace'] for x in vertices)
    rho = sqrt_upper(trace/alpha)
    relative = rho/(1-rho) if rho < 1 else None
    # sqrt(.001) is checked through its exact square.
    step_ok = relative is not None and relative*relative <= F(1,1000)
    steady_ok = all(x['steady_matrix_spd'] for x in vertices)
    # Supplemental diagnostics ONLY: all combinations of the two input columns.
    from_float = lambda x: np.asarray(x, dtype=float)
    sampled_step = sampled_steady = 0.
    for p in itertools.product(np.linspace(0,1,9), repeat=2):
        kh = from_float(k) + sum((p[i]*from_float(h[i]) for i in range(2)))
        a = from_float(v).T @ kh @ from_float(v)
        kl, ku = la.eigh(kh); al, au = la.eigh(a)
        for t in np.geomspace(1e-6, 1e4, 101):
            full = (ku*f(t,kl)) @ ku.T @ from_float(v)
            reduced = from_float(v) @ (au*f(t,al)) @ au.T
            err = full-reduced
            sampled_step = max(sampled_step, math.sqrt(max(0.,
                la.eigvalsh(err.T@err, full.T@full)[-1])))
        full = la.solve(kh, from_float(v), assume_a='pos')
        reduced = from_float(v) @ la.solve(a, np.eye(2), assume_a='pos')
        err = full-reduced
        sampled_steady = max(sampled_steady, math.sqrt(max(0.,
            la.eigvalsh(err.T@kh@err, full.T@kh@full)[-1])))
    return dict(kind='exact rational continuous box', coupling=eps,
                n=4, r=2, inputs=2, box=[[0,1],[0,1]], alpha=alpha,
                K0=k.tolist(), H=[j.tolist() for j in h], V=v.tolist(), G=v.tolist(),
                grounded_thermal_M_matrix=True,
                alpha_ldl_pivots=pivots,
                noncommuting_K0_H1=bool(np.any(k@h[0] != h[0]@k)),
                exact_rhs=2, factorizations=1, vertices=vertices,
                rho_rom=rho, relative_step_upper=relative,
                relative_steady_upper_squared=tau*tau/(1+tau*tau) if steady_ok else None,
                steady_surrogate_target=tau,
                step_pass=bool(step_ok), steady_pass=bool(steady_ok),
                rigorous_dual_acceptance=bool(step_ok and steady_ok),
                proof_uses_parameter_samples=False, proof_uses_time_samples=False,
                sampled_step_max=sampled_step, sampled_steady_max=sampled_steady,
                sampling_role='diagnostic only; exact rational gates are the proof')


def steady_counterexample():
    # Grounded thermal M-matrix, nonnegative source, diagonal Robin uncertainty.
    # Both G and the exact steady solution are in a fixed rational C-orthobasis.
    k = rational([[F(71,50),F(-1,5),F(-9,10)],
                  [F(-1,5),2,-1],[F(-9,10),-1,3]])
    h = rational(np.diag([1,0,0]))
    v = rational([[1,0],[0,F(3,5)],[0,F(4,5)]])
    g = rational([[1],[0],[0]])
    x = rational([[1],[F(3,10)],[F(2,5)]])
    a = v.T@k@v; b = v.T@g; d = k@v-v@a
    exact_identity = bool(np.all(k@x==g) and np.all(h@x==g)
                          and np.all(v@inverse(a)@b==x))
    db_squared = sum(q*q for q in (d@b).ravel())
    if not exact_identity or db_squared <= 0 or not positive_pivots(k)[0]:
        raise ValueError('counterexample algebra failed')
    # At h=1,t=1, certify a tolerance violation with an exact Taylor enclosure.
    # SPD semigroup contraction bounds the degree-N remainder by M^N/(N+1)!.
    kh=k+h; af=v.T@kh@v
    def step_series(matrix,source,n=60):
        term=source.copy();out=term.copy()
        for j in range(1,n):
            term=-(matrix@term)/F(j+1)
            out+=term
        m=max(sum(abs(x) for x in row) for row in matrix)
        radius=m**n/F(math.factorial(n+1))
        return out,radius
    xf,rf=step_series(kh,g)
    zr,rr=step_series(af,b)
    err=xf-v@zr
    w=rational([[0],[F(-4,5)],[F(3,5)]])
    error_lower=abs((w.T@err)[0,0])-rf-rr
    full_upper=sqrt_upper(sum(q*q for q in xf.ravel()))+rf
    violation=error_lower>0 and error_lower**2>F(1,1000)*full_upper**2
    if not violation:
        raise ValueError('exact transient tolerance violation not enclosed')
    maximum = 0.; witness = None
    for p in np.linspace(0,1,11):
        kh=np.asarray(k+p*h,dtype=float);vf=np.asarray(v,dtype=float)
        af=vf.T@kh@vf;gf=np.asarray(g,dtype=float);bf=vf.T@gf
        kl,ku=la.eigh(kh);al,au=la.eigh(af)
        for t in np.geomspace(1e-5,1e3,201):
            full=(ku*f(t,kl))@ku.T@gf
            reduced=vf@(au*f(t,al))@au.T@bf
            ratio=float(la.norm(full-reduced)/la.norm(full))
            if ratio>maximum:maximum=ratio;witness=[float(p),float(t)]
    return dict(kind='exact obstruction to steady-input-only certification',
                K0=k.tolist(), H=h.tolist(), V=v.tolist(), G=g.tolist(),
                continuous_parameter=[0,1],steady_solution='[1, 3/10, 2/5]^T / (1+h)',
                steady_error_identically_zero=exact_identity,
                second_error_derivative_norm_squared=db_squared,
                transient_error_not_identically_zero=True,
                exact_step_tolerance_violation=True,
                exact_witness_parameter_time=[1,1], taylor_degree=60,
                witness_relative_error_lower=error_lower/full_upper,
                witness_relative_error_lower_display=float(error_lower/full_upper),
                witness_remainder_full=rf,witness_remainder_rom=rr,
                proof_uses_parameter_samples=False,proof_uses_time_samples=False,
                sampled_step_max=maximum,sampled_witness_parameter_time=witness,
                sampling_role='diagnostic; algebra and rational Taylor enclosure are the proof')


def case1(mesh, archive, basis, domain):
    k,c,g,h,ranges,meta = case1_reconstruction(mesh)
    v = np.load(archive)[basis]
    cd = c.diagonal(); r = v.shape[1]
    coords = ranges[:,0] if domain else np.mean(ranges,axis=1)
    kl = operator(k,h,coords); solve = Solver(kl)
    alpha = decay_lower(kl,cd,solve)
    ar = v.T@(kl@v)
    d = kl@v - (cd[:,None]*v)@ar
    rhs = [d]
    if domain:
        rhs += [j@v-(cd[:,None]*v)@(v.T@(j@v)) for j in h]
    lifts = [solve.solve(q) for q in rhs]
    defects = [q-kl@y for q,y in zip(rhs,lifts)]
    b = v.T@g
    r0 = g-(cd[:,None]*v)@b
    ahi = v.T@(operator(k,h,ranges[:,1])@v) if domain else ar
    eta0 = norm(r0/np.sqrt(cd)[:,None])/la.svdvals(b)[-1] * max(1.,la.eigvalsh(ahi)[-1]/alpha)
    points = list(itertools.product(*ranges)) if domain else [coords]
    vals=[]
    for p in points:
        delta = np.asarray(p)-coords
        dp = d.copy(); y = lifts[0].copy(); res = defects[0].copy()
        if domain:
            for i,x in enumerate(delta):
                dp += x*rhs[i+1]; y += x*lifts[i+1]; res += x*defects[i+1]
        # Exact-arithmetic energy bound from an inexact inverse:
        # ||Klo^-1/2 D||F <= ||Klo^1/2 Y||F + ||C^-1/2 res||F/sqrt(alpha).
        q_upper = (math.sqrt(max(0.,float(np.sum(y*(kl@y)))))
                   + la.norm(res/np.sqrt(cd)[:,None],'fro')/math.sqrt(alpha))**2
        rho_energy = math.sqrt(q_upper/alpha)+eta0
        rho_point = la.norm(np.sqrt(cd)[:,None]*y,'fro') + la.norm(res/np.sqrt(cd)[:,None],'fro')/alpha + eta0
        vals.append(dict(parameter=np.asarray(p).tolist(), rho_energy=rho_energy,
                         rho_point=rho_point if not domain else None,
                         lift_frobenius=la.norm(np.sqrt(cd)[:,None]*y,'fro') if not domain else None,
                         lift_operator=norm(np.sqrt(cd)[:,None]*y) if not domain else None,
                         cg_correction_frobenius=la.norm(res/np.sqrt(cd)[:,None],'fro')/alpha,
                         dual_trace_upper=q_upper))
    rho = max(x['rho_energy'] if domain else x['rho_point'] for x in vals)
    return dict(kind='continuous box surrogate' if domain else 'necessary pointwise static-lift gate',
                n=len(cd),r=r,mesh_mm=mesh,archive=archive,basis=basis,
                box=ranges.tolist() if domain else None,alpha=alpha,rho_rom=rho,
                relative_step_upper=rho/(1-rho) if rho<1 else None,
                step_pass=bool(rho/(1-rho)<=math.sqrt(.001)) if rho<1 else False,
                floating_point_certified=False,source_projection_correction=eta0,
                orthogonality_defect=norm(v.T@(cd[:,None]*v)-np.eye(r)),
                counts=solve.counts(),vertices=vals,model=meta,
                rejection_scope='this sufficient certificate; does not disprove actual ROM accuracy')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toy',action='store_true')
    p.add_argument('--mesh-mm',type=float,default=10.)
    p.add_argument('--archive')
    p.add_argument('--basis',default='candidate')
    p.add_argument('--domain',action='store_true')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result=([exact_toy('1/10000'),exact_toy('1/100'),steady_counterexample()]
            if args.toy else case1(args.mesh_mm,args.archive,args.basis,args.domain))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,default=lambda x:str(x) if isinstance(x,F) else x)+'\n')
    keys=('kind','coupling','n','r','rho_rom','relative_step_upper','step_pass',
          'steady_pass','counts','exact_step_tolerance_violation',
          'witness_relative_error_lower_display')
    items=result if isinstance(result,list) else [result]
    print(json.dumps([{k:x[k] for k in keys if k in x} for x in items],
                     default=lambda x:str(x) if isinstance(x,F) else x))


if __name__ == '__main__':
    main()
