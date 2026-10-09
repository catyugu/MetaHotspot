"""Exact-arithmetic continuous affine-cell, all-time step perturbation bound.

Uses relative quadratic-form perturbations, not an absolute stiffness Lipschitz
constant. See AFFINE_STEP_BRIDGE_PROOF.md. Floating point is not interval verified.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import scipy.linalg as la
from steady_step_audit import _change, _dense, _norm, step_all_time_bound


def affine_step_variation_bound(K, C, G, H, radius, intervals=2048, *, full_spectrum=None):
    """Bound sup_h,t,u ||X_h(t)u-X_0(t)u||_C / ||X_0(t)u||_C."""
    if not isinstance(intervals,(int,np.integer)) or intervals<1:
        raise ValueError('time interval count must be a positive integer')
    rates, U = la.eigh(_dense(K), _dense(C)) if full_spectrum is None else full_spectrum
    B = U.T@G
    X = B/rates[:, None]
    root = np.sqrt(rates)
    terms = [(U.T@(_dense(J)@U))/(root[:, None]*root[None, :])*r
             for J, r in zip(H, radius)]
    rho = max(_norm(sum(s*A for s,A in zip(signs,terms)))
              for signs in itertools.product([-1,1],repeat=len(terms))) if terms else 0.
    if rho >= 1:
        return {'bound':float('inf'),'rho':rho,'reason':'relative form radius >= 1'}
    if rho == 0:
        return {'bound':0.,'rho':0.,'intervals':0}
    alpha = (1-rho)*rates[0]
    eta = rho/np.sqrt(1-rho)
    # sqrt(lambda)*f_s(lambda) <= sqrt(s); semigroup smoothing is
    # ||A^.5 exp(-A t)|| <= 1/sqrt(2 e t).
    short_constant = np.pi/(2*np.sqrt(2*np.e))*eta
    long_constant = np.pi/(2*np.e)*eta
    def state(t):
        return (-np.expm1(-rates*t)/rates)[:,None]*B
    def long(Z):
        return eta/np.sqrt(alpha)*_norm((root[:,None]*X)@Z)+long_constant*_norm(X@Z)
    t0 = 1e-7/rates[-1]
    T = 35/alpha
    Z0 = _change((state(t0)/t0).T@(state(t0)/t0))
    worst = short_constant*_norm(B@Z0)
    FT = state(T); ZT = _change(FT.T@FT)
    worst = max(worst,long(ZT))
    edges = np.geomspace(t0,T,intervals+1)
    for a,b in zip(edges[:-1],edges[1:]):
        F = state(a); Z = _change(F.T@F)
        worst = max(worst,min(short_constant*b*_norm(B@Z),long(Z)))
    return {'bound':float(worst),'rho':float(rho),'intervals':intervals,
            'scope':'continuous affine cell; every t>0; all input combinations',
            'floating_point_certified':False,
            'full_spectral_eigendecompositions':int(full_spectrum is None),
            'relative_form_corner_norms':2**len(radius)}


def continuous_step_bridge(K,C,G,H,V,low,high,center_threshold=.03):
    center=(np.asarray(low)+high)/2; radius=(np.asarray(high)-low)/2
    A=_dense(K)+sum(h*_dense(J) for h,J in zip(center,H))
    fixed=step_all_time_bound(A,C,G,V,center_threshold)
    full=affine_step_variation_bound(A,C,G,H,radius)
    reduced=affine_step_variation_bound(V.T@A@V,V.T@_dense(C)@V,V.T@G,
                                        [V.T@_dense(J)@V for J in H],radius)
    e,df,dr=fixed['bound'],full['bound'],reduced['bound']
    bound=(e+df+(1+e)*dr)/(1-df) if df<1 and fixed['accepted_analytic'] else float('inf')
    return {'bound':float(bound),'center':fixed,'full_variation':full,
            'reduced_variation':reduced,'low':np.asarray(low).tolist(),'high':np.asarray(high).tolist(),
            'scope':'continuous HTC cell; every t>0; all input combinations; final SVD basis',
            'floating_point_certified':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--validate',action='store_true');args=p.parse_args()
    z=np.load(args.archive);results=[]
    low,high=z['low'],z['high']; center=(low+high)/2
    for scale in [1.,.25,.1]:
        result=continuous_step_bridge(z['K'],z['C'],z['G'],z['H'],z['final_basis'],
                 center+scale*(low-center),center+scale*(high-center))
        result['scale']=scale;result['threshold']=float(2*np.sqrt(z['epsilon']))
        result['accepted_analytic']=bool(result['bound']<=result['threshold'])
        if args.validate and scale == .25:
            points=[*itertools.product(*zip(result['low'],result['high'])),
                    *np.random.default_rng(20261007).uniform(result['low'],result['high'],(12,len(low)))]
            worst=0.
            for h in points:
                A=z['K']+sum(x*J for x,J in zip(h,z['H']))
                rates,U=la.eigh(A,z['C']); B=U.T@z['G']
                V=z['final_basis']; rr,W=la.eigh(V.T@A@V,V.T@z['C']@V)
                W=V@W;Br=W.T@z['G'];D=U.T@z['C']@W
                for t in np.geomspace(1e-10/rates[-1],100/rates[0],120):
                    F=(-np.expm1(-rates*t)/rates)[:,None]*B
                    R=(-np.expm1(-rr*t)/rr)[:,None]*Br
                    worst=max(worst,_norm((F-D@R)@_change(F.T@F)))
            result['independent_reference']={'sampled_maximum':worst,
                'parameter_points':len(points),'times_per_point':120,
                'scope':'validation samples only; not the continuous certificate'}
        results.append(result)
        print(scale,result['bound'],result['center']['bound'],result['full_variation']['bound'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':
    main()
