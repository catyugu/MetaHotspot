"""Source-only common Galerkin space and rigorous modal-cone comparison tail.

This is a deliberately exposed sufficient construction, not an optimized cone.
No parameter/time samples are used to infer the certificate. See proof note.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
from case1_system import case1_reconstruction
from numerics import Solver, operator, c_basis, sym, counts
from positive_source_tail import EnclosedSolve, opnorm


def build(a):
    clock=time.perf_counter()
    K,C,G,H,ranges,meta=case1_reconstruction(a.mesh_mm)
    c=C.diagonal()
    G=G/np.sqrt(np.sum(G*G/c[:,None],axis=0))[None,:]
    prep=Solver(operator(K,H,ranges[:,0]),rtol=2e-13)
    w=prep.solve(c);alpha=float(np.min((prep.A@w)/(c*w)))
    raw=G/c[:,None]
    rate=float(la.eigvalsh(sym(raw.T@(operator(K,H,ranges[:,1])@raw)))[-1])
    shifts=np.r_[0,np.geomspace(alpha,rate,a.poles-1)]
    S=[raw];cost=[prep.counts()]
    for p in itertools.product([0,.5,1],repeat=2):
        h=np.exp(np.log(ranges[:,0])+(np.log(ranges[:,1])-np.log(ranges[:,0]))*p)
        for s in shifts:
            solver=Solver(operator(K,H,h)+s*C,rtol=2e-13)
            S.append(solver.solve(G));cost.append(solver.counts())
    V=c_basis(np.column_stack(S),c)
    a.candidate.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.candidate,V=V,G=G,ranges=ranges)
    record=dict(n=len(c),V_order=V.shape[1],alpha=alpha,input_fast_rate=rate,
        poles=shifts.tolist(),costs=counts(cost),seconds=time.perf_counter()-clock,
        metadata=meta,scope='source-only common-space construction; training is not certification; no final SVD')
    a.candidate.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def run(a):
    if a.build:build(a)
    clock=time.perf_counter()
    K,C,_,H,_,meta=case1_reconstruction(a.mesh_mm)
    data=np.load(a.candidate);V=data['V'];G=data['G'];ranges=data['ranges']
    c=C.diagonal();sc=np.sqrt(c);r=V.shape[1];B=V.T@G
    R0=G-c[:,None]*(V@B)
    r0=float(la.norm(R0/sc[:,None],2))
    As=[sym(V.T@(J@V)) for J in [K,*H]]
    prep=Solver(operator(K,H,ranges[:,0]),rtol=2e-13)
    w=prep.solve(c)
    if np.min(w)<=0 or np.min(prep.A@w)<=0:raise ValueError('positive supersolution failure')
    rows=[];cost=[prep.counts()]
    hc=np.sqrt(ranges[:,0]*ranges[:,1])
    for width in a.widths:
        lo=ranges[:,0] if width<0 else np.maximum(ranges[:,0],hc*np.exp(-width))
        hi=ranges[:,1] if width<0 else np.minimum(ranges[:,1],hc*np.exp(width))
        Ah=As[0]+sum(h*A for h,A in zip(hi,As[1:]))
        lam,U=la.eigh(Ah);VU=V@U;BU=U.T@B
        Pabs=sum((b-a)*np.abs(U.T@J@U) for a,b,J in zip(lo,hi,As[1:]))
        M=np.diag(lam)-Pabs
        root=np.sqrt(lam)
        rho=float(la.eigvalsh(Pabs/root[:,None]/root[None,:])[-1])
        row=dict(width=width,lo=lo.tolist(),hi=hi.tolist(),modal_comparison_rho=rho,
                 modal_comparison_min=float(la.eigvalsh(M)[0]),comparison_exists=bool(rho<1))
        if rho<1:
            qbar=la.solve(M,np.abs(BU),assume_a='pos')
            if qbar.min() < -1e-9:raise ValueError('comparison inverse lost positivity')
            qbar=np.maximum(qbar,0.)
            Dabs=np.zeros_like(V);Dnorm=0.
            for h in itertools.product(*zip(lo,hi)):
                A=As[0]+sum(x*J for x,J in zip(h,As[1:]))
                D=operator(K,H,h)@VU-c[:,None]*(VU@(U.T@A@U))
                Dabs=np.maximum(Dabs,np.abs(D))
                Dnorm=max(Dnorm,float(la.norm(D/sc[:,None],2)))
            forcing=np.abs(R0)+Dabs@qbar
            solver=EnclosedSolve(operator(K,H,lo),w)
            _,envelope,correction=solver.solve(forcing)
            cost.append(solver.solver.counts())
            steady=opnorm(envelope,c)
            raw=G/c[:,None]
            rate=float(la.eigvalsh(sym(raw.T@(operator(K,H,hi)@raw)))[-1])
            eta1=Dnorm*la.norm(B,2)/2
            crossing=2*steady/(r0+np.sqrt(r0*r0+4*eta1*steady)) if steady>0 else 0.
            lower=-np.expm1(-rate*crossing)/rate
            relative=steady/lower if lower>0 else r0
            row.update(dict(residual_envelope_C=steady,
                uniform_all_time_relative_Galerkin_C_upper=float(relative),
                near_zero_crossing_time=float(crossing),near_zero_linear=r0,
                near_zero_quadratic=float(eta1),input_rate_upper=rate,
                inverse_solve_defect_C=opnorm(correction,c),
                input_reachable_modal_envelope_norm=float(la.norm(qbar,2)),
                passes_step_target=bool(relative<=np.sqrt(.001))))
        rows.append(row);print(json.dumps(row),flush=True)
    out=dict(n=len(c),V_order=r,metadata=meta,rows=rows,costs=counts(cost),
        candidate_costs=json.loads(a.candidate.with_suffix('.json').read_text()),
        seconds=time.perf_counter()-clock,
        scope='When comparison_exists, exact-arithmetic proof covers continuous cell, every t>0 and arbitrary signed constant inputs for THIS Galerkin V. Evaluated in ordinary float with actual CG defects, no outward rounding. No steady K-energy certificate.')
    a.output.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mesh-mm',type=float,default=10)
    p.add_argument('--poles',type=int,default=4)
    p.add_argument('--build',action='store_true')
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--widths',type=float,nargs='+',default=[.01,.1,.5,-1])
    run(p.parse_args())
