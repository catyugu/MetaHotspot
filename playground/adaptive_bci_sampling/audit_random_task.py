"""Independent small-FOM spectral audit of the new all-time certificate.

An experiment, not a unit-test suite. Time samples only check the theorem's
implementation; they do not supply its all-time acceptance guarantee.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from case1_system import case1_reconstruction
from numerics import Solver,operator,sym,relative,f
from probabilistic_extraction import parameters,corners
from diagonal_time_certificate import DiagonalTimeCertificate
from affine_decay import AffineDecay
from graph_inverse_lower import GraphInverseLower, column_partition


def run(a):
    started=time.perf_counter()
    K,C,G,H,ranges,meta=case1_reconstruction(a.mesh_mm)
    if len(G)>2000:
        raise ValueError('dense independent spectrum is restricted to small FOM')
    data=np.load(a.candidate)
    V=data['V']; sc=np.sqrt(C.diagonal())
    so=Solver(operator(K,H,ranges[:,0]));z=so.solve(C.diagonal())
    D=(so.A@z)/z;alpha=float(min(D/C.diagonal()))
    diagonal_margin=float(la.eigvalsh(so.A.toarray()-np.diag(D))[0])
    metadata=json.loads(a.candidate.with_suffix('.json').read_text())
    decay=None
    if metadata.get('configuration',{}).get('decay_budget',0):
        decay=AffineDecay(K,C,H,alpha)
        for z,h in zip(data['decay_vectors'],data['decay_anchors']):
            decay.add(z,h)
    width=metadata.get('configuration',{}).get('graph_block_width',0)
    inverse_lower=GraphInverseLower(so.A,z,D,column_partition(meta['shape'],width)) if width else None
    cert=DiagonalTimeCertificate(K,C,H,G,V,D,alpha,decay=decay,inverse_lower=inverse_lower)
    evaluate=(cert.evaluate_matrix if metadata.get('configuration',{}).get('certificate_mode')=='matrix' else cert.evaluate)
    hp=np.vstack([corners(ranges),parameters(ranges,np.random.default_rng(a.seed),a.parameters)])
    rows=[]
    for h in hp:
        A=operator(K,H,h)
        lam,U=la.eigh(A.toarray()/sc[:,None]/sc[None,:])
        F=U.T@(G/sc[:,None])
        lr,Ur=la.eigh(sym(V.T@(A@V)),sym(V.T@(C@V)))
        W=V@Ur;Fr=W.T@G
        initial=relative(G/C.diagonal()[:,None]-W@Fr,G/C.diagonal()[:,None],C)
        X=(U@(F/lam[:,None]))/sc[:,None]
        Xr=W@(Fr/lr[:,None])
        steady=relative(X-Xr,X,A)
        maxstep=max(initial,relative(X-Xr,X,C));argmax=None
        times=np.geomspace(1e-7/lam[-1],40/lam[0],a.times)
        for t in times:
            S=(U@(f(t,lam)[:,None]*F))/sc[:,None]
            Sr=W@(f(t,lr)[:,None]*Fr)
            error=relative(S-Sr,S,C)
            if error>maxstep:
                maxstep=error;argmax=float(t)
        bound=evaluate(h,early_reject=False)
        rate=alpha if decay is None else decay(h)
        rows.append(dict(h=h.tolist(),initial=initial,steady=steady,sampled_step=maxstep,
                         argmax=argmax,bound=bound,decay_lower=rate,
                         decay_bound_holds=bool(rate<=lam[0]+1e-10),
                         steady_bound_holds=steady<=bound['steady_bound']+1e-8,
                         step_bound_holds=bound['step_bound'] is None or maxstep<=bound['step_bound']+1e-8))
    out=dict(n=len(G),metadata=meta,alpha=alpha,diagonal_ground_state_minimum_margin=diagonal_margin,
             parameters=len(hp),times=a.times,rows=rows,seconds=time.perf_counter()-started,
             scope='independent FOM spectrum at sampled HTC/time, including exact initial/steady limits; certificate itself encloses ALL time')
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(n=len(G),steady=max(r['steady'] for r in rows),step=max(r['sampled_step'] for r in rows),
                          violations=sum(not(r['steady_bound_holds'] and r['step_bound_holds'] and r['decay_bound_holds']) for r in rows),
                          seconds=out['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--mesh-mm',type=float,default=10)
    p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--parameters',type=int,default=16)
    p.add_argument('--times',type=int,default=128)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.parameters<0 or a.times<2:
        p.error('nonnegative parameter count and at least two time samples required')
    run(a)
