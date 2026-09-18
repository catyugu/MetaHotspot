"""Positive coefficient interpolation matching scalar diffusion/mass limits.

The scalar bound is NOT an operator bound with spatially varying weights.
Product-convolution and interpolated inverse preconditioners are prior art.
This independent representation screen uses exact small-matrix spectra only;
it does not hide dense Cholesky/eigendecomposition in an online algorithm.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import scipy.linalg as la
from operators import Backbone, assemble, coefficients, build_backbone
from compression import exact_oracle


def weights(a, count, kind):
    a=np.asarray(a,float)
    if count<2 or a.ndim!=2 or not np.all(np.isfinite(a)) or np.any(a<=0):
        raise ValueError('positive 2-D coefficients and at least two nodes required')
    if kind not in ('harmonic','arithmetic'):raise ValueError('unknown interpolation')
    low=float(a.min()); high=float(a.max())
    if high/low<1+1e-14:return np.array([low]),np.ones((1,)+a.shape)
    nodes=np.geomspace(low,high,count)
    # Both interpolants use exactly the same positive coefficient nodes.
    z=1/a if kind=='harmonic' else a
    zn=1/nodes if kind=='harmonic' else nodes
    order=np.argsort(zn); sorted_z=zn[order]
    idx=np.clip(np.searchsorted(sorted_z,z,side='right')-1,0,count-2)
    t=(z-sorted_z[idx])/(sorted_z[idx+1]-sorted_z[idx])
    w=np.zeros((count,)+a.shape)
    for j in range(count):
        w[order[j]]=np.where(idx==j,1-t,0)+np.where(idx+1==j,t,0)
    return nodes,w


def scalar_bound(nodes):
    ratio=np.max(np.asarray(nodes)[1:]/np.asarray(nodes)[:-1]) if len(nodes)>1 else 1.
    return float(((np.sqrt(ratio)-1)/(np.sqrt(ratio)+1))**2)


class Blend:
    def __init__(self,a,layer,shift,count,kind):
        self.nodes,w=weights(a,count,kind); self.root=np.sqrt(w)
        self.factors=[Backbone(v*np.asarray(layer),np.ones_like(a),shift) for v in self.nodes]
    def apply(self,b):
        result=np.zeros_like(b,dtype=float)
        for w,p in zip(self.root,self.factors):
            d=w.ravel() if b.ndim==1 else w.reshape(-1,1)
            result+=d*p.apply(d*b)
        return result


def run(seed,smoke,out):
    rows=[]; start=time.perf_counter()
    for n in ([6] if smoke else [12,24,36]):
        for family in (['smooth'] if smoke else ['smooth','layered','inclusion']):
            k,l=coefficients(n,family,seed)
            for shift in ([20.] if smoke else [0.,20.,2000.,200000.]):
                a=assemble(k,shift); identity=np.eye(n*n)
                for method in ['single_scaled','harmonic5','arithmetic5','harmonic9']:
                    t0=time.perf_counter()
                    if method=='single_scaled':
                        p=build_backbone(k,l,shift,'jump_aware')
                        ev,_=exact_oracle(a,p); count=1; bound=None
                    else:
                        count=9 if method.endswith('9') else 5
                        kind='harmonic' if method.startswith('harmonic') else 'arithmetic'
                        p=Blend(k/l[None,:],l,shift,count,kind)
                        matrix=p.apply(identity)
                        chol=la.cholesky((matrix+matrix.T)*.5,lower=True,check_finite=False)
                        h=chol.T@(a@chol)
                        ev=la.eigvalsh((h+h.T)*.5,check_finite=False)
                        bound=scalar_bound(p.nodes) if kind=='harmonic' else None
                    deviations=np.sort(abs(ev-1))[::-1]
                    row=dict(seed=seed,n=n,N=n*n,family=family,shift=shift,method=method,
                        backbone_calls=count,scalar_bound=bound,lambda_min=float(ev.min()),
                        lambda_max=float(ev.max()),oracle_s=time.perf_counter()-t0)
                    for degree in [1,2,3]:
                        for eps in [.01,.001]:
                            row[f'optimal_rank_m{degree}_eps{eps}']=int(np.sum(deviations>eps**(1/degree)))
                    # Warmup and repeated factor-only application, no source-dependent construction.
                    rhs=np.cos(np.arange(n*n)*.217);p.apply(rhs)
                    times=[]
                    for _ in range(5):
                        t0=time.perf_counter();p.apply(rhs);times.append(time.perf_counter()-t0)
                    row['apply_s']=float(np.median(times));rows.append(row)
                with (out/'blend.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                print(f'blend {n=} {family=} {shift=} complete',flush=True)
    (out/'metadata.json').write_text(json.dumps(dict(seed=seed,smoke=smoke,elapsed_s=time.perf_counter()-start,
        rows=len(rows),scope='Exact finite-matrix representation screen, not a practical inverse construction'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);run(args.seed,args.smoke,args.output_dir)
