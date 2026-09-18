"""Interpolate aligned inverse factors BEFORE forming their Gram product.

This keeps cross-node terms removed by a sum of independently weighted inverses.
No assertion of worldwide novelty or practical large-N speed is made here.
"""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from operators import Backbone,assemble,coefficients,build_backbone
from blend import weights,Blend
from compression import exact_oracle

class Coherent:
    def __init__(self,a,layer,shift,count,coordinate='root'):
        argument=np.sqrt(a) if coordinate=='root' else a
        nodes,self.w=weights(argument,count,'harmonic')
        self.nodes=nodes**2 if coordinate=='root' else nodes
        self.factors=[Backbone(v*np.asarray(layer),np.ones_like(a),shift) for v in self.nodes]
    def _d(self,w,b):return w.ravel() if b.ndim==1 else w.reshape(-1,1)
    def q(self,b):
        return sum(self._d(w,b)*p.q(b) for w,p in zip(self.w,self.factors))
    def qt(self,b):
        return sum(p.qt(self._d(w,b)*b) for w,p in zip(self.w,self.factors))
    def apply(self,b):return self.q(self.qt(b))

class IncoherentRoot(Coherent):
    """Same nodes/weights as Coherent, but remove all cross-factor products."""
    def apply(self,b):
        return sum(self._d(np.sqrt(w),b)*p.apply(self._d(np.sqrt(w),b)*b) for w,p in zip(self.w,self.factors))


def run(seed,smoke,out):
    rows=[];started=time.perf_counter()
    for n in ([6] if smoke else [12,24,36]):
        for family in (['layered'] if smoke else ['smooth','layered','inclusion']):
            k,l=coefficients(n,family,seed);a=k/l[None,:]
            for shift in ([20.] if smoke else [0.,20.,2000.,200000.]):
                matrix=assemble(k,shift);identity=np.eye(n*n)
                for name in ['single_scaled','incoherent_harmonic5','incoherent_root5','coherent_harmonic5','coherent_root5','coherent_root9']:
                    t0=time.perf_counter()
                    if name=='single_scaled':p=build_backbone(k,l,shift,'jump_aware');count=1
                    elif name=='incoherent_harmonic5':p=Blend(a,l,shift,5,'harmonic');count=5
                    elif name=='incoherent_root5':p=IncoherentRoot(a,l,shift,5);count=5
                    else:
                        count=9 if name.endswith('9') else 5
                        p=Coherent(a,l,shift,count,'harmonic' if 'harmonic' in name else 'root')
                    if name.startswith('incoherent'):
                        inverse=p.apply(identity);chol=la.cholesky((inverse+inverse.T)*.5,lower=True,check_finite=False)
                        h=chol.T@(matrix@chol);ev=la.eigvalsh((h+h.T)*.5,check_finite=False)
                    else:ev,_=exact_oracle(matrix,p)
                    if ev.min()<=0:raise RuntimeError('nonpositive oracle eigenvalue')
                    dev=np.sort(abs(ev-1))[::-1]
                    row=dict(seed=seed,n=n,N=n*n,family=family,shift=shift,method=name,
                        inverse_factor_pairs=count,lambda_min=float(ev.min()),lambda_max=float(ev.max()),
                        oracle_s=time.perf_counter()-t0,uncorrected_m2_energy_error=float(dev[0]**2))
                    for m in [1,2,3]:
                        for eps in [.01,.001]:row[f'optimal_rank_m{m}_eps{eps}']=int(np.sum(dev>eps**(1/m)))
                    rhs=np.cos(np.arange(n*n)*.217);p.apply(rhs);times=[]
                    for _ in range(5):
                        t0=time.perf_counter();p.apply(rhs);times.append(time.perf_counter()-t0)
                    row['apply_s']=float(np.median(times));rows.append(row)
                with (out/'coherent.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                print(f'coherent {n=} {family=} {shift=} complete',flush=True)
    (out/'metadata.json').write_text(json.dumps(dict(seed=seed,smoke=smoke,elapsed_s=time.perf_counter()-started,rows=len(rows),scope='Exact representation screen; no practical corrected inverse or physical model'),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);run(args.seed,args.smoke,args.output_dir)
