"""Rebuild the SAME input-driven eight-pole V used by coupled enclosure.
Training chooses V only; no continuum or time acceptance is inferred.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from numerics import invroot, sym, counts
from port_basis import ports, group_basis, hpoints
from case1_system import case1_reconstruction
from numerics import Solver,operator,decay_lower,c_basis

def run(a):
 t=time.perf_counter();K,C,G,H,ranges,meta=case1_reconstruction(a.mesh_mm);c=C.diagonal();sc=np.sqrt(c)
 G=G@invroot((G/sc[:,None]).T@(G/sc[:,None]));Q,groups,_=ports(H)
 prep=Solver(operator(K,H,ranges[:,0]));alpha=decay_lower(prep.A,c,prep)
 raw=G/c[:,None];rate=float(la.eigvalsh(sym(raw.T@(operator(K,H,ranges[:,1])@raw)))[-1]);tr=[];cost=[]
 for h in hpoints(ranges,[0,.5,1]):
  for s in np.r_[0,np.geomspace(alpha,rate,3)]:
   sol=Solver(operator(K,H,h)+s*C);X=sol.solve(G);tr.append(np.asarray(Q.T@X)@invroot((sc[:,None]*X).T@(sc[:,None]*X)));cost.append(sol.counts())
 Z,_=group_basis(np.column_stack(tr),groups,a.rank);QZ=np.asarray(Q@Z);snaps=[G/c[:,None],np.ones((len(c),1))];build=[]
 for s in np.r_[0,np.geomspace(alpha,rate,7)]:
  sol=Solver(prep.A+s*C);snaps.append(sol.solve(np.column_stack([G,QZ])));build.append(sol.counts())
 V=c_basis(np.column_stack(snaps),c);a.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output,V=V,G=G,ranges=ranges,Z=Z)
 out={'n':len(c),'metadata':meta,'V_order':V.shape[1],'port_order':Z.shape[1],'alpha':alpha,'input_fast_rate':rate,
 'costs':{'preparation':prep.counts(),'training':counts(cost),'basis':counts(build)},'seconds':time.perf_counter()-t,
 'scope':'candidate regeneration only, inherited extraction cost is not free'}
 a.output.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,default=10);p.add_argument('--rank',type=int,default=16);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
