"""Independent BE step reference with a spectrum-independent time error bound.
Point/time audits only; no continuous-parameter or all-time claim.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from case1_system import case1_reconstruction
from numerics import Solver,operator,f
from numerics import sym,relative,counts

def run(a):
 start=time.perf_counter();data=np.load(a.candidate);meta=json.loads(a.candidate.with_suffix('.json').read_text());V=data['V'];G=data['G'];ranges=data['ranges']
 K,C,_,H,_,_=case1_reconstruction(a.mesh_mm);c=C.diagonal();sc=np.sqrt(c);alpha=meta['alpha'];rows=[];cost=[]
 hs=[np.sqrt(ranges[:,0]*ranges[:,1]),np.array([ranges[0,0],ranges[1,1]]),np.array([ranges[0,1],ranges[1,0]])]
 for h in hs:
  Kh=operator(K,H,h);A=sym(V.T@(Kh@V));l,U=la.eigh(A);B=U.T@(V.T@G)
  for t in a.times:
   dt=t/a.steps;sol=Solver(C+dt*Kh);X=np.zeros_like(G);cg=0.;rho=1/(1+dt*alpha)
   for j in range(a.steps):
    rhs=c[:,None]*X+dt*G;X=sol.solve(rhs);defect=rhs-sol.A@X;cg=rho*(cg+la.norm(defect/sc[:,None],2))
   XR=V@(U@(f(t,l)[:,None]*B));eps=relative(X-XR,X,C);den=la.svdvals(sc[:,None]*X)[-1]
   be=t/(2*a.steps)*((a.steps-1)/a.steps)**(a.steps-1)*la.norm(G/sc[:,None],2);delta=(be+cg)/den
   upper=(eps+delta)/(1-delta) if delta<1 else None;lower=max(0.,eps-delta)/(1+delta)
   row=dict(h=h.tolist(),t=t,steps=a.steps,reference_relative=eps,BE_absolute_bound=be,CG_absolute_bound=cg,reference_error_relative_bound=delta,true_relative_lower=lower,true_relative_upper=upper,point_pass=bool(upper is not None and upper<=np.sqrt(.001)),point_fail=bool(lower>np.sqrt(.001)))
   if len(c)<=2000:
    ls,Us=la.eigh(Kh.toarray()/sc[:,None]/sc[None,:]);XF=(Us@(f(t,ls)[:,None]*(Us.T@(G/sc[:,None]))))/sc[:,None]
    row.update(small_oracle_relative=relative(XF-XR,XF,C),small_oracle_BE_absolute=la.norm(sc[:,None]*(XF-X),2),small_oracle_bound_holds=bool(la.norm(sc[:,None]*(XF-X),2)<=be+cg))
   rows.append(row);cost.append(sol.counts());print(json.dumps(row),flush=True)
 out=dict(n=len(c),V_order=V.shape[1],alpha=alpha,rows=rows,costs=counts(cost),seconds=time.perf_counter()-start,scope='fixed parameters and times; exact-arithmetic BE inequality evaluated with ordinary floating point; native_validated=false')
 a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,default=10);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=64);p.add_argument('--times',type=float,nargs='+',default=[.01,.05,1.]);run(p.parse_args())
