"""Continuous-cell deterministic D-weighted reconstruction tail.
Common storage is only an output-integral certificate, NOT original step acceptance.
"""
import argparse,itertools,json,time,math
from pathlib import Path
import numpy as np
import scipy.linalg as la
from case1_system import case1_reconstruction
from numerics import Solver,operator
from numerics import sym,counts

def root(Q):
 l,U=la.eigh(sym(Q));return (np.sqrt(np.maximum(l,0))[:,None]*U.T)
def bernstein(values):
 out=values.copy()
 for axis in range(2):
  z=np.moveaxis(out,axis,0);z[1]=2*z[1]-.5*(z[0]+z[2]);out=np.moveaxis(z,0,axis)
 return out

def run(a):
 clock=time.perf_counter();d=np.load(a.candidate);V=d['V'];G=d['G'];ranges=d['ranges'];r=V.shape[1];m=G.shape[1]
 K,C,_,H,_,_=case1_reconstruction(a.mesh_mm);c=C.diagonal();alpha=json.loads(a.candidate.with_suffix('.json').read_text())['alpha'];B=V.T@G
 As=[sym(V.T@(J@V)) for J in [K,*H]];Ds=[J@V-c[:,None]*V@A for J,A in zip([K,*H],As)]
 M=np.column_stack(Ds);sol=Solver(operator(K,H,ranges[:,0]));Y=sol.solve(M);defect=M-sol.A@Y
 Q=sym(M.T@Y+Y.T@M-Y.T@(sol.A@Y))+(defect/np.sqrt(c)[:,None]).T@(defect/np.sqrt(c)[:,None])/alpha
 # A posteriori variational inverse Gram upper bound; no exact solves assumed.
 del M,Y,defect;rows=[]
 def AA(h):return As[0]+h[0]*As[1]+h[1]*As[2]
 def NN(h):
  T=np.vstack([np.eye(r),h[0]*np.eye(r),h[1]*np.eye(r)]);return sym(T.T@Q@T)
 center=np.sqrt(ranges[:,0]*ranges[:,1])
 for width in a.widths:
  lo=np.maximum(ranges[:,0],center*np.exp(-width));hi=np.minimum(ranges[:,1],center*np.exp(width))
  if width<0:lo=ranges[:,0];hi=ranges[:,1]
  hc=(lo+hi)/2;Ac=AA(hc);bc=float(la.eigvalsh(Ac)[0]);beta=float(la.eigvalsh(AA(lo))[0]);zc=la.solve(Ac,B,assume_a='pos');Nc=NN(hc)
  def LL(h):
   Ah=AA(h);E=Ah-Ac;return np.block([[Ah,-E],[np.zeros_like(Ah),Ac]])
  Lc=LL(hc);Nbig=np.zeros((2*r,2*r));Nbig[:r,:r]=Nc;P0=sym(la.solve_continuous_lyapunov(Lc.T,Nbig))
  verts=[np.array(x) for x in itertools.product(*zip(lo,hi))];emax=max(la.norm(AA(h)-Ac,2) for h in verts);kappa=1+emax**2/(beta*bc);sdiag=np.r_[np.ones(r),np.full(r,kappa)];ss=np.sqrt(sdiag);d0=min(beta,bc)
  grid=np.empty((3,3,2*r,2*r))
  for i,j in itertools.product(range(3),repeat=2):
   h=lo+(hi-lo)*np.array([i,j])/2;L=LL(h);N=np.zeros_like(Nbig);N[:r,:r]=NN(h);grid[i,j]=sym(N-L.T@P0-P0@L)
  controls=bernstein(grid);worst=max(float(la.eigvalsh(T/ss[:,None]/ss[None,:],subset_by_index=[2*r-1,2*r-1])[0]) for T in controls.reshape(-1,2*r,2*r));delta=max(0.,worst)/d0
  P=P0+delta*np.diag(sdiag);RP=root(P[r:,r:]);AP=la.solve(Ac,RP.T,assume_a='pos').T
  affine=max(la.norm(RP@zc-AP@(AA(h)-Ac)@zc,2) for h in verts)
  f1=max(la.norm(AP@(AA(h)-Ac),2) for h in verts);f2=max(la.norm((AA(h)-Ac)@zc,2) for h in verts);rem=f1*f2/beta;uniform=(affine+rem)**2
  samples=[]
  for h in [hc,*verts]:
   L=LL(h);N=np.zeros_like(Nbig);N[:r,:r]=NN(h);Ph=sym(la.solve_continuous_lyapunov(L.T,N));z=la.solve(AA(h),B,assume_a='pos');actual=max(0.,float(la.eigvalsh(sym(z.T@Ph[r:,r:]@z))[-1]));storage=max(0.,float(la.eigvalsh(sym(z.T@P[r:,r:]@z))[-1]));samples.append(dict(h=h.tolist(),exact_reduced_output_integral_upper_gram=actual,common_storage_at_input=storage))

  ACroot=root(Ac);l,U=la.eigh(Ac);Ai=(U/np.sqrt(l))@U.T
  rho=max(la.norm(Ai@(AA(h)-Ac)@Ai,2) for h in verts)
  neumann=None
  if rho<1:
   Lfirst=np.block([[Ac,np.zeros_like(Ac),-As[1]],[np.zeros_like(Ac),Ac,-As[2]],[np.zeros_like(Ac),np.zeros_like(Ac),Ac]])
   ogrid=np.zeros((3,3,3*r,3*r))
   for i,j in itertools.product(range(3),repeat=2):
    h=lo+(hi-lo)*np.array([i,j])/2;T=np.vstack([np.eye(r),h[0]*np.eye(r),h[1]*np.eye(r)])
    ogrid[i,j,:,:r]=(h[0]-hc[0])*T;ogrid[i,j,:,r:2*r]=(h[1]-hc[1])*T
   amps=[]
   for O in bernstein(ogrid).reshape(-1,3*r,3*r):
    PF=sym(la.solve_continuous_lyapunov(Lfirst.T,sym(O.T@Q@O)));J=sym(zc.T@PF[2*r:,2*r:]@zc);amps.append(math.sqrt(max(0.,float(la.eigvalsh(J)[-1]))))
   mout=max(math.sqrt(max(0.,float(la.eigvalsh(sym(Ai@NN(h)@Ai))[-1]))) for h in verts)
   dz=max(la.norm(Ai@(AA(h)-Ac)@zc,2) for h in verts)/math.sqrt(bc)/(1-rho)
   remainder=mout*(rho*dz+rho*rho/(1-rho)*(la.norm(zc,2)+dz))/math.sqrt(2)
   neumann=dict(relative_perturbation=rho,first_order_Bernstein_amplitude=max(amps),remainder_amplitude=remainder,uniform_integral_bound=(max(amps)+remainder)**2,steady_source_difference_bound=dz)
  row=dict(width=width,input_driven_neumann=neumann,lo=lo.tolist(),hi=hi.tolist(),beta=beta,center_beta=bc,Emax=emax,kappa=kappa,storage_repair_delta=delta,uniform_input_output_integral_bound=uniform,affine_amplitude_bound=affine,remainder_amplitude_bound=rem,sample_max_output_integral=max(x['exact_reduced_output_integral_upper_gram'] for x in samples),samples=samples,storage_min_eigenvalue=float(la.eigvalsh(P,subset_by_index=[0,0])[0]),unrepaired_control_max=worst)
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
 out=dict(n=len(c),V_order=r,rows=rows,costs=sol.counts(),seconds=time.perf_counter()-clock,scope='Continuous effective-HTC cell, fixed HTC, arbitrary signed constant inputs, all-time integrated D-weighted proxy tail ONLY. Ordinary-float evaluation, no outward rounding; native_validated=false. Not a complete original ROM step/steady certificate.')
 a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,default=10);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--widths',type=float,nargs='+',default=[.01,.1,.5,-1]);run(p.parse_args())
