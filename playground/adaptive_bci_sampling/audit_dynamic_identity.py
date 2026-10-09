"""Small-model independent identity / infinite-time quadrature research audit."""
import argparse,json
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.integrate import quad_vec
from case1_system import case1_reconstruction
from numerics import operator
from numerics import sym

def run(a):
 d=np.load(a.candidate);V=d['V'];G=d['G'];ranges=d['ranges'];K,C,_,H,_,_=case1_reconstruction(10);c=C.diagonal();r=V.shape[1];B=V.T@G;hc=np.sqrt(ranges[:,0]*ranges[:,1]);Ac=sym(V.T@(operator(K,H,hc)@V));lc,Uc=la.eigh(Ac);rows=[]
 for h in [hc,hc*np.exp([.01,-.01]),hc*np.exp([.1,-.1])]:
  Kh=operator(K,H,h);Ah=sym(V.T@(Kh@V));lh,Uh=la.eigh(Ah);E=Ah-Ac;z=la.solve(Ah,B,assume_a='pos');D=Kh@V-c[:,None]*V@Ah;R0=G-c[:,None]*V@B
  N=sym(D.T@la.solve(Kh.toarray(),D,assume_a='pos'));L=np.block([[Ah,-E],[np.zeros_like(Ah),Ac]]);Nb=np.zeros_like(L);Nb[:r,:r]=N;P=sym(la.solve_continuous_lyapunov(L.T,Nb));J=sym(z.T@P[r:,r:]@z)
  def fun(tau):
   t=tau/lh[0];w=(Uc@(np.exp(-lc*t)[:,None]*(Uc.T@z)))-(Uh@(np.exp(-lh*t)[:,None]*(Uh.T@z)));return w.T@N@w/lh[0]
  Jquad,err=quad_vec(fun,0,np.inf,epsabs=1e-12,epsrel=1e-8)
  defects=[]
  for t in [.01,1.,100.]:
   vc=Uc@(np.exp(-lc*t)[:,None]*(Uc.T@z));vh=Uh@(np.exp(-lh*t)[:,None]*(Uh.T@z));p=z-vc;w=vc-vh;q=E@vc
   lhs=G-c[:,None]*V@(Ac@vc)-Kh@(V@p)-c[:,None]*V@q-D@w
   rhs=R0-D@p-D@w;defects.append(float(la.norm(lhs-rhs)/max(la.norm(G),1e-30)))
  rows.append(dict(h=h.tolist(),quadrature_lyapunov_relative=float(la.norm(J-Jquad)/max(la.norm(J),1e-30)),quadrature_error_estimate=float(err),lyapunov_gram_max=float(la.eigvalsh(J)[-1]),cancellation_relative_defects=defects))
 a.output.write_text(json.dumps(dict(n=len(c),rows=rows,dense_small_FOM_rhs=3*r,scope='Small-model independent numerical audit only; no continuum acceptance or outward rounding'),indent=2)+'\n');print(json.dumps(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
