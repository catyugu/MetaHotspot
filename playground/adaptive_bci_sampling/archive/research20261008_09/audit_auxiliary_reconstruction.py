"""Small-model equation / time-enclosure experiment, not a unit-test suite."""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.integrate import quad_vec

from case1_system import case1_reconstruction
from residual_reconstruction import operator
from auxiliary_energy import DefectEnvelope


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();K,C,G,H,ranges,meta=case1_reconstruction(10.);saved=np.load(args.archive)
    V=saved['V'];W=saved['W'][:,:16];G=saved['G'];c=C.diagonal();r=V.shape[1];k=W.shape[1]
    h=np.sqrt(ranges[:,0]*ranges[:,1]);A=operator(K,H,h);ar=V.T@(A@V);aw=W.T@(A@W);B=V.T@G
    D=A@V-c[:,None]*(V@ar);R0=G-c[:,None]*(V@B);proj=W.T@D
    Rbar=R0-c[:,None]*(W@(W.T@R0));Da=np.column_stack([D-c[:,None]*(W@proj),A@W-c[:,None]*(W@aw)])
    L=np.block([[ar,np.zeros((r,k))],[proj,aw]]);b=np.vstack([B,W.T@R0]);yi=la.solve(L,b)
    rates,T=la.eig(L);rates=rates.real;T=T.real
    root=la.solve_triangular(la.cholesky(A.toarray(),lower=True),np.column_stack([Rbar,Da]),lower=True)
    stationary=root@np.vstack([np.eye(G.shape[1]),-yi])
    env=DefectEnvelope(stationary,[root[:,G.shape[1]:]],np.ones((1,1,1)),T[None,:,:],rates[None,:],la.solve(T,yi)[None,:,:])
    records=[]
    for a,btime in [(0.,.1/rates.max()),(1/rates.max(),1.2/rates.max()),(1.,1.2),(100.,120.)]:
        def gram(t):
            y=yi-la.expm(-t*L)@yi;d=root@np.vstack([np.eye(G.shape[1]),-y]);return d.T@d
        actual,error=quad_vec(gram,a,btime,epsabs=1e-12,epsrel=1e-10)
        bound=env.integral_matrix_upper(a,btime)[0]
        records.append(dict(interval=[a,btime],minimum_eigenvalue_upper_minus_quadrature=float(la.eigvalsh((bound-actual+bound.T-actual.T)/2)[0]),
                            quadrature_error_estimate=float(error),relative_width=float(la.norm(bound-actual)/max(la.norm(actual),1e-300))))
    # Evaluate the two residual identities at a parameter not among the four nodes.
    box=np.column_stack([h*np.exp(-.1),h*np.exp(.1)]);axes=[box[i] for i in range(len(H))]
    nodes=np.array([[x,y] for x in axes[0] for y in axes[1]])
    fraction=np.array([.37,.61]);hp=box[:,0]+fraction*(box[:,1]-box[:,0]);Ap=operator(K,H,hp)
    weights=np.array([(fraction[0] if i else 1-fraction[0])*(fraction[1] if j else 1-fraction[1]) for i in [0,1] for j in [0,1]])
    S=np.column_stack([V,W]);t=.7;P=Pd=pr=prd=0.;RF=Rbar.copy();RR=np.zeros_like(B)
    for weight,hj in zip(weights,nodes):
        Aj=operator(K,H,hj);aj=V.T@(Aj@V);wj=W.T@(Aj@W);dj=Aj@V-c[:,None]*(V@aj)
        Lj=np.block([[aj,np.zeros((r,k))],[W.T@dj,wj]]);yj=la.solve(Lj,np.eye(len(Lj))-la.expm(-t*Lj))@b;dy=b-Lj@yj
        P+=weight*(S@yj);Pd+=weight*(S@dy);pr+=weight*yj[:r];prd+=weight*dy[:r]
        RF+=weight*((c[:,None]*S)@Lj-Ap@S)@yj
        RR+=weight*(aj-V.T@(Ap@V))@yj[:r]
    directF=G-c[:,None]*Pd-Ap@P;directR=B-prd-(V.T@(Ap@V))@pr
    out=dict(metadata=meta,scope='small dense oracle checks; not a machine or continuous-domain certificate',
             full_residual_identity_relative=float(la.norm(directF-RF)/max(la.norm(G),1e-300)),
             reduced_residual_identity_relative=float(la.norm(directR-RR)/max(la.norm(B),1e-300)),
             time_enclosure_comparisons=records,
             interpretation='near-zero negative upper-minus-reference eigenvalues expose floating roundoff; these comparisons are implementation evidence, not outward-rounded certification; the analytic proof supplies continuum inequalities')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out),flush=True)
