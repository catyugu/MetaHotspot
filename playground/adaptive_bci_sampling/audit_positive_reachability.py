"""Independent small-model oracle audit, not a unit test or continuum proof."""
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.special import gammainc
from numerics import f, invroot, relative


def run(a):
    start=time.perf_counter();n=24;m=2;degree=3
    c=np.geomspace(.2,2,n)
    edge=np.geomspace(.1,3,n-1)
    K=np.diag(np.r_[edge,0]+np.r_[0,edge])-np.diag(edge,1)-np.diag(edge,-1)
    H=[np.diag(np.r_[1.,np.zeros(n-1)]),np.diag(np.r_[np.zeros(n-1),1.])]
    lo=np.array([.1,.2]);hi=np.array([2.,4.])
    G=np.zeros((n,m));G[2,0]=np.sqrt(c[2]);G[-3,1]=np.sqrt(c[-3])
    sc=np.sqrt(c)
    def op(h):return K+h[0]*H[0]+h[1]*H[1]
    Kh=op(hi);Kl=op(lo);delta=Kh-Kl
    S=la.solve(Kh,G,assume_a='pos');partial=S.copy()
    for _ in range(degree):
        S=la.solve(Kh,delta@S,assume_a='pos');partial+=S
    terminal=la.solve(Kl,delta@S,assume_a='pos')
    subtraction=la.solve(Kl,G,assume_a='pos')-partial
    terminal_identity=la.norm(terminal-subtraction)/la.norm(terminal)
    dmax=float(np.max(np.diag(delta)/c))
    rows=[]
    for p in [[0,0],[.3,.8],[1,0],[1,1]]:
        h=lo+(hi-lo)*p;Dh=Kh-op(h)
        A=op(h)/sc[:,None]/sc[None,:]
        l,U=la.eigh(A);gs=G/sc[:,None]
        # Independent exact constant-input cascade: j=0..degree, then m constant states.
        dim=(degree+1)*n+m
        L=np.zeros((dim,dim))
        for j in range(degree+1):
            z=slice(j*n,(j+1)*n)
            L[z,z]=-Kh/c[:,None]
            if j:L[z,slice((j-1)*n,j*n)]=Dh/c[:,None]
        L[:n,-m:]=G/c[:,None]
        initial=np.zeros((dim,m));initial[-m:]=np.eye(m)
        for t in [1e-5,1e-3,.01,.1,1,10,100]:
            X=(U@(f(t,l)[:,None]*(U.T@gs)))/sc[:,None]
            cascade=(la.expm(t*L)@initial)[:(degree+1)*n]
            truncated=sum(cascade[j*n:(j+1)*n] for j in range(degree+1))
            E=X-truncated
            # Errors of order t^(degree+2) can be below oracle roundoff.
            scale=max(1.,float(np.max(X)))
            tol=2e-10*scale
            poisson=gammainc(degree+1,dmax*t)*X
            rows.append(dict(h=h.tolist(),t=t,min_tail=float(E.min()),
                terminal_margin=float(np.min(terminal-E)),
                poisson_margin=float(np.min(poisson-E)),
                inequality_oracle_holds=bool(E.min()>=-tol and np.min(terminal-E)>=-tol and np.min(poisson-E)>=-tol),
                relative_parameter_tail=relative(E,X,np.diag(c))))
    # Source-error manifold spectrum: diagnostic only, entirely different n=364 Case1.
    candidate=np.load(a.candidate);V=candidate['V'];G=candidate['G'];ranges=candidate['ranges']
    from case1_system import case1_reconstruction
    from numerics import operator, sym
    K,C,_,H,_,_=case1_reconstruction(10);c=C.diagonal();sc=np.sqrt(c);blocks=[];maxerror=0.;worst=None
    for p in itertools.product([0,.5,1],repeat=2):
        h=np.exp(np.log(ranges[:,0])+(np.log(ranges[:,1])-np.log(ranges[:,0]))*p)
        Kh=operator(K,H,h);l,U=la.eigh(Kh.toarray()/sc[:,None]/sc[None,:])
        ar,ur=la.eigh(sym(V.T@(Kh@V)));B=ur.T@(V.T@G)
        for t in np.geomspace(1e-5,1e5,31):
            X=(U@(f(t,l)[:,None]*(U.T@(G/sc[:,None]))))/sc[:,None]
            XR=V@(ur@(f(t,ar)[:,None]*B));E=sc[:,None]*(X-XR)
            T=invroot((sc[:,None]*X).T@(sc[:,None]*X))
            blocks.append(E@T);err=float(la.norm(E@T,2))
            if err>maxerror:maxerror=err;worst=dict(h=h.tolist(),t=float(t),relative_error=err)
    s=la.svdvals(np.column_stack(blocks));energy=np.cumsum(s*s)/np.sum(s*s)
    spectrum=dict(sampled_h_count=9,sampled_time_count=31,n=364,V_order=V.shape[1],
        max_sample_relative_error=maxerror,
        worst_sample=worst,
        rank_for_99_percent_energy=int(np.searchsorted(energy,.99)+1),
        rank_for_9999_percent_energy=int(np.searchsorted(energy,.9999)+1),
        singular_values=s.tolist(),scope='sampled empirical error spectrum ONLY; no continuous-domain tail inferred',
        dense_FOM_eigendecompositions=9)
    out=dict(positive_oracle_model=dict(n=n,degree=degree,lo=lo.tolist(),hi=hi.tolist()),
        terminal_identity_relative=terminal_identity,oracle_rows=rows,
        oracle_all_holds=all(x['inequality_oracle_holds'] for x in rows),
        case1_empirical_error_spectrum=spectrum,seconds=time.perf_counter()-start,
        dense_oracle_cost=dict(small_linear_rhs=12,small_exponentials=28,case1_eigendecompositions=9))
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['oracle_rows','case1_empirical_error_spectrum']}),flush=True)
    print(json.dumps({k:v for k,v in spectrum.items() if k!='singular_values'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
