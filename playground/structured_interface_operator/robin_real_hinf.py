"""Real rational Krylov moment/HTC derivative comparison + independent numeric Hinf.

Physical ports: top only; the Robin boundary is eliminated internally.
Only nonnegative REAL shifts used to build snapshots. Hinf uses a reduced
real Hamiltonian only for the independent fixed-HTC numerical audit.
No complex full-system enrichment; no continuum-HTC certificate.
"""
from __future__ import annotations
import argparse
from itertools import product
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import splu
from robin_model import make_model, k_of, orth_basis
from robin_followup import dt_n, scaled_error
from robin_lipschitz_probe import realize, at, sample_hinf


def hinf_crossing(a,b,c,d,gamma):
    rr=gamma**2*np.eye(b.shape[1])-d.T@d
    if la.eigvalsh(rr,subset_by_index=(0,0))[0]<=0:return True
    x=la.solve(rr,np.column_stack((d.T@c,b.T)),assume_a='pos')
    xc=x[:,:c.shape[1]];xb=x[:,c.shape[1]:]
    aa=a+b@xc
    ham=np.block([[aa,b@xb],[-(c.T@c+c.T@d@xc),-aa.T]])
    eig=la.eigvals(ham)
    scale=max(1.,la.norm(ham,1))
    return bool(np.any(np.abs(eig.real)<2e-9*scale))


def hinf_numeric(a,b,c,d):
    floor=la.eigvalsh(-a,subset_by_index=(0,0))[0]
    upper=la.norm(d,2)+la.norm(c,2)*la.norm(b,2)/floor
    low=la.norm(d,2)
    high=max(float(upper)*1.0000001,float(low)*1.0001)
    if hinf_crossing(a,b,c,d,high):return None
    for _ in range(29):
        mid=(low+high)/2
        if hinf_crossing(a,b,c,d,mid):low=mid
        else:high=mid
    return high


def moment_basis(model, rank, *, anchors, moments=3, derivatives=1,
                 shifts=(0.,.03,.4,5.)):
    top=model['top'];ni=np.setdiff1d(np.arange(model['n']),top)
    ci=model['c'][ni,:][:,ni]
    di=[hh[ni,:][:,ni] for hh in model['hh']]
    raw=np.column_stack((model['u'][ni,:model['nx']+2],
            np.column_stack([x[ni] for x in model['masks']]),
            model['y'][model['nx']:model['nx']+2,ni].T))
    # The first top-port columns are responses to imposed interface temperature:
    # Aig is negative in the nearest interior layer, so use -Aig.
    dc=None;add=[];lus=0;rhs=0
    for h in anchors:
        kh=k_of(model,h)
        for sigma in shifts:
            a=(kh+sigma*model['c']).tocsc()
            lu=splu(a[ni,:][:,ni].tocsc())
            aig=a[ni,:][:,top].toarray()
            b=raw.copy();b[:,:model['nx']]=-aig
            x=lu.solve(b)
            lus+=1;rhs+=b.shape[1]
            if dc is None and sigma==0:
                dc=orth_basis(x)
            else:
                add.append(x/max(la.norm(x,'fro'),1e-30))
            derivs=[]
            if derivatives:
                derivs=[-lu.solve(d@x) for d in di]
                rhs+=len(di)*x.shape[1]
                add.extend(y/max(la.norm(y,'fro'),1e-30) for y in derivs)
            for k in range(1,moments):
                x=lu.solve(ci@x)
                rhs+=x.shape[1]
                add.append(x/max(la.norm(x,'fro'),1e-30))
                if derivatives:
                    next_derivs=[]
                    for j,d in enumerate(di):
                        nxt=lu.solve(ci@derivs[j]-d@x)
                        rhs+=x.shape[1]
                        next_derivs.append(nxt)
                        add.append(nxt/max(la.norm(nxt,'fro'),1e-30))
                    derivs=next_derivs
    # Normalize each block and use a single shared real projection basis.
    assert dc is not None
    initial_rank=dc.shape[1]
    if rank < len(top)+initial_rank:
        raise ValueError('rank budget insufficient for initial DC directions')
    z=np.column_stack(add)
    z-=dc@(dc.T@z)
    tail=orth_basis(z)
    take=min(rank-len(top)-initial_rank,tail.shape[1])
    v=la.qr(np.column_stack((dc,tail[:,:take])),mode='economic')[0]
    p=np.zeros((model['n'],len(top)+v.shape[1]))
    p[top,np.arange(len(top))]=1.
    p[np.ix_(ni,np.arange(len(top),p.shape[1]))]=v
    return p,lus,rhs


def run(nx=12,ny=15,rank=72, nfreq=9):
    model=make_model(nx,ny)
    clo=(.008,.008);chi=(3.,3.)
    corners=list(product((.008,3.),repeat=2))
    train=corners+[(.4,.4)]
    rng=np.random.default_rng(20261018)
    hold=[tuple(np.exp(rng.uniform(np.log(.008),np.log(3.),2))) for i in range(10)]
    frequencies=np.r_[0.,np.geomspace(.003,40.,nfreq)]
    w=la.cholesky(dt_n(model,(.4,.4),0.),lower=True)
    refs={(i,float(freq)):dt_n(model,h,freq) for i,h in enumerate(hold)
          for freq in frequencies}
    for anchors in (([(.4,.4)],'nominal'),(corners,'corners')):
      hs,tag=anchors
      for m,d in ((1,0),(3,0),(1,1),(3,1)):
        p,lus,rhs=moment_basis(model,rank,anchors=hs,moments=m,derivatives=d)
        base,slopes=realize(model,p)
        numerical_hinfs=[]
        for h in train:
            ss=at(base,slopes,h)
            est=hinf_numeric(*ss)
            sampled=sample_hinf(base,slopes,h,nfreq=130)[0]
            numerical_hinfs.append((float(est**2) if est is not None else None,
                                    float(sampled**2),h))
        errs=[scaled_error(dt_n(model,h,fr,projector=p),refs[(i,float(fr))],w)
              for i,h in enumerate(hold) for fr in frequencies]
        print('REAL_MOMENT_HINF',{'method':tag,'moments':m,'HTC_derivatives':d,
              'rank':p.shape[1],'full_LU':lus,'real_RHS':rhs,
              'max_train_numerical_hinf_J_squared':max(x[0] for x in numerical_hinfs),
              'max_train_sampled_hinf_J_squared':max(x[1] for x in numerical_hinfs),
              'max_independent_holdout_DtN':max(errs),
              'numerical_hinf_samples':numerical_hinfs},flush=True)
    print('NOTE: fixed real shifts, not adaptive; Hamiltonian Hinf is unvalidated FLOAT; finite HTC training and finite heldout frequencies; NOT continuum-HTC certification.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--nx',type=int,default=12)
    p.add_argument('--ny',type=int,default=15)
    p.add_argument('--rank',type=int,default=72)
    p.add_argument('--nfreq',type=int,default=9)
    a=p.parse_args();run(a.nx,a.ny,a.rank,a.nfreq)