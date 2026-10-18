"""Analytic (exact-arithmetic) all-frequency Robin-HTC subbox Lipschitz bounds.

For a fixed embedded Galerkin basis, the Riesz residual realization
  J(s,h)=D+C(h)(sI-A(h))^-1 B
has D and B independent of h, A/C affine in two HTC parameters. Positive
Robin matrices imply an explicit resolvent lower spectral bound on every
subbox, hence an all-frequency Lipschitz majorant. We compare its numerical
values with an independent sampled Hinf diagnostic. Neither frequency sample
nor ordinary-float inequalities are verified numerical certificates.
"""
import argparse
from itertools import product
import numpy as np
from scipy import linalg as la
from robin_model import make_model, basis_for, k_of, schur
from robin_followup import parameter_taylor_basis


def realize(model, p, lower=(.008,.008)):
    n=model['n']; g=model['top']; inn=np.setdiff1d(np.arange(n),g)
    r=p.shape[1]-len(g)
    v=p[inn,len(g):]
    cii=model['c'][inn,:][:,inn].toarray()
    kref=k_of(model,lower)[inn,:][:,inn].toarray()
    kbase=model['k'][inn,:][:,inn].toarray()
    f=model['k'][inn,:][:,g].toarray()
    ki_terms=[mat[inn,:][:,inn].toarray() for mat in model['hh']]
    w=schur(k_of(model,(.4,.4)),g)
    lk=la.cholesky(kref,lower=True)
    lw=la.cholesky(w,lower=True)
    cr=v.T@cii@v
    lc=la.cholesky(cr,lower=True)
    lcinv=la.solve_triangular(lc,np.eye(r),lower=True)
    kr0=v.T@kbase@v
    kr_terms=[v.T@ki@v for ki in ki_terms]
    fr=v.T@f
    cf=cii@v@la.solve(cr,fr,assume_a='pos')
    rinf=f-cf
    bwin=la.solve_triangular(lw,fr.T,lower=True).T
    b=lcinv@bwin
    dw=la.solve_triangular(lw,rinf.T,lower=True).T
    d=la.solve_triangular(lk,dw,lower=True)
    av0=-lcinv@kr0@lcinv.T
    avterms=[-lcinv@kr@lcinv.T for kr in kr_terms]
    def cout_of(k,kr):
        q=k@v-(cii@v)@la.solve(cr,kr,assume_a='pos')
        q=q@lcinv.T
        return -la.solve_triangular(lk,q,lower=True)
    cv0=cout_of(kbase,kr0)
    cvterms=[cout_of(ki,kri) for ki,kri in zip(ki_terms,kr_terms)]
    assert la.norm(d)>0
    return (av0,b,cv0,d), (avterms,cvterms)


def at(base, slopes, h):
    a,b,c,d=base
    aj,cj=slopes
    return (a+sum(v*dv for v,dv in zip(h,aj)),b,
            c+sum(v*dv for v,dv in zip(h,cj)),d)


def sample_hinf(base, slopes, h, nfreq=160):
    a,b,c,d=at(base,slopes,h)
    poles=la.eigvalsh(-a)
    freq=np.r_[0.,np.geomspace(poles[0]/100.,poles[-1]*100.,nfreq)]
    out=0.;peak=0.
    eye=np.eye(len(a))
    for w in freq:
        val=la.norm(d+c@la.solve(1j*w*eye-a,b,assume_a='gen',check_finite=False),2)
        if val>out:out=float(val);peak=float(w)
    return out,peak


def box_bound(base, slopes, low, high, sample_nfreq=120):
    center=(np.array(low)+np.array(high))/2
    radius=(np.array(high)-np.array(low))/2
    av,b,cv,d=at(base,slopes,center)
    aj,cj=slopes
    min_pole=la.eigvalsh(-at(base,slopes,low)[0],subset_by_index=(0,0))[0]
    center_obs=sample_hinf(base,slopes,center,sample_nfreq)[0]
    cb=la.norm(cv,2)+sum(x*la.norm(y,2) for x,y in zip(radius,cj))
    norm_b=la.norm(b,2)
    lips=[(la.norm(cj_i,2)/min_pole
           +cb*la.norm(aj_i,2)/min_pole**2)*norm_b
          for aj_i,cj_i in zip(aj,cj)]
    # BOUND-UNVERIFIED-FLOAT only when center value is certified Hinf. Here
    # center_obs is a sampled lower bound, so the aggregate is diagnostic only.
    return dict(center_peak_sample=center_obs,
                lip_radius=float(np.dot(lips,radius)),
                proposed_enclosure_with_UNCERTIFIED_CENTER=center_obs+np.dot(lips,radius),
                min_pole=float(min_pole),lip_constants=tuple(lips))


def main(nx=12,ny=15,ranks=(56,72,96)):
    model=make_model(nx,ny);low=np.array((.008,.008));high=np.array((3.,3.))
    freqs=140
    for rank in ranks:
        for mode in ('derivative_1','corner','trace_exact'):
            if mode=='derivative_1':
                p=parameter_taylor_basis(model,rank,order=1)['p']
            elif mode=='corner':
                p=basis_for(model,'embedded',rank,
                            anchors=list(product((low[0],high[0]),repeat=2)))['p']
            else:
                p=basis_for(model,'trace_exact',rank)['p']
            base,slopes=realize(model,p)
            box=box_bound(base,slopes,low,high,sample_nfreq=freqs)
            worst=(0,None)
            for h in product(np.linspace(low[0],high[0],7),repeat=2):
                score=sample_hinf(base,slopes,h,nfreq=70)[0]
                if score>worst[0]:worst=(score,h)
            vals=[]
            for pieces in (1,2,4,8):
                vals.append((pieces,max(box_bound(base,slopes,
                    low+(high-low)*np.array((i,j))/pieces,
                    low+(high-low)*np.array((i+1,j+1))/pieces,
                    sample_nfreq=40)['proposed_enclosure_with_UNCERTIFIED_CENTER']
                    for i,j in product(range(pieces),repeat=2))))
            print('ROBUST_HTC_LIPSCHITZ',dict(mode=mode, rank=p.shape[1],
                  all_h_corner_to_corner=box,
                  peak_J_sampled_over_7x7_params=worst,
                  sampled_center_plus_analytic_lipschitz_subdivision=vals),flush=True)


if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--nx',type=int,default=12)
    q.add_argument('--ny',type=int,default=15)
    q.add_argument('--ranks',type=int,nargs='+',default=(56,72,96))
    z=q.parse_args();main(z.nx,z.ny,z.ranks)