"""Robin-HTC-parameter MOR research prototype for MetaHotspot agent/experiment1018.

Only the top edge is an exported physical connection interface. The bottom
Robin edge is always INTERNAL. It is either projected together with all
other internal unknowns (embedded) or retained exactly as internal boundary
states while reducing only the deeper interior (trace-exact).

All enrichment uses NONNEGATIVE REAL shifts and real-valued factorizations.
Finite parameter/frequency checks are diagnostic; analytic uniform bounds
are exact-arithmetic sufficient bounds, not verified floating-point proofs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import time
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import splu

from robin_model import (make_model, k_of, ambient_input, full_transfer,
                         reduce_model, rom_transfer, basis_for, orth_basis,
                         schur, reduced_schur_on_ports, robin_eliminate)


@dataclass
class Counts:
    factorizations: int = 0
    rhs: int = 0


def parameter_taylor_basis(model, rank, h_ref=(0.4, 0.4),
                           shifts=(0., 0.03, 0.4, 5.), order=1):
    """h derivatives at one anchor share each *real* shifted LU.

    X_0=A_ii(href,sigma)^-1 F, X_(k+1,j)=-A_ii^-1 D_j X_k.
    D_j>=0 is localized on the Robin boundary and never a material term.
    """
    top = model['top']; inner = np.setdiff1d(np.arange(model['n']), top)
    inputs = model['u'][inner, model['nx']:model['nx']+2]
    ambient_dirs = np.column_stack([d[inner] for d in model['masks']])
    sensor_dirs = model['y'][model['nx']:model['nx']+2, inner].T
    dc = None; candidates = []; count=Counts()
    for sigma in shifts:
        a = (k_of(model, h_ref)+sigma*model['c']).tocsc()
        lu = splu(a[inner,:][:,inner].tocsc())
        a_ip = a[inner,:][:,top].toarray()
        rhs = np.column_stack((-a_ip, inputs, ambient_dirs, sensor_dirs))
        x = lu.solve(rhs)
        count.factorizations += 1; count.rhs += rhs.shape[1]
        if sigma == 0:
            dc = orth_basis(x)
        else:
            candidates.append(x/la.norm(x, ord='fro'))
        # All derivatives also enter candidate pool; never force projection of h support.
        frontier = [x]
        for step in range(order):
            next_frontier=[]
            for prev in frontier:
                for d in model['hh']:
                    d_in = d[inner,:][:,inner]
                    nxt = -lu.solve(d_in@prev)
                    count.rhs += prev.shape[1]
                    if la.norm(nxt, ord='fro') > 1e-13:
                        candidates.append(nxt/la.norm(nxt,ord='fro'))
                        next_frontier.append(nxt)
            frontier=next_frontier
    assert dc is not None
    max_inner=rank-len(top)
    if dc.shape[1] > max_inner:
        raise ValueError('Cannot preserve nominal full-port DC input block at this rank')
    if candidates:
        z=np.column_stack(candidates)
        z-=dc@(dc.T@z)
        tail=orth_basis(z)
        take=min(max_inner-dc.shape[1], tail.shape[1])
        v=la.qr(np.column_stack((dc,tail[:,:take])),mode='economic')[0]
    else:
        v=dc
    p=np.zeros((model['n'],len(top)+v.shape[1]))
    p[top,np.arange(len(top))]=1.
    p[np.ix_(inner,np.arange(len(top),p.shape[1]))]=v
    return {'p':p,'dimension':p.shape[1], 'rank_inner':v.shape[1],
            'full_real_lu':count.factorizations,'full_real_rhs':count.rhs}


def dt_n(model, hs, omega, *, projector=None):
    a=(k_of(model,hs)+1j*omega*model['c']).tocsc()
    if projector is None:
        return schur(a,model['top'])
    return reduced_schur_on_ports(a,model['top'],projector)


def scaled_error(x, y, w_chol):
    z=la.solve_triangular(w_chol,x-y,lower=True,check_finite=False)
    z=la.solve_triangular(w_chol,z.T,lower=True,check_finite=False).T
    return float(la.norm(z,ord=2))


def uniform_robin_amplification(model, projector, frequencies=(0,.1,1,10)):
    """h-independent exact-arithmetic theorem for INTERNAL Robin elimination.

    For g=external top, b=Robin bottom, z=other internal unknowns,
    T(s) = Schur_z(K+sC) and Tr(s) is its Galerkin approximation.
    U(s,h)=Schur_b(T(s)+diag(0,H(h))). Uniformly for all h>=0, omega R:
      ||Ur-U|| <= [1+(M/alpha)^2] sup_w ||Tr(iw)-T(iw)||,
    where alpha=lambda_min(T_bb(0))>0 and
      M=||K_bg|| + ||Lz^-1 K_zb|| ||Lz^-1 K_zg||,
      Lz Lz^T=K_zz, with diagonal C/no off-block capacity coupling.
    This is a *loose sufficient bound*, NOT a posteriori error measurement.
    """
    g = model['top']; b = model['bottom']
    ab = np.r_[g,b]
    z = np.setdiff1d(np.arange(model['n']),ab)
    nk=len(g); k=model['k']; c=model['c']
    kzz=k[z,:][:,z].toarray()
    lz=la.cholesky(kzz,lower=True)
    kzb=k[z,:][:,b].toarray()
    kzg=k[z,:][:,g].toarray()
    kbg=k[b,:][:,g].toarray()
    a=la.solve_triangular(lz,kzb,lower=True,check_finite=False)
    d=la.solve_triangular(lz,kzg,lower=True,check_finite=False)
    m=float(la.norm(kbg,2)+la.norm(a,2)*la.norm(d,2))
    t0=schur(k,ab)
    alpha=float(la.eigvalsh(t0[nk:,nk:],subset_by_index=(0,0))[0])
    gamma=1.+(m/alpha)**2
    vz=projector[z,nk+len(b):]
    # This uniform theorem only applies to the trace-exact construction.
    if projector.shape[1] < nk+len(b) or not np.allclose(projector[b,nk:nk+len(b)],np.eye(len(b))):
        raise ValueError('The bound requires Robin boundary states exact and internal')
    # Certify alpha_r >= alpha numerically in the chosen floating-point prototype.
    tr0=reduced_schur_on_ports(k,ab,projector)
    alpha_r=float(la.eigvalsh(tr0[nk:,nk:],subset_by_index=(0,0))[0])
    # Finite sample to demonstrate conservatism, NOT the sup over frequencies.
    max_u=0.;max_direct=0.;max_ratio=0.;exact_closure=0.;identity=0.
    for w in frequencies:
        aw=(k+1j*w*c).tocsc()
        te=schur(aw,ab)
        tr=reduced_schur_on_ports(aw,ab,projector)
        delta=tr-te
        max_u=max(max_u,la.norm(delta,2))
        for h1,h2 in ((.008,.008),(.008,3.),(3.,.008),(3.,3.),(.2,1.)):
            hd=np.r_[np.full(len(b)//2,h1),np.full(len(b)-len(b)//2,h2)]
            xe=np.vstack((np.eye(nk),-la.solve(te[nk:,nk:]+np.diag(hd),te[nk:,:nk])))
            xr=np.vstack((np.eye(nk),-la.solve(tr[nk:,nk:]+np.diag(hd),tr[nk:,:nk])))
            ue=robin_eliminate(te,hd,nk)
            ur=robin_eliminate(tr,hd,nk)
            eff=ur-ue
            max_direct=max(max_direct,la.norm(eff,2))
            denom=la.norm(delta,2)
            if denom>1e-14:max_ratio=max(max_ratio,la.norm(eff,2)/denom)
            exact_closure=max(exact_closure, la.norm(dt_n(model,(h1,h2),w)-ue)/max(1.,la.norm(ue)))
            identity=max(identity,la.norm(eff-xr.T@delta@xe)/max(1.,la.norm(eff)))
    return dict(alpha=alpha,alpha_rom=alpha_r,M=m,
                analytic_h_independent_gamma=gamma,
                sampled_omega_only_max_auxiliary_error=max_u,
                sampled_h_omega_only_max_final_error=max_direct,
                sampled_ratio=max_ratio, sampled_identity_rel=identity,
                sampled_exact_condensation_rel=exact_closure)


def experiment(nx=12,ny=15,ranks=(56,72,96),n_holdout=10):
    t0=time.perf_counter(); model=make_model(nx,ny); m=nx
    hlo,hhi=.008,3.
    corners=list(product((hlo,hhi),repeat=2))
    rng=np.random.default_rng(20261018)
    holdout=[tuple(np.exp(rng.uniform(np.log(hlo),np.log(hhi),2))) for _ in range(n_holdout)]
    # A few off-grid points and extreme corners, frozen independently from training.
    cases=holdout+corners
    ws=np.r_[0.,np.geomspace(.003,40.,9)]
    w=la.cholesky(dt_n(model,(.4,.4),0.),lower=True)
    references={(j,float(om)):(dt_n(model,h,om),full_transfer(model,h,om))
                for j,h in enumerate(cases) for om in ws}
    print('CASE',{'physical_connection_dofs':m,'internal_robin_dofs':m,
                  'internal_htc_parameters':2,'full_dofs':model['n'],
                  'parameter_holdout':n_holdout,'stress_corners':len(corners),
                  'frequency_sample_count':len(ws),
                  'note':'frequency and HTC scans are numerical, NOT certified maxima'},flush=True)
    methods=('nominal_real','derivative_1','derivative_2','corner_real','internal_trace_exact')
    for rank in ranks:
        print('RANK_BUDGET',rank,flush=True)
        for method in methods:
            start=time.perf_counter()
            if method=='nominal_real':
                basis=basis_for(model,'embedded',rank,anchors=[(.4,.4)])
            elif method=='corner_real':
                basis=basis_for(model,'embedded',rank,anchors=corners)
            elif method=='derivative_1':
                basis=parameter_taylor_basis(model,rank,order=1)
            elif method=='derivative_2':
                basis=parameter_taylor_basis(model,rank,order=2)
            else:
                # Robin surface exact but INTERNAL; only true g port exported.
                try:
                    basis=basis_for(model,'trace_exact',rank)
                except ValueError as exc:
                    print('SKIP',dict(method=method,budget=rank,reason=str(exc)),flush=True)
                    continue
            reduced=reduce_model(model,basis['p'])
            errs=[];mimo=[];worst=(0.,None,None)
            for j,h in enumerate(cases):
                for om in ws:
                    ref_s,ref_t=references[(j,float(om))]
                    approx_s=dt_n(model,h,om,projector=basis['p'])
                    e=scaled_error(approx_s,ref_s,w)
                    r=la.norm(rom_transfer(reduced,h,om)-ref_t,2)/max(la.norm(ref_t,2),1e-12)
                    if j<n_holdout:
                        errs.append(e);mimo.append(r)
                    if e>worst[0]:worst=(e,h,float(om))
            kmin=float(min(la.eigvalsh(reduced['k']+sum(h*d for h,d in zip(ha,reduced['hs'])),
                                         subset_by_index=(0,0))[0] for ha in corners))
            cmin=float(la.eigvalsh(reduced['c'],subset_by_index=(0,0))[0])
            dense_robin=sum(np.count_nonzero(abs(a)>1e-10) for a in reduced['hs'])
            info=dict(method=method,budget=rank,actual=basis['dimension'],
                      full_real_lu=basis['full_real_lu'],full_real_rhs=basis['full_real_rhs'],
                      worst_holdout_scaled_dtn=max(errs),
                      median_holdout_scaled_dtn=float(np.median(errs)),
                      worst_holdout_relative_mimo=max(mimo),
                      worst_over_holdout_and_corners=(worst[0],worst[1],worst[2]),
                      projected_robin_nnz=dense_robin,
                      small_min_K_over_corners=kmin,small_min_C=cmin,
                      seconds=round(time.perf_counter()-start,3))
            print('RESULT',info,flush=True)
            if method=='internal_trace_exact' and rank==ranks[-1]:
                info=uniform_robin_amplification(model,basis['p'])
                print('TRACE_EXACT_UNIFORM_H_ANALYTIC_DIAGNOSTIC',info,flush=True)
    print('WALL_SECONDS',round(time.perf_counter()-t0,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--nx',type=int,default=12)
    p.add_argument('--ny',type=int,default=15)
    p.add_argument('--ranks',type=int,nargs='+',default=(56,72,96))
    p.add_argument('--holdout',type=int,default=10)
    a=p.parse_args()
    experiment(nx=a.nx,ny=a.ny,ranks=a.ranks,n_holdout=a.holdout)