"""Embedded Robin-HTC parametric model vs EXACT INTERNAL Robin-face states.

Standalone numerical research prototype informed by MetaHotspot agent/experiment1018
(a847444). Does not claim a rigorous error certificate or production performance.
Requires scipy/numpy. No complex full-order enrichment solves are used.
"""
from __future__ import annotations
import argparse
import time
import numpy as np
from scipy import linalg as la
from scipy import sparse as sp
from scipy.sparse.linalg import splu


def make_model(nx=12, ny=15):
    n=nx*ny
    rows=[]; cols=[]; values=[]
    def add_edge(i,j,k):
        for a,b,v in ((i,i,k),(j,j,k),(i,j,-k),(j,i,-k)):
            rows.append(a);cols.append(b);values.append(v)
    for y in range(ny):
        for x in range(nx):
            i=y*nx+x
            if x+1<nx:
                k=0.7*(1. + (0.8 if y>ny//2 else 0.) + (0.65 if x>=nx//2 and y<ny//2 else 0.))
                add_edge(i,i+1,k)
            if y+1<ny:
                k=0.58*(1. + (0.85 if y>ny//2 else 0.) + (0.7 if y<ny//2 and x<nx//2 else 0.))
                add_edge(i,i+nx,k)
    # An independent weak side cooling path makes the DC problem nonsingular at h=0.
    for y in range(3,ny-2):
        i=y*nx
        rows.append(i);cols.append(i);values.append(0.015)
    k=sp.coo_matrix((values,(rows,cols)),shape=(n,n)).tocsc()
    c=sp.diags(np.broadcast_to((0.5+0.045*np.arange(ny)[:,None]
                +0.15*(np.arange(nx)[None,:]>=nx//2)),(ny,nx)).ravel(),format='csc')
    top=np.arange(nx); bottom=np.arange(n-nx,n)
    masks=[]
    for sel in (bottom[:nx//2],bottom[nx//2:]):
        d=np.zeros(n);d[sel]=1.
        masks.append(d)
    hh=[sp.diags(d,format='csc') for d in masks]
    # Individual top-surface heat flux inputs, two independent internal powers,
    # and two Robin ambient-temperature inputs (h-dependent).
    u=np.zeros((n,nx+4))
    for j in range(nx):u[top[j],j]=1.
    sources=[(ny//3)*nx+nx//3,(ny//2)*nx+3*nx//4]
    for j,i in enumerate(sources):u[i,nx+j]=1.
    # h-dependent ambient columns are populated at solve time.
    y=np.zeros((nx+4,n))
    for j,i in enumerate(top):y[j,i]=1.
    for j,i in enumerate(sources):y[nx+j,i]=1.
    y[nx+2,bottom[:nx//2]]=2./nx
    y[nx+3,bottom[nx//2:]]=2./nx
    return dict(k=k,c=c,top=top,bottom=bottom,hh=hh,masks=masks,
                u=u,y=y,src=sources,n=n,nx=nx,ny=ny)


def ambient_input(model,hs):
    rhs=model['u'].copy()
    for j,d in enumerate(model['masks']):rhs[:,model['nx']+2+j]=hs[j]*d
    return rhs


def k_of(model,hs):
    return model['k']+sum(h*mat for h,mat in zip(hs,model['hh']))


def orth_basis(x,atol=1e-10):
    if x.size==0:return np.empty((x.shape[0],0))
    u,s,_=la.svd(x,full_matrices=False,check_finite=False)
    r=int(np.count_nonzero(s>atol*s[0])) if len(s) else 0
    return u[:,:r]


def basis_for(model,mode,rank,*,anchors=None,shifts=(0.,.03,.4,5.)):
    # These are exact physical states, not approximate low-dimensional thermal-face averages.
    top=model['top'];bot=model['bottom']
    # "ports" below are only auxiliary coordinates for the internal elimination.
    # The exported physical connection boundary remains TOP ONLY.
    ports=np.r_[top,bot] if mode=='trace_exact' else top
    interior=np.setdiff1d(np.arange(model['n']),ports)
    if mode=='trace_exact':
        anchors=[(0.,0.)]
    if anchors is None:anchors=[(.4,.4)]
    zero=[]; other=[]; solve_count=0; rhs_count=0
    for hs in anchors:
        km=k_of(model,hs)
        for sigma in shifts:
            a=(km+sigma*model['c']).tocsc()
            aii=a[interior,:][:,interior].tocsc()
            aip=a[interior,:][:,ports].toarray()
            b0=model['u'][interior,:model['nx']+2]
            # Include ambient coupling directions without multiplying by h:
            # they remain valid for all nonnegative HTC amplitudes.
            ba=np.column_stack([d[interior] for d in model['masks']]) if mode!='trace_exact' else np.zeros((len(interior),0))
            # Output adjoint RHS for the two internal temperature sensors.
            bs=model['y'][model['nx']:model['nx']+2,interior].T
            rhs=np.column_stack((-aip,b0[:,model['nx']:],ba,bs))
            x=splu(aii).solve(rhs)
            solve_count+=1;rhs_count+=rhs.shape[1]
            if sigma==0:
                zero.append(x)
            else:
                other.append(x/la.norm(x,ord='fro'))
    # Preserve a nominal DC anchor, then use SVD to share remaining real shifts.
    first=orth_basis(zero[0])
    if first.shape[1]>rank-len(ports):
        raise ValueError(f'budget {rank} too small for {first.shape[1]} independent DC vectors + {len(ports)} ports')
    candidates=zero[1:]+other
    if len(candidates):
        z=np.column_stack(candidates)
        z-=first@(first.T@z)
        tail=orth_basis(z)
        remaining=min(rank-len(ports)-first.shape[1],tail.shape[1])
        sub=la.qr(np.column_stack((first,tail[:,:remaining])),mode='economic')[0]
    else:
        sub=first
    p=np.zeros((model['n'],len(ports)+sub.shape[1]))
    p[ports,np.arange(len(ports))]=1.
    p[np.ix_(interior,np.arange(len(ports),p.shape[1]))]=sub
    return dict(p=p,ports=ports,dimension=p.shape[1],rank_inner=sub.shape[1],
                full_real_lu=solve_count,full_real_rhs=rhs_count)


def reduce_model(model,p):
    # h enters only as affine, positive semidefinite projected face terms.
    return dict(k=p.T@(model['k']@p),c=p.T@(model['c']@p),
                hs=[p.T@(h@p) for h in model['hh']],
                u=p.T@model['u'],
                ambient=[p.T@d for d in model['masks']],
                y=model['y']@p)


def rom_transfer(r,hs,omega):
    a=r['k']+sum(h*m for h,m in zip(hs,r['hs']))+1j*omega*r['c']
    rhs=r['u'].astype(complex,copy=True)
    for j,d in enumerate(r['ambient']):rhs[:,-2+j]=hs[j]*d
    return r['y']@la.solve(a,rhs,assume_a='gen',check_finite=False)


def full_transfer(model,hs,omega):
    a=(k_of(model,hs)+1j*omega*model['c']).tocsc()
    return model['y']@splu(a).solve(ambient_input(model,hs))


def schur(a,ports):
    n=a.shape[0]
    ins=np.setdiff1d(np.arange(n),ports)
    aii=a[ins,:][:,ins].tocsc()
    agi=a[ports,:][:,ins]
    aig=a[ins,:][:,ports].toarray()
    return a[ports,:][:,ports].toarray()-agi@splu(aii).solve(aig)


def reduced_schur_on_ports(a,ports,p):
    ints=np.setdiff1d(np.arange(a.shape[0]),ports)
    v=p[ints,len(ports):]
    ai=a[ints,:][:,ints]
    aip=a[ints,:][:,ports].toarray()
    agi=a[ports,:][:,ints]
    agg=a[ports,:][:,ports].toarray()
    return agg-(agi@v)@la.solve(v.T@(ai@v),v.T@aip)


def robin_eliminate(s,h,ntop):
    a=s[ntop:,ntop:]+np.diag(h)
    return s[:ntop,:ntop]-s[:ntop,ntop:]@la.solve(a,s[ntop:,:ntop])


def audit_identities(model,trace_exact_p):
    ports=np.r_[model['top'],model['bottom']]
    h=(.012,2.25);w=.47
    a0=(model['k']+1j*w*model['c']).tocsc()
    s=schur(a0,ports)
    sr=reduced_schur_on_ports(a0,ports,trace_exact_p)
    hh=np.r_[h[0]*np.ones(model['nx']//2),h[1]*np.ones(model['nx']-model['nx']//2)]
    f0=robin_eliminate(s,hh,model['nx'])
    fr=robin_eliminate(sr,hh,model['nx'])
    a=(k_of(model,h)+1j*w*model['c']).tocsc()
    exact=schur(a,model['top'])
    closure_identity=la.norm(exact-f0)/(1+la.norm(exact))
    left=np.vstack((np.eye(model['nx']),-la.solve(sr[model['nx']:,model['nx']:]+np.diag(hh),sr[model['nx']:,:model['nx']])))
    right=np.vstack((np.eye(model['nx']),-la.solve(s[model['nx']:,model['nx']:]+np.diag(hh),s[model['nx']:,:model['nx']])))
    residual_identity=la.norm(fr-f0-left.T@(sr-s)@right)/(1+la.norm(fr-f0))
    s0=schur(model['k'],ports)
    lam0=la.eigvalsh(s0[model['nx']:,model['nx']:])[0]
    # Dynamic Dirichlet-to-Neumann is accretive above its static Schur part.
    herm=(s[model['nx']:,model['nx']:]+s[model['nx']:,model['nx']:].conj().T)/2
    acc_floor=la.eigvalsh(herm-s0[model['nx']:,model['nx']:])[0]
    param_rank=[int(np.linalg.matrix_rank(hh.toarray())) for hh in model['hh']]
    return dict(exact_port_closure_rel=closure_identity,
                exact_error_propagation_rel=residual_identity,
                alpha_static_q_block=lam0,
                dynamic_accretive_margin=acc_floor,
                robin_parameter_matrix_ranks=param_rank)


def run(nx=12,ny=15,ranks=(56,72,96,120),n_holdout=12):
    t0=time.perf_counter()
    model=make_model(nx,ny)
    anchors_nom=[(.4,.4)]
    hlow,hhigh=.008,3.
    anchors_corners=[(hlow,hlow),(hlow,hhigh),(hhigh,hlow),(hhigh,hhigh)]
    rng=np.random.default_rng(20261018)
    hold=[tuple(np.exp(rng.uniform(np.log(hlow),np.log(hhigh),2))) for _ in range(n_holdout)]
    freqs=np.r_[0.,np.geomspace(.003,40.,8)]
    # Anchor corners are NOT in the holdout, but are additionally checked as stress cases.
    cases=hold+anchors_corners
    real={}
    for j,h in enumerate(cases):
        for w in freqs:real[(j,w)]=full_transfer(model,h,w)
    print('MODEL',dict(nodes=model['n'],top_port_dofs=nx,robin_face_dofs=nx,
                       independent_htc_parameters=2,frequencies=len(freqs),
                       holdout_htc=len(hold),stress_htc=len(anchors_corners)),flush=True)
    methods={'trace_exact':None,'top_nominal':anchors_nom,'top_corners':anchors_corners}
    for rank in ranks:
        print('\nTOTAL REDUCED UNKNOWN BUDGET',rank,flush=True)
        for name,anchors in methods.items():
            t1=time.perf_counter()
            mode='trace_exact' if name=='trace_exact' else 'embedded'
            basis=basis_for(model,mode,rank,anchors=anchors)
            red=reduce_model(model,basis['p'])
            errs=[]; errs_stress=[]; witness=None
            for j,h in enumerate(cases):
                for w in freqs:
                    ref=real[j,w]; pred=rom_transfer(red,h,w)
                    err=la.norm(pred-ref,2)/max(la.norm(ref,2),1e-14)
                    if j<n_holdout:errs.append(err)
                    else:errs_stress.append(err)
                    if witness is None or err>witness[0]:witness=(err,h,w)
            diag_nnz=sum(np.count_nonzero(abs(mat)>1e-10) for mat in red['hs'])
            psize=basis['p'].shape[1]
            print(name,dict(total=psize,inner=basis['rank_inner'],
                full_real_lu=basis['full_real_lu'],full_real_rhs=basis['full_real_rhs'],
                max_relative_spectral_holdout=float(max(errs)),
                median_relative_holdout=float(np.median(errs)),
                max_relative_spectral_stress=float(max(errs_stress)),
                worst_parameter_frequency=witness,
                robin_projected_nnz=int(diag_nnz),
                build_seconds=round(time.perf_counter()-t1,2)),flush=True)
            if name=='trace_exact' and rank==ranks[0]:
                print('EXACT IDENTITIES',audit_identities(model,basis['p']),flush=True)
            # Structural passivity check (positive K and C), at both extreme h.
            if rank==ranks[0]:
                kmin=min(la.eigvalsh(red['k']+sum(h*m for h,m in zip(ha,red['hs'])))[0] for ha in anchors_corners)
                cmin=la.eigvalsh(red['c'])[0]
                print('POSITIVE STRUCTURE',dict(min_K_at_htc_corners=kmin,min_C=cmin),flush=True)
    print('ELAPSED',round(time.perf_counter()-t0,2),flush=True)
    print('CAUTION: pointwise-frequency / finite-parameter holdout, not H_inf nor a continuum HTC certificate.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--nx',type=int,default=12)
    parser.add_argument('--ny',type=int,default=15)
    parser.add_argument('--ranks',type=int,nargs='+',default=[56,72,96,120])
    parser.add_argument('--holdout',type=int,default=12)
    args=parser.parse_args()
    run(args.nx,args.ny,args.ranks,args.holdout)