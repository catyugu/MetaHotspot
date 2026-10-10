"""Decision experiment for one-snapshot Robin coverage; not an extractor.

No shared response spaces. Operational certificates need boundary quadratic
forms and one common Collatz supersolution. Dense solves/eigensystems below
are explicitly charged to small-model reference audits, not certificates.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq

from case1_system import case1_reconstruction
from numerics import Solver, operator, decay_lower, f, relative, c_basis
from probabilistic_extraction import stock
from metahotspot.macromodel import utils


def draw(ranges, rng, size):
    return np.exp(np.log(ranges[:, 0]) + rng.random((size, len(ranges))) *
                  np.log(ranges[:, 1] / ranges[:, 0]))


def production_post(S,c):
    """Use stock normalized-snapshot cutoff AND its exact constant mode."""
    post,_=utils._snapshot_svd_basis(S,.001)
    constant=utils.orthonormalize_block(post,np.ones((len(c),1))/np.sqrt(len(c)))
    if constant.shape[1]: post=np.column_stack([post,constant])
    return c_basis(post,c)


class Anchor:
    def __init__(self, K, C, H, g, hc, shift, alpha):
        self.h = np.asarray(hc)
        self.shift = shift
        A = operator(K, H, hc) + shift * C
        solver = Solver(A, rtol=1e-12)
        self.x = solver.solve(g)
        self.cost = solver.counts()
        self.trace = np.array([self.x @ (J @ self.x) for J in H])
        self.q0 = float(2*g @ self.x - self.x @ (A @ self.x))
        defect = g-A @ self.x
        self.defect_norm=float(la.norm(defect/np.sqrt(C.diagonal())))
        self.alpha=alpha
        self.defect=self.defect_norm/np.sqrt(alpha+shift)
        self.g = g

    def bound(self, h):
        h = np.atleast_2d(h)
        dh = h-self.h
        # Boundary groups must be disjoint: exact diagonal Robin pseudoinverse.
        b = (np.sqrt(np.sum(dh*dh/h*self.trace, axis=1)) + self.defect)**2
        q = self.q0-dh @ self.trace
        return np.sqrt(np.divide(b, q+b, out=np.full_like(b, np.inf), where=q>0))

    def box(self, ranges, tolerance):
        energy = float(self.h @ self.trace)
        cap = float(np.max(np.log(ranges[:, 1]/ranges[:, 0])))
        def bound(r):
            t = np.expm1(r)
            b = (np.sqrt(np.exp(r)*t*t*energy)+self.defect)**2
            q = self.q0-t*energy
            return np.sqrt(b/(q+b)) if q>0 else np.inf
        radius = (cap if bound(cap)<=tolerance else
                  brentq(lambda r: bound(r)-tolerance, 0, cap))
        lo = np.maximum(np.log(ranges[:, 0]), np.log(self.h)-radius)
        hi = np.minimum(np.log(ranges[:, 1]), np.log(self.h)+radius)
        mass = float(np.prod((hi-lo)/np.log(ranges[:, 1]/ranges[:, 0])))
        return dict(log_radius=float(radius), guaranteed_box_mass=mass,
                    log_lo=lo.tolist(),log_hi=hi.tolist(),defect_energy_bound=self.defect)


def rank1_error(A, g, x):
    full = la.solve(A, g, assume_a='pos', check_finite=False)
    reduced = x*float(g @ x/(x @ A @ x))
    e = full-reduced
    return float(np.sqrt(max(0., e @ A @ e/(g @ full))))


def ray_audit(anchor, K, C, H, ranges, tolerance):
    directions = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]])
    rows = []
    for direction in directions:
        available = []
        for i,d in enumerate(direction):
            if d>0: available.append(np.log(ranges[i,1]/anchor.h[i]))
            if d<0: available.append(np.log(anchor.h[i]/ranges[i,0]))
        cap = min(available)
        def value(r, exact=False):
            h = anchor.h*np.exp(r*direction)
            if not exact: return float(anchor.bound(h)[0])
            return rank1_error((operator(K,H,h)+anchor.shift*C).toarray(), anchor.g, anchor.x)
        def first_crossing(exact):
            # Local first crossing only: no assumption of global monotonicity.
            previous = 0.
            for r in np.geomspace(1e-7, max(cap,1e-7), 28):
                if value(r,exact)>tolerance:
                    return float(brentq(lambda z:value(z,exact)-tolerance, previous,r))
                previous = r
            return float(cap)
        certified = first_crossing(False)
        exact = first_crossing(True)
        rows.append(dict(direction=direction.tolist(), certified_radius=certified,
                         reference_first_crossing=exact,
                         reference_to_certified=exact/max(certified,1e-30)))
    return rows


def coverage_stream(K,C,H,g,ranges,alpha,seed,count,tolerance,samples):
    rng = np.random.default_rng(seed)
    hp = draw(ranges,rng,samples)
    bounds = np.full(samples,np.inf)
    anchors = []
    rows = []
    start = time.perf_counter()
    for k in range(count):
        uncovered = np.flatnonzero(bounds>tolerance)
        if not len(uncovered): break
        # Empirical discovery pool, NOT an independent final risk certificate.
        h = hp[rng.choice(uncovered)]
        anchor = Anchor(K,C,H,g,h,0.,alpha)
        anchors.append(anchor)
        bounds = np.minimum(bounds,anchor.bound(hp))
        box = anchor.box(ranges,tolerance)
        rows.append(dict(rhs=k+1, sampled_union_coverage=float(np.mean(bounds<=tolerance)),
                         new_box_mass=box['guaranteed_box_mass']))
    hp_fresh=draw(ranges,rng,65536)
    fresh=np.min(np.array([a.bound(hp_fresh) for a in anchors]),axis=0)
    return dict(history=rows, discovery_pool=samples, fresh_samples=len(hp_fresh),
                fresh_union_hits=int(np.sum(fresh<=tolerance)),
                fresh_union_coverage=float(np.mean(fresh<=tolerance)),seconds=time.perf_counter()-start,
                cost={key:sum(a.cost[key] for a in anchors) for key in anchors[0].cost},
                scope='reused discovery pool; sampled union coverage is not a risk certificate')


def affine_bound(anchors,K,H,g,h,C=None,shift=0.):
    """Minimum boundary-residual affine combination, sum(w)=1.

    Same source and shift are essential. This is a witness for the Galerkin
    energy error, not a substitute Galerkin solution. Defects retained through
    a conservative quadratic penalty; no extra inverse actions.
    """
    h=np.atleast_2d(h);X=np.column_stack([a.x for a in anchors])
    hc=np.array([a.h for a in anchors]);k=len(anchors)
    M=np.array([X.T@(J@X) for J in H]);kx=X.T@(K@X);gx=g@X
    B=np.zeros((len(h),k,k))
    for i in range(len(H)):
        dh=hc[None,:,i]-h[:,None,i]
        B+=dh[:,:,None]*M[i][None,:,:]*dh[:,None,:]/h[:,None,None,i]
    # (sum |w_j| defect_j)^2 <= k sum w_j^2 defect_j^2.
    penalty=k*np.array([a.defect_norm for a in anchors])**2/(anchors[0].alpha+shift)
    slope=np.zeros((k,k))
    if any(a.shift!=shift for a in anchors):
        if C is None: raise ValueError('C required for mixed-shift witnesses')
        ds=np.array([a.shift-shift for a in anchors])
        slope=ds[:,None]*(X.T@(C@X))*ds[None,:]/(anchors[0].alpha+shift)
    # Nonnegative ridge is included in the certificate, not silently discarded.
    ridge=np.maximum(np.max(np.abs(B),axis=(1,2))*1e-13,1e-24)
    J=B+slope[None,:,:]+np.eye(k)[None,:,:]*(penalty[None,:,None]+ridge[:,None,None])
    ones=np.ones((len(h),k,1))
    z=np.linalg.solve(J,ones)[:,:,0]
    w=z/np.sum(z,axis=1)[:,None]
    bb=np.einsum('bi,bij,bj->b',w,B,w)
    dd=np.sum(w*w*penalty[None,:],axis=1)
    # triangle inequality between exact boundary residual and actual CG defect
    bb+=ridge*np.sum(w*w,axis=1)
    ss=np.einsum('bi,ij,bj->b',w,slope,w)
    energy=(np.sqrt(np.maximum(bb,0))+np.sqrt(np.maximum(dd,0))+np.sqrt(np.maximum(ss,0)))**2
    reduced=kx[None,:,:]+np.einsum('bi,ijk->bjk',h,M)
    if shift:
        if C is None: raise ValueError('C required for nonzero target shift')
        reduced+=shift*(X.T@(C@X))[None,:,:]
    q=2*w@gx-np.einsum('bi,bij,bj->b',w,reduced,w)
    bound=np.sqrt(np.divide(energy,q+energy,out=np.full(len(h),np.inf),where=q>0))
    return bound,w


def affine_stream(K,C,H,g,ranges,alpha,seed,count,tolerance,samples,audit):
    rng=np.random.default_rng(seed);anchors=[];rows=[];cost=[]
    start=time.perf_counter();checks=0
    for iteration in range(count):
        # Fresh independent parameters for the current frozen space.
        hp=draw(ranges,rng,samples)
        if anchors:
            upper,_=affine_bound(anchors,K,H,g,hp);checks+=len(hp)
            failures=np.flatnonzero(upper>tolerance)
            rows.append(dict(rhs=len(anchors),samples=len(hp),failures=len(failures),
                             sampled_coverage=float(np.mean(upper<=tolerance))))
            if not len(failures): break
            h=hp[failures[0]]  # first random failure, no greedy search
        else: h=hp[0]
        anchor=Anchor(K,C,H,g,h,0.,alpha);anchors.append(anchor);cost.append(anchor.cost)
    hp=draw(ranges,rng,4096)
    upper,w=affine_bound(anchors,K,H,g,hp);checks+=len(hp)
    X=np.column_stack([a.x for a in anchors]);V=c_basis(X,C.diagonal())
    post,_=utils._snapshot_svd_basis(X,.001)
    records=[]
    for index,h in enumerate(hp[:audit]):
        A=operator(K,H,h).toarray();full=la.solve(A,g,assume_a='pos')
        row=dict(h=h.tolist(),bound=float(upper[index]))
        for name,W in [('pre',V),('post',post)]:
            xr=W@la.solve(W.T@A@W,W.T@g,assume_a='pos')
            row[name]=float(np.sqrt((full-xr)@A@(full-xr)/(g@full)))
        trial=X@w[index]
        row['trial']=float(np.sqrt((full-trial)@A@(full-trial)/(g@full)))
        if row['pre']>row['bound']*(1+1e-5)+1e-10:
            raise RuntimeError('operational bound violated by independent oracle')
        records.append(row)
    return dict(rhs=len(anchors),orders=dict(pre=V.shape[1],post=post.shape[1]),history=rows,
                fresh_samples=len(hp),fresh_failures=int(np.sum(upper>tolerance)),
                upper_max=float(np.max(upper)),checks=checks,audit=records,seconds=time.perf_counter()-start,
                cost={key:sum(a[key] for a in cost) for key in cost[0]},
                scope='per-source steady only; finite fresh-sample checks, no final risk acceptance or transient certificate')


def mixed_stream(K,C,G,H,ranges,alpha,plans,seed,tolerance,audit,stock_pre,stock_post):
    """Try residual witnesses in the extraction loop, strictly source-separated."""
    rng=np.random.default_rng(seed);snapshots=[];cost=[];histories=[];allanchors=[]
    start=time.perf_counter();checks=0;cap=96
    for port,plan in enumerate(plans):
        anchors=[];history=[]
        for shift in np.r_[0.,plan['shifts_per_s']]:
            for attempt in range(cap):
                hp=draw(ranges,rng,64)
                if anchors:
                    upper,_=affine_bound(anchors,K,H,G[:,port],hp,C,float(shift));checks+=len(hp)
                    failures=np.flatnonzero(upper>tolerance)
                    if not len(failures):
                        history.append(dict(shift=float(shift),rhs=len(anchors),attempts=attempt,screen_max=float(np.max(upper))))
                        break
                    h=hp[failures[0]]
                else: h=hp[0]
                if len(anchors)>=cap: break
                anchor=Anchor(K,C,H,G[:,port],h,float(shift),alpha)
                anchors.append(anchor);cost.append(anchor.cost);snapshots.append(anchor.x)
            else: raise RuntimeError('mixed-shift iteration cap')
            if len(anchors)>=cap:
                history.append(dict(shift=float(shift),rhs=len(anchors),cap_reached=True));break
        histories.append(history);allanchors.append(anchors)
    training_seconds=time.perf_counter()-start
    validation_start=time.perf_counter()
    hp=draw(ranges,rng,4096);joint=np.zeros(len(hp));perport=[]
    for port,(anchors,plan) in enumerate(zip(allanchors,plans)):
        v=np.zeros(len(hp))
        for shift in np.r_[0.,plan['shifts_per_s']]:
            upper,_=affine_bound(anchors,K,H,G[:,port],hp,C,float(shift))
            v=np.maximum(v,upper);checks+=len(hp)
        perport.append(dict(rhs=len(anchors),fresh_failures=int(np.sum(v>tolerance)),max_bound=float(v.max())))
        joint=np.maximum(joint,v)
    validation_seconds=time.perf_counter()-validation_start
    audit_start=time.perf_counter()
    raw=c_basis(np.column_stack([*snapshots,np.ones(len(G))]),C.diagonal())
    post=production_post(np.column_stack(snapshots),C.diagonal())
    accuracy=audit_models(K,C,G,H,hp[:audit],{'mixed_pre':raw,'mixed_post':post,
                         'stock_pre':stock_pre,'stock_post':c_basis(stock_post,C.diagonal())})
    return dict(rhs=len(cost),source_counts=[len(a) for a in allanchors],history=histories,
                fresh_samples=len(hp),joint_failures=int(np.sum(joint>tolerance)),per_source=perport,
                orders=dict(pre=raw.shape[1],post=post.shape[1]),accuracy=accuracy,checks=checks,
                training_seconds=training_seconds,validation_seconds=validation_seconds,
                audit_seconds=time.perf_counter()-audit_start,
                seconds=time.perf_counter()-start,cost={key:sum(a[key] for a in cost) for key in cost[0]},
                scope='finite real-shift residual witness, not a full-time step certificate; screening is not final risk acceptance')


def audit_models(K,C,G,H,hp,models):
    c=C.diagonal();sqrtc=np.sqrt(c)
    l0=la.eigvalsh(operator(K,H,hp[0]).toarray()/sqrtc[:,None]/sqrtc[None,:])[0]
    l1=la.eigvalsh(operator(K,H,hp[0]).toarray()/sqrtc[:,None]/sqrtc[None,:])[-1]
    times=np.geomspace(1e-3/l1,20/l0,48);rows=[]
    for h in hp:
        A=operator(K,H,h).toarray();l,Q=la.eigh(A/sqrtc[:,None]/sqrtc[None,:]);B=Q.T@(G/sqrtc[:,None])
        values={name:0. for name in models};late={};decompositions={}
        for name,V in models.items():
            lr,Qr=la.eigh(V.T@A@V,V.T@(c[:,None]*V))
            decompositions[name]=(V@Qr,lr,Qr.T@V.T@G)
            full=la.solve(A,G,assume_a='pos');rom=V@la.solve(V.T@A@V,V.T@G,assume_a='pos')
            late[name]=relative(full-rom,full,A)
        for t in times:
            full=(Q@(f(t,l)[:,None]*B))/sqrtc[:,None]
            for name,(W,lr,Br) in decompositions.items():
                values[name]=max(values[name],relative(full-W@(f(t,lr)[:,None]*Br),full,C))
        rows.append(dict(h=h.tolist(),sampled_step=values,steady=late))
    return dict(times=times.tolist(),rows=rows,scope='sampled time audit only')


def step_audit(K,C,G,H,ranges,baseline,pre,post,seed,points):
    rng=np.random.default_rng(seed)
    hp=np.vstack([np.sqrt(ranges[:,0]*ranges[:,1]),draw(ranges,rng,points)])
    c=C.diagonal();sqrtc=np.sqrt(c)
    l0=la.eigvalsh(operator(K,H,ranges[:,0]).toarray()/sqrtc[:,None]/sqrtc[None,:])[0]
    l1=la.eigvalsh(operator(K,H,ranges[:,1]).toarray()/sqrtc[:,None]/sqrtc[None,:])[-1]
    times=np.geomspace(1e-3/l1,20/l0,48)
    # Parameter-centred block, per-source extraction, no cross-source reuse.
    hc=np.sqrt(ranges[:,0]*ranges[:,1]);snapshots=[];locals_=[];cost=[]
    for j,plan in enumerate(baseline['per_port_plans']):
        local=[]
        for s in np.r_[0.,plan['shifts_per_s']]:
            sol=Solver(operator(K,H,hc)+s*C,rtol=1e-12)
            local.append(sol.solve(G[:,j]));cost.append(sol.counts())
        local.append(G[:,j]/c)  # free diagonal initial-slope direction
        locals_.append(np.column_stack(local));snapshots.extend(local)
    raw=c_basis(np.column_stack([*snapshots,np.ones(len(G))]),c)
    compressed=production_post(np.column_stack(snapshots),c)
    models={'stock_pre':pre,'stock_post':c_basis(post,c),'anchor_block_pre':raw,'anchor_block_post':compressed}
    records=[]
    for h in hp:
        A=operator(K,H,h).toarray()
        l,Q=la.eigh(A/sqrtc[:,None]/sqrtc[None,:]);B=Q.T@(G/sqrtc[:,None])
        values={name:0. for name in models};late={}
        decompositions={}
        for name,V in models.items():
            lr,Qr=la.eigh(V.T@A@V,V.T@(c[:,None]*V))
            decompositions[name]=(V@Qr,lr,Qr.T@V.T@G)
            full=la.solve(A,G,assume_a='pos')
            rom=V@la.solve(V.T@A@V,V.T@G,assume_a='pos')
            late[name]=relative(full-rom,full,A)
        for t in times:
            full=(Q@(f(t,l)[:,None]*B))/sqrtc[:,None]
            for name,(W,lr,Br) in decompositions.items():
                rom=W@(f(t,lr)[:,None]*Br)
                values[name]=max(values[name],relative(full-rom,full,C))
        records.append(dict(h=h.tolist(),sampled_all_input_step=values,steady_all_input=late))
    return dict(times=times.tolist(),records=records,orders={k:v.shape[1] for k,v in models.items()},
                anchor_block_rhs=len(cost),anchor_block_cost={key:sum(a[key] for a in cost) for key in cost[0]},
                per_source_snapshot_counts=[a.shape[1] for a in locals_],
                scope='48 sampled times and sampled parameters; not all-time or continuous-domain certification')


def main(a):
    start=time.perf_counter()
    K,C,G,H,ranges,metadata=case1_reconstruction(a.mesh_mm)
    if K.shape[0]>2000: raise ValueError('dense oracle restricted to small models')
    overlap=sum((J.diagonal()>0).astype(int) for J in H)
    if np.max(overlap)>1: raise ValueError('certificate requires disjoint boundary groups')
    setup=Solver(operator(K,H,ranges[:,0]),rtol=1e-12)
    alpha=decay_lower(operator(K,H,ranges[:,0]),C.diagonal(),setup)
    pre,post,baseline=stock(K,C,G,H,ranges,a.seed)
    rows=[]
    for j,plan in enumerate(baseline['per_port_plans']):
        shifts=np.r_[0.,min(plan['shifts_per_s']),np.median(plan['shifts_per_s']),max(plan['shifts_per_s'])]
        for fraction in [.15,.5,.85]:
            hc=np.exp(np.log(ranges[:,0])+fraction*np.log(ranges[:,1]/ranges[:,0]))
            for s in shifts:
                anchor=Anchor(K,C,H,G[:,j],hc,float(s),alpha)
                row=dict(port=j,log_fraction=fraction,shift=float(s),h=hc.tolist(),
                         cost=anchor.cost,**anchor.box(ranges,a.tolerance))
                if fraction==.5 and s==0:
                    row['ray_audit']=ray_audit(anchor,K,C,H,ranges,a.tolerance)
                rows.append(row)
    streams=[coverage_stream(K,C,H,G[:,j],ranges,alpha,a.seed+j,a.anchors,a.tolerance,a.samples)
             for j in range(G.shape[1])]
    affine=[affine_stream(K,C,H,G[:,j],ranges,alpha,a.seed+100+j,min(a.anchors,24),a.tolerance,256,32)
            for j in range(G.shape[1])]
    dynamic=step_audit(K,C,G,H,ranges,baseline,pre,post,a.seed+999,a.audit)
    mixed=mixed_stream(K,C,G,H,ranges,alpha,baseline['per_port_plans'],a.seed+200,a.tolerance,a.audit,pre,post) if a.mixed else None
    out=dict(model=metadata,n=K.shape[0],seed=a.seed,ranges=ranges.tolist(),tolerance=a.tolerance,
             alpha_lower=alpha,common_certificate_setup=setup.counts(),stock=baseline,
             anchor_rows=rows,coverage_streams=streams,affine_streams=affine,mixed_stream=mixed,dynamic_audit=dynamic,seconds=time.perf_counter()-start,
             arithmetic='exact-arithmetic inequalities, measured CG defects; no directed rounding',
             verdict='one-anchor boundary-diagonal certificate gate; no complete extraction algorithm claimed')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(n=out['n'],seed=a.seed,seconds=out['seconds'],
        stock_rhs=baseline['full_rhs'],
        steady_center_box_mass=[r['guaranteed_box_mass'] for r in rows if r['shift']==0 and r['log_fraction']==.5],
        union_coverage=[s['history'][-1]['sampled_union_coverage'] for s in streams],
        fresh_union_coverage=[s['fresh_union_coverage'] for s in streams],
        affine=[{k:s[k] for k in ['rhs','fresh_failures','upper_max','orders']} for s in affine],
        mixed=None if mixed is None else {k:mixed[k] for k in ['rhs','source_counts','joint_failures','seconds','orders']},
        sampled_step_max={k:max(r['sampled_all_input_step'][k] for r in dynamic['records']) for k in dynamic['orders']})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mesh-mm',type=float,default=10)
    p.add_argument('--seed',type=int,default=20261010)
    p.add_argument('--tolerance',type=float,default=.001)
    p.add_argument('--anchors',type=int,default=32)
    p.add_argument('--samples',type=int,default=8192)
    p.add_argument('--audit',type=int,default=24)
    p.add_argument('--mixed',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
