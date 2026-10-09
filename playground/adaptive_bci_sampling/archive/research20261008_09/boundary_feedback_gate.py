"""Input-driven Robin feedback structural experiments, not continuum acceptance.

Reference diffusion is kept fixed; a group-separated port space preserves PSD
feedback and has no parameter polynomial. Small FOM spectral step oracles are
restricted to n<=2000. Formal references and basis extraction use AMG-CG.
The uniform domination number is a deterministic sufficient steady tail in
exact arithmetic; the sampled relative errors are deliberately labelled.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower, c_basis, f


def sym(a):
    return (a+a.T)/2


def opnorm(a):
    return float(la.svdvals(a)[0]) if a.size else 0.


def invroot(q):
    l,v=la.eigh(sym(q))
    if l[0]<=0:
        raise ValueError('dependent source response')
    return (v/np.sqrt(l))@v.T


def relative(e,x,metric=None):
    xx=x if metric is None else metric@x
    ee=e if metric is None else metric@e
    return float(np.sqrt(max(0.,la.eigvalsh(sym(e.T@ee),sym(x.T@xx))[-1])))


def counts(rows):
    if not rows:
        return {}
    return {k:sum(r[k] for r in rows) for k in rows[0]}


def hpoints(ranges,levels):
    for p in itertools.product(levels,repeat=len(ranges)):
        yield np.exp(np.log(ranges[:,0])+np.asarray(p)*np.log(ranges[:,1]/ranges[:,0]))


def ports(H):
    q=[];groups=[];ids=[]
    for i,h in enumerate(H):
        d=h.diagonal(); ix=np.flatnonzero(d)
        q.append(sp.csc_matrix((np.sqrt(d[ix]),(ix,np.arange(len(ix)))),shape=(len(d),len(ix))))
        groups.extend([i]*len(ix));ids.append(ix.tolist())
    return sp.hstack(q,format='csc'),np.asarray(groups),ids


def group_basis(traces,groups,rank):
    z=[];spectra=[]
    for i in np.unique(groups):
        ix=np.flatnonzero(groups==i)
        u,s,_=la.svd(traces[ix],full_matrices=False)
        r=min(rank,u.shape[1]); zz=np.zeros((len(groups),r));zz[ix]=u[:,:r]
        z.append(zz);spectra.append(s.tolist())
    return np.column_stack(z),spectra


def run(args):
    if args.ranks!=sorted(set(args.ranks)) or min(args.ranks)<=0:
        raise ValueError('ranks must be unique, positive and increasing')
    if args.poles<2 or args.check_poles<2:
        raise ValueError('at least two poles required')
    start=time.perf_counter(); ledger=[]
    K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm)
    c=C.diagonal();sc=np.sqrt(c);n=len(c)
    if args.step_oracle and n>2000:
        raise ValueError('dense step oracle only for small models')
    # Input coordinate normalization is reversible, preserves arbitrary signed u.
    G=G@invroot((G/sc[:,None]).T@(G/sc[:,None]))
    Q,groups,ids=ports(H); qrank=Q.shape[1]
    low=operator(K,H,ranges[:,0]); prep=Solver(low)
    alpha=decay_lower(low,c,prep);ledger.append(prep.counts())
    raw=G/c[:,None]
    source_rate=float(la.eigvalsh(sym(raw.T@(operator(K,H,ranges[:,1])@raw)))[-1])
    poles=np.r_[0.,np.geomspace(alpha,source_rate,args.poles-1)]
    check_poles=np.r_[0.,np.geomspace(alpha/3,source_rate/3,args.check_poles-1)]
    traces=[];states=[];train_info=[]
    train_cache=[]
    for h in hpoints(ranges,[0.,.5,1.]):
        for s in poles:
            A=operator(K,H,h)+s*C;sol=Solver(A);X=sol.solve(G)
            x=sc[:,None]*X;change=invroot(x.T@x)
            traces.append(np.asarray(Q.T@X)@change)
            states.append(x@change)
            train_cache.append((h,float(s),X))
            train_info.append(sol.counts())
    traces=np.column_stack(traces);states=np.column_stack(states)
    Ustate,svstate,_=la.svd(states,full_matrices=False)
    # Source-only fixed-reference baseline, four (or requested) diffusion poles.
    source_snaps=[G/c[:,None],np.ones((n,1))]; source_counts=[]
    for s in poles:
        sol=Solver(low+s*C);source_snaps.append(sol.solve(G));source_counts.append(sol.counts())
    V0=c_basis(np.column_stack(source_snaps),c);B0=V0.T@G
    # Error-space diagnostic on the same training set: projection only.
    errors=[]
    for h,s,X in train_cache:
        Ar=sym(V0.T@(operator(K,H,h)@V0))+s*np.eye(V0.shape[1])
        errors.append((sc[:,None]*(X-V0@la.solve(Ar,B0,assume_a='pos')))
                      @invroot((sc[:,None]*X).T@(sc[:,None]*X)))
    Uerr,sverr,_=la.svd(np.column_stack(errors),full_matrices=False)
    rank_list=args.ranks
    models=[]; basis_counts=[]
    for rank in rank_list:
        Z,spectra=group_basis(traces,groups,rank)
        gr=[min(rank,int((groups==i).sum()),traces.shape[1]) for i in range(len(H))]
        QZ=np.asarray(Q@Z);snaps=list(source_snaps)
        for s in poles:
            sol=Solver(low+s*C);snaps.append(sol.solve(QZ));basis_counts.append(sol.counts())
        V=c_basis(np.column_stack(snaps),c)
        models.append(dict(rank_per_group=rank,Z=Z,QZ=QZ,V=V,group_ranks=gr,
                           spectra=spectra,feedback_C=0.,galerkin_C=0.,feedback_K=0.,galerkin_K=0.,
                           port_projection=0.,state_projection=0.,error_projection=0.,
                           max_feedback_reference_CG_additive=0.))
    # One fixed reference solve per pole, reused across all parameters/ranks.
    # Group basis column order differs across ranks, so build an explicit union.
    union=la.orth(np.column_stack([m['QZ'] for m in models]))
    ref_cache={};ref_counts=[]
    for s in check_poles:
        sol=Solver(low+s*C);F=sol.solve(G);T=sol.solve(union)
        ref_cache[float(s)]=(F,T)
        ref_counts.append(sol.counts())
    reference_counts=[]; max_ref_cg=0.;baseline=0.
    for h in hpoints(ranges,[.125,.375,.625,.875]):
        for s in check_poles:
            A=operator(K,H,h)+s*C;sol=Solver(A);X=sol.solve(G)
            reference_counts.append(sol.counts());x=sc[:,None]*X
            den=float(la.svdvals(x)[-1]); defect=G-A@X
            cg=opnorm(defect/sc[:,None])/(alpha+s)/den;max_ref_cg=max(max_ref_cg,cg)
            if cg>=1:
                raise ValueError('reference CG correction exceeds denominator')
            F,Tunion=ref_cache[float(s)]
            Ar0=sym(V0.T@(operator(K,H,h)@V0))+s*np.eye(V0.shape[1])
            X0=V0@la.solve(Ar0,B0,assume_a='pos');baseline=max(baseline,relative(sc[:,None]*(X-X0),x))
            e0=sc[:,None]*(X-X0)
            for m in models:
                QZ=m['QZ'];T=Tunion@(union.T@QZ)
                Sm=sym(QZ.T@T);Fm=QZ.T@F
                d=np.concatenate([np.full(r,h[i]-ranges[i,0]) for i,r in enumerate(m['group_ranks'])])
                root=np.sqrt(np.maximum(d,0))
                coef=root[:,None]*la.solve(np.eye(len(d))+root[:,None]*Sm*root[None,:],root[:,None]*Fm,assume_a='pos')
                Xhat=F-T@coef
                # Residual relative to the compressed PSD operator includes all
                # fixed-reference CG defects and the finite-port solve defect.
                L=low+s*C
                residual=G-L@Xhat-QZ@(d[:,None]*(QZ.T@Xhat))
                approx_cg=opnorm(residual/sc[:,None])/(alpha+s)/den
                m['max_feedback_reference_CG_additive']=max(m['max_feedback_reference_CG_additive'],approx_cg)
                vf=relative(sc[:,None]*(X-Xhat),x)
                m['feedback_C']=max(m['feedback_C'],(vf+cg+approx_cg)/(1-cg))
                V=m['V'];Ar=sym(V.T@(operator(K,H,h)@V))+s*np.eye(V.shape[1])
                Xv=V@la.solve(Ar,V.T@G,assume_a='pos')
                vg=relative(sc[:,None]*(X-Xv),x)
                m['galerkin_C']=max(m['galerkin_C'],(vg+cg)/(1-cg))
                if s==0:
                    m['feedback_K']=max(m['feedback_K'],relative(X-Xhat,X,operator(K,H,h)))
                    m['galerkin_K']=max(m['galerkin_K'],relative(X-Xv,X,operator(K,H,h)))
                # A pressure trace proxy for the input-driven flux space;
                # groupwise constant delta commutes with the group projector.
                port=np.asarray(Q.T@X)@invroot(x.T@x)
                Z=m['Z'];m['port_projection']=max(m['port_projection'],opnorm(port-Z@(Z.T@port)))
                r=min(m['V'].shape[1],Ustate.shape[1]);us=Ustate[:,:r]
                m['state_projection']=max(m['state_projection'],relative(x-us@(us.T@x),x))
                er=Uerr[:,:min(m['rank_per_group'],Uerr.shape[1])]
                m['error_projection']=max(m['error_projection'],relative(e0-er@(er.T@e0),x))
    out=dict(n=n,metadata=meta,parameters=ranges.tolist(),boundary_ranks=[int((groups==i).sum()) for i in range(len(H))],
             alpha=alpha,source_rate=source_rate,poles=poles.tolist(),check_poles=check_poles.tolist(),
             training_parameters=9,held_out_parameters=16,training_state_spectrum=svstate.tolist(),
             training_error_spectrum=sverr.tolist(),source_reference_order=V0.shape[1],
             sampled_baseline_resolvent_C=baseline,max_reference_CG_correction=max_ref_cg,
             continuous_parameter_certified=False,all_time_step_certified=False,floating_point_certified=False,
             counts={'preparation':counts(ledger),'training':counts(train_info),'source_baseline':counts(source_counts),
                     'port_state_basis_all_ranks':counts(basis_counts),'fixed_reference_checks':counts(ref_counts),
                     'held_out_reference':counts(reference_counts)},
             full_eigendecompositions=0,full_direct_factorizations=0,models=[])
    if args.step_oracle:
        dense_low=low.toarray(); invc=1/sc
        # Arbitrary port source-to-state transfer: fixed reference, full port
        # columns. These singular values are complete at each listed pole.
        arbitrary=[];RQ=None;passive_bounds=[]
        for s in poles:
            M=dense_low+s*np.diag(c)
            both=la.solve(M,np.column_stack([G,Q.toarray()]),assume_a='pos')
            Fstate,RQ=both[:,:G.shape[1]],both[:,G.shape[1]:];out['full_direct_factorizations']+=1
            arbitrary.append({'s':float(s),'singular_values':la.svdvals(sc[:,None]*RQ).tolist()})
            S=sym(Q.T@RQ);Si=la.solve(S,np.eye(qrank),assume_a='pos')
            Fport=np.asarray(Q.T@Fstate)
            Shigh=operator(K,H,ranges[:,1]).toarray()+s*np.diag(c)
            Xhigh=la.solve(Shigh,G,assume_a='pos');out['full_direct_factorizations']+=1
            # G^T X is a lower bound on the C-state response singular value
            # after normalizing G^T C^-1 G=I, by duality and Loewner order.
            d=float(la.eigvalsh(sym(G.T@Xhigh))[0])
            sl,su=la.eigh(S);Sih=(su/np.sqrt(sl))@su.T
            state_gain=opnorm(sc[:,None]*RQ@Sih)
            rows=[]
            for m in models:
                Z=m['Z'];P=np.eye(qrank)-Z@Z.T;Sz=sym(Z.T@S@Z)
                szl,szu=la.eigh(Sz);szih=(szu/np.sqrt(szl))@szu.T
                R0=P@Fport;L=P@S@Z@szih;b=szih@(Z.T@Fport)
                Q0=sym(R0.T@Si@R0)
                k2=opnorm(Sih@L)**2;Q1=k2*sym(b.T@b)
                # Young is optimized only over a scalar grid; every entry is
                # an upper bound, so no global optimization claim is needed.
                best=min(float(la.eigvalsh(sym((1+t)*Q0+(1+1/t)*Q1))[-1]) for t in np.geomspace(1e-5,1e5,121))
                sampled_innovation=0.;identity_defect=0.;identity_absolute=0.
                for h in list(hpoints(ranges,[.125,.375,.625,.875]))+list(map(np.asarray,itertools.product(*ranges))):
                    dp=np.asarray([h[g]-ranges[g,0] for g in groups]);dz=np.concatenate([np.full(r,h[i]-ranges[i,0]) for i,r in enumerate(m['group_ranks'])])
                    rz=np.sqrt(np.maximum(dz,0))
                    coef=rz[:,None]*la.solve(np.eye(len(dz))+rz[:,None]*Sz*rz[None,:],rz[:,None]*(Z.T@Fport),assume_a='pos')
                    residual=P@(Fport-S@Z@coef)
                    sampled_innovation=max(sampled_innovation,opnorm(Sih@residual))
                    rr=np.sqrt(np.maximum(dp,0))
                    actual_coef=rr[:,None]*la.solve(np.eye(len(dp))+rr[:,None]*S*rr[None,:],rr[:,None]*Fport,assume_a='pos')
                    error=-RQ@(actual_coef-Z@coef)
                    predicted=-RQ@(rr[:,None]*la.solve(np.eye(len(dp))+rr[:,None]*S*rr[None,:],rr[:,None]*residual,assume_a='pos'))
                    identity_delta=opnorm(sc[:,None]*(error-predicted))
                    reference_norm=opnorm(sc[:,None]*Fstate)
                    identity_absolute=max(identity_absolute,identity_delta/reference_norm)
                    error_norm=opnorm(sc[:,None]*error)
                    if error_norm>1e-10*reference_norm:
                        identity_defect=max(identity_defect,identity_delta/error_norm)
                rows.append({'rank_per_group':m['rank_per_group'],
                             'source_innovation_M':float(np.sqrt(max(0.,la.eigvalsh(Q0)[-1]))),
                             'feedback_invariance_defect':float(np.sqrt(k2)),
                             'uniform_error_M_per_normalized_input':float(np.sqrt(max(0.,best))),
                             'uniform_resolvent_relative_C_upper':float(state_gain*np.sqrt(max(0.,best))/d),
                             'sampled_innovation_M':sampled_innovation,
                             'feedback_error_identity_relative_defect_above_floor':identity_defect,
                             'feedback_error_identity_reference_normalized_defect':identity_absolute,
                             'identity_relative_floor':1e-10,
                             'scope':'all delta>=0 error in reference M; relative bound restricted to original continuous HTC box',
                             'all_time_step_certified':False,'floating_point_certified':False})
            passive_bounds.append({'s':float(s),'relative_C_denominator_lower':d,'state_gain':state_gain,'models':rows})
        out['arbitrary_port_transfer']=arbitrary
        out['passive_innovation_continuous_parameter_bounds']=passive_bounds
        RQ=la.solve(dense_low,Q.toarray(),assume_a='pos');out['full_direct_factorizations']+=1
        Sfull=sym(Q.T@RQ)
        times=np.geomspace(1e-5/source_rate,40/alpha,args.times)
        # Oracle checks use grid disjoint from training. Original parameter
        # corners are added to expose limiting feedback behaviour.
        for h in list(hpoints(ranges,[.125,.375,.625,.875]))+list(map(np.asarray,itertools.product(*ranges))):
            Af=operator(K,H,h).toarray();rates,U=la.eigh(invc[:,None]*Af*invc[None,:]);out['full_eigendecompositions']+=1
            b=U.T@(G/sc[:,None])
            for m in models:
                d=np.concatenate([np.full(r,h[i]-ranges[i,0]) for i,r in enumerate(m['group_ranks'])])
                QZ=m['QZ'];Ah=dense_low+(QZ*d[None,:])@QZ.T
                lh,Uh=la.eigh(invc[:,None]*Ah*invc[None,:]);out['full_eigendecompositions']+=1
                bh=Uh.T@(G/sc[:,None]);V=m['V'];vw=sc[:,None]*V
                lv,Uv=la.eigh(sym(V.T@Af@V));bv=Uv.T@(V.T@G)
                m.setdefault('sampled_step_feedback_C',0.);m.setdefault('sampled_step_galerkin_C',0.)
                for t in times:
                    x=U@(f(t,rates)[:,None]*b)
                    xh=Uh@(f(t,lh)[:,None]*bh)
                    xv=vw@Uv@(f(t,lv)[:,None]*bv)
                    m['sampled_step_feedback_C']=max(m['sampled_step_feedback_C'],relative(x-xh,x))
                    m['sampled_step_galerkin_C']=max(m['sampled_step_galerkin_C'],relative(x-xv,x))
        # Analytic uniform operator domination for the feedback surrogate:
        # L(h) <= Lmax and Khat(h)>=Klow, theta=max eig(Lmax,Klow).
        for m in models:
            Z=m['Z'];P=np.eye(qrank)-Z@Z.T
            dmax=np.asarray([ranges[g,1]-ranges[g,0] for g in groups])
            Lroot=P*np.sqrt(dmax)[None,:]
            theta=float(la.eigvalsh(sym(Lroot.T@Sfull@Lroot))[-1])
            m['uniform_feedback_steady_K_relative_bound']=max(0.,theta)
        out['times_per_point']=len(times);out['step_parameter_points']=20;out['time_range']=[float(times[0]),float(times[-1])]
    for m in models:
        row={k:v for k,v in m.items() if k not in {'Z','QZ','V','spectra'}}
        row['port_order']=m['Z'].shape[1];row['primal_order']=m['V'].shape[1]
        row['group_trace_singular_values']=m['spectra'];out['models'].append(row)
    out['seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    if args.save_space:
        args.save_space.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.save_space,G=G,ranges=ranges,V0=V0,**{f'Z{m["rank_per_group"]}':m['Z'] for m in models},**{f'V{m["rank_per_group"]}':m['V'] for m in models})
    print(json.dumps({'n':n,'models':[{k:v for k,v in r.items() if 'singular' not in k} for r in out['models']], 'counts':out['counts'],'seconds':out['seconds']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mesh-mm',type=float,default=10.)
    p.add_argument('--poles',type=int,default=4);p.add_argument('--check-poles',type=int,default=3)
    p.add_argument('--ranks',type=int,nargs='+',default=[4,8,16,32])
    p.add_argument('--times',type=int,default=48);p.add_argument('--step-oracle',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--save-space',type=Path)
    run(p.parse_args())
