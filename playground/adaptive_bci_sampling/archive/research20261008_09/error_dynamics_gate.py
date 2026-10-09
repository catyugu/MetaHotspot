"""Pointwise/all-time gate for input-driven auxiliary error spaces.

Exact-arithmetic majorants with actual algebraic residuals. Ordinary floating
point, not outward-rounded certification. No continuum-parameter claim.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from case1_system import case1_reconstruction
from residual_reconstruction import Solver, c_basis, decay_lower, norm, operator, f, build_basis


def psd_norm(M):
    return max(0., float(la.eigvalsh((M + M.T) / 2)[-1]))


def auxiliary(A, c, G, V, W, alpha, time_ratio=1.08, dense_audit=False, reference=None):
    """Certify original V, using W only for its error bound, at fixed A."""
    r, k = V.shape[1], W.shape[1]
    Ar = V.T @ (A @ V); Ar = (Ar + Ar.T) / 2
    B = V.T @ G
    R0 = G - (c[:, None] * V) @ B
    D = A @ V - (c[:, None] * V) @ Ar
    Ae = W.T @ (A @ W); Ae = (Ae + Ae.T) / 2
    d = W.T @ D; b0 = W.T @ R0
    L = np.block([[Ar, np.zeros((r, k))], [d, Ae]])
    b = np.vstack([B, b0]); yinf = la.solve(L, b)
    M = np.column_stack([-D + (c[:, None] * W) @ d,
                         (c[:, None] * W) @ Ae - A @ W])
    delta0 = R0 - (c[:, None] * W) @ b0
    Mw = M / np.sqrt(c)[:, None]
    delta_inf = delta0 + M @ yinf
    lr, U = la.eigh(Ar); le, E = la.eigh(Ae)
    zr = U.T @ yinf[:r]; xe = E.T @ yinf[r:]
    de = E.T @ d @ U
    def transient(t, derivative=0):
        er, ee = np.exp(-lr*t), np.exp(-le*t)
        diff = le[:, None] - lr[None, :]
        # Stable divided difference, including repeated rates.
        lo = np.minimum(le[:, None], lr[None, :])
        gap = np.abs(diff)
        conv = np.exp(-lo*t) * np.divide(-np.expm1(-gap*t), gap,
                   out=np.full_like(gap, t), where=gap != 0)
        if derivative:
            hi=np.maximum(le[:,None],lr[None,:])
            powers=sum((lo**(derivative-1-j)*hi**j for j in range(derivative)),np.zeros_like(lo))
            conv=(-lo)**derivative*conv+(-1)**(derivative+1)*np.exp(-hi*t)*powers
            er=er*(-lr)**derivative; ee=ee*(-le)**derivative
        return np.vstack([U @ (er[:, None]*zr),
                          E @ (ee[:, None]*xe - (de*conv) @ zr)])
    def state(t):
        # Avoid cancellation in the short-time branch.
        if t*norm(L) < .05:
            term=t*b; out=term.copy()
            for j in range(1, 14):
                term=(-t/(j+1))*(L@term); out += term
            return out
        return yinf-transient(t)
    S = np.column_stack([np.zeros((k,r)), np.eye(k)])
    # QR before the Gram avoids squaring cancellation between large full maps.
    _,root=la.qr(Mw,mode='economic')
    ext=np.column_stack([delta0/np.sqrt(c)[:,None],Mw]); _,rroot=la.qr(ext,mode='economic')
    del ext
    mz, mx = norm(root[:,:r]), norm(root[:,r:]); dn=norm(d)
    stats={'quadrature_subdivisions':0,'max_relative_quadrature_remainder':0.}
    nodes,weights=np.polynomial.legendre.leggauss(8)
    import math
    gc=math.exp(4*math.lgamma(9)-math.log(17)-3*math.lgamma(17))
    def derivative_integral(a,t,kind,depth=0):
        dt=t-a; F=root if kind=='delta' else S
        fz,fx=norm(F[:,:r]),norm(F[:,r:])
        Q=np.zeros((B.shape[1],B.shape[1]))
        for node,weight in zip(nodes,weights):
            v=F@transient(a+(node+1)*dt/2,1)
            Q += weight*dt/2*(v.T@v)
        caps=[]
        for j in range(17):
            v=transient(a,j+1); vz=norm(v[:r]); vx=norm(v[r:])
            caps.append(fz*vz+fx*(vx+dn*dt*vz))
        remainder=gc*dt**17*sum(math.comb(16,j)*caps[j]*caps[16-j] for j in range(17))
        value=psd_norm(Q)
        if remainder>1e-5*max(value,1e-40) and depth<5:
            stats['quadrature_subdivisions']+=1
            mid=(a+t)/2
            return derivative_integral(a,mid,kind,depth+1)+derivative_integral(mid,t,kind,depth+1)
        stats['max_relative_quadrature_remainder']=max(stats['max_relative_quadrature_remainder'],remainder/max(value,1e-40))
        return value+remainder
    beta=float(lr[0]); betae=float(le[0]) if k else beta; tiny=1e-7/lr[-1]
    end=40/min(beta,betae)
    grid=np.r_[tiny,np.geomspace(tiny,end,int(np.ceil(np.log(end/tiny)/np.log(time_ratio)))+1)[1:]]
    early_der=mz*norm(B)+mx*(norm(b0)+dn*tiny*norm(B))
    early_eta=norm(b0)+.5*tiny*dn*norm(B)
    early=(norm(delta0/np.sqrt(c)[:,None])+.5*tiny*early_der+early_eta)*np.exp(lr[-1]*tiny)/float(la.svdvals(B)[-1])
    maximum=early; tailmax=early; etamax=0.; worst=tiny
    tail=float(tiny*norm(delta0/np.sqrt(c)[:,None])+.5*tiny**2*early_der)
    za=transient(tiny)
    for a,t in zip(grid[:-1],grid[1:]):
        dt=t-a; zt=transient(t); ya=state(a)
        rd=norm(rroot@np.vstack([np.eye(B.shape[1]),ya]))
        I=derivative_integral(a,t,'delta')
        residual_cap=rd+np.sqrt(dt*I)
        nexttail=np.exp(-alpha*dt)*tail+float(f(dt,alpha))*residual_cap
        localtail=tail+float(f(dt,alpha))*residual_cap
        J=derivative_integral(a,t,'eta')
        eta=norm(ya[r:])+np.sqrt(dt*J)
        denom=float(la.svdvals(f(a,lr)[:,None]*(U.T@B))[-1])
        rho=(eta+localtail)/denom
        if rho>maximum: maximum=rho; worst=float(a)
        tailmax=max(tailmax,localtail/denom); etamax=max(etamax,eta/denom)
        tail=nexttail; za=zt
    # All infinite future: triangular semigroup bounds, not time samples.
    zn=norm(yinf[:r]); xn=norm(yinf[r:])
    ztail=np.exp(-beta*end)*zn
    # Integral tail coupling bounded by exp(-min(beta,betae)*t)*t;
    # this function decreases for t>=end by construction.
    xtail=np.exp(-betae*end)*xn+dn*zn*end*np.exp(-min(beta,betae)*end)
    residual_tail=norm(delta_inf/np.sqrt(c)[:,None])+mz*ztail+mx*xtail
    denend=float(la.svdvals(f(end,lr)[:,None]*(U.T@B))[-1])
    tail_ratio=(xn+xtail+max(tail,residual_tail/alpha))/denend
    maximum=max(maximum,tail_ratio)
    tailmax=max(tailmax,max(tail,residual_tail/alpha)/denend)
    steady_eta=np.sqrt(psd_norm(yinf[r:].T@Ae@yinf[r:]))
    steady_tail=norm(delta_inf/np.sqrt(c)[:,None])/np.sqrt(alpha)
    # All-input denominator uses smallest eigenvalue, not maximum.
    steady_den=np.sqrt(max(0.,float(la.eigvalsh(B.T@la.solve(Ar,B))[0])))
    sr=(steady_eta+steady_tail)/steady_den
    audit={}
    if dense_audit:
        if len(c)>2000:raise ValueError('Dense audit restricted to small diagnostic models')
        Af=(sp.diags(1/np.sqrt(c))@A@sp.diags(1/np.sqrt(c))).toarray()
        lf,Uf=la.eigh(Af); gf=Uf.T@(G/np.sqrt(c)[:,None])
        errmax=0.; remmax=0.; bestmax=0.; snapshots=[]
        for t in np.geomspace(tiny,end,180):
            X=Uf@(f(t,lf)[:,None]*gf)
            y=state(t); XR=np.sqrt(c)[:,None]*(V@y[:r]); eh=np.sqrt(c)[:,None]*(W@y[r:])
            err=X-XR; Q=X.T@X
            errmax=max(errmax,np.sqrt(max(0.,float(la.eigvalsh(err.T@err,Q)[-1]))))
            # Generalized symmetric eigenvalues, avoiding nonsymmetric norm.
            rem=err-eh
            remmax=max(remmax,np.sqrt(max(0.,float(la.eigvalsh(rem.T@rem,Q)[-1]))))
            best=err-(np.sqrt(c)[:,None]*W)@((np.sqrt(c)[:,None]*W).T@err)
            bestmax=max(bestmax,np.sqrt(max(0.,float(la.eigvalsh(best.T@best,Q)[-1]))))
            den=float(la.svdvals(X)[-1]); snapshots.append(err/den)
        singular=la.svdvals(np.column_stack(snapshots))
        total=float(np.sum(singular**2))
        audit=dict(sampled_true_step_relative=errmax,sampled_remaining_relative=remmax,
            sampled_best_projection_relative=bestmax,normalized_error_snapshot_singular_values=singular.tolist(),
            ranks_energy_99=int(np.searchsorted(np.cumsum(singular**2),.99*total)+1),
            ranks_energy_9999=int(np.searchsorted(np.cumsum(singular**2),.9999*total)+1),
            all_time_bound_dominates_samples=bool(errmax<=maximum/(1-maximum)) if maximum<1 else None,
            tail_bound_dominates_samples=bool(remmax<=tailmax/(1-maximum)) if maximum<1 else None,
            full_eigendecompositions=1,scope='small diagnostic model only, 180 time samples')
    if reference is not None:
        RA=reference.T@(A@reference); RA=(RA+RA.T)/2
        rl,RU=la.eigh(RA); rb=RU.T@(reference.T@G)
        # A thin QR preserves differences without subtracting nearly equal norms.
        joined=np.sqrt(c)[:,None]*np.column_stack([V,reference,W])
        _,RR=la.qr(joined,mode='economic'); del joined
        RV=RR[:,:r]; RF=RR[:,r:r+reference.shape[1]]; RW=RR[:,r+reference.shape[1]:]
        snapshots=[]; em=0.; rm=0.; bm=0.
        for t in np.geomspace(tiny,end,180):
            X=RF@(RU@(f(t,rl)[:,None]*rb)); y=state(t)
            err=X-RV@y[:r]; rem=err-RW@y[r:]; best=err-RW@(RW.T@err)
            Q=X.T@X
            vals=[np.sqrt(max(0.,float(la.eigvalsh(e.T@e,Q)[-1]))) for e in [err,rem,best]]
            em=max(em,vals[0]); rm=max(rm,vals[1]); bm=max(bm,vals[2])
            snapshots.append(err/float(la.svdvals(X)[-1]))
        sing=la.svdvals(np.column_stack(snapshots)); total=np.sum(sing**2)
        audit=dict(sampled_reference_step_relative=em,sampled_reference_remaining_relative=rm,
            sampled_reference_best_projection_relative=bm,
            normalized_error_snapshot_singular_values=sing.tolist(),
            ranks_energy_99=int(np.searchsorted(np.cumsum(sing**2),.99*total)+1),
            ranks_energy_9999=int(np.searchsorted(np.cumsum(sing**2),.9999*total)+1),
            scope='180 sampled times, richer rational reference; not an exact FOM oracle',
            full_eigendecompositions=0)
    return dict(auxiliary_order=k,step_error_over_rom=maximum,
        step_relative_bound=maximum/(1-maximum) if maximum<1 else None,
        steady_error_over_rom=sr,steady_relative_bound=sr/(1-sr) if sr<1 else None,
        step_tail_over_rom_max=tailmax,auxiliary_over_rom_max=etamax,
        remaining_error_relative_bound=tailmax/(1-maximum) if maximum<1 else None,
        low_dimension_gate_passed=bool(maximum<1 and tailmax/(1-maximum)<=np.sqrt(.001)),
        infinite_tail_over_rom=tail_ratio,worst_time=worst,steady_tail=steady_tail,
        residual_initial=norm(delta0/np.sqrt(c)[:,None]),
        residual_steady=norm(delta_inf/np.sqrt(c)[:,None]),time_intervals=len(grid)-1,
        accepted=bool(maximum<=np.sqrt(.001)/(1+np.sqrt(.001)) and sr<=.001/1.001),
        floating_point_certified=False,scope='fixed parameter, every t>0, all constant input combinations',dense_audit=audit,**stats)


def run(args):
    K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm); c=C.diagonal()
    V=np.load(args.archive)[args.basis]
    # Whiten inputs once: same nonsingular change for all h.
    B=V.T@G; l,U=la.eigh(B.T@B); G=G@((U/np.sqrt(l))@U.T)
    center=np.sqrt(ranges[:,0]*ranges[:,1])
    points={'center':center,'low':ranges[:,0],'high':ranges[:,1],
            'low_high':np.array([ranges[0,0],ranges[1,1]]),
            'high_low':np.array([ranges[0,1],ranges[1,0]])}
    report=dict(n=len(c),primal_order=V.shape[1],metadata=meta,
        basis=args.basis,archive=str(args.archive),points=[],reference_scope='independent full steady only',
        continuous_parameter_certified=False,floating_point_certified=False)
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name in args.points:
        start=time.perf_counter(); h=points[name]; A=operator(K,H,h)
        base=Solver(A); alpha=decay_lower(A,c,base)
        scale=sp.diags(1/np.sqrt(c)); upper=float(np.max(np.asarray(abs(scale@A@scale).sum(axis=1))))
        Ar=V.T@(A@V); Ar=(Ar+Ar.T)/2; B=V.T@G
        D=A@V-(c[:,None]*V)@Ar; R0=G-(c[:,None]*V)@B
        # Nested pole sequence: start with exact steady error forcing.
        positive=np.geomspace(alpha/10,upper*10,args.max_shifts-1)
        # Spread early prefixes across the full spectral interval.
        indices=[]
        # Bit-reversal-like breadth-first bisection order.
        queue=[(0,len(positive)-1)]
        while queue:
            lo,hi=queue.pop(0)
            if lo>hi:continue
            mid=(lo+hi)//2; indices.append(mid); queue.extend([(lo,mid-1),(mid+1,hi)])
        shifts=np.r_[0.,positive[indices]]
        greedy_candidates=np.geomspace(alpha/50,upper*50,96)
        selected=[]; selection_scores=[]
        snapshots=[]; counters=[]; rows=[]; steady_direct=None
        reference=None; reference_record={}
        if args.reference_shifts:
            rs=np.r_[0.,np.geomspace(alpha/10,upper*10,args.reference_shifts-1)]
            reference,rcounts=build_basis(K,C,G,H,np.column_stack([h,h]),rs,parameter_points=[h])
            ref_cert=auxiliary(A,c,G,reference,np.zeros((len(c),0)),alpha,args.time_ratio)
            reference_record=dict(order=reference.shape[1],counts=rcounts,certificate=ref_cert)
        baseline=auxiliary(A,c,G,V,np.zeros((len(c),0)),alpha,args.time_ratio)
        baseline.update(shifts=0,construction_rhs=1,seconds=time.perf_counter()-start)
        rows.append(baseline)
        print('GATE',len(c),name,0,0,baseline['step_relative_bound'],baseline['steady_relative_bound'],baseline['step_tail_over_rom_max'],flush=True)
        # The first snapshot's true defect is retained; no exact-solve assumption.
        for j,shift in enumerate(shifts,1):
            if args.pole_mode=='greedy' and snapshots:
                W=c_basis(np.column_stack(snapshots),c)
                Ae=W.T@(A@W); Ae=(Ae+Ae.T)/2
                maps=np.column_stack([R0,D,c[:,None]*W,A@W])/np.sqrt(c)[:,None]
                _,rr=la.qr(maps,mode='economic'); del maps
                wr0=W.T@R0; wd=W.T@D
                scores=[]
                for s in greedy_candidates:
                    z=la.solve(Ar+s*np.eye(len(Ar)),B,assume_a='pos')
                    xi=la.solve(Ae+s*np.eye(len(Ae)),wr0-wd@z,assume_a='pos')
                    coef=np.vstack([np.eye(B.shape[1]),-z,-s*xi,-xi])
                    scores.append(norm(rr@coef)/((alpha+s)*float(la.svdvals(z)[-1])))
                for s in selected:
                    scores[int(np.argmin(abs(greedy_candidates-s)))]=-1.
                ii=int(np.argmax(scores)); shift=float(greedy_candidates[ii])
                selection_scores.append(float(scores[ii]))
            selected.append(float(shift))
            source=R0-D@la.solve(Ar+shift*np.eye(len(Ar)),B,assume_a='pos')
            solve=base if shift==0 else Solver(A+shift*C)
            snapshots.append(solve.solve(source)); counters.append(solve.counts().copy())
            if j==1:
                Q=B.T@la.solve(Ar,B,assume_a='pos'); ql,qu=la.eigh(Q)
                qi=(qu/np.sqrt(ql))@qu.T
                S0=snapshots[0]; defect=source-A@S0
                sr=np.sqrt(max(0.,float(la.eigvalsh(S0.T@(A@S0),Q)[-1])))
                sr+=norm((defect/np.sqrt(c)[:,None])@qi)/np.sqrt(alpha)
                steady_direct=float(sr/np.sqrt(1+sr*sr))
            if j not in args.checkpoints:continue
            W=c_basis(np.column_stack(snapshots),c)
            row=auxiliary(A,c,G,V,W,alpha,args.time_ratio,args.dense_audit,
                          reference if j==args.max_shifts else None)
            row.update(shifts=j,construction_rhs=1+4*j,
                       cg_iterations=sum(x['cg_iterations'] for x in counters),
                       seconds=time.perf_counter()-start,
                       steady_zero_snapshot_relative_bound=steady_direct,
                       joint_accepted_with_zero_snapshot_steady=bool(
                           row['step_relative_bound'] is not None and row['step_relative_bound']<=np.sqrt(.001) and steady_direct<=.001))
            rows.append(row)
            print('GATE',len(c),name,j,W.shape[1],row['step_relative_bound'],row['steady_relative_bound'],row['step_tail_over_rom_max'],flush=True)
        # Independent full steady state: count the four reference RHS separately.
        X=base.solve(G); XR=V@la.solve(Ar,B,assume_a='pos'); e=X-XR
        trueK=np.sqrt(max(0.,float(la.eigvalsh(e.T@(A@e),X.T@(A@X))[-1])))
        trueC=np.sqrt(max(0.,float(la.eigvalsh(e.T@(c[:,None]*e),X.T@(c[:,None]*X))[-1])))
        # CG defect bound makes the steady reference limitation explicit.
        defect=G-A@X
        ref_abs=norm(defect/np.sqrt(c)[:,None])/np.sqrt(alpha)
        ref_min=np.sqrt(max(0.,float(la.eigvalsh(X.T@(A@X))[0])))
        report['points'].append(dict(name=name,h=h.tolist(),alpha=alpha,upper=upper,
            rows=rows,steady_reference_K_relative=trueK,steady_reference_C_relative=trueC,
            steady_reference_K_defect_over_min=ref_abs/ref_min,
            reference_rhs=4,total_gate_rhs=1+4*args.max_shifts,
            richer_reference=reference_record,
            shifts=selected,pole_mode=args.pole_mode,selection_scores=selection_scores,
            seconds=time.perf_counter()-start))
        save()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mesh-mm',type=float,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--basis',default='candidate')
    p.add_argument('--points',nargs='+',default=['center','low','high'])
    p.add_argument('--max-shifts',type=int,default=12)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[2,4,6,8,12])
    p.add_argument('--time-ratio',type=float,default=1.08)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dense-audit',action='store_true')
    p.add_argument('--reference-shifts',type=int,default=0)
    p.add_argument('--pole-mode',choices=['fixed','greedy'],default='fixed')
    run(p.parse_args())
