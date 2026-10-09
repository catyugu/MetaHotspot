"""Common input-weighted residual Gramian: continuum/all-time gate, not a new extractor.

See COMMON_RESIDUAL_GRAMIAN_PROOF.md. No floating-point certification claim.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import itertools
import json
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import scipy.linalg as la

from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower, build_basis, norm, f


def sym(a):
    return (a + a.T) / 2


def exact_toy():
    # Klo = I + grounded graph Laplacian; diagonal Robin terms do not commute.
    # V=(e1,e2), C=I, G=V, D has entries -coupling on omitted coordinates.
    coupling = F(1, 10000)
    mu, alpha, a = F(1, 4), F(1), F(4)
    p = coupling**2
    # D^T Klo^-1 D <= coupling^2 I; Ar >= I.
    lmi_margin = 2 * (1-mu)*p - p
    # L <= 4 follows from 1/(2 mu alpha)=2 and 1/(4 mu^2)=4:
    # sqrt(2)+sqrt(6) < 4, since sqrt(12)<4.
    # q <= sqrt(p)*(1/sqrt(2)+a L) <= 17 coupling.
    q = 17 * coupling
    step_threshold = F(1, 1000)
    steady_q_squared = a*p/(2*mu)
    # q/(1-q) <= sqrt(epsilon); square to use exact rational comparisons.
    step_pass = q*q <= step_threshold*(1-q)**2
    # Galerkin K orthogonality: true relative^2 <= s^2/(1+s^2).
    steady_pass = steady_q_squared <= F(1, 1000000)*(1+steady_q_squared)
    return dict(scope='[0,1]^2, every t>0, all real constant input combinations',
                matrices={'C':'I4', 'G':'[e1,e2]', 'V':'[e1,e2]',
                          'K0':[['1+c','0','-c','0'],['0','2+c','0','-c'],
                                ['-c','0','1+c','0'],['0','-c','0','2+c']],
                          'H1':'diag(1,0,0,0)', 'H2':'diag(0,1,0,0)'},
                coupling=str(coupling), P='c^2 I2', alpha=str(alpha), mu=str(mu),
                gramian_lmi_margin=str(lmi_margin), step_error_over_rom_upper=str(q),
                step_true_relative_upper=str(q/(1-q)),
                steady_error_over_rom_squared_upper=str(steady_q_squared),
                accepted=bool(step_pass and steady_pass), exact_rational_certified=True,
                full_parameter_box=True, arithmetic='Python Fraction, structural Loewner proof')


def solve_storage(As, Qs, B, mu, beta, mode):
    """Small SDP followed by explicit eigenvalue-defect repair.

    SDP solver status alone never constitutes a feasibility certificate.
    """
    import cvxpy as cp
    r = len(As[0])
    if mode=='span':
        templates=[sym(la.solve_continuous_lyapunov(a-mu*np.eye(r),q)) for a,q in zip(As,Qs)]
        templates.append(np.eye(r))
        scale=max(max(norm(q) for q in Qs),1e-30)
        templates=[t/scale for t in templates]
        weights=cp.Variable(len(templates),nonneg=True)
        z=cp.Variable(nonneg=True)
        PP=sum((weights[i]*t for i,t in enumerate(templates)))
        constraints=[PP>>0, B.T@PP@B << z*np.eye(B.shape[1])]
        for a,q in zip(As,Qs):
            fixed=[sym(a@t+t@a-2*mu*t) for t in templates]
            constraints.append(sum((weights[i]*t for i,t in enumerate(fixed)))-q/scale>>0)
        problem=cp.Problem(cp.Minimize(z),constraints)
        start=time.perf_counter()
        try:
            # Dense PSD KKT factorization is prohibitive even with few weights.
            # Iterative SCS is only a candidate generator; repair below decides feasibility.
            problem.solve(solver='SCS',max_iters=300,eps=1e-5)
            status=problem.status
        except Exception as exc:
            status=type(exc).__name__+': '+str(exc)
        elapsed=time.perf_counter()-start
        if PP.value is not None and np.all(np.isfinite(PP.value)):
            P=sym(PP.value*scale)
            shift=max(0.,-float(la.eigvalsh(P)[0]));P+=shift*np.eye(r)
            before=min(float(la.eigvalsh(sym(a@P+P@a-2*mu*P-q))[0]) for a,q in zip(As,Qs))
            repair=max(0.,-before)/(2*(beta-mu))*(1+1e-10)
            P+=repair*np.eye(r)
            after=min(float(la.eigvalsh(sym(a@P+P@a-2*mu*P-q))[0]) for a,q in zip(As,Qs))
            return P,dict(mode=mode,solver_status=status,seconds=elapsed,positive_shift=shift,
                          lmi_defect_before_repair=before,isotropic_repair=repair,lmi_min_after_repair=after,
                          weights=weights.value.tolist())
        P=max(norm(q) for q in Qs)/(2*(beta-mu))*np.eye(r)
        return P,dict(mode=mode,solver_status=status,seconds=elapsed,fallback='analytic isotropic storage')
    center = sum(As)/len(As)
    rates, U = la.eigh(sym(center))
    aa = [sym(U.T @ a @ U) for a in As]
    qq = [sym(U.T @ q @ U) for q in Qs]
    bb = U.T @ B
    scale = max(max(norm(q) for q in qq), 1e-30)
    # modal storage normalization improves badly conditioned thermal SDPs
    d = 1/np.sqrt(rates)
    D = np.diag(d)
    P0 = cp.Variable(r, nonneg=True) if mode == 'diagonal' else cp.Variable((r,r), symmetric=True)
    PP = D @ (cp.diag(P0) if mode == 'diagonal' else P0) @ D
    z = cp.Variable(nonneg=True)
    constraints = [] if mode == 'diagonal' else [P0 >> 0]
    for a, q in zip(aa, qq):
        constraints.append(a @ PP + PP @ a - 2*mu*PP - q/scale >> 0)
    constraints.append(bb.T @ PP @ bb << z*np.eye(B.shape[1]))
    problem = cp.Problem(cp.Minimize(z + 1e-8*cp.trace(PP)), constraints)
    start = time.perf_counter()
    try:
        problem.solve(solver='CLARABEL', max_iter=100)
        status = problem.status
    except Exception as exc:
        status = type(exc).__name__ + ': ' + str(exc)
    elapsed = time.perf_counter()-start
    if PP.value is None or not np.all(np.isfinite(PP.value)):
        P = max(norm(q) for q in Qs)/(2*(beta-mu))*np.eye(r)
        return P, dict(solver_status=status, seconds=elapsed, fallback='isotropic analytic feasible storage')
    P = sym(U @ (PP.value*scale) @ U.T)
    # Exact-arithmetic repair formula; computed defect/eigenvalues still floating point.
    positive_shift = max(0., -float(la.eigvalsh(P)[0]))
    P += positive_shift*np.eye(r)
    before = min(float(la.eigvalsh(sym(a@P+P@a-2*mu*P-q))[0]) for a,q in zip(As,Qs))
    repair = max(0., -before)/(2*(beta-mu))
    repair *= 1+1e-10
    P += repair*np.eye(r)
    after = min(float(la.eigvalsh(sym(a@P+P@a-2*mu*P-q))[0]) for a,q in zip(As,Qs))
    return P, dict(solver_status=status, seconds=elapsed, positive_shift=positive_shift,
                   lmi_defect_before_repair=before, isotropic_repair=repair,
                   lmi_min_after_repair=after, mode=mode)


def certify(K,C,G,H,V,box,mode='diagonal'):
    c = C.diagonal(); r = V.shape[1]
    # The theorem requires exact inclusion. In float, enforce and measure it and
    # include its actual residual contribution separately in the error bound.
    B0 = V.T@G
    l,U=la.eigh(sym(B0.T@B0))
    change=(U/np.sqrt(l))@U.T
    G=G@change; B=V.T@G
    R0=G-c[:,None]*V@B
    r0=norm(R0/np.sqrt(c)[:,None])
    Alo=operator(K,H,box[:,0]); Ahi=operator(K,H,box[:,1])
    solver=Solver(Alo)
    alpha=decay_lower(Alo,c,solver)
    Amid=operator(K,H,np.mean(box,axis=1))
    Ac=sym(V.T@(Amid@V))
    Ai=[sym(V.T@(J@V)) for J in H]
    D0=Amid@V-(c[:,None]*V)@Ac
    Ds=[J@V-(c[:,None]*V)@a for J,a in zip(H,Ai)]
    lifts=solver.solve(np.column_stack([D0,*Ds]))
    Y=[lifts[:,i*r:(i+1)*r] for i in range(len(H)+1)]
    defects=[d-Alo@y for d,y in zip([D0,*Ds],Y)]
    As=[]; Qs=[]; Qlower=[]; repairs=[]; vertices=[]
    for h in itertools.product(*box):
        h=np.asarray(h); dh=h-np.mean(box,axis=1)
        av=np.r_[1.,dh]
        a=Ac+sum((x*j for x,j in zip(dh,Ai)),np.zeros_like(Ac))
        d=sum((x*j for x,j in zip(av,[D0,*Ds])),np.zeros_like(D0))
        y=sum((x*j for x,j in zip(av,Y)),np.zeros_like(D0))
        e=sum((x*j for x,j in zip(av,defects)),np.zeros_like(D0))
        correction=norm(e/np.sqrt(c)[:,None])**2/alpha
        # d^T A^-1 d = sym(d^T y)+sym(y^T e)+e^T A^-1 e.
        core=sym(d.T@y)+sym(y.T@e)
        q=core+correction*np.eye(r)
        Qlower.append(core)
        As.append(sym(a));Qs.append(sym(q));repairs.append(correction);vertices.append(h.tolist())
    beta=float(la.eigvalsh(sym(V.T@(Alo@V)))[0])
    a=float(la.eigvalsh(sym(V.T@(Ahi@V)))[-1])
    mu=beta/4
    # Necessary obstruction for *every* common storage P and every 0<mu<beta:
    # P >= fixed-vertex zero-weight observability Gramian; L(mu)>=L(beta).
    plowers=[]
    for av,qlo in zip(As,Qlower):
        Ph=la.solve_continuous_lyapunov(av,qlo)
        plowers.append(max(0.,float(la.eigvalsh(sym(B.T@Ph@B))[-1])))
    plower=max(plowers)
    best_L=1/np.sqrt(2*beta*alpha)+np.sqrt(1/(2*beta*alpha)+1/(4*beta*beta))
    sigma=float(la.svdvals(B)[-1])
    highAr=sym(V.T@(Ahi@V))
    high_energy_min=float(la.eigvalsh(sym(B.T@la.solve(highAr,B,assume_a='pos')))[0])
    unavoidable=np.sqrt(plower)*best_L*sigma/high_energy_min
    if mode=='gate':
        # Do not pay for an SDP after a mathematical necessary gate fails.
        P=max(norm(q) for q in Qs)/(2*(beta-mu))*np.eye(r)
        sd=dict(mode='gate',fallback='analytic isotropic upper; SDP not executed')
    else:
        P,sd=solve_storage(As,Qs,B,mu,beta,mode)
    p=max(0.,float(la.eigvalsh(sym(B.T@P@B))[-1]))
    L=1/np.sqrt(2*mu*alpha)+np.sqrt(1/(2*mu*alpha)+1/(4*mu*mu))
    crossing=np.sqrt(2)*L
    # Exact crossing envelope, f(a,t); zero-time limit included.
    rational_den=float(la.eigvalsh(sym(crossing*B.T@la.solve(np.eye(r)+crossing*highAr,B,assume_a='pos')))[0])/norm(B)
    qstep=np.sqrt(p)*L/rational_den
    # R0 term is bounded by r0*f(alpha,t), relative to sigma*f(a,t).
    qstep+=r0/sigma*max(1.,a/alpha)
    qsteady=(np.sqrt(p/(2*mu))+r0/np.sqrt(alpha))/np.sqrt(high_energy_min)
    return dict(n=len(c),order=r,box=box.tolist(),vertices=vertices,
                alpha=alpha,beta=beta,mu=mu,reduced_upper=a,
                input_gramian_bound=p,input_residual_Cdual=r0,
                denominator_rational_at_crossover=rational_den,
                denominator_steady_K_lower=np.sqrt(high_energy_min),
                fixed_vertex_gramian_lower_bounds=plowers,
                all_storage_all_mu_step_envelope_lower=unavoidable,
                common_scalar_gramian_family_ruled_out=bool(unavoidable>np.sqrt(.001)/(1+np.sqrt(.001))),
                envelope_L=L,crossover_time=crossing,
                step_error_over_rom=qstep,
                step_relative_bound=qstep/(1-qstep) if qstep<1 else None,
                steady_error_over_rom=qsteady,
                steady_relative_bound=qsteady/np.sqrt(1+qsteady*qsteady),
                accepted=bool(qstep<1 and qstep/(1-qstep)<=np.sqrt(.001)
                              and qsteady/np.sqrt(1+qsteady*qsteady)<=.001),
                counts=solver.counts(),CG_gram_corrections=repairs,storage=sd,
                scope='continuous specified box; every positive time; all constant input combinations',
                floating_point_certified=False)


def run(args):
    if args.toy:
        out=exact_toy()
    else:
        start=time.perf_counter()
        K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm)
        center=np.sqrt(ranges[:,0]*ranges[:,1])
        if args.archive:
            V=np.load(args.archive)[args.basis]
            extraction={'scope':'reused archive; construction cost must be attributed separately'}
        else:
            lo=operator(K,H,ranges[:,0]); sol=Solver(lo)
            alpha=decay_lower(lo,C.diagonal(),sol)
            # Input spectrum, not FOM upper eigenvalue, places the training poles.
            g=G/np.sqrt(C.diagonal())[:,None]
            source_rate=norm((G/C.diagonal()[:,None]).T@(operator(K,H,ranges[:,1])@(G/C.diagonal()[:,None])))/float(la.eigvalsh(g.T@g)[0])
            shifts=np.r_[0.,np.geomspace(alpha/10,source_rate*10,args.shifts-1)]
            points=[center] if args.training=='center' else [center,*map(np.asarray,itertools.product(*ranges))]
            V,extraction=build_basis(K,C,G,H,ranges,shifts,points)
            extraction=dict(counts=extraction,preparation=sol.counts(),shifts=shifts.tolist(),
                            training_points=[h.tolist() for h in points])
        if args.save_basis:
            args.save_basis.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.save_basis,candidate=V)
        out=dict(metadata=meta,extraction=extraction,results=[],floating_point_certified=False)
        for width in args.widths:
            box=np.column_stack([center-width*(center-ranges[:,0]),center+width*(ranges[:,1]-center)])
            row=certify(K,C,G,H,V,box,args.storage)
            row['width_fraction']=width;out['results'].append(row)
            print('COMMON_GRAMIAN',len(C.diagonal()),V.shape[1],width,row['step_error_over_rom'],row['steady_relative_bound'],flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        out['seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--toy',action='store_true')
    parser.add_argument('--mesh-mm',type=float,default=10.)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--basis',default='candidate')
    parser.add_argument('--save-basis',type=Path)
    parser.add_argument('--training',choices=['center','corners'],default='corners')
    parser.add_argument('--shifts',type=int,default=8)
    parser.add_argument('--widths',type=float,nargs='+',default=[0.,.01,1.])
    parser.add_argument('--storage',choices=['gate','span','diagonal','dense'],default='gate')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
