"""All-time, all-input auxiliary-error tail certificate at fixed parameters.

This is a deterministic tail of a solved auxiliary dynamics, NOT a projection
tail. Parameter samples never certify the parameter continuum. Exact arithmetic
theory, actual AMG-CG defects, ordinary floating evaluation (no outward rounding).
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

from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower, norm, f, c_basis
from velocity_defect import TaylorModalEnvelope


def sym(a):return (a+a.T)/2


def generalized(a,b):return np.sqrt(max(0.,float(la.eigvalsh(sym(a),sym(b))[-1])))


class DefectEnvelope(TaylorModalEnvelope):
    def __init__(self,constant,*args):
        self.constant=constant
        super().__init__(*args)

    def matrices(self,t,derivative=0):
        out=super().matrices(t,derivative)
        return out+self.constant[None,:,:] if derivative==1 else out


def certify_point(K,C,G,H,V,W,h,time_ratio=1.2):
    start=time.perf_counter();c=C.diagonal();r=V.shape[1];k=W.shape[1];m=G.shape[1]
    if norm(V.T@(c[:,None]*V)-np.eye(r))>1e-8 or norm(W.T@(c[:,None]*W)-np.eye(k))>1e-8:raise ValueError('C orthonormal spaces required')
    Ah=operator(K,H,h);solver=Solver(Ah);alpha=decay_lower(Ah,c,solver)
    ar=sym(V.T@(Ah@V));aw=sym(W.T@(Ah@W));B=V.T@G
    ell,U=la.eigh(B.T@B);change=(U/np.sqrt(ell))@U.T;G=G@change;B=B@change
    D=Ah@V-c[:,None]*(V@ar);R0=G-c[:,None]*(V@B)
    projected=W.T@D;Dw=Ah@W-c[:,None]*(W@aw)
    Da=np.column_stack([D-c[:,None]*(W@projected),Dw])
    Rbar=R0-c[:,None]*(W@(W.T@R0))
    L=np.block([[ar,np.zeros((r,k))],[projected,aw]])
    rhs=np.vstack([B,W.T@R0]);yinfty=la.solve(L,rhs)
    rates,T=la.eig(L);imag=float(np.max(np.abs(rates.imag)))
    if imag>1e-7*max(1.,float(np.max(np.abs(rates)))):raise ValueError('complex numerical spectrum')
    rates=rates.real;T=T.real
    if rates.min()<=0:raise ValueError('stable auxiliary system required')
    condition=float(np.linalg.cond(T))
    if condition>1e10:raise ValueError('ill-conditioned modal representation; use a Schur implementation')
    # One joint inverse, retaining all cross terms and actual CG defects.
    M=np.column_stack([Rbar,Da]);Y=solver.solve(M);E=M-Ah@Y
    Qlow=sym(M.T@Y+Y.T@M-Y.T@(Ah@Y))
    Qup=sym(Qlow+(E.T@(E/c[:,None]))/alpha)
    ell,U=la.eigh(Qup);negative=float(min(0.,ell[0]))
    root=np.sqrt(np.maximum(ell,0))[:,None]*U.T
    stationary_coeff=np.vstack([np.eye(m),-yinfty])
    trial=Y@stationary_coeff;defect=E@stationary_coeff
    rc=sym(1.001*(trial.T@(c[:,None]*trial))+1001*(defect.T@(defect/c[:,None]))/alpha**2)
    P=sym(la.solve_continuous_lyapunov(L.T,Qup[m:,m:]))
    lyap_defect=sym(Qup[m:,m:]-(L.T@P+P@L))
    delta=max(0.,float(la.eigvalsh(lyap_defect)[-1]))
    if delta:P+=1.1*delta*sym(la.solve_continuous_lyapunov(L.T,np.eye(len(L))))
    jinfty=sym(yinfty.T@P@yinfty)
    a=np.sqrt(max(0.,float(la.eigvalsh(rc)[-1])))
    b=np.sqrt(max(0.,float(la.eigvalsh(rc+jinfty)[-1])))
    theta=max(1e-12,min(1e12,b/max(a,1e-300)))
    global_tail=sym((1+theta)*rc+(1+1/theta)*(rc+jinfty))
    coeff=np.ones((1,1,1));rates=rates[None,:];vectors=T[None,:,:]
    defect_env=DefectEnvelope(root@stationary_coeff,[root[:,m:]],coeff,vectors,rates,la.solve(T,yinfty)[None,:,:])
    # W is C orthonormal, so eta itself is an isometric output coordinate.
    output=np.column_stack([np.zeros((k,r)),np.eye(k)])
    error_env=TaylorModalEnvelope([output],coeff,vectors,rates,la.solve(T,rhs)[None,:,:])
    # A second deterministic tail keeps the input-driven Riesz-lifted velocity.
    # It reuses the joint inverse, including its actual equation defect.
    lifted=la.qr(np.sqrt(c)[:,None]*Y[:,m:],mode='economic')[1]
    lifted_defect=la.qr(E[:,m:]/np.sqrt(c)[:,None],mode='economic')[1]
    velocity_env=TaylorModalEnvelope([lifted],coeff,vectors,rates,la.solve(T,rhs)[None,:,:])
    velocity_defect_env=TaylorModalEnvelope([lifted_defect],coeff,vectors,rates,la.solve(T,rhs)[None,:,:])
    rbar_norm=norm(Rbar/np.sqrt(c)[:,None])
    lr,Ur=la.eigh(ar);br=Ur.T@B
    def denominator(t):
        z=f(t,lr)[:,None]*br;return sym(z.T@z)
    dmax=norm(D/np.sqrt(c)[:,None])
    # Cover a resolved initial interval analytically. An excessively tiny
    # cutover needlessly magnifies the sqrt(t) energy enclosure of CG roundoff.
    tiny=min(.01/lr[-1],.001/max(dmax,1e-300));end=35/rates.min()
    grid=np.r_[0.,np.geomspace(tiny,end,int(np.ceil(np.log(end/tiny)/np.log(time_ratio)))+1)]
    cumulative=np.zeros((m,m));iv=icg=0.
    maximum=(tiny*dmax/2+norm(R0/np.sqrt(c)[:,None]))*np.exp(lr[-1]*tiny)
    worst=tiny;worst_terms={}
    for a,b in zip(grid[:-1],grid[1:]):
        cumulative+=defect_env.integral_matrix_upper(a,b)[0]
        vi=float(velocity_env.integral_controls(a,b)[0]);ci=float(velocity_defect_env.integral_controls(a,b)[0])
        if a>0:
            qz=denominator(a);ehat=error_env.matrices(a)[0]
            main=generalized(ehat.T@ehat,qz)
            variation=generalized((b-a)*error_env.integral_matrix_upper(a,b)[0],qz)
            energy_tail=min(generalized(cumulative,qz),generalized(global_tail,qz))
            yh=velocity_env.matrices(a)[0];den=np.sqrt(la.eigvalsh(qz)[0])
            riesz_tail=generalized(yh.T@yh,qz)+(iv+2*vi+2*(icg+ci)/alpha+float(f(b,alpha))*rbar_norm)/den
            tail=min(energy_tail,riesz_tail)
            value=main+variation+tail
            if value>maximum:maximum=value;worst=a;worst_terms=dict(auxiliary_error=main,time_variation=variation,deterministic_tail=tail,energy_tail=energy_tail,riesz_tail=riesz_tail)
        iv+=vi;icg+=ci
    qz=denominator(end);ehat=output@yinfty
    den=np.sqrt(la.eigvalsh(qz)[0]);yh=velocity_env.matrices(np.inf)[0]
    rt=generalized(yh.T@yh,qz)+(iv+2*velocity_env.tail(end)+2*(icg+velocity_defect_env.tail(end))/alpha+rbar_norm/alpha)/den
    tailvalue=(generalized(ehat.T@ehat,qz)+error_env.tail(end)/den+min(generalized(global_tail,qz),rt))
    if tailvalue>maximum:maximum=tailvalue;worst=end;worst_terms={'infinite_time_envelope':tailvalue}
    return dict(n=len(c),primal_order=r,error_space_order=k,parameter=np.asarray(h).tolist(),
                error_over_rom_bound=float(maximum),bound=float(maximum/(1-maximum)) if maximum<1 else None,
                passed_step=bool(maximum<=np.sqrt(.001)/(1+np.sqrt(.001))),
                worst_time=float(worst),worst_terms=worst_terms,stationary_tail_C_upper=float(np.sqrt(max(0.,la.eigvalsh(rc)[-1]))),
                infinite_transient_dual_energy=float(max(0.,la.eigvalsh(jinfty)[-1])),
                full_decay_lower=alpha,modal_condition=condition,modal_imaginary_part=imag,
                joint_dual_gram_negative_roundoff=negative,lyapunov_repair=delta,counts=solver.counts(),
                time_intervals=len(grid)-1,seconds=time.perf_counter()-start,
                analytic_initial_interval_end=float(tiny),
                tail_methods=['input_matrix_energy','input_driven_Riesz_velocity'],
                full_eigendecompositions=0,full_direct_factorizations=0,
                continuous_parameter_certified=False,all_time_inequalities=True,floating_point_certified=False,
                scope='fixed supplied parameter, every t>0, arbitrary signed input combinations')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--mesh-mm',type=float,default=10.);p.add_argument('--ranks',type=int,nargs='+',default=[16,64])
    p.add_argument('--promote-rank',type=int,default=0,help='add leading common error directions to the pre-SVD primal basis; use following directions as auxiliary W')
    p.add_argument('--points',choices=['center','corners'],default='center');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm)
    archive=np.load(args.archive);V=archive['V'];W=archive['W'];G=archive['G'] if 'G' in archive else G
    points=[np.sqrt(ranges[:,0]*ranges[:,1])]
    if args.points=='corners':points+=list(map(np.asarray,itertools.product(*ranges)))
    original_order=V.shape[1]
    if args.promote_rank:
        V=c_basis(np.column_stack([V,W[:,:args.promote_rank]]),C.diagonal());W=W[:,args.promote_rank:]
    report=dict(metadata=meta,original_order=original_order,promoted_error_directions=args.promote_rank,runs=[])
    for k in args.ranks:
        for h in points:
            row=certify_point(K,C,G,H,V,W[:,:k],h);report['runs'].append(row)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print('AUXILIARY_ENERGY',len(V),k,h,row['error_over_rom_bound'],row['seconds'],flush=True)
