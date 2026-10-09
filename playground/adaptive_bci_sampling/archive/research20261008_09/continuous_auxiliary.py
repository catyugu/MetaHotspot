"""Continuous parameter/time energy-tail proof for a common auxiliary space.

Polynomial auxiliary trajectories are trials, not assumed exact. Full and ROM
equation defects are bounded jointly in the INPUT coordinates. No Riesz-lift
operator norm on arbitrary reduced states occurs. Ordinary floats, no rounding.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower, norm, f, c_basis
from polynomial_trajectory import interpolation_controls, product_controls
from velocity_defect import TaylorModalEnvelope


def sym(x):return (x+x.T)/2
def maxnorm(matrices):return max(norm(x) for x in matrices)
def maxgram(grams):return np.sqrt(max(0.,max(float(la.eigvalsh(sym(g))[-1]) for g in grams)))


class StepResidual(TaylorModalEnvelope):
    def __init__(self,maps,coeff,vectors,rates,inputs,constant):
        self.stationary=constant.copy()
        # Residual = stationary - sum N_j exp(-L_j t) L_j^-1 b.
        super().__init__([-a for a in maps],coeff,vectors,rates,inputs/rates[:,:,None])
        self.stationary-=super().matrices(0.,1)

    def matrices(self,t,derivative=0):
        out=super().matrices(t,derivative)
        return out+self.stationary if derivative==1 else out

    def transient(self):
        return TaylorModalEnvelope(self.maps,self.coeff,self.vectors,self.rates,self.inputs)


def certify_box(K,C,G,H,V,W,box,degree=2,time_ratio=1.2,inverse_ranges=None,inverse_cache=None):
    start=time.perf_counter();c=C.diagonal();r=V.shape[1];k=W.shape[1];m=G.shape[1];dim=len(H)
    if np.any(c<=0) or (C-sp.diags(c)).nnz:raise ValueError('positive diagonal C required')
    asym=K-K.T;off=K-sp.diags(K.diagonal())
    if (asym.nnz and np.max(np.abs(asym.data))>1e-12) or np.any(off.data>0):raise ValueError('symmetric M-matrix conduction required')
    if norm((K@np.ones(len(c)))[:,None])>1e-8*max(1.,norm(K.diagonal()[:,None])):raise ValueError('conservative conduction required')
    for J in H:
        if np.any(J.diagonal()<0) or (J-sp.diags(J.diagonal())).nnz:raise ValueError('nonnegative diagonal Robin terms required')
    if norm(V.T@(c[:,None]*V)-np.eye(r))>1e-8 or norm(W.T@(c[:,None]*W)-np.eye(k))>1e-8:raise ValueError('C orthonormal spaces required')
    if np.any(box[:,0]<=0) or np.any(box[:,1]<box[:,0]):raise ValueError('positive ordered box required')
    mid=np.mean(box,axis=1);width=box[:,1]-box[:,0];Ac=operator(K,H,mid)
    low=operator(K,H,box[:,0])
    Ar=sym(V.T@(Ac@V));Ai=[sym(V.T@(J@V)) for J in H]
    B=V.T@G;ell,U=la.eigh(B.T@B);change=(U/np.sqrt(ell))@U.T;G=G@change;B=B@change
    R0=G-c[:,None]*(V@B);Rbar=R0-c[:,None]*(W@(W.T@R0))
    S=np.column_stack([V,W]);CS=c[:,None]*S
    # QR reduces a redundant operator span without dropping any directions.
    inverse_box=box if inverse_ranges is None else inverse_ranges
    reused=bool(inverse_cache and 'root' in inverse_cache)
    if reused:
        if inverse_cache['V'] is not V or inverse_cache['W'] is not W or not np.array_equal(inverse_cache['G'],G):raise ValueError('inverse cache/model mismatch')
        if inverse_ranges is not None and not np.array_equal(inverse_ranges,inverse_cache['inverse_box']):raise ValueError('inverse cache/reference box mismatch')
        inverse_box=inverse_cache['inverse_box']
        root=inverse_cache['root'];solver=inverse_cache['solver'];w=inverse_cache['w']
        negative=inverse_cache['negative'];span=inverse_cache['span'];inverse_rhs=inverse_cache['inverse_rhs']
        before=solver.counts()
    else:
        inverse_low=operator(K,H,inverse_box[:,0]);solver=Solver(inverse_low);before={key:0 for key in solver.counts()}
        w=solver.solve(c)
        if np.any(w<=0):raise ValueError('positive coercivity supersolution required')
        inverse_alpha=float(np.min((inverse_low@w)/(c*w)))
        if inverse_alpha<=0:raise ValueError('positive inverse coercivity lower required')
        # Use K0 in the span so the very same inverse controls every cell.
        M=np.column_stack([Rbar,CS,K@S,*[J@S for J in H]])
        basis,coordinate=la.qr(M,mode='economic');Y=solver.solve(basis);E=basis-inverse_low@Y
        Q=sym(basis.T@Y+Y.T@basis-Y.T@(inverse_low@Y)+(E.T@(E/c[:,None]))/inverse_alpha)
        ell,U=la.eigh(Q);negative=float(min(0.,ell[0]));root=(np.sqrt(np.maximum(ell,0))[:,None]*U.T)@coordinate
        span=M.shape[1];inverse_rhs=basis.shape[1]
        if inverse_cache is not None:
            inverse_cache.update(root=root,solver=solver,w=w,negative=negative,span=span,inverse_rhs=inverse_rhs,V=V,W=W,G=G.copy(),inverse_box=inverse_box.copy())
    if np.any(box[:,0]<inverse_box[:,0]):raise ValueError('inverse reference must be a Loewner lower operator')
    alpha=float(np.min((low@w)/(c*w)))
    if alpha<=0:raise ValueError('positive local coercivity lower required')
    p=r+k;root0=root[:,:m];rootCS=root[:,m:m+p];rootK0=root[:,m+p:m+2*p]
    rootHi=[root[:,m+(2+i)*p:m+(3+i)*p] for i in range(dim)]
    rootKS=rootK0+sum((h*a for h,a in zip(mid,rootHi)),np.zeros_like(rootK0))
    zero=(0,)*dim;one={zero:1.}
    delta=[{zero:-width[i]/2,tuple(int(j==i) for j in range(dim)):width[i]} for i in range(dim)]
    nodes,base=interpolation_controls(dim,degree,degree+1)
    coeff=np.stack([base,*[product_controls(dim,degree,degree+1,d) for d in delta]],axis=2)
    rates=[];vectors=[];inputs=[];Lnodes=[];arnodes=[];maxcondition=0.
    rhs=np.vstack([B,W.T@R0]);select=np.column_stack([np.eye(r),np.zeros((r,k))])
    for node in nodes:
        h=mid+(node-.5)*width;Ah=operator(K,H,h);ar=sym(V.T@(Ah@V));aw=sym(W.T@(Ah@W))
        D=Ah@V-c[:,None]*(V@ar)
        L=np.block([[ar,np.zeros((r,k))],[W.T@D,aw]])
        ell,T=la.eig(L)
        if np.max(np.abs(ell.imag))>1e-7*max(1.,np.max(np.abs(ell))) or ell.real.min()<=0:raise ValueError('invalid auxiliary modal spectrum')
        T=T.real;condition=float(np.linalg.cond(T));maxcondition=max(maxcondition,condition)
        if condition>1e10:raise ValueError('modal conditioning requires Schur implementation')
        rates.append(ell.real);vectors.append(T);inputs.append(la.solve(T,rhs));Lnodes.append(L);arnodes.append(ar)
    rates=np.asarray(rates);vectors=np.asarray(vectors);inputs=np.asarray(inputs)
    fullmaps=[np.array([rootCS@L-rootKS for L in Lnodes]),*[-a for a in rootHi]]
    full=StepResidual(fullmaps,coeff,vectors,rates,inputs,np.broadcast_to(root0,(len(coeff),*root0.shape)))
    arlow=sym(V.T@(low@V));arhigh=sym(V.T@(operator(K,H,box[:,1])@V));beta=float(la.eigvalsh(arlow)[0])
    chol=la.cholesky(arlow,lower=True)
    reducedmaps=[np.array([la.solve_triangular(chol,(a-Ar)@select,lower=True) for a in arnodes]),
                 *[-la.solve_triangular(chol,a@select,lower=True) for a in Ai]]
    reduced=StepResidual(reducedmaps,coeff,vectors,rates,inputs,np.zeros((len(coeff),r,m)))
    fulltrans=full.transient();redtrans=reduced.transient()
    response=TaylorModalEnvelope([select],base[:,:,None],vectors,rates,inputs)
    error=TaylorModalEnvelope([np.column_stack([np.zeros((k,r)),np.eye(k)])],base[:,:,None],vectors,rates,inputs)
    # A Bernstein convex combination bounds both finite integrals and stationary
    # defects. Infinite transient integrals are integrated, not constant defects.
    Dmid=Ac@V-c[:,None]*(V@Ar);Di=[J@V-c[:,None]*(V@a) for J,a in zip(H,Ai)]
    dmax=max(norm((Dmid+sum(((x-.5)*w*d for x,w,d in zip(v,width,Di)),np.zeros_like(Dmid)))/np.sqrt(c)[:,None]) for v in itertools.product([0.,1.],repeat=dim))
    tiny=min(.01/float(la.eigvalsh(arhigh)[-1]),.001/max(dmax,1e-300));end=35/rates.min()
    grid=np.r_[0.,np.geomspace(tiny,end,int(np.ceil(np.log(end/tiny)/np.log(time_ratio)))+1)]
    jf=np.zeros((len(coeff),m,m));jr=jf.copy();jft=jf.copy();jrt=jf.copy()
    rf=maxnorm(full.stationary)/np.sqrt(alpha);rr=maxnorm(reduced.stationary)/np.sqrt(beta)
    # First pass supplies all-time stationary/transient envelopes.
    full_segments=[];red_segments=[];response_segments=[];error_segments=[]
    for a,b in zip(grid[:-1],grid[1:]):
        fs=full.integral_matrix_upper(a,b);rs=reduced.integral_matrix_upper(a,b)
        full_segments.append(fs);red_segments.append(rs)
        jft+=fulltrans.integral_matrix_upper(a,b);jrt+=redtrans.integral_matrix_upper(a,b)
        response_segments.append(response.integral_controls(a,b));error_segments.append(error.integral_controls(a,b))
    # For the omitted infinite tail, exp(-lambda*T) weights /sqrt(2*lambda)
    # bound each control's L2 norm by the triangle inequality.
    def infinity_energy(env,j):
        tail=np.einsum('cnr,nr->c',env.weights,np.exp(-end*env.rates)/np.sqrt(2*env.rates),optimize=True)
        return maxgram(j)+float(np.max(tail))
    tf=rf+np.hypot(rf,infinity_energy(fulltrans,jft))
    tr=rr+np.hypot(rr,infinity_energy(redtrans,jrt))
    maximum=(tiny*dmax/2+norm(R0/np.sqrt(c)[:,None]))*np.exp(la.eigvalsh(arhigh)[-1]*tiny);worst=tiny;worst_parts={}
    # Exact ROM norm can be bounded from its polynomial trial without a global
    # reduced operator Lipschitz norm. Center below is a trial, not ground truth.
    center_weights=np.array([np.prod([math.comb(degree+1,j) for j in index])/2**(dim*(degree+1))
                             for index in itertools.product(range(degree+2),repeat=dim)])
    # Response/error controls share degree+1, same order as full/reduced residual.
    for i,(a,b) in enumerate(zip(grid[:-1],grid[1:])):
        jf+=full_segments[i];jr+=red_segments[i]
        if a==0:continue
        tailf=min(tf,maxgram(jf));tailr=min(tr,maxgram(jr))
        zcontrols=response.matrices(a);center=np.einsum('c,coi->oi',center_weights,zcontrols)
        deviation=maxnorm(zcontrols-center[None,:,:]);variation=float(np.max(response_segments[i]))
        denominator=max(0.,float(la.svdvals(center)[-1])-deviation-variation-tailr)
        rational=a*(B.T@la.solve(np.eye(r)+a*arhigh,B,assume_a='pos'))
        denominator=max(denominator,float(la.eigvalsh(sym(rational))[0]))
        main=float(np.max(error.values(a)))+float(np.max(error_segments[i]))
        value=(main+tailf+tailr)/denominator
        if value>maximum:maximum=value;worst=a;worst_parts=dict(auxiliary_error=main,full_tail=tailf,reduced_interpolation_tail=tailr,denominator=denominator)
    zcontrols=response.matrices(np.inf);center=np.einsum('c,coi->oi',center_weights,zcontrols)
    denominator=max(float(la.svdvals(center)[-1])-maxnorm(zcontrols-center[None,:,:])-response.tail(end)-tr,
                    float(la.eigvalsh(sym(end*(B.T@la.solve(np.eye(r)+end*arhigh,B,assume_a='pos'))))[0]))
    late=(error.limit()+error.tail(end)+tf+tr)/denominator
    if late>maximum:maximum=late;worst=end;worst_parts=dict(infinite_time_envelope=late,full_tail=tf,reduced_interpolation_tail=tr,denominator=denominator)
    # Steady K-orthogonality of the ORIGINAL V gives the relative conversion.
    awhigh=sym(W.T@(operator(K,H,box[:,1])@W));energyroot=la.cholesky(awhigh,lower=False)
    eta=error.matrices(np.inf)
    sn=maxnorm(np.einsum('ko,coi->cki',energyroot,eta))+maxnorm(full.stationary)+maxnorm(reduced.stationary)
    sd=np.sqrt(float(la.eigvalsh(sym(B.T@la.solve(arhigh,B,assume_a='pos')))[0]));sr=sn/sd
    return dict(n=len(c),primal_order=r,error_space_order=k,degree=degree,parameter_nodes=len(nodes),
                supplied_parameter_box=box.tolist(),error_over_rom_bound=float(maximum),
                bound=float(maximum/(1-maximum)) if maximum<1 else None,steady_bound=float(sr/np.hypot(1,sr)),
                full_stationary_C_bound=rf,reduced_stationary_C_bound=rr,
                full_global_tail=tf,reduced_global_tail=tr,worst_time=float(worst),worst_terms=worst_parts,
                counts={key:solver.counts()[key]-before[key] for key in before},inverse_operator_span=span,inverse_rhs_after_QR=inverse_rhs,
                inverse_parameter_box=inverse_box.tolist(),global_inverse_reused=reused,
                modal_condition=maxcondition,joint_dual_gram_negative_roundoff=negative,
                full_eigendecompositions=0,full_direct_factorizations=0,floating_point_certified=False,
                analytic_initial_interval_end=float(tiny),
                continuous_parameter_inequalities=True,all_time_inequalities=True,seconds=time.perf_counter()-start,
                passed=bool(maximum<=np.sqrt(.001)/(1+np.sqrt(.001)) and sr/np.hypot(1,sr)<=.001))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--mesh-mm',type=float,default=10.);p.add_argument('--rank',type=int,default=16)
    p.add_argument('--degree',type=int,default=2);p.add_argument('--widths',type=float,nargs='+',default=[.001,.01])
    p.add_argument('--promote-rank',type=int,default=0)
    p.add_argument('--log-radius',type=float,help='instead certify both axes within exp(+/- radius) of geometric center')
    p.add_argument('--global-inverse',action='store_true',help='one common inverse Gram for the original entire box, reused across supplied cells')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm);saved=np.load(args.archive)
    V=saved['V'];W=saved['W'][:,:args.rank];G=saved['G'] if 'G' in saved else G
    original_order=V.shape[1]
    if args.promote_rank:
        V=c_basis(np.column_stack([V,saved['W'][:,:args.promote_rank]]),C.diagonal())
        W=saved['W'][:,args.promote_rank:args.promote_rank+args.rank]
    report=dict(metadata=meta,method='common auxiliary trajectories / input matrix energy tail',original_order=original_order,promoted_error_directions=args.promote_rank,runs=[]);cache={}
    for width in ([None] if args.log_radius is not None else args.widths):
        center=np.sqrt(ranges[:,0]*ranges[:,1])
        box=None if width is None else np.column_stack([center-width*(center-ranges[:,0]),center+width*(ranges[:,1]-center)])
        if args.log_radius is not None:
            if args.log_radius<=0:raise ValueError('positive log radius required')
            box=np.column_stack([np.maximum(ranges[:,0],center*np.exp(-args.log_radius)),np.minimum(ranges[:,1],center*np.exp(args.log_radius))])
        row=certify_box(K,C,G,H,V,W,box,args.degree,inverse_ranges=ranges if args.global_inverse else None,inverse_cache=cache if args.global_inverse else None)
        row['width_fraction']=width;row['log_radius']=args.log_radius;report['runs'].append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('CONTINUOUS_AUXILIARY',len(V),args.rank,width,row['error_over_rom_bound'],row['steady_bound'],row['seconds'],flush=True)
