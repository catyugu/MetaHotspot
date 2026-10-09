"""Positive source-seeded Dyson tail. No ROM acceptance is inferred.

The terminal solve bounds a continuous parameter cell and every t>=0.
All solves use AMG-CG; actual defects are enclosed with a positive supersolution.
Run from the repository root; see POSITIVE_SOURCE_TAIL_PROOF.md.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq
from scipy.special import gammainc
from case1_system import case1_reconstruction
from numerics import Solver, operator, counts


def opnorm(X, c):
    return float(la.norm(np.sqrt(c)[:, None]*X, 2))


class EnclosedSolve:
    def __init__(self, A, w):
        self.solver = Solver(A, rtol=2e-13)
        self.w = w
        self.aw = np.asarray(A@w)
        if np.min(w) <= 0 or np.min(self.aw) <= 0:
            raise ValueError('positive supersolution check failed')
        self.max_correction = 0.

    def solve(self, B):
        Y = self.solver.solve(B)
        F = B-self.solver.A@Y
        eta = np.max(np.abs(F)/self.aw[:, None], axis=0)
        correction = self.w[:, None]*eta
        self.max_correction = max(self.max_correction, float(correction.max()))
        return Y, np.maximum(Y+correction, 0.), correction


def cell(lo, hi, K, C, G, H, w, orders, shifts):
    c = C.diagonal()
    delta = sum((b-a)*J.diagonal() for a,b,J in zip(lo, hi, H))
    dmax=float(np.max(delta/c))
    gfield=G/c[:,None]
    input_rate=float(la.eigvalsh(gfield.T@(operator(K,H,hi)@gfield))[-1])
    rows, costs = [], []
    for s in shifts:
        low = EnclosedSolve(operator(K,H,lo)+s*C, w)
        high = EnclosedSolve(operator(K,H,hi)+s*C, w)
        Xlo, Xlo_up, _ = low.solve(G)
        Xhi, S, correction = high.solve(G)
        partial = Xhi.copy()
        # A verified componentwise Collatz contraction, NOT a Ritz estimate.
        _, Tw_up, _ = high.solve((delta*w)[:,None])
        rho = float(np.max(Tw_up[:,0]/w))
        for j in range(max(orders)+1):
            if j in orders:
                raw_tail, tail_up, corr = low.solve(delta[:,None]*S)
                amplitude = opnorm(tail_up, c)
                all_time_relative=None;crossing=None
                if s==0:
                    if amplitude==0 or dmax==0:
                        all_time_relative=0.;crossing=0.
                    else:
                        target=amplitude*dmax
                        bound=max(1.,2*target+2*(j+2))
                        v=brentq(lambda v:v*gammainc(j+1,v)-target,0.,bound,xtol=1e-14)
                        crossing=v/dmax
                        lower=-np.expm1(-input_rate*crossing)/input_rate
                        all_time_relative=amplitude/lower
                exact_difference = Xlo-partial
                # Both are source-response upper bounds, not relative step bounds.
                geometric = None
                if rho < 1:
                    coeff = np.max(S/w[:,None],axis=0)*rho/(1-rho)
                    geometric = opnorm(w[:,None]*coeff,c)
                gram_lower = 0.5*(G.T@Xhi+Xhi.T@G)
                # correction is from the INITIAL high solve (kept separately).
                gram_lower -= np.eye(G.shape[1])*la.norm(G.T@correction,2)
                denom = float(la.eigvalsh(gram_lower)[0])
                rows.append(dict(shift=float(s),degree=j,
                    terminal_tail_C=amplitude,
                    uniform_all_time_relative_parameter_tail_upper=all_time_relative,
                    near_zero_to_steady_crossing_time=crossing,
                    delta_max_rate=dmax,input_rate_upper=input_rate,
                    subtraction_audit_C=opnorm(exact_difference,c),
                    lower_endpoint_response_C=opnorm(Xlo,c),
                    tail_fraction_of_low_response=amplitude/opnorm(Xlo,c),
                    uniform_steady_C_lower=max(0.,denom),
                    uniform_relative_steady_tail_upper=amplitude/denom if denom>0 else None,
                    scalar_geometric_tail_C=geometric,collatz_rho=rho,
                    source_tail_vs_scalar=(geometric/amplitude if geometric is not None and amplitude>0 else None),
                    terminal_solve_correction_C=opnorm(corr,c),
                    min_tail_entry=float(tail_up.min())))
            if j<max(orders):
                raw, S, _ = high.solve(delta[:,None]*S)
                partial += raw
        costs.extend([low.solver.counts(),high.solver.counts()])
    return dict(lo=lo.tolist(),hi=hi.tolist(),rows=rows),costs


def run(a):
    start=time.perf_counter()
    K,C,G,H,ranges,meta=case1_reconstruction(a.mesh_mm)
    c=C.diagonal()
    # Only positive diagonal rescaling: preserve input cone and arbitrary signed span.
    G=G/np.sqrt(np.sum(G*G/c[:,None],axis=0))[None,:]
    off=K.copy();off.setdiag(0);off.eliminate_zeros()
    if off.nnz and off.data.max()>0:raise ValueError('not an M-matrix thermal graph')
    if np.min(G)<0:raise ValueError('negative source')
    prep=Solver(operator(K,H,ranges[:,0]),rtol=2e-13)
    w=prep.solve(c)
    aw=prep.A@w
    if np.min(w)<=0 or np.min(aw)<=0:raise ValueError('supersolution failure')
    alpha=float(np.min(aw/(c*w)))
    center=np.sqrt(ranges[:,0]*ranges[:,1])
    cells=[]
    if a.cover_factor:
        axes=[np.geomspace(x,y,int(np.ceil(np.log(y/x)/np.log(a.cover_factor)))+1) for x,y in ranges]
        for i,j in itertools.product(range(len(axes[0])-1),range(len(axes[1])-1)):
            cells.append((np.array([axes[0][i],axes[1][j]]),np.array([axes[0][i+1],axes[1][j+1]])))
    else:
        for width in a.widths:
            cells.append((ranges[:,0],ranges[:,1]) if width<0 else
                (np.maximum(ranges[:,0],center*np.exp(-width)),np.minimum(ranges[:,1],center*np.exp(width))))
    rows=[];cost=[prep.counts()]
    for i,(lo,hi) in enumerate(cells):
        row,cc=cell(lo,hi,K,C,G,H,w,a.orders,a.shifts)
        rows.append(row);cost.extend(cc)
        print(json.dumps(dict(cell=i+1,total_cells=len(cells),n=len(c),
            worst_tail_fraction=max(x['tail_fraction_of_low_response'] for x in row['rows']),
            last_degree=row['rows'][-1])),flush=True)
    out=dict(n=len(c),metadata=meta,alpha=alpha,ranges=ranges.tolist(),orders=a.orders,
        cells=rows,costs=counts(cost),seconds=time.perf_counter()-start,
        cover_factor=a.cover_factor,
        floating_point='ordinary float; explicit CG defects; no outward rounding',
        scope='s=0: continuous-cell all-time absolute AND Poisson/Jensen relative C-norm PARAMETER-DYSON tail, arbitrary signed constant sources. s>=0: continuous-cell shifted resolvent tail. Does NOT certify omitted spatial directions or original Galerkin step error.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cells'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mesh-mm',type=float,default=10)
    p.add_argument('--orders',type=int,nargs='+',default=[0,1,2,4,8])
    p.add_argument('--widths',type=float,nargs='+',default=[.01,.1,.5,-1])
    p.add_argument('--shifts',type=float,nargs='+',default=[0,1,100])
    p.add_argument('--cover-factor',type=float,default=None)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
