"""Train a COMMON actual-input error space; audit disjoint parameters/shifts.

Four-input resolvent residuals drive a separate auxiliary Ritz reconstruction.
Positive terminal solves bound every discarded direction at each audit point.
The audit points are NOT a continuous-domain certificate; no such claim is made.
"""
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from case1_system import case1_reconstruction
from numerics import Solver, operator, sym, invroot, relative, counts
from positive_source_tail import EnclosedSolve, opnorm


def run(a):
    start=time.perf_counter();data=np.load(a.candidate);V=data['V'];G=data['G'];ranges=data['ranges']
    meta=json.loads(a.candidate.with_suffix('.json').read_text())
    K,C,_,H,_,model=case1_reconstruction(a.mesh_mm);c=C.diagonal();sc=np.sqrt(c)
    B=V.T@G
    low=Solver(operator(K,H,ranges[:,0]),rtol=2e-13)
    w=low.solve(c)
    terminal=EnclosedSolve(low.A,w)
    cost=[low.counts()];traincost=[];auditcost=[];blocks=[]
    shifts=np.geomspace(meta['alpha'],meta['input_fast_rate'],7)
    def hpoint(p):return np.exp(np.log(ranges[:,0])+np.log(ranges[:,1]/ranges[:,0])*p)
    for p in itertools.product([0.,1.],repeat=2):
        h=hpoint(p);Kh=operator(K,H,h);Ar=sym(V.T@(Kh@V))
        for s in shifts[[1,3,5]]:
            solver=EnclosedSolve(Kh+s*C,w);X,_,correction=solver.solve(G)
            XR=V@la.solve(Ar+s*np.eye(Ar.shape[0]),B,assume_a='pos')
            E=sc[:,None]*(X-XR)
            blocks.append(E@invroot((sc[:,None]*X).T@(sc[:,None]*X)))
            traincost.append(solver.solver.counts())
    Q,sv,_=la.svd(np.column_stack(blocks),full_matrices=False)
    ranks=[q for q in [8,16,32] if q<=Q.shape[1]]
    Us={r:Q[:,:r]/sc[:,None] for r in ranks}
    rows=[]
    # Disjoint parameter nodes AND disjoint real shifts; selection never certifies the continuum.
    tests=np.sqrt(shifts[[0,2,4]]*shifts[[1,3,5]])
    for p in itertools.product([.25,.75],repeat=2):
        h=hpoint(p);Kh=operator(K,H,h);Ar=sym(V.T@(Kh@V))
        for s in tests:
            M=Kh+s*C;solver=EnclosedSolve(M,w);X,_,correction=solver.solve(G)
            XR=V@la.solve(Ar+s*np.eye(Ar.shape[0]),B,assume_a='pos')
            actual=X-XR;residual=G-M@XR
            den=la.svdvals(sc[:,None]*X)[-1];uncertainty=opnorm(correction,c)/den
            item=dict(h=h.tolist(),shift=float(s),original_relative=relative(actual,X,C),ranks=[])
            for r,U in Us.items():
                Au=sym(U.T@(M@U));rhs=U.T@residual
                proxy=U@la.solve(Au,rhs,assume_a='pos')
                # Exact full error after auxiliary reconstruction equals M^-1 remainder.
                remainder=G-M@(XR+proxy)
                _,envelope,tailcorrection=terminal.solve(np.abs(remainder))
                q=relative(actual-proxy,X,C)
                certified=opnorm(envelope,c)/(den*(1-uncertainty)) if uncertainty<1 else None
                item['ranks'].append(dict(rank=r,
                    sampled_reconstruction_relative=q,
                    reference_defect_relative=uncertainty,
                    true_relative_upper=(q+uncertainty)/(1-uncertainty) if uncertainty<1 else None,
                    positive_terminal_relative_upper=certified,
                    best_projection_relative=relative(actual-U@(U.T@(c[:,None]*actual)),X,C),
                    terminal_defect_C=opnorm(tailcorrection,c)))
            rows.append(item);auditcost.append(solver.solver.counts())
    energy=np.cumsum(sv*sv)/np.sum(sv*sv)
    result=dict(n=len(c),V_order=V.shape[1],metadata=model,train_h_count=4,
        train_shift_count=3,audit_h_count=4,audit_shift_count=3,
        source_error_singular_values=sv.tolist(),
        rank_for_99_percent_training_energy=int(np.searchsorted(energy,.99)+1),
        rank_for_9999_percent_training_energy=int(np.searchsorted(energy,.9999)+1),
        rows=rows,costs=dict(preparation=counts(cost),training=counts(traincost),
            independent_references=counts(auditcost),deterministic_point_tails=terminal.solver.counts()),
        seconds=time.perf_counter()-start,
        scope='COMMON input-driven error space; exact residual auxiliary Ritz proxy; deterministic discarded-direction tails at named real-shift/HTC audit points ONLY. Not a continuum or all-time step certificate. Ordinary float; actual CG defects; no outward rounding.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','source_error_singular_values']}),flush=True)
    for r in ranks:
        rr=[x for row in rows for x in row['ranks'] if x['rank']==r]
        print(json.dumps(dict(rank=r,worst_actual_upper=max(x['true_relative_upper'] for x in rr),
            worst_terminal_upper=max(x['positive_terminal_relative_upper'] for x in rr))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,default=10)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
