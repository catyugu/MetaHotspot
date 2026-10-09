"""Certify the uncompressed snapshot space at epsilon / sqrt(epsilon).

Tolerance never selects a rank here. Pivoted QR removes only numerical linear
 dependencies; this is not tolerance-based SVD compression. Historical final-SVD
experiments remain separate. Generated data stays in ignored results directories.
"""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import scipy.linalg as la
from rational_dynamic import AffineSystem
from run_rational_dynamic import case1_reconstruction, Operators, stock_utils, build_parametric_basis
from steady_step_audit import steady_cell_bound, _change, _norm
from affine_step_bridge import continuous_step_bridge


def certification_targets(epsilon):
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    return float(epsilon),float(np.sqrt(epsilon))


def uncompressed_snapshot_basis(snapshots, include_constant=True):
    S=np.asarray(snapshots,dtype=float)
    norms=la.norm(S,axis=0)
    if np.any(norms == 0):
        raise ValueError('zero snapshot column')
    S=S/norms
    if include_constant:
        S=np.column_stack([S,np.ones(S.shape[0])/np.sqrt(S.shape[0])])
    Q,R,_=la.qr(S,mode='economic',pivoting=True)
    diagonal=np.abs(np.diag(R))
    rank=int(np.sum(diagonal>np.finfo(float).eps*max(S.shape)*diagonal[0]))
    return Q[:,:rank]


def reference_check(K,C,G,H,V,low,high):
    import itertools
    points=[*itertools.product(*zip(low,high)),
            *np.random.default_rng(20261007).uniform(low,high,(12,len(low)))]
    worst=0.
    for h in points:
        A=K+sum(x*J for x,J in zip(h,H))
        rates,U=la.eigh(A,C);B=U.T@G
        rr,W=la.eigh(V.T@A@V,V.T@C@V);W=V@W
        Br=W.T@G;D=U.T@C@W
        for t in np.geomspace(1e-10/rates[-1],100/rates[0],120):
            F=(-np.expm1(-rates*t)/rates)[:,None]*B
            R=(-np.expm1(-rr*t)/rr)[:,None]*Br
            worst=max(worst,_norm((F-D@R)@_change(F.T@F)))
    return {'sampled_maximum':worst,'parameter_points':len(points),'times_per_point':120,
            'scope':'independent validation only; not continuous acceptance'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=20260805)
    p.add_argument('--epsilon',type=float,default=.001)
    p.add_argument('--mesh-mm',type=float,default=10.)
    p.add_argument('--fraction',type=float,default=.0002)
    p.add_argument('--scales',type=float,nargs='+',default=[1.,.25,.1])
    p.add_argument('--validate',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();steady_target,step_target=certification_targets(args.epsilon)
    K,C,G,H,ranges,metadata=case1_reconstruction(args.mesh_mm)
    snapshots=[]; original=stock_utils._snapshot_svd_basis
    def capture(A,tol):
        snapshots.append(A.copy());return original(A,tol)
    started=time.perf_counter()
    with patch.object(stock_utils,'_snapshot_svd_basis',capture):
        stock,stats=build_parametric_basis(Operators(K,C,np.zeros(K.shape[0])),G,H,ranges,
                                           tolerance=args.epsilon,seed=args.seed)
    V=uncompressed_snapshot_basis(snapshots[0])
    extraction_seconds=time.perf_counter()-started
    center=np.sqrt(ranges[:,0]*ranges[:,1])
    low=center-args.fraction*(center-ranges[:,0]); high=center+args.fraction*(ranges[:,1]-center)
    midpoint=(low+high)/2
    system=AffineSystem(K,C,G,H,V)
    report={'seed':args.seed,'metadata':metadata,'epsilon':args.epsilon,
        'basis_stage':'pre-SVD snapshot span; pivoted QR only',
        'raw_snapshot_columns':snapshots[0].shape[1],'raw_order':V.shape[1],
        'stock_compressed_order_for_context_only':stock.shape[1],
        'steady_threshold':steady_target,'step_threshold':step_target,
        'extraction_rhs':sum(row['full_solves'] for row in stats['history']),
        'extraction_seconds':extraction_seconds,'experiments':[],
        'post_svd_certified':False,'floating_point_certified':False}
    started=time.perf_counter()
    for scale in args.scales:
        lo=midpoint+scale*(low-midpoint);hi=midpoint+scale*(high-midpoint)
        steady=steady_cell_bound(system,lo,hi)
        step=continuous_step_bridge(K,C,G,H,V,lo,hi,center_threshold=.1*step_target)
        accepted=steady['bound']<=steady_target and step['bound']<=step_target
        result={'scale':scale,'steady':steady,'step':step,'accepted_analytic':bool(accepted)}
        if args.validate and accepted:
            result['independent_reference']=reference_check(K.toarray(),C.toarray(),G,
                                      [J.toarray() for J in H],V,lo,hi)
        report['experiments'].append(result)
        print(args.seed,scale,V.shape[1],steady['bound'],step['bound'],accepted,flush=True)
    report['certification_and_validation_seconds']=time.perf_counter()-started
    report['steady_riesz_rhs_total']=sum(r['steady']['counts']['riesz_rhs'] for r in report['experiments'])
    report['step_full_generalized_eigendecompositions_total']=2*len(report['experiments'])
    report['step_full_corner_norms_total']=2**len(low)*len(report['experiments'])
    report['all_requested_scales_accepted']=all(r['accepted_analytic'] for r in report['experiments'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez(args.output.with_suffix('.npz'),raw_basis=V,K=K.toarray(),C=C.toarray(),G=G,
             H=np.array([J.toarray() for J in H]),ranges=ranges,epsilon=args.epsilon,low=low,high=high)

if __name__=='__main__':
    main()
