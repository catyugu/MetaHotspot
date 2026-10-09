"""User target: steady K-energy <=2*epsilon; step C-energy <=2*sqrt(epsilon).

Steady Bernstein certificate covers a continuous HTC cell. Step certificate
covers every t>=0 at each audited HTC point, with analytic initial/tail bounds
and derivative enclosures between endpoints. It does NOT cover unsampled HTC.
All statements assume exact arithmetic; dense eigensolves have no rounding guard.
"""
import argparse
import itertools
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from rational_dynamic import AffineSystem, polynomial_controls, _axis_transform
from run_rational_dynamic import case1_reconstruction, Operators, stock_utils, build_parametric_basis
from svd_dynamic_guard import svd_candidate_coordinates


def _dense(A):
    return A.toarray() if sp.issparse(A) else np.asarray(A)


def _change(Q):
    return la.solve_triangular(la.cholesky(Q, lower=True).T, np.eye(len(Q)), lower=False)


def _norm(A):
    return float(la.svdvals(A)[0])


def steady_cell_bound(system, low, high, *, return_witness=False, factor=None):
    d, V, G = len(low), system.V, system.G
    shape = (2,)*d
    q = np.empty((*shape, V.shape[1], G.shape[1]))
    for index in np.ndindex(shape):
        h = np.where(index, high, low)
        q[index] = la.solve(V.T@(system.operator(h)@V), V.T@G, assume_a='pos')
    for axis in range(d):
        q = _axis_transform(q, np.array([[1., 0.], [-1., 1.]]), axis)
    residual = np.zeros((*((3,)*d), G.shape[0], G.shape[1]))
    residual[(0,)*d] = G
    Ka = system.operator(low)
    for index in np.ndindex(shape):
        x = V@q[index]
        residual[index] -= Ka@x
        for axis, H in enumerate(system.H):
            nxt = list(index); nxt[axis] += 1
            residual[tuple(nxt)] -= (high[axis]-low[axis])*(H@x)
    controls = polynomial_controls(residual, d).reshape(-1, G.shape[0], G.shape[1])
    new_factor = factor is None
    if new_factor:
        factor = sla.splu(sp.csc_matrix(Ka))
    rhs = controls.transpose(1, 0, 2).reshape(G.shape[0], -1)
    actions = factor.solve(rhs).reshape(G.shape[0], -1, G.shape[1]).transpose(1, 0, 2)
    Qlow = G.T@V@la.solve(V.T@(system.operator(high)@V), V.T@G, assume_a='pos')
    change = _change((Qlow+Qlow.T)/2)
    grams = np.einsum('bni,bnj->bij', controls@change, actions@change)
    bound = np.sqrt(max(0., np.linalg.eigvalsh((grams+grams.swapaxes(-1,-2))/2)[:,-1].max()))
    result = {'bound': float(bound), 'counts': {'riesz_factorizations':int(new_factor),
        'riesz_rhs':int(rhs.shape[1]), 'reduced_node_solves':int(2**d)},
        'scope':'all input combinations; continuous HTC cell; steady K-energy',
        'floating_point_certified':False}
    if return_witness:
        values, vectors = np.linalg.eigh((grams+grams.swapaxes(-1,-2))/2)
        worst = int(np.argmax(values[:,-1]))
        # This Riesz action has already been paid for by the certificate.
        direction = actions[worst] @ change @ vectors[worst,:,-1]
        return result, {'direction':direction, 'control_index':worst}
    return result


def step_all_time_bound(K, C, G, V, threshold, *, full_spectrum=None):
    K, C = _dense(K), _dense(C)
    rates, U = la.eigh(K, C) if full_spectrum is None else full_spectrum
    rr, Ur = la.eigh(V.T@K@V, V.T@C@V)
    W = V@Ur
    B, Br, D = U.T@G, W.T@G, U.T@C@W
    def state(t):
        f = (-np.expm1(-rates*t)/rates)[:,None]*B
        r = (-np.expm1(-rr*t)/rr)[:,None]*Br
        return f, r, f-D@r
    def result(bound, accepted, intervals, reason):
        return {'bound':float(bound), 'accepted_analytic':bool(accepted),
            'intervals':intervals, 'reason':reason, 'full_spectral_eigendecompositions':int(full_spectrum is None),
            'scope':'every t>=0, one fixed HTC point, all input combinations, C-energy',
            'floating_point_certified':False}
    Z = _change(B.T@B)
    initial = _norm((B-D@Br)@Z)
    Fs, Rs = B/rates[:,None], Br/rr[:,None]
    Zs = _change(Fs.T@Fs)
    steady = _norm((Fs-D@Rs)@Zs)
    if max(initial, steady) > threshold:
        return result(max(initial, steady), False, 0, 'initial or infinite-time lower witness')
    t0 = 1e-5/max(rates[-1], rr[-1])
    T = 25/min(rates[0], rr[0])
    F0, _, _ = state(t0)
    Z0 = _change((F0/t0).T@(F0/t0))
    initial_bound = _norm((B-D@Br)@Z0)+t0*.5*(la.norm((rates[:,None]*B)@Z0)+la.norm((rr[:,None]*Br)@Z0))
    FT, _, _ = state(T); ZT = _change(FT.T@FT)
    tail = _norm((Fs-D@Rs)@ZT)+np.exp(-rates[0]*T)*la.norm(Fs@ZT)+np.exp(-rr[0]*T)*la.norm(Rs@ZT)
    if max(initial_bound, tail) > threshold:
        return result(max(initial_bound, tail), False, 0, 'initial/tail enclosure unresolved')
    edges = np.geomspace(t0, T, 129)
    pending = list(zip(edges[:-1], edges[1:]))
    worst, checked, accepted_intervals = max(initial_bound, tail), 0, 0
    while pending:
        a,b = pending.pop(); mid = .5*(a+b)
        Fa,_,_ = state(a); Z = _change(Fa.T@Fa)
        Fm,_,Em = state(mid)
        value = _norm(Em@Z)
        derivative = la.norm((np.exp(-rates*a)[:,None]*B)@Z)+la.norm((np.exp(-rr*a)[:,None]*Br)@Z)
        upper = value+.5*(b-a)*derivative
        checked += 1
        if upper <= threshold:
            worst = max(worst, upper); accepted_intervals += 1
        elif _norm(Em@_change(Fm.T@Fm)) > threshold:
            return result(max(worst, upper), False, checked, 'interior-time lower witness')
        elif checked >= 50000:
            return result(max(worst, upper), False, checked, 'diagnostic enclosure budget exhausted')
        else:
            pending.extend([(a,mid),(mid,b)])
    return result(worst, True, accepted_intervals, 'analytic initial, interval derivative and infinite tail bounds')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=20260805)
    p.add_argument('--mesh-mm', type=float, default=10.)
    p.add_argument('--epsilon', type=float, default=.001)
    p.add_argument('--fraction', type=float, default=.0002)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stock-sweep', action='store_true')
    args=p.parse_args()
    if args.stock_sweep:
        stock_sweep(args)
        return
    K,C,G,H,ranges,metadata=case1_reconstruction(args.mesh_mm)
    snapshots=[]; original=stock_utils._snapshot_svd_basis
    def capture(A,tol):
        snapshots.append(A.copy()); return original(A,tol)
    with patch.object(stock_utils,'_snapshot_svd_basis',capture):
        stock,stats=build_parametric_basis(Operators(K,C,np.zeros(K.shape[0])),G,H,ranges,tolerance=args.epsilon,seed=args.seed)
    S=snapshots[0]/la.norm(snapshots[0],axis=0)
    U,s,_=la.svd(S,full_matrices=False)
    aug=np.column_stack([S,np.ones(K.shape[0])/np.sqrt(K.shape[0])])
    W,sw,_=la.svd(aug,full_matrices=False)
    W=W[:,sw>np.finfo(float).eps*max(aug.shape)*sw[0]]
    center=np.sqrt(ranges[:,0]*ranges[:,1]);low=center-args.fraction*(center-ranges[:,0]);high=center+args.fraction*(ranges[:,1]-center)
    A=K+sum(h*term for h,term in zip(center,H)); factor=sla.splu(A.tocsc()); X=factor.solve(G)
    reference_Q=G.T@X
    def steady_point(V):
        E=X-V@la.solve(V.T@(A@V),V.T@G,assume_a='pos')
        return float(np.sqrt(max(0.,la.eigvalsh(E.T@(A@E),reference_Q)[-1])))
    report={'seed':args.seed,'metadata':metadata,'epsilon':args.epsilon,
        'steady_threshold':2*args.epsilon,'step_threshold':2*np.sqrt(args.epsilon),
        'extraction_rhs':sum(row['full_solves'] for row in stats['history']),
        'extraction_stats':stats,'stock_order':stock.shape[1],'raw_order':W.shape[1],
        'stock_steady_center':steady_point(stock),'rank_checks':[],
        'selection_full_rhs':G.shape[1], 'certificate_full_rhs':0,
        'low':low.tolist(),'high':high.tolist(),'step_certified_continuous_htc':False,
        'floating_point_certified':False}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    started=time.perf_counter()
    kept0=int((s>=args.epsilon*s[0]).sum())
    for kept in range(kept0,len(s)+1):
        V=W@svd_candidate_coordinates(U,W,kept)
        error=steady_point(V)
        entry={'kept':kept,'order':V.shape[1],'steady_center':error}
        report['rank_checks'].append(entry)
        if error>report['steady_threshold']:
            continue
        system=AffineSystem(K,C,G,H,V)
        cert=steady_cell_bound(system,low,high)
        entry['steady_cell_certificate']=cert
        report['certificate_full_rhs']+=cert['counts']['riesz_rhs']
        if cert['bound']>report['steady_threshold']:
            continue
        report['selected_order']=V.shape[1]
        points=[low,center,high,*np.random.default_rng(20261008).uniform(low,high,(4,len(low)))]
        report['step_point_certificates']=[{'h':h.tolist(),**step_all_time_bound(system.operator(h),C,G,V,report['step_threshold'])} for h in points]
        report['all_audited_step_points_accepted']=all(row['accepted_analytic'] for row in report['step_point_certificates'])
        report['stock_step_center']=step_all_time_bound(A,C,G,stock,report['step_threshold'])
        report['total_inverse_rhs']=report['extraction_rhs']+report['selection_full_rhs']+report['certificate_full_rhs']
        report['full_dense_spectral_eigendecompositions']=len(points)+1
        report['seconds_selection_and_audit']=time.perf_counter()-started
        np.savez(args.output.with_suffix('.npz'),final_basis=V,K=K.toarray(),C=C.toarray(),G=G,H=np.array([term.toarray() for term in H]),ranges=ranges,epsilon=args.epsilon,low=low,high=high)
        save();print(args.seed,report['selected_order'],cert['bound'],report['total_inverse_rhs'],report['all_audited_step_points_accepted'],flush=True)
        return
    report['reason']='no continuous steady certificate passed';save()


def stock_sweep(args):
    """Preset tighter-tolerance comparison, counting rejected extraction runs too."""
    K,C,G,H,ranges,metadata=case1_reconstruction(args.mesh_mm)
    center=np.sqrt(ranges[:,0]*ranges[:,1])
    low=center-args.fraction*(center-ranges[:,0]);high=center+args.fraction*(ranges[:,1]-center)
    A=K+sum(h*term for h,term in zip(center,H));X=sla.splu(A.tocsc()).solve(G);Q=G.T@X
    report={'epsilon':args.epsilon,'steady_threshold':2*args.epsilon,
        'step_threshold':2*np.sqrt(args.epsilon),'common_reference_rhs':G.shape[1],
        'metadata':metadata,'experiments':[],'step_certified_continuous_htc':False,
        'scope':'preset tighter tolerances, first passing run per seed; failed runs retained',
        'floating_point_certified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for seed in [20260805,20260806,20261007]:
        for ratio in [.5,.2,.1,.05]:
            tol=args.epsilon*ratio
            V,stats=build_parametric_basis(Operators(K,C,np.zeros(K.shape[0])),G,H,ranges,tolerance=tol,seed=seed)
            E=X-V@la.solve(V.T@(A@V),V.T@G,assume_a='pos')
            err=float(np.sqrt(max(0.,la.eigvalsh(E.T@(A@E),Q)[-1])))
            row={'seed':seed,'internal_tolerance':tol,'basis_order':V.shape[1],
                'extraction_rhs':sum(r['full_solves'] for r in stats['history']),
                'stats':stats,'steady_center':err,'center_accepted':bool(err<=2*args.epsilon)}
            if row['center_accepted']:
                system=AffineSystem(K,C,G,H,V)
                cert=steady_cell_bound(system,low,high)
                row['steady_cell_certificate']=cert
                row['steady_cell_accepted']=bool(cert['bound']<=2*args.epsilon)
                if row['steady_cell_accepted']:
                    points=[low,center,high,*np.random.default_rng(20261008).uniform(low,high,(4,len(low)))]
                    row['step_point_certificates']=[{'h':h.tolist(),**step_all_time_bound(system.operator(h),C,G,V,report['step_threshold'])} for h in points]
                    row['all_audited_step_points_accepted']=all(r['accepted_analytic'] for r in row['step_point_certificates'])
                    row['full_dense_spectral_eigendecompositions']=len(points)
            report['experiments'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(seed,tol,row['extraction_rhs'],V.shape[1],err,flush=True)
            if row.get('steady_cell_accepted') and row.get('all_audited_step_points_accepted'):
                break


if __name__=='__main__':
    main()
