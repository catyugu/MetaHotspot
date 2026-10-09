"""Same-operator, same-domain end-to-end comparison with current random extractor.

Run with one BLAS thread. Original random solver is unchanged. Optional matched
sparse-direct random baseline separates solver effects from extraction decisions.
Generated operator archives and raw JSON stay in ignored results/.
"""
import argparse
import itertools
import json
import platform
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from certified_extraction import extract_certified_rom, ExtractedROM, _Audit, _compress, _json_safe
from pre_svd_audit import uncompressed_snapshot_basis
from steady_step_audit import _change, _norm
from run_rational_dynamic import case1_reconstruction, synthetic, Operators, stock_utils, build_parametric_basis


def model(name):
    if name.startswith('case1'):
        K,C,G,H,ranges,metadata=case1_reconstruction(8. if 'mesh8' in name else 10.)
    else:
        K,C,G,H,ranges,metadata=synthetic()
    if name.endswith('local'):
        center=np.sqrt(ranges[:,0]*ranges[:,1])
        low=center-.0002*(center-ranges[:,0]);high=center+.0002*(ranges[:,1]-center)
        mid=(low+high)/2;ranges=np.column_stack([mid+.25*(low-mid),mid+.25*(high-mid)])
    return K,C,G,H,ranges,metadata


def project(K,C,G,H,V,report):
    return ExtractedROM(V,V.T@(K@V),V.T@(C@V),V.T@G,[V.T@(J@V) for J in H],report)


def reference_metrics(K,C,G,H,ranges,bases,parameter_seed=20261008):
    K,C=K.toarray(),C.toarray();H=[J.toarray() for J in H]
    points=[*itertools.product(*ranges),np.mean(ranges,axis=1),
             *np.random.default_rng(parameter_seed).uniform(ranges[:,0],ranges[:,1],(12,len(H)))]
    results={name:dict(steady_K_relative=0.,steady_C_relative=0.,step_C_relative=0.,
             nodal_step_steady_normalized_relative=0.,min_reduced_decay_rate=float('inf')) for name in bases}
    for h in points:
        A=K+sum(x*J for x,J in zip(h,H));rates,U=la.eigh(A,C);B=U.T@G
        Fs=B/rates[:,None];Xs=U@Fs;Qk=G.T@Xs;Qc=Fs.T@Fs
        times=np.geomspace(1e-9/rates[-1],40/rates[0],120)
        reduced={}
        for name,V in bases.items():
            rr,W=la.eigh(V.T@A@V,V.T@C@V);W=V@W
            Br=W.T@G;D=U.T@C@W;Xr=W@(Br/rr[:,None]);E=Xs-Xr
            r=results[name]
            r['steady_K_relative']=max(r['steady_K_relative'],float(np.sqrt(max(0.,la.eigvalsh(E.T@A@E,Qk)[-1]))))
            r['steady_C_relative']=max(r['steady_C_relative'],_norm((Fs-D@(Br/rr[:,None]))@_change(Qc)))
            r['min_reduced_decay_rate']=min(r['min_reduced_decay_rate'],float(rr[0]))
            reduced[name]=(rr,W,Br,D)
        nodal_den=np.maximum(np.abs(Xs),1e-12*np.max(np.abs(Xs),axis=0,keepdims=True))
        for t in times:
            F=(-np.expm1(-rates*t)/rates)[:,None]*B;Z=_change(F.T@F)
            for name,(rr,W,Br,D) in reduced.items():
                R=(-np.expm1(-rr*t)/rr)[:,None]*Br;E=F-D@R
                results[name]['step_C_relative']=max(results[name]['step_C_relative'],_norm(E@Z))
                # Positive-source thermal example, per-node/per-port normalization.
                value=float(np.max(np.abs(U@E)/nodal_den))
                results[name]['nodal_step_steady_normalized_relative']=max(results[name]['nodal_step_steady_normalized_relative'],value)
    for r in results.values():
        r['parameter_points']=len(points);r['times_per_point']=120
        r['scope']='shared independent samples; all-input energy metrics; nodal metric per source'
    return results


def hankel_metric(K,C,G,H,V,h):
    """Standard collocated port Hankel operator norm, at one parameter only.

    P is the joint controllability Gramian. Hankel error is self-adjoint;
    its nonzero eigenvalues equal those of sqrt(P) diag(I,-I) sqrt(P).
    This is a port metric, not used for state-field acceptance.
    """
    A=(K+sum(x*J for x,J in zip(h,H))).toarray();C=C.toarray()
    rates,U=la.eigh(A,C);B=U.T@G
    rr,W=la.eigh(V.T@A@V,V.T@C@V);W=V@W;Br=W.T@G
    q=np.r_[rates,rr];E=np.vstack([B,Br]);P=(E@E.T)/(q[:,None]+q[None,:])
    lam,Z=la.eigh((P+P.T)/2);root=(Z*np.sqrt(np.maximum(lam,0.)))@Z.T
    signs=np.r_[np.ones(len(rates)),-np.ones(len(rr))]
    error=la.eigvalsh((root*signs[None,:])@root)
    reference=la.eigvalsh((B@B.T)/(rates[:,None]+rates[None,:]))[-1]
    return float(np.max(np.abs(error))/reference)


def online_metrics(K,C,G,H,rom,ranges,repeats=5):
    h=np.mean(ranges,axis=1);u=np.ones(G.shape[1]);times=np.geomspace(1e-6,1e5,120)
    A=K+sum(x*J for x,J in zip(h,H));full_steady=[];reduced_steady=[];full_step=[];reduced_step=[]
    for _ in range(repeats):
        start=time.perf_counter();sla.splu(A.tocsc()).solve(G@u);full_steady.append(time.perf_counter()-start)
        start=time.perf_counter();rom.steady(h,u);reduced_steady.append(time.perf_counter()-start)
        start=time.perf_counter();r,U=la.eigh(A.toarray(),C.toarray());b=U.T@G@u
        U@((-np.expm1(-times[:,None]*r)/r*b).T);full_step.append(time.perf_counter()-start)
        start=time.perf_counter();rom.step(h,u,times);reduced_step.append(time.perf_counter()-start)
    return {'full_steady_median_seconds':float(np.median(full_steady)),
            'rom_steady_with_reconstruction_median_seconds':float(np.median(reduced_steady)),
            'full_step_new_parameter_median_seconds':float(np.median(full_step)),
            'rom_step_new_parameter_with_reconstruction_median_seconds':float(np.median(reduced_step)),
            'operator_bytes':sum(A.nbytes for A in [rom.K0,rom.C,rom.G,*rom.H]),
            'reconstruction_bytes':rom.V.nbytes,'repeats':repeats}


def random_extract(K,C,G,H,ranges,epsilon,seed,solver='stock'):
    snapshots=[];original_svd=stock_utils._snapshot_svd_basis;original_solve=stock_utils.spd_solve
    original_amg=stock_utils._rs_preconditioner;original_cg=stock_utils.spla.cg
    counters={'full_snapshot_rhs':0,'direct_factorizations':0,'amg_builds':0,'cg_iterations':0}
    def capture(A,tol):snapshots.append(A.copy());return original_svd(A,tol)
    def amg(A):counters['amg_builds']+=1;return original_amg(A)
    def cg(A,b,**kwargs):
        previous=kwargs.get('callback')
        def callback(x):
            counters['cg_iterations']+=1
            if previous is not None:previous(x)
        kwargs['callback']=callback;return original_cg(A,b,**kwargs)
    def solve(A,b,**kwargs):
        counters['full_snapshot_rhs']+=1
        if solver=='direct':
            counters['direct_factorizations']+=1;return sla.splu(A.tocsc()).solve(b)
        return original_solve(A,b,**kwargs)
    start=time.perf_counter()
    with patch.object(stock_utils,'_snapshot_svd_basis',capture),patch.object(stock_utils,'spd_solve',solve),\
         patch.object(stock_utils,'_rs_preconditioner',amg),patch.object(stock_utils.spla,'cg',cg):
        final,stats=build_parametric_basis(Operators(K,C,np.zeros(len(G))),G,H,ranges,
                                          tolerance=epsilon,seed=seed)
    elapsed=time.perf_counter()-start;raw=uncompressed_snapshot_basis(snapshots[0])
    return raw,final,snapshots[0],stats,counters,elapsed



def strict_random_comparison(K,C,G,H,ranges,epsilon,seed,solver):
    trials=[];selected=None;selected_raw=None
    for ratio in [.1,.05,.01]:
        raw,final,S,stats,counters,elapsed=random_extract(K,C,G,H,ranges,epsilon*ratio,seed,solver)
        audit=_Audit(K,C,G,H);start=time.perf_counter()
        row,_=audit.evaluate(raw,ranges[:,0],ranges[:,1],epsilon,np.sqrt(epsilon))
        certificate_seconds=time.perf_counter()-start
        guarded=final;guard_seconds=0.;small={};attempts=[]
        if row['accepted']:
            start=time.perf_counter()
            guarded,checks,attempts,small=_compress(K,C,G,H,raw,np.column_stack([S,np.ones((len(G),1))]),
                                     [(ranges[:,0],ranges[:,1])],epsilon,2048)
            guard_seconds=time.perf_counter()-start
            selected=project(K,C,G,H,guarded,{'status':'certified random comparison'})
            selected_raw=raw
        trials.append({'ratio':ratio,'internal_tolerance':epsilon*ratio,'raw_order':raw.shape[1],
            'stock_final_order':final.shape[1],'guarded_final_order':guarded.shape[1],
            'accepted':bool(row['accepted']),'extraction_seconds':elapsed,
            'certificate_seconds':certificate_seconds,'guard_seconds':guard_seconds,
            'end_to_end_seconds':elapsed+certificate_seconds+guard_seconds,
            'full_inverse_rhs':counters['full_snapshot_rhs']+audit.counts['certificate_riesz_rhs'],
            'snapshot_rhs':counters['full_snapshot_rhs'],'extraction_counts':counters,
            'certificate_counts':audit.counts,'compression_small_model_counts':small,
            'raw_check':row,'guard_attempts':attempts})
        if selected is not None:break
    return {'seed':seed,'solver':solver,'trials':trials,'accepted':selected is not None,
            'tuning_total_seconds':sum(t['end_to_end_seconds'] for t in trials),
            'tuning_total_full_inverse_rhs':sum(t['full_inverse_rhs'] for t in trials)},selected,selected_raw


def benchmark(name,seeds,epsilon,repeats,matched_direct,max_cells,output):
    K,C,G,H,ranges,metadata=model(name)
    report={'model':name,'metadata':metadata,'n':len(G),'inputs':G.shape[1],
            'ranges':ranges.tolist(),'epsilon':epsilon,'parameter_seed':20261008,
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'blas_threads_requested':1,'new_runs':[],'random_runs':[],'strict_random_runs':[]}
    bases={};roms={};new=None
    for repeat in range(repeats):
        start=time.perf_counter();new=extract_certified_rom(K,C,G,H,ranges,epsilon=epsilon,
                                                   max_cells=max_cells,max_enrichments=64)
        row=new.report.copy();row['end_to_end_seconds']=time.perf_counter()-start
        report['new_runs'].append(row)
        print(name,'new',repeat,row['status'],row['raw_order'],row['final_order'],row['end_to_end_seconds'],flush=True)
    bases['new_raw']=new.raw_basis;bases['new_final']=new.V;roms['new_final']=new
    new.save(output.with_name(output.stem+'_new_operators.npz'))
    for seed in seeds:
        for solver in (['stock','direct'] if matched_direct else ['stock']):
            raw,final,S,stats,counters,elapsed=random_extract(K,C,G,H,ranges,epsilon,seed,solver)
            key=f'random_{solver}_{seed}';bases[key+'_raw']=raw;bases[key+'_final']=final
            original=project(K,C,G,H,final,{'status':'uncertified random baseline'})
            roms[key+'_final']=original
            audit=_Audit(K,C,G,H);start=time.perf_counter()
            check,_=audit.evaluate(raw,ranges[:,0],ranges[:,1],epsilon,np.sqrt(epsilon))
            raw_seconds=time.perf_counter()-start;guard_seconds=0.;guard_counts={};guard_attempts=[]
            guarded=final;guard_certified=False
            if check['accepted']:
                start=time.perf_counter()
                augmented=np.column_stack([S,np.ones((len(G),1))])
                guarded,checks,guard_attempts,guard_counts=_compress(K,C,G,H,raw,augmented,
                                      [(ranges[:,0],ranges[:,1])],epsilon,2048)
                guard_seconds=time.perf_counter()-start;guard_certified=True
                bases[key+'_guarded']=guarded;roms[key+'_guarded']=project(K,C,G,H,guarded,{'status':'guarded'})
            row={'seed':seed,'solver':solver,'extraction_seconds':elapsed,'raw_order':raw.shape[1],
                'stock_final_order':final.shape[1],'guarded_final_order':guarded.shape[1],
                'raw_certified':bool(check['accepted']),'guarded_final_certified':guard_certified,
                'extraction_stats':stats,'extraction_counts':counters,'raw_check':check,
                'raw_certificate_counts':audit.counts,'raw_certificate_seconds':raw_seconds,
                'compression_small_model_counts':guard_counts,'guard_seconds':guard_seconds,
                'guard_attempts':guard_attempts,
                'same_target_end_to_end_seconds':elapsed+raw_seconds+guard_seconds,
                'same_target_full_inverse_rhs':counters['full_snapshot_rhs']+audit.counts['certificate_riesz_rhs']}
            report['random_runs'].append(row)
            print(name,key,elapsed,raw.shape[1],final.shape[1],guarded.shape[1],guard_certified,flush=True)
            if not check['accepted'] and check['reason']!='geometry':
                tuned,selected,selected_raw=strict_random_comparison(K,C,G,H,ranges,epsilon,seed,solver)
                report['strict_random_runs'].append(tuned)
                if selected is not None:
                    bases[key+'_strict_raw']=selected_raw;bases[key+'_strict_final']=selected.V
                    roms[key+'_strict_final']=selected
                print(name,key,'strict',tuned['accepted'],[t['internal_tolerance'] for t in tuned['trials']],flush=True)

    start=time.perf_counter();report['reference_metrics']=reference_metrics(K,C,G,H,ranges,bases)
    report['online_metrics']={key:online_metrics(K,C,G,H,rom,ranges) for key,rom in roms.items()}
    h=np.mean(ranges,axis=1)
    report['center_port_hankel_relative']={key:hankel_metric(K,C,G,H,rom.V,h) for key,rom in roms.items()}
    report['validation_seconds']=time.perf_counter()-start
    report['validation_is_not_continuous_certificate']=True
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['case1-local','case1-full','case1-mesh8-local','chain-local','chain-full'],default='case1-local')
    p.add_argument('--seeds',type=int,nargs='+',default=[20260805,20260806,20261007])
    p.add_argument('--epsilon',type=float,default=.001);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--matched-direct',action='store_true');p.add_argument('--max-cells',type=int,default=16)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    benchmark(a.model,a.seeds,a.epsilon,a.repeats,a.matched_direct,a.max_cells,a.output)

if __name__=='__main__':main()
