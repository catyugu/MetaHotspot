"""Large-model comparison. FOM solves AMG-CG, no full dense spectra."""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from residual_reconstruction import build_basis,certify,Solver,operator,c_basis,decay_lower,steady_chord
from case1_system import case1_reconstruction
from metahotspot._compiled_data import Operators
from metahotspot.macromodel.utils import build_parametric_basis
from metahotspot.macromodel import utils as stock_utils
from unittest.mock import patch


def run(mesh,output,shifts_count,random_baseline,widths):
    K,C,G,H,ranges,meta=case1_reconstruction(mesh)
    center=np.sqrt(ranges[:,0]*ranges[:,1]);start=time.perf_counter()
    Ac=operator(K,H,center);spec=Solver(Ac);lower=decay_lower(Ac,C.diagonal(),spec)
    import scipy.sparse as sp
    scale=sp.diags(1/np.sqrt(C.diagonal()))
    upper=float(np.max(np.asarray(abs(scale@Ac@scale).sum(axis=1))))
    shifts=np.r_[0.,np.geomspace(lower/10,upper*10,shifts_count-1)]
    V,counts=build_basis(K,C,G,H,ranges,shifts,parameter_points=[center])
    for key,value in spec.counts().items():counts[key]+=value
    extraction_seconds=time.perf_counter()-start
    report=dict(n=len(G),mesh_mm=mesh,metadata=meta,shifts=shifts.tolist(),full_ranges=ranges.tolist(),
                seed=20261008,epsilon=.001,step_target=np.sqrt(.001),basis_order=V.shape[1],
                extraction_counts=counts,extraction_seconds=extraction_seconds,certificates=[],baseline={})
    def save():output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez(output.with_suffix('.npz'),candidate=V)
    save();print('EXTRACT',mesh,len(G),V.shape[1],counts,extraction_seconds,flush=True)
    for width in widths:
        box=np.column_stack([center-width*(center-ranges[:,0]),center+width*(ranges[:,1]-center)])
        start=time.perf_counter();row=certify(K,C,G,H,V,box,time_ratio=1.1)
        row['width_fraction']=width;row['ranges']=box.tolist();row['accepted_analytic']=bool(row['bound']<=np.sqrt(.001))
        row['steady_chord']=steady_chord(K,C,G,H,V,box)
        report['certificates'].append(row);save()
        print('CERT',mesh,width,row['bound'],row['center_bound'],row['parameter_penalty_infinity'],row['seconds'],flush=True)
    if random_baseline:
        raw=[];old=stock_utils._snapshot_svd_basis
        oldcg=stock_utils.spla.cg;oldamg=stock_utils._rs_preconditioner
        counters=dict(cg_iterations=0,amg_setups=0)
        def capture(S,t):raw.append(S.copy());return old(S,t)
        def cg(A,b,**kw):
            cb=kw.get('callback')
            def count(x):
                counters['cg_iterations']+=1
                if cb:cb(x)
            kw['callback']=count;return oldcg(A,b,**kw)
        def amg(A):counters['amg_setups']+=1;return oldamg(A)
        start=time.perf_counter()
        with patch.object(stock_utils,'_snapshot_svd_basis',capture),patch.object(stock_utils.spla,'cg',cg),patch.object(stock_utils,'_rs_preconditioner',amg):
            final,stats=build_parametric_basis(Operators(K,C,np.zeros(len(G))),G,H,ranges,tolerance=.001,seed=20261008)
        # Augmentation is explicitly labeled: original stock does not protect initial slope.
        rawV=c_basis(np.column_stack([raw[0],G/C.diagonal()[:,None],np.ones((len(G),1))]),C.diagonal())
        report['baseline']=dict(extraction_seconds=time.perf_counter()-start,stats=stats,counters=counters,
                                stock_final_order=final.shape[1],augmented_raw_order=rawV.shape[1],
                                initial_slope_augmented=True,certificates=[])
        np.savez(output.with_suffix('.npz'),candidate=V,random_raw_augmented=rawV,random_final=final)
        save();print('RANDOM',mesh,report['baseline']['extraction_seconds'],final.shape[1],rawV.shape[1],flush=True)
        for width in [widths[0],widths[-1]]:
            box=np.column_stack([center-width*(center-ranges[:,0]),center+width*(ranges[:,1]-center)])
            row=certify(K,C,G,H,rawV,box,time_ratio=1.1);row['width_fraction']=width
            row['steady_chord']=steady_chord(K,C,G,H,rawV,box)
            report['baseline']['certificates'].append(row);save()
            print('RANDOM_CERT',mesh,width,row['bound'],row['center_bound'],flush=True)
    # Independent full steady audit: samples only, CG residual correction is diagnostic.
    metrics={}
    bases={'candidate':V}
    if random_baseline:bases.update(random_raw_augmented=rawV,random_final=final)
    for name,v in bases.items():metrics[name]=dict(steady_K_relative=0.,steady_C_relative=0.)
    reference_counts=[]
    for h in [ranges[:,0],center,ranges[:,1]]:
        A=operator(K,H,h);solve=Solver(A);X=solve.solve(G);reference_counts.append(solve.counts())
        for name,v in bases.items():
            R=v@la.solve(v.T@(A@v),v.T@G,assume_a='pos');E=X-R
            Qk=X.T@(A@X);Qc=X.T@(C@X)
            metrics[name]['steady_K_relative']=max(metrics[name]['steady_K_relative'],float(np.sqrt(max(0,la.eigvalsh(E.T@(A@E),Qk)[-1]))))
            metrics[name]['steady_C_relative']=max(metrics[name]['steady_C_relative'],float(np.sqrt(max(0,la.eigvalsh(E.T@(C@E),Qc)[-1]))))
    report['independent_steady_samples']=metrics;report['reference_counts']=reference_counts;save()
    print('STEADY',mesh,metrics,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,default=1.5)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shifts',type=int,default=12)
    p.add_argument('--random-baseline',action='store_true');p.add_argument('--widths',type=float,nargs='+',default=[0.,.00005,.005,1.])
    a=p.parse_args();run(a.mesh_mm,a.output,a.shifts,a.random_baseline,a.widths)
