"""Predeclared structural oracle and independent-source cost comparisons."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import resource
import time
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator, cg, splu
from operators import assemble, coefficients, build_backbone
from compression import exact_oracle, adaptive_inverse, refine


def save(path, rows):
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def sources(n,seed):
    """Held-out physical RHSs. Never read by basis/backbone construction."""
    rng=np.random.default_rng(seed+700001)
    x,y=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n,indexing='ij')
    result=[]; labels=[]
    for i in range(3):
        cx,cy=rng.uniform(.15,.85,2)
        z=np.exp(-((x-cx)**2+(y-cy)**2)/(.10+.04*i)**2)
        result.append(z.ravel()); labels.append('smooth')
    for i in range(3):
        z=rng.standard_normal((n,n)); result.append(z.ravel()); labels.append('signed')
    # Position samples are mesh-independent; use a single-cell concentrated RHS.
    for i in range(6):
        cx,cy=rng.uniform(.05,.95,2); z=np.zeros((n,n)); z[min(int(cx*n),n-1),min(int(cy*n),n-1)]=1
        result.append(z.ravel()); labels.append('point')
    return np.column_stack([z/np.linalg.norm(z) for z in result]),labels


def oracle_study(seed,smoke,out):
    rows=[]; grids=[6] if smoke else [12,24,36]
    for n in grids:
        for family in ['smooth','layered','inclusion']:
            k,l=coefficients(n,family,seed)
            for shift in ([20.] if smoke else [0.,20.,2000.]):
                a=assemble(k,shift)
                for kind in ['homogeneous','raw_layer','scaled_uniform','jump_aware']:
                    start=time.perf_counter(); bb=build_backbone(k,l,shift,kind)
                    vals,_=exact_oracle(a,bb); dev=np.sort(abs(vals-1))[::-1]
                    row=dict(seed=seed,n=n,N=n*n,family=family,shift=shift,backbone=kind,
                        defect_norm=float(dev[0]),lambda_min=float(vals.min()),lambda_max=float(vals.max()),
                        oracle_seconds=time.perf_counter()-start,k_min=float(k.min()),k_max=float(k.max()))
                    for degree in [1,2,3]:
                        for target in [.1,.01,.001]:
                            row[f'optimal_rank_m{degree}_eps{target}']=int(np.sum(dev>target**(1/degree)))
                    for rank in [0,8,16,32,64]:
                        if rank<len(dev): row[f'optimal_tail_r{rank}']=float(dev[rank])
                    rows.append(row)
                save(out/'oracle.csv',rows)
                print(f'oracle n={n} {family} shift={shift} complete',flush=True)
    return rows


def solve_batch(a,b,preconditioner=None,rtol=1e-2,inverse=None,degree=1):
    count=0; solutions=[]
    for j in range(b.shape[1]):
        if inverse is not None:
            x=refine(a,inverse,b[:,j],degree)
        else:
            def callback(_):
                nonlocal count
                count+=1
            x,info=cg(a,b[:,j],M=preconditioner,rtol=rtol,atol=0.,maxiter=3000,callback=callback)
            if info!=0: raise RuntimeError(f'CG failure {info}')
        solutions.append(x)
    return np.column_stack(solutions),count


def metrics(a,x,ref,b):
    e=x-ref
    energy=np.sqrt(np.maximum(0,np.sum(e*(a@e),axis=0))/np.sum(ref*(a@ref),axis=0))
    linf=np.max(abs(e),axis=0)/np.max(abs(ref),axis=0)
    residual=np.linalg.norm(b-a@x,axis=0)/np.linalg.norm(b,axis=0)
    return energy,linf,residual


def online_study(seed,smoke,out):
    if not smoke:
        import pyamg
    rows=[]; errors=[]; construction=[]
    grids=[12] if smoke else [48,96]
    repeats=1 if smoke else 3
    for n in grids:
        for family in (['layered'] if smoke else ['smooth','layered','inclusion']):
            k,l=coefficients(n,family,seed)
            for shift in ([20.] if smoke else [0.,20.,2000.]):
                a=assemble(k,shift)
                start=time.perf_counter(); lu=splu(a.tocsc()); lu_setup=time.perf_counter()-start
                start=time.perf_counter(); raw=build_backbone(k,l,shift,'raw_layer'); raw_setup=time.perf_counter()-start
                start=time.perf_counter(); bb=build_backbone(k,l,shift,'jump_aware'); bb_setup=time.perf_counter()-start
                inv,info=adaptive_inverse(a,bb,tail_target=.09,cap=64)
                b,labels=sources(n,seed)
                ref=lu.solve(b)
                ref_res=float(np.max(np.linalg.norm(b-a@ref,axis=0)/np.linalg.norm(b,axis=0)))
                if ref_res>1e-9: raise RuntimeError('reference residual exceeds declared limit')
                construction.append(dict(seed=seed,n=n,family=family,shift=shift,**info))
                with (out/'construction.json').open('w') as f: json.dump(construction,f,indent=2)
                raw_op=LinearOperator(a.shape,matvec=raw.apply,dtype=float)
                bb_op=LinearOperator(a.shape,matvec=bb.apply,dtype=float)
                methods={}
                for name,p,setup in [('raw_layer',raw_op,raw_setup),('jump_aware',bb_op,bb_setup)]:
                    for tol in [1e-2,1e-6]:
                        methods[f'{name}_cg_{tol}']=(lambda p=p,tol=tol:solve_batch(a,b,preconditioner=p,rtol=tol),setup)
                if not smoke:
                    start=time.perf_counter(); ml=pyamg.ruge_stuben_solver(a,interpolation='direct'); amg_setup=time.perf_counter()-start
                    mp=ml.aspreconditioner(cycle='V')
                    for tol in [1e-2,1e-6]:
                        methods[f'amg_cg_{tol}']=(lambda tol=tol:solve_batch(a,b,preconditioner=mp,rtol=tol),amg_setup)
                methods['sparse_lu']=(lambda:(np.column_stack([lu.solve(b[:,j]) for j in range(b.shape[1])]),0),lu_setup)
                methods['backbone_m2']=(lambda:solve_batch(a,b,inverse=bb.apply,degree=2),bb_setup)
                for degree in [1,2,3]:
                    methods[f'adaptive_m{degree}']=(lambda degree=degree:solve_batch(a,b,inverse=inv,degree=degree),bb_setup+info['setup_s'])
                times={name:[] for name in methods}; answers={}
                names=list(methods)
                # One untimed warmup per method; timing order reversed and rotated.
                for name in names: answers[name]=methods[name][0]()
                for repeat in range(repeats):
                    order=names if repeat%2==0 else names[::-1]
                    order=order[repeat:]+order[:repeat]
                    for name in order:
                        start=time.perf_counter(); answers[name]=methods[name][0](); times[name].append(time.perf_counter()-start)
                for name in names:
                    x,iters=answers[name]; en,li,rr=metrics(a,x,ref,b)
                    setup=methods[name][1]; online=float(np.median(times[name]))
                    row=dict(seed=seed,n=n,N=n*n,family=family,shift=shift,method=name,
                        sources=b.shape[1],max_energy_error=float(en.max()),max_linf_error=float(li.max()),
                        max_residual=float(rr.max()),reference_max_residual=ref_res,cg_iterations=iters,
                        online_batch_s=online,online_per_rhs_s=online/b.shape[1],setup_s=setup,
                        cold_12_rhs_s=setup+online,cost_100_rhs_s=setup+100*online/b.shape[1],
                        cost_1000_rhs_s=setup+1000*online/b.shape[1],
                        correction_rank=info['rank'] if name.startswith('adaptive') else 0,
                        tail_observed=info['tail_observed'] if name.startswith('adaptive') else '',
                        inverse_storage_bytes=bb.storage_bytes+info['update_bytes'] if name.startswith('adaptive') else '',
                        target_energy_pass=bool(en.max()<=.01))
                    rows.append(row)
                    for j,label in enumerate(labels):
                        errors.append(dict(seed=seed,n=n,family=family,shift=shift,method=name,rhs=j,rhs_kind=label,
                            energy_error=float(en[j]),linf_error=float(li[j]),residual=float(rr[j])))
                save(out/'online.csv',rows); save(out/'rhs_errors.csv',errors)
                print(f'online n={n} {family} shift={shift} rank={info["rank"]} tail={info["tail_observed"]} complete',flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    oracle=oracle_study(args.seed,args.smoke,out)
    online=online_study(args.seed,args.smoke,out)
    meta=dict(seed=args.seed,smoke=args.smoke,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        elapsed_s=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        oracle_rows=len(oracle),online_rows=len(online),threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        scope='Dimensionless 2-D shifted FVM inverse; not a transient solver, native timing, or novelty proof')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)


if __name__=='__main__': main()
