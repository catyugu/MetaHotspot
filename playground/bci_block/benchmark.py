"""Memory-bounded native BCI comparison; all extraction algorithms are charged."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from block_extract import extract as block_extract
from algorithms import extract as shared_extract
from measures import effective_h, split_ports
from reference_backend import LinearSolver
from contracts import reduced_history, recover, measure, from_steps

METHODS=('stock_fantastic','shared_tangent_cached','full_block','ritz_single','ritz_block')
TOLS=(1e-2,1e-3,1e-4)


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))


def append(path,obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj,allow_nan=False)+'\n')


def load_data(path,ports):
    K=sp.load_npz(path/'K.npz'); C=sp.load_npz(path/'C.npz')
    H=[sp.load_npz(path/f'H{j}.npz') for j in range(2)]
    with np.load(path/'data.npz') as f: data={k:f[k] for k in f.files}
    if ports==16:
        data['G'],data['power']=split_ports(data['G'],data['power'],data['centers'])
    elif ports!=4: raise ValueError('only 4 and 16 declared ports')
    np.testing.assert_allclose(data['G'].sum(axis=0),1.,atol=1e-13)
    for h,col in ((1.,0),(1e4,1)):
        np.testing.assert_allclose(effective_h(np.full(2,h),data['half_over_k']),data['ranges'][:,col])
    return K,C,H,data


def build_worker(path,ports,method,tol,seed,repeat,output,cache):
    key=f'{method}_{tol:g}'
    target=output/f'extract_{key}_{repeat}.json'
    try:
        from metahotspot.macromodel import utils as u
        K,C,H,data=load_data(path,ports); G=data['G']
        core=u.normalized_operators(K,C,G@data['power'])
        t=time.perf_counter()
        if method=='stock_fantastic':
            V,stats=u.build_parametric_basis(core,G,H,data['ranges'],tolerance=tol,
                    max_order=u.MAX_ORDER,probe_rounds=10,seed=seed)
            stats['full_solves']=stats['pre_svd_order']
        elif method=='shared_tangent_cached':
            V,stats=shared_extract(core,G,H,data['ranges'],tolerance=tol,
                    max_order=u.MAX_ORDER,probe_rounds=10,seed=seed,method=method)
        else:
            V,stats=block_extract(core,G,H,data['ranges'],tolerance=tol,
                    max_order=u.MAX_ORDER,probe_rounds=10,seed=seed,method=method)
        extraction=time.perf_counter()-t
        t=time.perf_counter()
        cp,k0,f,fb,ab=u.project_bci(core,G,H,V,boundary_epsilon=1e-3)
        projection=time.perf_counter()-t
        summary={'status':'success','method':method,'tolerance':tol,'repeat':repeat,
                 'rank':V.shape[1],'boundary_rank':fb.shape[1],'basis_bytes':V.nbytes,
                 'extraction_seconds':extraction,'projection_seconds':projection,
                 'offline_seconds':extraction+projection,
                 'basis_sha256':hashlib.sha256(V.tobytes()).hexdigest(),
                 'peak_process_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'algorithm':stats}
        # Work cache is not an uploaded artifact; reduced operators and basis
        # hashes are retained in the artifact. Regenerate full bases with seeds.
        np.savez(cache/f'{key}.npz',V=V,C=cp.toarray(),K0=k0.toarray(),F=f,fb=fb,ab=np.stack(ab))
        np.savez_compressed(output/f'reduced_{key}.npz',C=cp.toarray(),K0=k0.toarray(),F=f,fb=fb,ab=np.stack(ab))
        dump(target,summary)
    except Exception as exc:
        dump(target,{'status':'failed','method':method,'tolerance':tol,'repeat':repeat,
                     'exception':str(exc),'traceback':traceback.format_exc()})


def builds(args,methods,tols,n):
    result=[]; failed=set()
    for tol in tols:
        for repeat in (0,1):
            order=methods if repeat==0 else methods[::-1]
            for method in order:
                key=f'{method}_{tol:g}'; target=args.output/f'extract_{key}_{repeat}.json'
                if key in failed:
                    row={'status':'not_repeated_after_failure','method':method,'tolerance':tol,'repeat':repeat}
                else:
                    process=mp.get_context('spawn').Process(target=build_worker,args=(args.data,args.ports,method,tol,args.seed,repeat,args.output,args.cache))
                    start=time.perf_counter(); process.start()
                    budget=360 if n>50000 else 180
                    process.join(budget)
                    if process.is_alive():
                        process.terminate(); process.join()
                        row={'status':'wall_budget_exceeded','method':method,'tolerance':tol,'repeat':repeat,'budget_seconds':budget}
                    elif target.exists(): row=json.loads(target.read_text())
                    else: row={'status':'process_failed','method':method,'tolerance':tol,'repeat':repeat,'exitcode':process.exitcode}
                    row['worker_wall_seconds']=time.perf_counter()-start
                    if row['status']!='success': failed.add(key)
                dump(target,row); append(args.output/'builds.jsonl',row); result.append(row)
                print(f"BUILD p={args.ports} {method} tol={tol:g} repeat={repeat} status={row['status']} rank={row.get('rank')} offline={row.get('offline_seconds')}",flush=True)
    return result,[f'{m}_{tol:g}' for tol in tols for m in methods if f'{m}_{tol:g}' not in failed]


def timed(fn,repeats=3):
    samples=[]; value=None
    for _ in range(repeats):
        t=time.perf_counter(); value=fn(); samples.append(time.perf_counter()-t)
    return value,float(np.median(samples))


def reference(K,C,G,dt,steps,powers=None):
    t=time.perf_counter(); A=(K+C/dt).tocsc(); factor=LinearSolver(A)
    setup=time.perf_counter()-t
    X=np.zeros((steps+1,K.shape[0],G.shape[1] if powers is None else 1))
    c=C.diagonal()[:,None]
    t=time.perf_counter()
    for j in range(1,steps+1):
        rhs=c*X[j-1]/dt+(G if powers is None else (G@powers[j])[:,None])
        X[j]=factor.solve(rhs,x0=X[j-1])
    solve=time.perf_counter()-t
    residual=0.
    for j in (1,steps//2,steps):
        rhs=c*X[j-1]/dt+(G if powers is None else (G@powers[j])[:,None])
        # Individual RHS checks, not a norm dominated by the largest source.
        residual=max(residual,float(np.max(la.norm(A@X[j]-rhs,axis=0)/np.maximum(la.norm(rhs,axis=0),1e-30))))
    if residual>1e-9: raise RuntimeError(f'reference residual {residual:g}')
    return X,{'setup_seconds':setup,'solve_seconds':solve,'residual':max(residual,factor.residual),'backend':factor.backend,'iterations':factor.iterations}


def evaluate(args,keys,K,C,H,data,hcases):
    from metahotspot.macromodel import utils as u
    G=data['G']; power=data['power']; c=C.diagonal(); fine=K.shape[0]>50000
    rng=np.random.default_rng(args.seed+900000)
    slow=np.repeat(rng.uniform(.05,1.,(9,args.ports)),5,axis=0)[:41]*power
    fast=np.repeat(rng.integers(0,2,(11,args.ports)),10,axis=0)[:101]*power
    slow[0]=0.; fast[0]=0.
    dump(args.output/'loads.json',{'slow':slow.tolist(),'fast':fast.tolist()})
    for hi,h in enumerate(hcases):
        p=effective_h(h,data['half_over_k']); Kh=(K+sum(a*b for a,b in zip(p,H))).tocsc()
        t=time.perf_counter(); L=LinearSolver(Kh); Xss=L.solve(G)
        steady_time=time.perf_counter()-t
        ssres=float(np.max(la.norm(Kh@Xss-G,axis=0)/la.norm(G,axis=0)))
        if ssres>1e-9: raise RuntimeError('steady reference not converged')
        S,refstats=reference(Kh,C,G,50.,40)
        append(args.output/'references.jsonl',{'h_id':hi,'profile':'unit_steps','steady_seconds':steady_time,'steady_residual':ssres,**refstats})
        # The slow profile is exact superposition of THIS same time-discrete
        # linear FOM, not a ROM, regression surrogate or continuous-time truth.
        Xslow=from_steps(S,slow)
        fast_ref=None
        if not fine or hi==4:
            fast_ref,fs=reference(Kh,C,G,.1,100,fast)
            append(args.output/'references.jsonl',{'h_id':hi,'profile':'mixed_fast',**fs})
        for key in keys if hi%2==0 else keys[::-1]:
            with np.load(args.cache/f'{key}.npz') as saved: model={k:saved[k] for k in saved.files}
            V=model['V']; F=model['F']; cp=model['C']; k0=sp.csc_matrix(model['K0'])
            Kr,closure=timed(lambda:u.assemble_reduced_k(k0,model['fb'],model['ab'],p).toarray())
            method,tolerance=key.rsplit('_',1)
            base={'method':method,'tolerance':float(tolerance),'n':K.shape[0],'ports':args.ports,'seed':args.seed,'h_id':hi,'physical_h':h.tolist(),'rank':V.shape[1]}
            Zss,ss_solve=timed(lambda:la.solve(Kr,F,assume_a='pos',check_finite=False))
            Appss,ss_recovery=timed(lambda:V@Zss)
            append(args.output/'metrics.jsonl',{**base,'profile':'steady','closure_seconds':closure,'solve_seconds':ss_solve,'recovery_seconds':ss_recovery,**measure(Xss[None],Appss[None],G,c,Xss)})
            profiles=[('unit_steps',50.,40,None,S,Xss),('mixed_slow',50.,40,slow,Xslow,None)]
            if fast_ref is not None: profiles.append(('mixed_fast',.1,100,fast,fast_ref,None))
            for name,dt,steps,P,truth,denom in profiles:
                Z,solve=timed(lambda:reduced_history(cp,Kr,F,dt,steps,P))
                _,porttime=timed(lambda:np.einsum('rp,trm->tpm',F,Z,optimize=True))
                App,reconstruction=timed(lambda:recover(V,Z))
                _,scan=timed(lambda:App.max(axis=1))
                row={**base,'profile':name,'closure_seconds':closure,'solve_seconds':solve,
                     'junction_seconds':porttime,'recovery_seconds':reconstruction,'scan_seconds':scan,
                     'online_junction_seconds':closure+solve+porttime,
                     'online_field_peak_seconds':closure+solve+reconstruction+scan,
                     **measure(truth,App,G,c,denom)}
                append(args.output/'metrics.jsonl',row)
                if hi==4 and name=='unit_steps':
                    # Exactly the original Case1 combined-power endpoint metrics.
                    xr=Xss@power; xa=Appss@power
                    tr=np.einsum('tnp,p->tn',truth,power); ta=np.einsum('tnp,p->tn',App,power)
                    original=u.accuracy_summary(xr+308.15,xa+308.15,tr+308.15,ta+308.15,308.15)
                    t=time.perf_counter()
                    ts,stockZ=u.solve_rom_transient(sp.csc_matrix(cp),sp.csc_matrix(Kr),F,lambda _t:power,50.,2000.)
                    stocktime=time.perf_counter()-t
                    common=np.einsum('trp,p->tr',Z,power)
                    agreement=float(la.norm(stockZ-common)/max(la.norm(common),1e-30))
                    if agreement>1e-6: raise RuntimeError('stock and common BDF1 backend disagree')
                    append(args.output/'original_contract.jsonl',{**base,**original,'stock_backend_seconds':stocktime,'backend_relative_difference':agreement})
                del App
            print(f'VALIDATED h={hi} {key}',flush=True)
            del model,V
        del S,Xslow,fast_ref


def main():
    from metahotspot.macromodel import utils as u
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--ports',type=int,required=True); ap.add_argument('--seed',type=int,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True); args.cache.mkdir(parents=True,exist_ok=True)
    K,C,H,data=load_data(args.data,args.ports); n=K.shape[0]; fine=n>50000
    methods=tuple(m for m in METHODS if not(fine and m=='ritz_single'))
    tols=(1e-2,1e-3) if fine else TOLS
    hcases=np.array([[1.,1.],[1.,1e4],[1e4,1.],[1e4,1e4],[50.,1000.]])
    if not fine: hcases=np.vstack((hcases,10**np.random.default_rng(args.seed+700000).uniform(0,4,(2,2))))
    protocol={'seed':args.seed,'n':n,'ports':args.ports,'methods':methods,'tolerances':tols,
              'max_order':u.MAX_ORDER,'block_cap':4,'gain_fraction':.9,'probes':10,
              'physical_h':hcases.tolist(),'effective_ranges':data['ranges'].tolist(),
              'stock_sha256':hashlib.sha256(Path(u.__file__).read_bytes()).hexdigest(),
              'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.iterdir()},
              'note':'Original native Case1; 1 mm is the original mesh, 2.5 mm is a matched coarse screen. Fine fast pulse is tested only at nominal h.'}
    dump(args.output/'protocol.json',protocol)
    rows,keys=builds(args,methods,tols,n)
    evaluate(args,keys,K,C,H,data,hcases)
    dump(args.output/'completion.json',{'completed':True,'complete_models':keys,
          'failed_build_records':sum(r['status']!='success' for r in rows),
          'peak_parent_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    # Recorded resource/numerical failures are scientific outcomes, not silent
    # passes. CI itself fails if any construction could not complete its budget.
    if any(r['status']!='success' for r in rows): raise SystemExit(2)

if __name__=='__main__': main()
