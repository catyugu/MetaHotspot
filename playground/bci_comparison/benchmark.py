"""Native Case1 comparisons against the UNMODIFIED repository BCI extractor."""
from __future__ import annotations
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from algorithms import extract, dense_be
from measures import effective_h, split_ports, errors
from reference_backend import LinearSolver

METHODS=('stock_fantastic','shared_column','shared_tangent','shared_tangent_cached','krylov_tangent')
TOLS=(1e-2,1e-3,1e-4)
REPEATS=2


def timed(function, repeats=3):
    values=[]; out=None
    for _ in range(repeats):
        t=time.perf_counter(); out=function(); values.append(time.perf_counter()-t)
    return out,float(np.median(values))


def reduced_step(C,K,F,dt,steps):
    lu=la.cho_factor(C/dt+K,check_finite=False)
    X=np.zeros((steps+1,C.shape[0],F.shape[1]))
    for i in range(1,steps+1):
        X[i]=la.cho_solve(lu,C@X[i-1]/dt+F,check_finite=False)
    return X


def recover(V,Z):
    if Z.ndim==2: return (Z@V.T)[:,:,None]
    R=V@Z.transpose(1,0,2).reshape(V.shape[1],-1)
    return R.reshape(V.shape[0],Z.shape[0],Z.shape[2]).transpose(1,0,2)


def reference(K,C,G,dt,steps,powers=None):
    t=time.perf_counter(); L=(K+C/dt).tocsc(); factor=LinearSolver(L)
    prep=time.perf_counter()-t; c=C.diagonal()
    m=G.shape[1] if powers is None else 1
    X=np.zeros((steps+1,K.shape[0],m)); residual=0.
    t=time.perf_counter()
    for i in range(1,steps+1):
        rhs=c[:,None]*X[i-1]/dt+(G if powers is None else (G@powers[i])[:,None])
        X[i]=factor.solve(rhs,x0=X[i-1])
    solve=time.perf_counter()-t
    # Reference audit is excluded from its solve time, never from ROM costs.
    for i in (1,steps//2,steps):
        rhs=c[:,None]*X[i-1]/dt+(G if powers is None else (G@powers[i])[:,None])
        residual=max(residual,float(la.norm(L@X[i]-rhs)/max(la.norm(rhs),1e-30)))
    if residual>1e-9: raise RuntimeError(f'reference residual {residual}')
    return X,{'setup_seconds':prep,'solve_seconds':solve,'max_checked_residual':max(residual,factor.residual),'backend':factor.backend,'iterations':factor.iterations}


def main():
    from metahotspot.macromodel import utils as stock
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--port-counts',type=int,nargs='+',default=[4,16]); ap.add_argument('--extra-h',type=int,default=5)
    args=ap.parse_args(); args.output.mkdir(exist_ok=True,parents=True)
    K=sp.load_npz(args.data/'K.npz'); C=sp.load_npz(args.data/'C.npz')
    H=[sp.load_npz(args.data/f'H{i}.npz') for i in range(2)]
    data=np.load(args.data/'data.npz'); G4=data['G']; power4=data['power']
    ranges=data['ranges']; half=data['half_over_k']
    np.testing.assert_allclose(effective_h(np.array([1.,1.]),half),ranges[:,0],rtol=1e-12)
    np.testing.assert_allclose(effective_h(np.array([1e4,1e4]),half),ranges[:,1],rtol=1e-12)
    rng=np.random.default_rng(args.seed+700000)
    hcases=np.vstack((np.array([[1.,1.],[1.,1e4],[1e4,1.],[1e4,1e4],[50.,1000.]]),10**rng.uniform(0,4,(args.extra_h,2))))
    meta={'seed':args.seed,'n':K.shape[0],'methods':METHODS,'tolerances':TOLS,
          'probe_rounds':10,'port_counts':args.port_counts,'physical_HTC_cases':hcases.tolist(),
          'physical_HTC_range':[[1.,1e4],[1.,1e4]],'effective_ranges':ranges.tolist(),
          'data_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.iterdir()},
          'stock_source_sha256':hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest(),
          'native_case_path':'playground/bci_rom_testcase1/model_case1.py',
          'notes':'Same native Case1 geometry; 1 mm dataset matches the stock reproduction mesh. No commercial ROM used as truth.'}
    (args.output/'protocol.json').write_text(json.dumps(meta,indent=2))
    summaries=[]; metrics=[]; references=[]; compatibility=[]
    for port_count in args.port_counts:
        G,power=(G4,power4) if port_count==4 else split_ports(G4,power4,data['centers'])
        core=stock.normalized_operators(K,C,G@power)
        np.testing.assert_allclose(G.sum(axis=0),1.,rtol=1e-12)
        # Validation power sequences are frozen independently of construction.
        rg=np.random.default_rng(args.seed+800000+port_count)
        slow=np.repeat(rg.uniform(.05,1.,(9,port_count)),5,axis=0)[:41]*power
        fast=np.repeat(rg.integers(0,2,(11,port_count)),10,axis=0)[:101]*power
        models={}; builds={}
        for tol in TOLS:
            for repeat in range(REPEATS):
                order=METHODS if repeat==0 else METHODS[::-1]
                for method in order:
                    t=time.perf_counter()
                    if method=='stock_fantastic':
                        V,s=stock.build_parametric_basis(core,G,H,ranges,tolerance=tol,
                                max_order=1024,probe_rounds=10,seed=args.seed)
                        s['full_solves']=s['pre_svd_order']; s['basis_bytes']=V.nbytes
                    else:
                        V,s=extract(core,G,H,ranges,tolerance=tol,seed=args.seed,method=method)
                    build_s=time.perf_counter()-t
                    t=time.perf_counter()
                    cp,k0,f,fb,ab=stock.project_bci(core,G,H,V,boundary_epsilon=1e-3)
                    project_s=time.perf_counter()-t
                    key=(method,tol)
                    builds.setdefault(key,[]).append({'build':build_s,'project':project_s,'order':V.shape[1]})
                    models[key]={'V':V,'C':cp.toarray(),'K0':k0,'F':f,'fb':fb,'ab':ab,'info':s}
                    print(f'ports={port_count} tol={tol} repeat={repeat} {method}: rank={V.shape[1]} time={build_s:.3f}',flush=True)
                    (args.output/f'extract_p{port_count}_{method}_{tol}_{repeat}.json').write_text(json.dumps(s,indent=2))
            for method in METHODS:
                model=models[(method,tol)]; s=model['info']; b=builds[(method,tol)]
                summary={'ports':port_count,'method':method,'tolerance':tol,'n':K.shape[0],
                         'rank':model['V'].shape[1], 'boundary_rank':model['fb'].shape[1],
                         'extraction_seconds':float(np.median([x['build'] for x in b])),
                         'projection_seconds':float(np.median([x['project'] for x in b])),
                         'repeat_orders':[x['order'] for x in b], 'basis_bytes':model['V'].nbytes,
                         'full_solves':s['full_solves'],'pre_compression_order':s.get('pre_compression_order',s.get('pre_svd_order')),
                         'krylov_cycles':s.get('krylov_cycles',0),'preconditioners':s.get('preconditioners',s['full_solves'])}
                summaries.append(summary)
                basis_data={'V':model['V']} if K.shape[0]<=50000 else {'basis_sha256':hashlib.sha256(model['V'].tobytes()).hexdigest(),'basis_shape':model['V'].shape}
                np.savez_compressed(args.output/f'rom_p{port_count}_{method}_{tol}.npz',**basis_data,C=model['C'],K0=model['K0'].toarray(),F=model['F'],fb=model['fb'],ab=np.stack(model['ab']))
        # All methods and tolerances see the same held-out references.
        for hi,h in enumerate(hcases):
            p=effective_h(h,half); Kh=K+sum(x*M for x,M in zip(p,H))
            t=time.perf_counter(); lu=LinearSolver(Kh); Xss=lu.solve(G)
            ss_time=time.perf_counter()-t
            residual=la.norm(Kh@Xss-G)/la.norm(G)
            if residual>1e-9: raise RuntimeError('steady reference not converged')
            profiles=[('unit_steps',50.,40,None),('mixed_slow',50.,40,slow),('mixed_fast',.1,100,fast)]
            for profile,dt,steps,P in profiles:
                Xref,ref_info=reference(Kh,C,G,dt,steps,P)
                denominator=Xss if P is None else np.max(np.abs(Xref),axis=0)
                references.append({'ports':port_count,'h_id':hi,'profile':profile,
                                   'steady_setup_solve_seconds':ss_time,**ref_info})
                order=list(models.items()); order=order if hi%2==0 else order[::-1]
                for (method,tol),model in order:
                    # Parameter closure and conversion to dense are both charged.
                    Kr,assembly_s=timed(lambda:stock.assemble_reduced_k(model['K0'],model['fb'],model['ab'],p).toarray())
                    cp=model['C']; f=model['F']; V=model['V']
                    if P is None:
                        solve=lambda:reduced_step(cp,Kr,f,dt,steps)
                    else:
                        solve=lambda:dense_be(cp,Kr,f,P,dt)
                    Z,solve_s=timed(solve)
                    Xapp,recovery_s=timed(lambda:recover(V,Z))
                    _,ports_s=timed(lambda:np.einsum('np,tnm->tpm',G,Xapp))
                    # Low-dimensional output projection, not a full-field shortcut.
                    if P is None:
                        _,reduced_ports_s=timed(lambda:np.einsum('rp,trm->tpm',f,Z))
                    else:
                        _,reduced_ports_s=timed(lambda:Z@f)
                    _,peak_s=timed(lambda:Xapp.max(axis=1))
                    row={'ports':port_count,'method':method,'tolerance':tol,'n':K.shape[0],
                         'h_id':hi,'physical_h':h.tolist(),'effective_h':p.tolist(),'profile':profile,
                         'assembly_seconds':assembly_s,'reduced_solve_seconds':solve_s,
                         'reduced_ports_seconds':reduced_ports_s,'recovery_seconds':recovery_s,
                         'peak_scan_seconds':peak_s,**errors(Xref,Xapp,G,C.diagonal(),denominator)}
                    if profile=='unit_steps':
                        zss=la.solve(Kr,f,assume_a='pos',check_finite=False); appss=V@zss
                        row['steady_errors']=errors(Xss[None],appss[None],G,C.diagonal(),Xss)
                        fluxref=np.array([x*(M.diagonal()@Xss) for x,M in zip(p,H)])
                        fluxapp=np.array([x*(M.diagonal()@appss) for x,M in zip(p,H)])
                        row['steady_boundary_flux_absolute_W']=float(np.max(np.abs(fluxref-fluxapp)))
                    metrics.append(row)
                    if hi==4 and profile=='unit_steps':
                        t=time.perf_counter()
                        ts,stock_Z=stock.solve_rom_transient(sp.csc_matrix(cp),sp.csc_matrix(Kr),f,lambda _:power,50.,2000.)
                        stock_time=time.perf_counter()-t
                        dense_Z,dense_time=timed(lambda:dense_be(cp,Kr,f,np.tile(power,(41,1)),50.))
                        mismatch=float(np.max(np.abs((stock_Z-dense_Z)@V.T))/max(np.max(np.abs(dense_Z@V.T)),1e-14))
                        if mismatch>2e-5: raise RuntimeError(f'stock/common time solver mismatch: {mismatch}')
                        compatibility.append({'ports':port_count,'method':method,'tolerance':tol,
                                              'stock_seconds':stock_time,'common_seconds':dense_time,
                                              'field_relative_mismatch':mismatch})
                print(f'validated ports={port_count} h={hi} profile={profile}',flush=True)
            # Save progressively, so failures do not erase previous evidence.
            (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2))
        (args.output/'summary.json').write_text(json.dumps(summaries,indent=2))
        (args.output/'references.json').write_text(json.dumps(references,indent=2))
        (args.output/'stock_backend_audit.json').write_text(json.dumps(compatibility,indent=2))
    (args.output/'memory.json').write_text(json.dumps({'process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                    'note':'includes reference fields and all stored ROMs; not per-method memory'}))

if __name__=='__main__': main()
