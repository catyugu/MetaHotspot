"""Direct state-space studies against unchanged repository BCI FANTASTIC.

The original native FVM is the accuracy reference. The actual library ROM,
not an approximate-inverse solver, is the performance comparator. Extra parent
extraction, optimization, parameter-local construction and decoder costs count.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from geometry import whiten, make_bank, pod_basis, loss_gradient, optimize_space, query_space, balanced_space
from numerics import timed, dense_be, modal_be, recover, output, reference, effective_h, split_ports, errors
from reference_backend import LinearSolver

TOLS=(1e-2,1e-3,1e-4)
FRACTIONS=(.35,.5,.7)


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def array_bytes(model):
    return int(sum(np.asarray(model[k]).nbytes for k in ('C','F','fb','V'))+
               model['K0'].data.nbytes+model['K0'].indices.nbytes+model['K0'].indptr.nbytes+
               sum(a.nbytes for a in model['ab']))


def projected_parent(parent,W):
    return {'C':W.T@parent['C']@W, 'K0':sp.csc_matrix(W.T@(parent['K0']@W)),
            'F':W.T@parent['F'],'fb':W.T@parent['fb'],'ab':parent['ab'],
            'V':parent['V']@W}


def main():
    from metahotspot.macromodel import utils as stock
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--ports',type=int,choices=(4,16),default=4)
    ap.add_argument('--seed',type=int,required=True); ap.add_argument('--extra-h',type=int,default=5)
    args=ap.parse_args(); out=args.output; out.mkdir(exist_ok=True,parents=True)
    source_hash=hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest()
    K=sp.load_npz(args.data/'K.npz'); C=sp.load_npz(args.data/'C.npz')
    H=[sp.load_npz(args.data/f'H{i}.npz') for i in range(2)]
    d=np.load(args.data/'data.npz'); G,power=d['G'],d['power']
    if args.ports==16:G,power=split_ports(G,power,d['centers'])
    ranges,half=d['ranges'],d['half_over_k']; core=stock.normalized_operators(K,C,G@power)
    np.testing.assert_allclose(G.sum(axis=0),1.,rtol=1e-12)
    np.testing.assert_allclose(effective_h([1.,1.],half),ranges[:,0],rtol=1e-12)
    np.testing.assert_allclose(effective_h([1e4,1e4],half),ranges[:,1],rtol=1e-12)
    rng=np.random.default_rng(args.seed+710000)
    hcases=np.vstack(([[1.,1.],[1.,1e4],[1e4,1.],[1e4,1e4],[50.,1000.]],10**rng.uniform(0,4,(args.extra_h,2))))
    training_h=np.array(list(itertools.product((1.,100.,1e4),repeat=2)))
    rg=np.random.default_rng(args.seed+810000)
    slow=np.repeat(rg.uniform(.05,1.,(9,args.ports)),5,axis=0)[:41]*power
    fast=np.repeat(rg.integers(0,2,(11,args.ports)),10,axis=0)[:101]*power
    slow[0]=0; fast[0]=0
    protocol={'seed':args.seed,'ports':args.ports,'n':K.shape[0], 'tolerances':TOLS,'fractions':FRACTIONS,
              'training_physical_h':training_h.tolist(),'validation_physical_h':hcases.tolist(),
              'effective_ranges':ranges.tolist(),'stock_source_sha256':source_hash,
              'native_case':'playground/bci_rom_testcase1/model_case1.py',
              'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data.iterdir())},
              'note':'Parameter values are constant within a trajectory. Corners overlap the deterministic construction bank; random validation and loads do not. Query-local models do not implement time-varying-HTC state transfer.'}
    write(out/'protocol.json',protocol)
    models={}; summaries=[]; extract_records=[]; common_costs={}
    for repeat in range(2):
        for tol in (TOLS if repeat==0 else TOLS[::-1]):
            start=time.perf_counter()
            V,s=stock.build_parametric_basis(core,G,H,ranges,tolerance=tol,
                     max_order=stock.MAX_ORDER,probe_rounds=10,seed=args.seed)
            extraction=time.perf_counter()-start
            start=time.perf_counter(); cp,k0,f,fb,ab=stock.project_bci(core,G,H,V,boundary_epsilon=1e-3)
            projection=time.perf_counter()-start
            key=f'stock_{tol:g}'
            models[key]={'C':cp.toarray(),'K0':k0,'F':f,'fb':fb,'ab':ab,'V':V}
            rec={'method':key,'repeat':repeat,'rank':V.shape[1],'extraction_seconds':extraction,
                 'projection_seconds':projection,'full_response_solves':s['pre_svd_order']}
            extract_records.append(rec); write(out/f'{key}_extraction_{repeat}.json',{**s,**rec})
            print(json.dumps(rec),flush=True)
    for tol in TOLS:
        key=f'stock_{tol:g}'; m=models[key]; rr=[x for x in extract_records if x['method']==key]
        cost=float(np.median([x['extraction_seconds']+x['projection_seconds'] for x in rr]))
        summaries.append({'method':key,'kind':'stock','rank':m['V'].shape[1],'offline_seconds':cost,
                          'basis_bytes':m['V'].nbytes,'deployment_bytes':array_bytes(m),
                          'full_response_solves':rr[-1]['full_response_solves'],
                          'repeat_ranks':[x['rank'] for x in rr]})
    parent_key='stock_0.0001'; original=models[parent_key]
    parent_cost=next(s['offline_seconds'] for s in summaries if s['method']==parent_key)
    start=time.perf_counter(); T=whiten(original['C']); parent=projected_parent(original,T)
    constant=la.solve(T,original['V'].T@np.ones(K.shape[0]),check_finite=False)[:,None]
    constant/=la.norm(constant)
    whitening=time.perf_counter()-start; R=len(parent['C'])
    np.testing.assert_allclose(parent['C'],np.eye(R),atol=1e-10)
    start=time.perf_counter()
    Ks=[stock.assemble_reduced_k(parent['K0'],parent['fb'],parent['ab'],effective_h(h,half)).toarray() for h in training_h]
    lmin=min(la.eigvalsh(A,check_finite=False)[0] for A in Ks)
    lmax=max(la.eigvalsh(A,check_finite=False)[-1] for A in Ks)
    shifts=np.r_[0.,np.geomspace(max(lmin/10,1e-10),lmax*10,12)]
    bank=make_bank(Ks,parent['F'],shifts); P=pod_basis(bank,constant)
    bank_pod=time.perf_counter()-start
    targets=sorted(set(min(R-1,max(args.ports+3,int(np.ceil(frac*R)))) for frac in FRACTIONS))
    common_costs={'whitening_seconds':whitening,'bank_and_pod_seconds':bank_pod,'parent_rank':R,
                  'shifts':shifts.tolist(),'targets':targets,'pod_available_rank':P.shape[1]}
    write(out/'common_costs.json',common_costs)
    if max(targets)>P.shape[1]:raise RuntimeError('construction snapshot rank too small for fixed targets')
    # Equal-rank POD and spectral minimax, both inheriting exactly the parent BCI closure.
    for r in targets:
        for method in ('pod','spectral_minimax'):
            rows=[]; m=None; W=None
            for repeat in range(2):
                start=time.perf_counter()
                if method=='pod':W=P[:,:r].copy(); hist=[]
                else:W,hist=optimize_space(P[:,:r],bank,constant)
                optimization=time.perf_counter()-start
                start=time.perf_counter(); m=projected_parent(parent,W); projection=time.perf_counter()-start
                rows.append(optimization+projection)
                write(out/f'{method}_r{r}_construction_{repeat}.json',{'optimization_seconds':optimization,'projection_seconds':projection,'history':hist})
            key=f'{method}_r{r}'; models[key]=m
            f0,_,w0=loss_gradient(P[:,:r],bank,.002); f1,_,w1=loss_gradient(W,bank,.002)
            summaries.append({'method':key,'kind':'global','rank':W.shape[1],
                              'offline_seconds':parent_cost+whitening+bank_pod+float(np.median(rows)),
                              'extra_offline_seconds':whitening+bank_pod+float(np.median(rows)),
                              'parent_rank':R,'basis_bytes':m['V'].nbytes,'deployment_bytes':array_bytes(m),
                              'training_worst_before':w0,'training_worst_after':w1,
                              'training_loss_before':f0,'training_loss_after':f1})
            np.savez_compressed(out/f'{key}_coordinates.npz',W=W)
            print(json.dumps(summaries[-1]),flush=True)
    for r in targets:
        for method in ('dc_augment','harmonic','local_balanced'):
            key=f'{method}_r{r}'
            summaries.append({'method':key,'kind':'query_local','rank_budget':r,'rank':r,
                              'offline_seconds':parent_cost+whitening+bank_pod,
                              'basis_bytes':parent['V'].nbytes,'deployment_bytes':array_bytes(parent)+P.nbytes,
                              'note':'Parent decoder retained; a smaller query state does not mean less stored field data.'})
    write(out/'summary.json',summaries)
    # Save reproducible parent matrices and basis (hash only at the fine mesh).
    saved={'C':original['C'],'K0':original['K0'].toarray(),'F':original['F'],'fb':original['fb'],'ab':np.stack(original['ab'])}
    if K.shape[0]<=50000:saved['V']=original['V']
    else:saved['V_sha256']=hashlib.sha256(original['V'].tobytes()).hexdigest(); saved['V_shape']=original['V'].shape
    np.savez_compressed(out/'parent.npz',**saved)
    metrics=[]; query_records=[]; stock_audit=[]; refs=[]
    for hi,h in enumerate(hcases):
        p=effective_h(h,half); Kh=K+sum(x*M for x,M in zip(p,H))
        factor=LinearSolver(Kh); Xss=factor.solve(G)
        residual=float(la.norm(Kh@Xss-G)/la.norm(G))
        if residual>1e-9:raise RuntimeError(f'steady reference residual {residual}')
        closed={}
        for key,m in models.items():
            Kr,seconds=timed(lambda:stock.assemble_reduced_k(m['K0'],m['fb'],m['ab'],p).toarray())
            closed[key]={'C':m['C'],'K':Kr,'F':m['F'],'V':m['V'],'preparation_seconds':seconds,'decoder_seconds':0.}
        Kparent,parent_close=timed(lambda:stock.assemble_reduced_k(parent['K0'],parent['fb'],parent['ab'],p).toarray())
        Xparent=la.solve(Kparent,parent['F'],assume_a='pos',check_finite=False)
        for r in targets:
            for method in ('dc_augment','harmonic','local_balanced'):
                def construct():
                    if method=='local_balanced':W=balanced_space(Kparent,parent['F'],r,constant)
                    else:W,_=query_space(Kparent,parent['F'],P,r,constant,method)
                    return W,W.T@parent['C']@W,W.T@Kparent@W,W.T@parent['F']
                (W,Cr,Kr,Fr),prepare=timed(construct)
                decoder,decoder_s=timed(lambda:parent['V']@W)
                dc=W@la.solve(Kr,Fr,assume_a='pos',check_finite=False)
                dc_defect=float(la.norm(dc-Xparent)/la.norm(Xparent))
                if method!='local_balanced' and dc_defect>1e-8:raise RuntimeError(f'DC identity failure {dc_defect}')
                key=f'{method}_r{r}'
                closed[key]={'C':Cr,'K':Kr,'F':Fr,'V':decoder,'preparation_seconds':parent_close+prepare,'decoder_seconds':decoder_s}
                query_records.append({'method':key,'h_id':hi,'rank':W.shape[1],'parent_dc_relative':dc_defect,
                                      'preparation_seconds':parent_close+prepare,'decoder_seconds':decoder_s,
                                      'decoder_workspace_bytes':decoder.nbytes})
                np.savez_compressed(out/f'query_h{hi}_{key}.npz',W=W)
        for key,m in closed.items():
            # Uniform representation, reciprocal ports, and strictly positive closed systems.
            np.testing.assert_allclose(G.T@m['V'],m['F'].T,rtol=1e-9,atol=1e-10)
            if la.eigvalsh(m['C'])[0]<=0 or la.eigvalsh(m['K'])[0]<=0:raise RuntimeError('lost definiteness')
        profiles=(('unit_steps',50.,40,None),('mixed_slow',50.,40,slow),('mixed_fast',.1,100,fast))
        for profile,dt,steps,powers in profiles:
            Xref,ref_info=reference(Kh,C,G,dt,steps,powers)
            denominator=Xss if powers is None else np.max(np.abs(Xref),axis=0)
            refs.append({'h_id':hi,'profile':profile,**ref_info})
            names=list(closed); names=names if hi%2==0 else names[::-1]
            for key in names:
                m=closed[key]; cp,kr,f,V=m['C'],m['K'],m['F'],m['V']
                Z,dense_s=timed(lambda:dense_be(cp,kr,f,powers,dt,steps=steps))
                _,dense_output=timed(lambda:output(f,Z))
                Xapp,recover_s=timed(lambda:recover(V,Z))
                _,peak_s=timed(lambda:Xapp.max(axis=1))
                (Y,U,Fm),modal_s=timed(lambda:modal_be(cp,kr,f,powers,dt,steps=steps))
                _,modal_output=timed(lambda:output(Fm,Y))
                Vm,modal_decoder=timed(lambda:V@U)
                Xmodal,modal_recover=timed(lambda:recover(Vm,Y))
                mismatch=float(np.max(np.abs(Xmodal-Xapp))/max(np.max(np.abs(Xapp)),1e-14))
                if mismatch>2e-7:raise RuntimeError(f'dense/modal mismatch {mismatch}')
                prepare=m['preparation_seconds']; decode=m['decoder_seconds']
                row={'method':key,'h_id':hi,'h':h.tolist(),'profile':profile,'rank':len(cp),
                     'preparation_seconds':prepare,'decoder_setup_seconds':decode,
                     'dense_solve_seconds':dense_s,'dense_output_seconds':dense_output,
                     'field_recovery_seconds':recover_s,'peak_scan_seconds':peak_s,
                     'modal_solve_seconds':modal_s,'modal_output_seconds':modal_output,
                     'modal_decoder_seconds':modal_decoder,'modal_recovery_seconds':modal_recover,
                     'junction_query_seconds':prepare+min(dense_s+dense_output,modal_s+modal_output),
                     'fullfield_query_seconds':prepare+decode+min(dense_s+recover_s+peak_s,modal_s+modal_decoder+modal_recover+peak_s),
                     'dense_modal_mismatch':mismatch,
                     **errors(Xref,Xapp,G,C.diagonal(),denominator,trajectory_normalization=powers is not None)}
                if powers is None:
                    Zss=la.solve(kr,f,assume_a='pos',check_finite=False); appss=V@Zss
                    row['steady_errors']=errors(Xss[None],appss[None],G,C.diagonal(),Xss)
                    fluxref=np.array([p[j]*(H[j].diagonal()@Xss) for j in range(2)])
                    fluxapp=np.array([p[j]*(H[j].diagonal()@appss) for j in range(2)])
                    row['steady_flux_error_W']=float(np.max(np.abs(fluxref-fluxapp)))
                metrics.append(row)
                if hi==4 and profile=='unit_steps':
                    start=time.perf_counter()
                    _,Zstock=stock.solve_rom_transient(sp.csc_matrix(cp),sp.csc_matrix(kr),f,lambda _:power,50.,2000.)
                    actual_s=time.perf_counter()-start
                    Zd=dense_be(cp,kr,f,np.tile(power,(41,1)),50.)
                    mismatch_stock=float(np.max(np.abs((Zstock-Zd)@V.T))/max(np.max(np.abs(Zd@V.T)),1e-14))
                    if mismatch_stock>2e-5:raise RuntimeError('original ROM time integration mismatch')
                    stock_audit.append({'method':key,'actual_stock_solver_seconds':actual_s,'field_mismatch':mismatch_stock})
                del Xapp,Xmodal,Vm,Z,Y
            write(out/'metrics.json',metrics); write(out/'references.json',refs)
            write(out/'query_setup.json',query_records); write(out/'stock_backend_audit.json',stock_audit)
            print(f'validated h={hi} profile={profile} models={len(closed)}',flush=True)
            del Xref
        del closed,Xss
    if hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest()!=source_hash:raise RuntimeError('stock source modified')
    write(out/'completed.json',{'completed':True,'rows':len(metrics),'models':len(summaries),'n':K.shape[0],
                               'ports':args.ports,'seed':args.seed,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               'memory_note':'process RSS includes all compared models and reference histories, not per-method workspace'})

if __name__=='__main__':main()
