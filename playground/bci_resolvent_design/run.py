"""Physical, stock-BCI comparison of two resolvent-design hypotheses.

See README for the fixed protocol. Stock routines are called without replacing
any solver, eigensolver, sampling rule, compression or boundary representation.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from design import (Family, effective, log_derivative, make_pool, coarse_bank,
                    weighted_features, discrepancy_features, select_agenda,
                    random_agenda, bump_parameter, close_basis)
from evaluation import errors, march_dense, march_modal

BASE_SHA='0b4e5874d27571488d62568d12b7a68df7656aca'
UTILS_SHA256='9d44ae2807cd96d38d4968067f159b65ef1552bb5861129af1bf3b845d2698ea'
STOCK_TOLS=(1e-2,1e-3,1e-4,1e-5)
BUDGETS=(24,48,72,96)
CLOSE_TOLS=(1e-3,1e-4)
METHODS=('coarse_qr','defect_qr','random','jet','secant')


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False))


def load_family(path):
    path=Path(path)
    for name,digest in json.loads((path/'hashes.json').read_text()).items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'native export hash mismatch: {name}')
    d=dict(np.load(path/'data.npz'))
    K=sp.load_npz(path/'K.npz').tocsc(); C=sp.load_npz(path/'C.npz').tocsc()
    H=[sp.load_npz(path/f'H{j}.npz').tocsc() for j in range(len(d['half_over_k']))]
    f=Family(K,C,d['G'],H,d['half_over_k'],d['centers'])
    if not np.allclose(f.G.sum(axis=0),1.): raise ValueError('source ports not unit power')
    return f,d


def prepare(coarse,fine,method,seed,max_budget):
    started=time.perf_counter()
    upper=float(2*np.max(fine.K.diagonal()/fine.C.diagonal()))
    pool,initial=make_pool(fine.G.shape[1],upper,seed)
    details={'pool_size':len(pool),'initial':initial,'upper_shift':upper}
    if method=='random':
        actions=random_agenda(len(pool),max_budget,initial,seed)
    else:
        jets=method in ('jet','secant')
        X,D,details['coarse_work']=coarse_bank(coarse,pool,jets)
        F,scale=weighted_features(X,coarse.C.diagonal())
        if method=='defect_qr':
            R,details['defect_work']=discrepancy_features(coarse,fine,pool,X,seed)
            F=np.vstack((F,R/scale[None,:]))
        features=None
        if jets:
            features=np.array([weighted_features(d,coarse.C.diagonal(),scale)[0] for d in D])
        actions=select_agenda(F,features,max_budget,initial)
    details['seconds']=time.perf_counter()-started
    details['agenda']=[asdict(a) for a in actions]
    details['agenda_sha256']=hashlib.sha256(json.dumps(details['agenda'],sort_keys=True).encode()).hexdigest()
    details['pool']=[asdict(s) for s in pool]
    return pool,actions,details


class FineSolves:
    """Identical fine AMG-CG kernel and bounded cache for every candidate.

    The stock extractor does NOT use this class. Secants use the SAME anchor
    preconditioner as their matched derivative, but the actual perturbed matrix.
    """
    def __init__(self):
        import pyamg
        self.pyamg=pyamg; self.cache=OrderedDict()
        self.solves=0; self.iterations=0; self.setups=0; self.max_residual=0.

    def solve(self,A,b,anchor,key,x0):
        if key not in self.cache:
            self.cache[key]=self.pyamg.ruge_stuben_solver(anchor.tocsr(),interpolation='direct').aspreconditioner(cycle='V')
            self.setups+=1
            if len(self.cache)>4: self.cache.popitem(last=False)
        self.cache.move_to_end(key); M=self.cache[key]
        niter=0
        def count(_):
            nonlocal niter
            niter+=1
        x,info=sla.cg(A,b,x0=x0,rtol=1e-6,atol=0.,maxiter=2000,M=M,callback=count)
        res=float(la.norm(A@x-b)/max(la.norm(b),1e-300))
        if info!=0 or res>1.01e-6: raise RuntimeError(f'fine solve failure info={info} residual={res}')
        self.solves+=1; self.iterations+=niter; self.max_residual=max(self.max_residual,res)
        return x


def wrap_model(V,family,utils):
    core=utils.normalized_operators(family.K,family.C,np.zeros(family.K.shape[0]))
    C,K,F,Fb,Ab=utils.project_bci(core,family.G,family.H,V,boundary_epsilon=1e-3)
    result={'V':V,'C':C.toarray(),'K0':K,'F':F,'Fb':Fb,'Ab':Ab}
    result['bytes']=sum(a.nbytes for a in (V,result['C'],K.data,K.indices,K.indptr,F,Fb,*Ab))
    return result


def extract_candidate(coarse,fine,method,seed,utils,budgets=BUDGETS,tolerances=CLOSE_TOLS):
    pool,actions,plan=prepare(coarse,fine,method,seed,max(budgets))
    solver=FineSolves(); snapshots=[]; bases={}; records=[]; response={}
    Q=np.empty((fine.K.shape[0],0)); work_time=0.
    for step,a in enumerate(actions,1):
        started=time.perf_counter(); sample=pool[a.sample]
        h=np.asarray(sample.physical_h); anchor=fine.operator(h,sample.shift); A=anchor
        key=(sample.physical_h,sample.shift)
        if a.derivative<0:
            rhs=fine.G[:,sample.port]
        else:
            if a.sample not in response: raise ValueError('unpaid derivative parent')
            if method=='secant':
                perturbed,delta=bump_parameter(h,a.derivative)
                A=fine.operator(perturbed,sample.shift); rhs=fine.G[:,sample.port]
            else:
                rhs=-log_derivative(h,fine.beta)[a.derivative]*(fine.H[a.derivative]@response[a.sample])
        if Q.shape[1]:
            estimate=Q@la.solve(Q.T@(A@Q),Q.T@rhs,assume_a='pos',check_finite=False)
        else: estimate=np.zeros(len(rhs))
        x=solver.solve(A,rhs,anchor,key,estimate)
        if a.derivative<0: response[a.sample]=x.copy()
        elif method=='secant': x=(x-response[a.sample])/delta
        snapshots.append(x)
        block=utils.orthonormalize_block(Q,x[:,None])
        if block.shape[1]: Q=np.column_stack((Q,block))
        work_time+=time.perf_counter()-started
        if step in budgets:
            for tol in tolerances:
                started=time.perf_counter(); V=close_basis(np.column_stack(snapshots),tol)
                model=wrap_model(V,fine,utils); closing=time.perf_counter()-started
                ident=f'{method}_b{step}_t{tol:g}'
                bases[ident]=model
                records.append({'id':ident,'method':method,'budget':step,'tolerance':tol,
                                'rank':V.shape[1],'boundary_rank':model['Fb'].shape[1],
                                'model_bytes':model['bytes'],'basis_bytes':V.nbytes,
                                'offline_seconds':plan['seconds']+work_time+closing,
                                'planning_seconds':plan['seconds'],'fine_seconds':work_time,
                                'closing_and_projection_seconds':closing,'full_solves':solver.solves,
                                'fine_iterations':solver.iterations,'fine_amg_setups':solver.setups,
                                'max_fine_relative_residual':solver.max_residual,
                                'basis_sha256':hashlib.sha256(V.tobytes()).hexdigest()})
    return bases,records,plan


def reduced_k(model,h,beta,utils):
    return utils.assemble_reduced_k(model['K0'],model['Fb'],model['Ab'],effective(h,beta)).toarray()


def query(model,h,beta,powers,dt,backend,full,utils):
    K=reduced_k(model,h,beta,utils)
    if backend=='dense':
        Z=march_dense(model['C'],K,model['F'],powers,dt)
        decoder=model['V']; output=model['F'].T
    elif backend=='modal':
        W,Z=march_modal(model['C'],K,model['F'],powers,dt)
        decoder=model['V']@W if full else None; output=model['F'].T@W
    else: raise ValueError('unknown backend')
    ports=np.einsum('pr,trm->tpm',output,Z,optimize=True)
    field=np.einsum('nr,trm->tnm',decoder,Z,optimize=True) if full else None
    peak=field.max(axis=1) if full else None
    return field,ports,peak


def input_profiles(power,seed):
    rng=np.random.default_rng(seed+717); m=len(power)
    units=np.tile(np.eye(m),(40,1,1))
    slow=np.repeat(rng.uniform(.1,1.7,(8,m)),5,axis=0)*power
    fast=np.repeat(rng.uniform(0.,2.,(20,m)),5,axis=0)*power
    return [('unit_steps',units,50.,False),('slow_mixed',slow[:,:,None],50.,True),('fast_mixed',fast[:,:,None],.1,True)]


def reference(family,h,profiles):
    K=family.operator(h,0.); started=time.perf_counter()
    lu=sla.splu(K); steady=lu.solve(family.G); worst=0.
    def residual(A,X,rhs):
        return float(np.max(np.linalg.norm(A@X-rhs,axis=0)/np.maximum(np.linalg.norm(rhs,axis=0),1e-300)))
    worst=max(worst,residual(K,steady,family.G)); result={}
    for name,P,dt,mixed in profiles:
        A=(K+family.C/dt).tocsc(); fac=sla.splu(A)
        X=np.zeros((len(P)+1,K.shape[0],P.shape[2]))
        for n,p in enumerate(P):
            rhs=family.C@X[n]/dt+family.G@p; X[n+1]=fac.solve(rhs)
            if n in (0,len(P)//2,len(P)-1): worst=max(worst,residual(A,X[n+1],rhs))
        result[name]=X
    if worst>1e-9: raise RuntimeError(f'full-order reference residual {worst}')
    return steady,result,{'seconds':time.perf_counter()-started,'max_relative_linear_residual':worst}


def summarize(records,metrics,timings):
    table={}
    for r in records:
        ident=r['id']
        if ident not in table:
            table[ident]={k:v for k,v in r.items() if k!='repeat'}
            table[ident]['offline_repeats']=[]
        table[ident]['offline_repeats'].append(r['offline_seconds'])
    for ident,row in table.items():
        row['offline_seconds']=statistics.median(row['offline_repeats'])
        ms=[m for m in metrics if m['id']==ident]
        for key in ('field_relative','junction_relative','peak_relative','peak_error_K','capacity_L2_relative','boundary_flow_absolute_W'):
            vals=[m[key] for m in ms if key in m]
            if vals: row['worst_'+key]=max(vals)
        row['qualified_field']=row['worst_field_relative']<=.01 and row['worst_junction_relative']<=.01
        row['qualified_ports']=row['worst_junction_relative']<=.01
        for full in (False,True):
            name='full_seconds' if full else 'ports_seconds'
            total=0.
            for profile in ('unit_steps','fast_mixed'):
                tm=[t for t in timings if t['id']==ident and t['full']==full and t['profile']==profile]
                total+=min(statistics.median(t['seconds'] for t in tm if t['backend']==backend) for backend in ('dense','modal'))
            row[name]=total
    comparisons={}
    for contract in ('field','ports'):
        flag='qualified_'+contract; stocks=[r for r in table.values() if r['method']=='stock' and r[flag]]
        if not stocks:
            comparisons[contract]={'stock_qualified':False}; continue
        stock_off=min(stocks,key=lambda r:r['offline_seconds']); stock_rank=min(r['rank'] for r in stocks)
        fastest_ports=min(r['ports_seconds'] for r in stocks); fastest_full=min(r['full_seconds'] for r in stocks)
        outcomes={}
        for method in METHODS:
            good=[r for r in table.values() if r['method']==method and r[flag]]
            if not good:
                outcomes[method]={'qualified':False}; continue
            best=min(good,key=lambda r:r['offline_seconds'])
            ratio=stock_off['offline_seconds']/best['offline_seconds']
            ports_speed=fastest_ports/best['ports_seconds']; full_speed=fastest_full/best['full_seconds']
            control='coarse_qr' if method=='defect_qr' else 'secant' if method=='jet' else None
            control_ratio=None
            if control:
                key=f"{control}_b{best['budget']}_t{best['tolerance']:g}"
                cr=table[key]
                objective='worst_field_relative' if contract=='field' else 'worst_junction_relative'
                control_ratio=cr[objective]/max(best[objective],1e-30)
            outcomes[method]={'qualified':True,'selected_id':best['id'],'offline_speedup':ratio,
                              'rank':best['rank'],'ports_speedup':ports_speed,'full_speedup':full_speed,
                              'matched_control_error_ratio':control_ratio,
                              'practical_gate':ratio>=2 and best['rank']<=1.25*stock_rank and ports_speed>=.8 and (contract=='ports' or full_speed>=.8),
                              'new_mechanism_gate':control_ratio is not None and control_ratio>=2}
        comparisons[contract]={'stock_qualified':True,'stock_offline_id':stock_off['id'],
                               'stock_min_rank':stock_rank,'methods':outcomes}
    return {'models':list(table.values()),'comparisons':comparisons}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    from metahotspot.macromodel import utils as u
    path=Path(u.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=UTILS_SHA256: raise RuntimeError('stock library changed')
    coarse,cd=load_family(args.native/'case1_5.0'); fine,fd=load_family(args.native/'case1_2.5')
    profiles=input_profiles(fd['power'],args.seed)
    core=u.normalized_operators(fine.K,fine.C,fine.G@fd['power'])
    ranges=effective(np.array([[1.,1.],[1e4,1e4]]),fine.beta).T
    models={}; records=[]; plans={}
    methods=('stock',)+METHODS
    for rep in range(2):
        for method in (methods if rep==0 else methods[::-1]):
            if method=='stock':
                bm={}; rr=[]
                for tol in (STOCK_TOLS if rep==0 else STOCK_TOLS[::-1]):
                    started=time.perf_counter()
                    V,details=u.build_parametric_basis(core,fine.G,fine.H,ranges,tolerance=tol,max_order=1024,probe_rounds=10,seed=args.seed)
                    b=wrap_model(V,fine,u); elapsed=time.perf_counter()-started
                    ident=f'stock_t{tol:g}'; bm[ident]=b
                    rr.append({'id':ident,'method':'stock','tolerance':tol,'budget':None,'rank':V.shape[1],
                               'boundary_rank':b['Fb'].shape[1],'model_bytes':b['bytes'],'basis_bytes':V.nbytes,
                               'offline_seconds':elapsed,'full_solves':details['pre_svd_order'],
                               'basis_sha256':hashlib.sha256(V.tobytes()).hexdigest()})
                    dump(args.output/f'{ident}_repeat{rep}.json',details)
            else:
                bm,rr,plan=extract_candidate(coarse,fine,method,args.seed,u)
                plans[f'{method}_{rep}']=plan
                dump(args.output/f'{method}_plan_repeat{rep}.json',plan)
            for row in rr:
                row['repeat']=rep; records.append(row); print('BUILD '+json.dumps(row),flush=True)
                if rep==0: models[row['id']]=bm[row['id']]
                else:
                    V=models[row['id']]['V']; W=bm[row['id']]['V']
                    if V.shape!=W.shape: raise RuntimeError('nondeterministic construction rank')
                    gap=la.norm(W-V@(V.T@W))/max(la.norm(W),1e-30)
                    if gap>1e-7: raise RuntimeError(f'nondeterministic subspace {row["id"]}: {gap}')
            dump(args.output/'builds.json',records)
    if plans['jet_0']['agenda']!=plans['secant_0']['agenda']: raise RuntimeError('unmatched jet/secant agendas')
    htests=[(1.,1.),(1.,1e4),(1e4,1.),(1e4,1e4),(50.,1000.)]
    htests.extend(tuple(h) for h in 10**np.random.default_rng(args.seed+1717).uniform(0,4,(5,2)))
    metrics=[]; refs=[]; times=[]; compat=[]
    for hi,h in enumerate(htests):
        steady,truth,refwork=reference(fine,h,profiles); refs.append({'h':h,**refwork})
        print(f'VALIDATE h={h}',flush=True)
        for ident,b in models.items():
            K=reduced_k(b,h,fine.beta,u)
            eigmin=float(la.eigvalsh(K,b['C'],check_finite=False,subset_by_index=[0,0])[0])
            if eigmin<=0: raise RuntimeError('nonpositive ROM')
            xs=b['V']@la.solve(K,b['F'],assume_a='pos',check_finite=False)
            row={'id':ident,'h_index':hi,'h':h,'profile':'steady','min_eigenvalue':eigmin,
                 **errors(steady[None],xs[None],fine.G,fine.C.diagonal(),steady,False)}
            row['boundary_flow_absolute_W']=float(np.max(np.abs(np.array([p*(H.diagonal()@(xs-steady)) for p,H in zip(effective(h,fine.beta),fine.H)]))))
            metrics.append(row)
            for name,P,dt,mixed in profiles:
                Z=march_dense(b['C'],K,b['F'],P,dt)
                X=np.einsum('nr,trm->tnm',b['V'],Z,optimize=True)
                metrics.append({'id':ident,'h_index':hi,'h':h,'profile':name,
                                **errors(truth[name],X,fine.G,fine.C.diagonal(),steady,mixed)})
            if hi==4:
                # Compatibility is checked for every model, never used as the fast competitor.
                power=fd['power']; P=np.tile(power[None,:,None],(40,1,1))
                start=time.perf_counter()
                ts,Zstock=u.solve_rom_transient(sp.csc_matrix(b['C']),sp.csc_matrix(K),b['F'],lambda t:power,50.,2000.)
                sec=time.perf_counter()-start; Z=march_dense(b['C'],K,b['F'],P,50.)[:,:,0]
                gap=float(np.max(np.abs(Zstock-Z))/max(np.max(np.abs(Z)),1e-14))
                if gap>1e-5: raise RuntimeError(f'stock/common BDF1 disagreement {ident}: {gap}')
                compat.append({'id':ident,'stock_wrapper_seconds':sec,'coefficient_relative_gap':gap})
                for name,P,dt,mixed in (profiles[0],profiles[2]):
                    for full in (False,True):
                        for backend in ('dense','modal'):
                            query(b,h,fine.beta,P,dt,backend,full,u)
                            for repeat in range(3):
                                start=time.perf_counter(); out=query(b,h,fine.beta,P,dt,backend,full,u)
                                elapsed=time.perf_counter()-start
                                times.append({'id':ident,'profile':name,'full':full,'backend':backend,'repeat':repeat,'seconds':elapsed})
                            # Untimed equation-equivalence check, for ports AND fields.
                            expected=query(b,h,fine.beta,P,dt,'dense',full,u)
                            for k in ([0,1,2] if full else [1]):
                                gap=float(np.max(np.abs(out[k]-expected[k]))/max(float(np.max(np.abs(expected[k]))),1e-14))
                                if gap>1e-8: raise RuntimeError(f'common backend disagreement {ident}: {gap}')
                            del out,expected
        dump(args.output/'metrics.json',metrics); dump(args.output/'timings.json',times)
    result=summarize(records,metrics,times)
    result.update({'completed':True,'base':BASE_SHA,'stock_utils_sha256':UTILS_SHA256,
                   'seed':args.seed,'fine_n':fine.K.shape[0],'coarse_n':coarse.K.shape[0],
                   'independent_sources':fine.G.shape[1],'htc_settings':htests,
                   'reference_audits':refs,'stock_backend_checks':compat,
                   'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   'note':'One original Case1 geometry; 9072-cell screen. 1 mm native data exported, not a completed fine numerical comparison. No continuous-parameter or PDE error certificate.'})
    dump(args.output/'summary.json',result); dump(args.output/'compatibility.json',compat)
    print('SUMMARY '+json.dumps(result['comparisons']),flush=True)


if __name__=='__main__': main()
