"""Certificate-driven affine thermal extraction, operators -> exported ROM.

Research implementation: dense spectral certification; sparse direct snapshots.
Exact-arithmetic analytic claims, ordinary floating-point evaluation. No random
acceptance and no silent shrinkage of the requested parameter box. See
CERTIFIED_EXTRACTION_ALGORITHM.md for proofs, stopping rules and costs.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from rational_dynamic import AffineSystem
from steady_step_audit import steady_cell_bound, step_all_time_bound, _change, _norm, _dense
from affine_step_bridge import affine_step_variation_bound
from pre_svd_audit import uncompressed_snapshot_basis


def certificate_composition(steady_raw, step_raw, steady_compression, step_compression):
    """Nested Galerkin steady Pythagoras and pointwise step triangle."""
    return (float(np.hypot(steady_raw,steady_compression)),
            float(step_raw+(1+step_raw)*step_compression))


def _json_safe(value):
    if isinstance(value,dict):return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_safe(v) for v in value]
    if isinstance(value,np.ndarray):return _json_safe(value.tolist())
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value


@dataclass
class ExtractedROM:
    V: np.ndarray
    K0: np.ndarray
    C: np.ndarray
    G: np.ndarray
    H: list[np.ndarray]
    report: dict
    raw_basis: np.ndarray | None = None

    def operator(self,h):
        if len(h)!=len(self.H):raise ValueError('parameter dimension mismatch')
        return self.K0+sum(x*J for x,J in zip(h,self.H))

    def steady(self,h,u):
        return self.V@la.solve(self.operator(h),self.G@np.asarray(u),assume_a='pos')

    def step(self,h,u,times):
        t=np.asarray(times,dtype=float).reshape(-1)
        if np.any(t<0):raise ValueError('negative time')
        rates,W=la.eigh(self.operator(h),self.C)
        b=W.T@self.G@np.asarray(u)
        f=-np.expm1(-t[:,None]*rates[None,:])/rates[None,:]
        return (self.V@W@(f*b).T).T

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        # Only reduced operators and reconstruction basis, not full matrices/snapshots.
        np.savez(path,V=self.V,K0=self.K0,C=self.C,G=self.G,
                 H=np.array(self.H).reshape(len(self.H),len(self.C),len(self.C)),
                 report_json=json.dumps(_json_safe(self.report),allow_nan=False))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            return cls(z['V'],z['K0'],z['C'],z['G'],list(z['H']),json.loads(str(z['report_json'])))


def _counts():
    return dict(full_snapshot_factorizations=0,full_snapshot_rhs=0,mass_factorizations=0,
                mass_rhs=0,certificate_factorizations=0,certificate_riesz_rhs=0,
                full_spectral_decompositions=0,full_corner_norms=0,
                reduced_spectral_decompositions=0,reduced_corner_norms=0,
                reduced_node_solves=0,center_time_intervals=0,sampled_witness_evaluations=0)


class _Audit:
    """Full-system geometry is fixed per cell; only reduced quantities change."""
    def __init__(self,K,C,G,H,variation_intervals=2048):
        self.K,self.C,self.G,self.H=K,C,G,H
        self.dK,self.dC=_dense(K),_dense(C)
        self.cache={};self.factors={};self.counts=_counts()
        self.variation_intervals=variation_intervals

    def operator(self,h):return self.K+sum(x*J for x,J in zip(h,self.H))

    def geometry(self,low,high):
        key=(tuple(low),tuple(high))
        if key not in self.cache:
            center=(low+high)/2;A=_dense(self.operator(center))
            spectrum=la.eigh(A,self.dC)
            self.counts['full_spectral_decompositions']+=1
            variation=affine_step_variation_bound(A,self.dC,self.G,self.H,(high-low)/2,
                                  self.variation_intervals,full_spectrum=spectrum)
            self.counts['full_corner_norms']+=2**len(low) if len(low) else 0
            self.cache[key]=(center,A,spectrum,variation)
        return self.cache[key]

    def witness(self,A,spectrum,V):
        rates,U=spectrum;B=U.T@self.G
        rr,Z=la.eigh(V.T@A@V,V.T@self.dC@V)
        W=V@Z;Br=W.T@self.G;D=U.T@self.dC@W
        self.counts['reduced_spectral_decompositions']+=1
        times=np.geomspace(1e-8/rates[-1],35/rates[0],96)
        worst=0.;chosen=float(times[0]);input_direction=np.zeros(self.G.shape[1])
        for t in times:
            F=(-np.expm1(-rates*t)/rates)[:,None]*B
            R=(-np.expm1(-rr*t)/rr)[:,None]*Br
            Zq=_change(F.T@F);E=(F-D@R)@Zq
            _,s,Ut=la.svd(E,full_matrices=False)
            if s[0]>worst:
                worst=float(s[0]);chosen=float(t);input_direction=Zq@Ut[0]
        self.counts['sampled_witness_evaluations']+=len(times)
        return worst,chosen,input_direction

    def evaluate(self,V,low,high,steady_target,step_target,*,use_witness_gate=True):
        if V.shape[1]==V.shape[0]:
            return {'accepted':True,'steady_bound':0.,'step_bound':0.,
                    'low':low.tolist(),'high':high.tolist(),'reason':'full independent state space'},None
        center,A,spectrum,full=self.geometry(low,high)
        df=full['bound']
        if not np.isfinite(df) or df>=1:
            return {'accepted':False,'reason':'geometry','low':low.tolist(),'high':high.tolist()},None
        reduced=affine_step_variation_bound(V.T@A@V,V.T@self.dC@V,V.T@self.G,
                  [V.T@_dense(J)@V for J in self.H],(high-low)/2,self.variation_intervals)
        self.counts['reduced_spectral_decompositions']+=1
        self.counts['reduced_corner_norms']+=2**len(low) if len(low) else 0
        dr=reduced['bound']
        budget=(step_target*(1-df)-df-dr)/(1+dr)
        row={'accepted':False,'low':low.tolist(),'high':high.tolist(),
             'full_variation':float(df),'reduced_variation':float(dr),'center_budget':float(budget)}
        if not np.isfinite(budget) or budget<=0:
            row['reason']='geometry';return row,None
        if use_witness_gate:
            peak,t,u=self.witness(A,spectrum,V)
            row['sampled_center_lower_witness']=peak
            if peak>.65*budget:
                row['reason']='time_witness';return row,{'h':center,'shift':1/t,'input':u}
        fixed=step_all_time_bound(A,self.dC,self.G,V,.95*budget,full_spectrum=spectrum)
        self.counts['reduced_spectral_decompositions']+=1
        self.counts['center_time_intervals']+=fixed['intervals']
        row['center_certificate']=fixed
        if not fixed['accepted_analytic']:
            row['reason']='time_enclosure'
            _,t,u=self.witness(A,spectrum,V)
            return row,{'h':center,'shift':1/t,'input':u}
        row['step_bound']=(fixed['bound']+df+(1+fixed['bound'])*dr)/(1-df)
        factor_key=tuple(low)
        if factor_key not in self.factors:
            self.factors[factor_key]=sla.splu(sp.csc_matrix(self.operator(low)))
            self.counts['certificate_factorizations']+=1
        result,witness=steady_cell_bound(AffineSystem(self.K,self.C,self.G,self.H,V),low,high,
                                  return_witness=True,factor=self.factors[factor_key])
        self.counts['certificate_riesz_rhs']+=result['counts']['riesz_rhs']
        self.counts['reduced_node_solves']+=result['counts']['reduced_node_solves']
        row['steady_certificate']=result;row['steady_bound']=result['bound']
        if result['bound']>steady_target:
            row['reason']='steady_control';return row,witness
        row['accepted']=row['step_bound']<=step_target;row['reason']='accepted' if row['accepted'] else 'geometry'
        return row,None

    def split_axis(self,low,high):
        _,_,(rates,U),_=self.geometry(low,high)
        root=np.sqrt(rates)
        scores=[(high[j]-low[j])*_norm((U.T@(_dense(J)@U))/(root[:,None]*root[None,:]))
                for j,J in enumerate(self.H)]
        return int(np.argmax(scores))


def _validate(K,C,G,H,ranges,epsilon):
    K,C=sp.csc_matrix(K),sp.csc_matrix(C);G=np.asarray(G,dtype=float)
    H=[sp.csc_matrix(J) for J in H];ranges=np.asarray(ranges,dtype=float).reshape(-1,2)
    n=K.shape[0]
    if K.shape!=(n,n) or C.shape!=(n,n) or G.ndim!=2 or G.shape[0]!=n:
        raise ValueError('incompatible system dimensions')
    if len(H)!=len(ranges) or any(J.shape!=(n,n) for J in H):raise ValueError('parameter dimensions')
    if not np.isfinite(epsilon) or not 0<epsilon<1:raise ValueError('epsilon must be in (0,1)')
    if not np.isfinite(ranges).all() or np.any(ranges[:,0]>ranges[:,1]):raise ValueError('invalid parameter box')
    if np.linalg.matrix_rank(G)!=G.shape[1]:raise ValueError('input columns must be independent')
    for J in [K,C,*H]:
        D=_dense(J)
        if not np.isfinite(D).all() or not np.allclose(D,D.T,rtol=1e-12,atol=1e-14):
            raise ValueError('operators must be finite symmetric matrices')
    la.cholesky(_dense(C))
    for J in H:
        d=_dense(J)
        if la.eigvalsh(d)[0] < -1e-12*max(1.,la.norm(d,2)):
            raise ValueError('Robin terms must be PSD for the steady certificate')
    low=K+sum(x*J for x,J in zip(ranges[:,0],H))
    la.cholesky(_dense(low))
    return K,C,G,H,ranges


def _compress(K,C,G,H,raw,S,cells,epsilon,variation_intervals):
    """Guard compression inside the already certified raw ROM only."""
    projected=(raw.T@(K@raw),raw.T@(C@raw),raw.T@G,[raw.T@(J@raw) for J in H])
    audit=_Audit(*projected,variation_intervals=variation_intervals)
    Sn=S/la.norm(S,axis=0);reference_scale=la.svdvals(Sn)[0];r=raw.shape[1]
    protected=[la.solve(projected[1],projected[2],assume_a='pos'),raw.T@np.ones((len(raw),1))]
    for low,high in cells:
        A=projected[0]+sum(x*J for x,J in zip((low+high)/2,projected[3]))
        protected.append(la.solve(A,projected[2],assume_a='pos'))
    P=uncompressed_snapshot_basis(np.column_stack(protected),include_constant=False)
    coordinates=raw.T@Sn
    remainder=coordinates-P@(P.T@coordinates)
    U,s,_=la.svd(remainder,full_matrices=False)
    remainder_rank=int(np.sum(s>np.finfo(float).eps*max(remainder.shape)*reference_scale))
    start=int(np.sum(s>=epsilon*reference_scale))
    a=np.sqrt(epsilon);target=a/(1+a)
    attempts=[]
    for kept in range(start,remainder_rank+1):
        coordinates=np.column_stack([P,U[:,:kept]])
        # QR rank threshold is machine precision, not an additional ROM tolerance.
        T=uncompressed_snapshot_basis(coordinates,include_constant=False)
        # Parameter refinement here operates ONLY on the raw reduced system.
        # It cannot invalidate or shrink the full-system raw certificate.
        guard_cells=[(low.copy(),high.copy()) for low,high in cells]
        checks=[];index=0
        while index<len(guard_cells):
            low,high=guard_cells[index]
            row,_=audit.evaluate(T,low,high,epsilon,target)
            refine_time = row['reason'] in ('time_witness','time_enclosure') and \
                          row.get('sampled_center_lower_witness',float('inf'))<.5*target
            if (row['reason']=='geometry' or refine_time) and len(guard_cells)<64 and len(low):
                axis=audit.split_axis(low,high);mid=(low[axis]+high[axis])/2
                h1=high.copy();h1[axis]=mid;l2=low.copy();l2[axis]=mid
                guard_cells[index:index+1]=[(low.copy(),h1),(l2,high.copy())]
                continue
            checks.append(row)
            if not row['accepted']:break
            index+=1
        attempts.append({'kept_svd':kept,'order':T.shape[1],
                         'guard_cells':len(guard_cells),
                         'last_reason':checks[-1]['reason'] if checks else 'no checks',
                         'accepted':len(checks)==len(guard_cells) and all(c['accepted'] for c in checks)})
        if attempts[-1]['accepted']:
            return raw@T,checks,attempts,audit.counts
    # Keeping the entire independent raw space is an exact identity compression.
    checks=[{'accepted':True,'steady_bound':0.,'step_bound':0.,'low':l.tolist(),'high':h.tolist(),
             'reason':'identity compression'} for l,h in cells]
    return raw,checks,attempts,audit.counts


def extract_certified_rom(K,C,G,H,ranges,*,epsilon=.001,max_enrichments=64,max_cells=16,
                          max_order=None,variation_intervals=2048):
    """Deterministic adaptive extraction with honest unresolved status on budgets.

    Never certifies only a shrunken subset of `ranges`. Return operators in every
    case, but callers must check report['final_certified'] before claiming the
    requested accuracy. Certification precedes compression in accepted cases.
    """
    validation_start=time.perf_counter()
    K,C,G,H,ranges=_validate(K,C,G,H,ranges,epsilon)
    validation_seconds=time.perf_counter()-validation_start
    if max_cells<1 or max_enrichments<0:raise ValueError('invalid resource budget')
    n,m=G.shape;cap=n if max_order is None else min(n,max_order)
    if cap<1:raise ValueError('max_order must be positive')
    counts=_counts();events=[];timings={'input_validation_seconds':validation_seconds,
        'snapshot_generation_seconds':0.,'raw_certificate_seconds':0.};started=time.perf_counter()
    audit=_Audit(K,C,G,H,variation_intervals)
    low,high=ranges[:,0].copy(),ranges[:,1].copy();center=(low+high)/2
    cells=[(low,high)]
    mass=sla.splu(C);forcing=mass.solve(G)
    counts['mass_factorizations']=1;counts['mass_rhs']=m
    A=K+sum(x*J for x,J in zip(center,H));factor=sla.splu(A.tocsc())
    steady=factor.solve(G);counts['full_snapshot_factorizations']=1;counts['full_snapshot_rhs']=m
    columns=[forcing,steady,np.ones((n,1))]
    S=np.column_stack(columns);V=uncompressed_snapshot_basis(S,include_constant=False)
    if V.shape[1]>cap:raise ValueError('max_order below initial independent snapshot span')
    timings['snapshot_generation_seconds']+=time.perf_counter()-started
    checks=[];raw_ok=False;enrichments=0;reason=''
    while True:
        checks=[];failed=None
        for index,(lo,hi) in enumerate(cells):
            audit_start=time.perf_counter()
            row,witness=audit.evaluate(V,lo,hi,epsilon,np.sqrt(epsilon))
            timings['raw_certificate_seconds']+=time.perf_counter()-audit_start
            checks.append(row)
            if not row['accepted']:
                failed=(index,row,witness);break
        if failed is None:
            raw_ok=True;break
        index,row,witness=failed;lo,hi=cells[index]
        if row['reason']=='geometry':
            if len(cells)>=max_cells or not len(H):
                reason='parameter cover budget exhausted';break
            axis=audit.split_axis(lo,hi);mid=(lo[axis]+hi[axis])/2
            if mid==lo[axis] or mid==hi[axis]:reason='parameter resolution exhausted';break
            h1=hi.copy();h1[axis]=mid;l2=lo.copy();l2[axis]=mid
            cells[index:index+1]=[(lo.copy(),h1),(l2,hi.copy())]
            events.append({'action':'split','axis':axis,'cell_count':len(cells)})
            continue
        if enrichments>=max_enrichments or V.shape[1]>=cap:
            reason='snapshot enrichment budget exhausted';break
        snapshot_start=time.perf_counter()
        if row['reason']=='steady_control':
            block=witness['direction'][:,None]
            action={'action':'riesz_direction','control_index':int(witness['control_index'])}
        else:
            h=witness['h'];shift=float(witness['shift'])
            operator=K+shift*C+sum(x*J for x,J in zip(h,H))
            factor=sla.splu(operator.tocsc());block=factor.solve(G)
            counts['full_snapshot_factorizations']+=1;counts['full_snapshot_rhs']+=m
            action={'action':'rational_block','h':h.tolist(),'shift':shift}
        previous=V.shape[1];proposed_S=np.column_stack([S,block])
        proposed_V=uncompressed_snapshot_basis(proposed_S,include_constant=False)
        if proposed_V.shape[1]>cap:
            action['rejected_proposed_order']=proposed_V.shape[1];events.append(action)
            timings['snapshot_generation_seconds']+=time.perf_counter()-snapshot_start
            reason='raw order budget exceeded';break
        S,V=proposed_S,proposed_V
        enrichments+=1;action['raw_order']=V.shape[1];events.append(action)
        timings['snapshot_generation_seconds']+=time.perf_counter()-snapshot_start
        if V.shape[1]==previous:
            reason='no independent enrichment direction; unresolved';break
    timings['raw_extraction_and_certification_seconds']=time.perf_counter()-started
    raw=V.copy();raw_counts={k:counts[k]+audit.counts[k] for k in counts}
    compression_counts={};compression_checks=[];attempts=[]
    started=time.perf_counter()
    if raw_ok:
        V,compression_checks,attempts,compression_counts=_compress(K,C,G,H,raw,S,cells,
                                                                  epsilon,variation_intervals)
    timings['compression_and_guard_seconds']=time.perf_counter()-started
    certificates=[]
    if raw_ok:
        for small_check in compression_checks:
            sl,sh=np.asarray(small_check['low']),np.asarray(small_check['high'])
            raw_check=next(row for row in checks if np.all(np.asarray(row['low'])<=sl)
                          and np.all(sh<=np.asarray(row['high'])))
            s,t=certificate_composition(raw_check['steady_bound'],raw_check['step_bound'],
                                        small_check['steady_bound'],small_check['step_bound'])
            certificates.append({'low':small_check['low'],'high':small_check['high'],
                'steady_raw_bound':raw_check['steady_bound'],'step_raw_bound':raw_check['step_bound'],
                'steady_compression_bound':small_check['steady_bound'],
                'step_compression_bound':small_check['step_bound'],
                'steady_final_bound':s,'step_final_bound':t})
    final_ok=raw_ok and all(x['steady_final_bound']<=2*epsilon*(1+1e-12) and
                       x['step_final_bound']<=2*np.sqrt(epsilon)*(1+1e-12) for x in certificates)
    report={'algorithm':'certificate-driven affine rational extraction','epsilon':epsilon,
        'status':'accepted' if final_ok else 'unresolved','reason':reason,
        'raw_certified':raw_ok,'final_certified':bool(final_ok),'floating_point_certified':False,
        'raw_steady_target':epsilon,'raw_step_target':float(np.sqrt(epsilon)),
        'compression_steady_target':epsilon,'compression_step_target':float(np.sqrt(epsilon)/(1+np.sqrt(epsilon))),
        'ranges':ranges.tolist(),'raw_order':raw.shape[1],'final_order':V.shape[1],
        'parameter_cover':[{'low':l.tolist(),'high':h.tolist()} for l,h in cells],
        'max_order':cap,'max_cells':max_cells,'max_enrichments':max_enrichments,
        'cell_count':len(cells),'enrichments':enrichments,'events':events,'raw_checks':checks,
        'certificates':certificates,'compression_attempts':attempts,'counts':raw_counts,
        'compression_small_model_counts':compression_counts,**timings,
        'full_inverse_rhs_total':raw_counts['full_snapshot_rhs']+raw_counts['certificate_riesz_rhs'],
        'scope':'requested entire fixed-HTC box; all constant inputs; all t>0; exact-arithmetic theorem'}
    report['snapshot_columns']=S.shape[1]
    report['snapshot_storage_bytes']=S.nbytes
    report['basis_storage_bytes']=V.nbytes
    report['input_validation_psd_eigenvalue_checks']=len(H)
    report['input_validation_psd_norms']=len(H)
    report['input_validation_cholesky_checks']=2
    return ExtractedROM(V,V.T@(K@V),V.T@(C@V),V.T@G,[V.T@(J@V) for J in H],_json_safe(report),raw)
