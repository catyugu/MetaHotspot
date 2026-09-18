"""Finite-pool spectral constraint exchange for nonlinear edge cubature.

The state basis is held fixed. Methods get the same candidate-state pool and
are evaluated on separately generated interior and vertex holdouts. A finite
pool/search is NOT a uniform-domain spectral guarantee. No Jacobian or PDE
solution-error guarantee is inferred from stiffness approximation.
"""
from __future__ import annotations
import itertools
import time
import numpy as np
from scipy import linalg as la
from scipy.optimize import linprog, nnls
from common import fingerprint, sym


class EdgeFamily:
    def __init__(self,nx:int,rank:int,seed:int,strength:str):
        if strength not in ('mild','strong') or rank > 6 or rank < 1 or nx < 4:
            raise ValueError('invalid edge-family parameters')
        x=(np.arange(nx)+.5)/nx
        xx,yy=np.meshgrid(x,x,indexing='ij')
        modes=[(1,1),(1,2),(2,1),(2,2),(1,3),(3,1)][:rank]
        phi=np.column_stack([(np.sin(i*np.pi*xx)*np.sin(j*np.pi*yy)).ravel() for i,j in modes])
        phi/=np.max(np.sum(np.abs(phi),axis=1))
        self.r=rank; self.nx=nx
        # Ambient ground has zero temperature rise and therefore zero basis row.
        phi=np.vstack((phi,np.zeros((1,rank))))
        ground=nx*nx; edges=[]
        for i in range(nx):
            for j in range(nx):
                u=i*nx+j
                if i+1<nx: edges.append((u,(i+1)*nx+j))
                if j+1<nx: edges.append((u,i*nx+j+1))
                if i in (0,nx-1): edges.append((u,ground))
                if j in (0,nx-1): edges.append((u,ground))
        ends=np.asarray(edges,dtype=int)
        groups=np.minimum((3*yy).astype(int),2).ravel()
        rng=np.random.default_rng(seed)
        k=np.r_[np.array([1.,4.,12.])[groups]*np.exp(rng.normal(scale=.12,size=nx*nx)),1.]
        beta=np.array([.2,-.3,.4] if strength=='mild' else [1.,-1.6,2.])[groups]
        beta=np.r_[beta,0.]
        self.left=phi[ends[:,0]]; self.right=phi[ends[:,1]]
        self.a=self.left-self.right
        self.kl=k[ends[:,0]]; self.kr=k[ends[:,1]]
        self.bl=beta[ends[:,0]]; self.br=beta[ends[:,1]]
        # Ground-side half resistance uses the boundary cell's nominal material.
        boundary=ends[:,1]==ground
        self.kr[boundary]=self.kl[boundary]
        self.edge_count=len(edges); self.scale=1.
        self.scale=rank/np.trace(self.matrix(np.zeros(rank)))

    def conductance(self,q,ids=None):
        if ids is None:
            left,right,kl,kr,bl,br=self.left,self.right,self.kl,self.kr,self.bl,self.br
        else:
            left,right,kl,kr,bl,br=(v[ids] for v in (self.left,self.right,self.kl,self.kr,self.bl,self.br))
        l=kl*np.exp(bl*(left@q)); r=kr*np.exp(br*(right@q))
        return self.scale*2.*l*r/(l+r)

    def matrix(self,q,ids=None,weights=None):
        a=self.a if ids is None else self.a[ids]
        g=self.conductance(q,ids)
        if weights is not None: g=g*weights
        return sym(a.T@(g[:,None]*a))


def worst_direction(reference,approximation):
    eig,v=la.eigh(sym(approximation-reference),reference,check_finite=False)
    k=int(np.argmax(np.abs(eig)))
    return float(abs(eig[k])),v[:,k]


def positive_greedy(a,b,budget):
    norms=la.norm(a,axis=0); ids=[]; w=np.empty(0); residual=b.copy()
    for _ in range(min(budget,a.shape[1])):
        score=(a.T@residual)/np.maximum(norms,1e-30)
        if ids: score[ids]=-np.inf
        k=int(np.argmax(score))
        if score[k] < 1e-13*max(la.norm(b),1.): break
        ids.append(k)
        w,_=nnls(a[:,ids],b,maxiter=max(300,30*len(ids)))
        residual=b-a[:,ids]@w
    return np.asarray(ids,dtype=int),w


def minimax_weights(a):
    rows,cols=a.shape
    objective=np.r_[np.zeros(cols),1.]
    constraints=np.vstack((np.column_stack((a,-np.ones(rows))),np.column_stack((-a,-np.ones(rows)))))
    rhs=np.r_[np.ones(rows),-np.ones(rows)]
    result=linprog(objective,A_ub=constraints,b_ub=rhs,bounds=(0.,None),method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:
        raise RuntimeError('minimax weight fit failed: '+result.message)
    return result.x[:-1],float(result.x[-1])


def pool_data(family,pool):
    g=np.asarray([family.conductance(q) for q in pool])
    k=np.einsum('pe,ei,ej->pij',g,family.a,family.a,optimize=True)
    return g,k


def cut_row(family,g,k,v):
    return g*(family.a@v)**2/(v@k@v)


def initial_cuts(family,g,k):
    eye=np.eye(family.r); vectors=list(eye)
    for i,j in itertools.combinations(range(family.r),2):
        vectors.extend([eye[i]+eye[j],eye[i]-eye[j]])
    return [cut_row(family,g[0],k[0],v) for v in vectors]


def evaluate_pool(family,g,k,ids,w):
    a=family.a[ids]
    kh=np.einsum('pe,e,ei,ej->pij',g[:,ids],w,a,a,optimize=True)
    return [worst_direction(ref,approx) for ref,approx in zip(k,kh)]


def spectral_exchange(family,pool,budget,rounds=10,fixed_ids=None,precomputed=None):
    g,k=pool_data(family,pool) if precomputed is None else precomputed
    cuts=initial_cuts(family,g,k); best=None
    for iteration in range(rounds):
        f=np.asarray(cuts)
        if fixed_ids is None:
            ids,_=positive_greedy(f,np.ones(len(f)),budget)
        else:
            ids=np.asarray(fixed_ids,dtype=int)
        w,t=minimax_weights(f[:,ids])
        errors=evaluate_pool(family,g,k,ids,w)
        worst=np.argsort([z[0] for z in errors])[-2:]
        value=float(errors[worst[-1]][0])
        if best is None or value<best[0]: best=(value,ids.copy(),w.copy(),iteration+1)
        if value < .025: break
        for i in worst:
            cuts.append(cut_row(family,g[i],k[i],errors[i][1]))
    return best[1],best[2],{'pool_states':len(pool),'rounds_run':iteration+1,
                           'selected_round':best[3],'training_spectral_max':best[0],
                           'constraint_rows':len(cuts),'fixed_support':fixed_ids is not None}


def matrix_features(family,g,k):
    indices=np.triu_indices(family.r)
    scale=np.where(indices[0]==indices[1],1.,np.sqrt(2.))
    blocks=[]
    for coeff,ref in zip(g,k):
        l=la.cholesky(ref,lower=True,check_finite=False)
        aw=la.solve_triangular(l,family.a.T,lower=True,check_finite=False).T
        block=(aw[:,indices[0]]*aw[:,indices[1]])*coeff[:,None]*scale
        blocks.append(block.T)
    target=np.tile(np.eye(family.r)[indices]*scale,len(g))
    return np.vstack(blocks),target


def run(seed,smoke=False):
    rank=3 if smoke else 6
    grids=[6] if smoke else [12,24,48]
    strengths=['mild'] if smoke else ['mild','strong']
    budgets=[8] if smoke else [12,24,48]
    rng=np.random.default_rng(seed+6001)
    pool=np.vstack((np.zeros(rank),rng.uniform(-1.,1.,(11 if smoke else 47,rank))))
    hold=np.vstack((np.random.default_rng(seed+9001).uniform(-1.,1.,(24 if smoke else 192,rank)),
                    np.array(list(itertools.product([-1.,1.],repeat=rank)))))
    bench=np.random.default_rng(seed+11001).uniform(-1.,1.,(24 if smoke else 64,rank))
    rows=[]; audits=[]
    for nx,strength in itertools.product(grids,strengths):
        start=time.perf_counter(); family=EdgeFamily(nx,rank,seed,strength)
        family_seconds=time.perf_counter()-start
        start=time.perf_counter(); g,k=pool_data(family,pool)
        pool_seconds=time.perf_counter()-start
        start=time.perf_counter(); features,target=matrix_features(family,g,k)
        feature_seconds=time.perf_counter()-start
        references=[family.matrix(q) for q in hold]
        full_times=[]
        for _ in range(3):
            start=time.perf_counter()
            for q in bench: family.matrix(q)
            full_times.append(time.perf_counter()-start)
        full_seconds=float(np.median(full_times))
        l=la.cholesky(k[0],lower=True)
        whitened=la.solve_triangular(l,family.a.T,lower=True).T
        lev=g[0]*np.sum(whitened**2,axis=1); lev/=lev.sum()
        for budget in budgets:
            methods={}; cost={}; meta={}
            start=time.perf_counter(); ids,w=positive_greedy(features,target,budget)
            greedy_seconds=time.perf_counter()-start
            methods['matrix_frobenius']=(ids,w)
            cost['matrix_frobenius']=pool_seconds+feature_seconds+greedy_seconds
            meta['matrix_frobenius']={}
            start=time.perf_counter()
            ids2,w2,info=spectral_exchange(family,pool,budget,rounds=2 if smoke else 10,
                                            fixed_ids=ids,precomputed=(g,k))
            methods['matrix_fixed_support_minimax']=(ids2,w2)
            cost['matrix_fixed_support_minimax']=cost['matrix_frobenius']+time.perf_counter()-start
            meta['matrix_fixed_support_minimax']=info
            start=time.perf_counter()
            ids3,w3,info=spectral_exchange(family,pool,budget,rounds=2 if smoke else 10,precomputed=(g,k))
            methods['spectral_exchange']=(ids3,w3)
            cost['spectral_exchange']=pool_seconds+time.perf_counter()-start
            meta['spectral_exchange']=info
            start=time.perf_counter()
            chosen=np.random.default_rng(seed+nx+budget).choice(family.edge_count,size=budget,replace=False,p=lev)
            weights,_=nnls(features[:,chosen],target,maxiter=30*budget)
            methods['nominal_leverage_nnls']=(chosen,weights)
            cost['nominal_leverage_nnls']=pool_seconds+feature_seconds+time.perf_counter()-start
            meta['nominal_leverage_nnls']={}
            for method,(selected,weights) in methods.items():
                positive=weights>1e-12
                selected,weights=selected[positive],weights[positive]
                errors=[]; force_errors=[]; min_eig=float('inf')
                for q,ref in zip(hold,references):
                    approx=family.matrix(q,selected,weights)
                    errors.append(worst_direction(ref,approx)[0])
                    min_eig=min(min_eig,float(la.eigvalsh(approx)[0]))
                    force=(approx-ref)@q
                    force_errors.append(float(np.sqrt(max(0.,force@la.solve(ref,force,assume_a='pos')))/
                                              max(np.sqrt(q@ref@q),1e-30)))
                online=[]
                for _ in range(3):
                    start=time.perf_counter()
                    for q in bench: family.matrix(q,selected,weights)
                    online.append(time.perf_counter()-start)
                seconds=float(np.median(online))
                train_errors=evaluate_pool(family,g,k,selected,weights)
                row={'nx':nx,'rank':rank,'strength':strength,'full_edges':family.edge_count,
                     'budget':budget,'active_edges':len(selected),'method':method,
                     'train_spectral_max':float(max(z[0] for z in train_errors)),
                     'holdout_spectral_max':float(max(errors)),
                     'holdout_spectral_p95':float(np.quantile(errors,.95)),
                     'holdout_force_energy_max':float(max(force_errors)),
                     'minimum_sampled_eigenvalue':min_eig,
                     'offline_s':cost[method], 'shared_family_setup_s':family_seconds,
                     'full_reduced_operator_batch_s':full_seconds,'sampled_operator_batch_s':seconds,
                     'online_operator_speedup':full_seconds/seconds,
                     'break_even_operator_calls':float(cost[method]/((full_seconds-seconds)/len(bench))) if full_seconds>seconds else None,
                     'training_feature_bytes':features.nbytes if method!='spectral_exchange' else 0,
                     'online_edge_data_bytes':int(len(selected)*(3*rank+5)*8),
                     'bench_queries':len(bench),'holdout_states':len(hold)}
                rows.append(row)
                audits.append({'nx':nx,'strength':strength,'budget':budget,'method':method,
                               'selected_edges':selected.tolist(),'weights':weights.tolist(),**meta[method]})
            print(f'spectral {nx=} {strength=} {budget=} complete',flush=True)
    return rows,{'pool_sha256':fingerprint(pool),'holdout_sha256':fingerprint(hold),
                 'pool_states':len(pool),'holdout_states':len(hold),'frozen_basis':True,
                 'holdout_used_for_selection':False,'uniform_domain_certification':False,
                 'timing_scope':'evaluate an already reduced stiffness operator, not a full PDE solve',
                 'online_uses_selected_edges_only':True,'selection_audits':audits}
