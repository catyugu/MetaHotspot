"""Low-fidelity resolvent geometry and dependency-aware Hermite sampling.

These are feasibility constructions, not new global error certificates. All
scientific fine solves and all stock BCI algorithms are invoked elsewhere.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def effective(h, beta):
    h=np.asarray(h,dtype=float); beta=np.asarray(beta,dtype=float)
    if np.any(h<0) or np.any(beta<0) or not np.all(np.isfinite(h+beta)):
        raise ValueError('invalid physical boundary coefficient')
    return h/(1+h*beta)


def log_derivative(h,beta):
    h=np.asarray(h,dtype=float); beta=np.asarray(beta,dtype=float)
    effective(h,beta)
    return h/(1+h*beta)**2


@dataclass(frozen=True)
class Sample:
    physical_h: tuple[float,...]
    shift: float
    port: int


@dataclass(frozen=True)
class Action:
    sample: int
    derivative: int = -1


@dataclass
class Family:
    K: sp.spmatrix
    C: sp.spmatrix
    G: np.ndarray
    H: list
    beta: np.ndarray
    centers: np.ndarray

    def operator(self,h,shift):
        A=self.K+shift*self.C
        for p,H in zip(effective(h,self.beta),self.H): A=A+p*H
        return A.tocsc()


def make_pool(ports,upper_shift,seed):
    if ports<1 or upper_shift<=1e-5: raise ValueError('invalid pool dimensions')
    rng=np.random.default_rng(seed+101)
    hs=list(itertools.product((1.,100.,10000.),repeat=2))
    hs.extend(tuple(row) for row in 10**rng.uniform(0,4,(8,2)))
    shifts=np.r_[0.,np.geomspace(1e-5,float(upper_shift),13)]
    pool=[Sample(tuple(h),float(s),p) for h in hs for s in shifts for p in range(ports)]
    initial=[i for i,x in enumerate(pool) if x.physical_h==(100.,100.) and x.shift==0.]
    return pool,initial


def coarse_bank(family,pool,with_jets=False):
    """Exact coarse sparse-LU solves; no fine states or validation data."""
    started=time.perf_counter(); n=family.K.shape[0]; m=family.G.shape[1]
    X=np.empty((n,len(pool))); p=len(family.H)
    D=np.empty((p,n,len(pool))) if with_jets else None
    grouped={}
    for i,s in enumerate(pool): grouped.setdefault((s.physical_h,s.shift),[]).append((i,s.port))
    nsolves=0
    for (h,shift),items in grouped.items():
        fac=sla.splu(family.operator(h,shift))
        Y=fac.solve(family.G); nsolves+=m
        jets=[]
        if with_jets:
            for d,H in zip(log_derivative(h,family.beta),family.H):
                jets.append(fac.solve(-d*(H@Y))); nsolves+=m
        for i,port in items:
            X[:,i]=Y[:,port]
            if with_jets:
                for j in range(p): D[j,:,i]=jets[j][:,port]
    return X,D,{'seconds':time.perf_counter()-started,'factorizations':len(grouped),'rhs_solves':nsolves}


def weighted_features(X,capacity,scale=None):
    """Capacity-L2 features orthogonal to the explicitly retained constant."""
    X=np.asarray(X,dtype=float); c=np.asarray(capacity,dtype=float)
    if np.any(c<=0): raise ValueError('capacity must be positive')
    sqrtc=np.sqrt(c); Y=sqrtc[:,None]*X
    norms=np.sqrt(np.sum(Y*Y,axis=0)) if scale is None else np.asarray(scale)
    if np.any(norms<=0) or not np.all(np.isfinite(norms)): raise ValueError('zero/invalid response scale')
    q=sqrtc/np.linalg.norm(sqrtc)
    Y-=q[:,None]*(q@Y)[None,:]
    return np.ascontiguousarray(Y/norms[None,:]),norms


def trilinear_lift(coarse_centers,fine_centers):
    """Cell-center interpolation, constant-preserving boundary clamping.

    This is a low-fidelity predictor only, never the fine operator or reference.
    The current benchmark has a complete Cartesian grid including its air cells.
    """
    c=np.asarray(coarse_centers,dtype=float); f=np.asarray(fine_centers,dtype=float)
    axes=[np.unique(c[:,k]) for k in range(3)]; shape=tuple(len(a) for a in axes)
    if np.prod(shape)!=len(c): raise ValueError('coarse grid is not complete Cartesian')
    coords=[np.searchsorted(axes[k],c[:,k]) for k in range(3)]
    flat=np.ravel_multi_index(coords,shape)
    if np.unique(flat).size!=len(c): raise ValueError('duplicate coarse cell')
    lookup=np.empty(len(c),dtype=int); lookup[flat]=np.arange(len(c))
    bounds=[]
    for k,a in enumerate(axes):
        if len(a)==1:
            bounds.append((np.zeros(len(f),int),np.zeros(len(f),int),np.zeros(len(f))))
        else:
            x=np.clip(f[:,k],a[0],a[-1]); hi=np.clip(np.searchsorted(a,x,side='right'),1,len(a)-1)
            lo=hi-1; weight=(x-a[lo])/(a[hi]-a[lo]); bounds.append((lo,hi,weight))
    rows=[]; cols=[]; vals=[]
    for choices in itertools.product((0,1),repeat=3):
        w=np.ones(len(f)); ix=[]
        for k,v in enumerate(choices):
            lo,hi,t=bounds[k]; ix.append(hi if v else lo); w*=t if v else 1-t
        col=lookup[np.ravel_multi_index(ix,shape)]; keep=w>0
        rows.extend(np.flatnonzero(keep)); cols.extend(col[keep]); vals.extend(w[keep])
    return sp.csr_matrix((vals,(rows,cols)),shape=(len(f),len(c)))


def discrepancy_features(coarse,fine,pool,X,seed,sketch_rows=128):
    """Sketch one diagonally scaled fine residual per coarse response.

    This adds inexpensive fine-grid information, not an inverse error estimate.
    All sparse actions, interpolation and sketching are charged to construction.
    """
    started=time.perf_counter(); P=trilinear_lift(coarse.centers,fine.centers)
    rng=np.random.default_rng(seed+303); n=fine.K.shape[0]
    rows=rng.integers(0,sketch_rows,n); signs=rng.choice([-1.,1.],n)
    S=sp.csr_matrix((signs,(rows,np.arange(n))),shape=(sketch_rows,n))
    kd=fine.K.diagonal(); cd=fine.C.diagonal(); hd=np.array([H.diagonal() for H in fine.H])
    result=np.empty((sketch_rows,len(pool)))
    # Bound memory: no fine N-by-pool bank is retained.
    grouped={}
    for i,s in enumerate(pool): grouped.setdefault((s.physical_h,s.shift),[]).append(i)
    for (h,shift),ids in grouped.items():
        Y=P@X[:,ids]
        extra=shift*cd+effective(h,fine.beta)@hd
        residual=fine.G[:,[pool[i].port for i in ids]]-fine.K@Y-extra[:,None]*Y
        diagonal=kd+extra
        if np.any(diagonal<=0): raise ValueError('nonpositive step diagonal')
        result[:,ids]=S@(np.sqrt(cd)[:,None]*(residual/diagonal[:,None]))
    return result,{'seconds':time.perf_counter()-started,'fine_operator_columns':len(pool),
                   'sketch_rows':sketch_rows,'interpolation_nnz':P.nnz}


def select_agenda(values,jets,max_solves,initial):
    """Greedy residual feature pivot per required fine RHS solve.

    A derivative requires its own base response. Both are appended to the agenda
    and both count; a prefix may finish immediately after that base response.
    """
    F=np.asarray(values,dtype=float); count=F.shape[1]
    blocks=[F]+([] if jets is None else [np.asarray(j) for j in jets])
    Z=np.ascontiguousarray(np.column_stack(blocks)); total=Z.shape[1]
    available=np.ones(total,dtype=bool); base_seen=set(); agenda=[]; Q=[]
    def append(col):
        sample=col%count; derivative=col//count-1
        if not available[col]: return
        agenda.append(Action(sample,derivative)); available[col]=False
        if derivative<0: base_seen.add(sample)
        v=Z[:,col].copy()
        if Q:
            qmat=np.column_stack(Q)
            for _ in range(2): v-=qmat@(qmat.T@v)
        norm=np.linalg.norm(v)
        if norm>1e-14:
            q=v/norm; Q.append(q)
            Z[:]-=q[:,None]*(q@Z)[None,:]
    for i in initial: append(int(i))
    while len(agenda)<max_solves and available.any():
        score=np.sum(Z*Z,axis=0)
        if jets is not None:
            missing=np.array([i not in base_seen for i in range(count)],dtype=float)
            score[count:]/=np.tile(1+missing,len(blocks)-1)
        score[~available]=-1.; col=int(np.argmax(score)); sample=col%count
        if col>=count and sample not in base_seen:
            append(sample)
            if len(agenda)>=max_solves: break
        append(col)
    if len(agenda)<max_solves: raise ValueError('pool exhausted before requested budget')
    return agenda[:max_solves]


def random_agenda(count,max_solves,initial,seed):
    rng=np.random.default_rng(seed+404); chosen=set(initial)
    rest=rng.permutation([i for i in range(count) if i not in chosen])
    return [Action(int(i)) for i in list(initial)+list(rest[:max_solves-len(initial)])]


def bump_parameter(h,j):
    h=np.asarray(h,dtype=float); out=h.copy(); delta=.25
    if h[j]*np.exp(delta)>1e4: delta=-delta
    out[j]*=np.exp(delta)
    if not 1.<=out[j]<=1e4: raise ValueError('secant leaves physical box')
    return out,delta


def close_basis(snapshots,tolerance):
    """Same normalized-response SVD principle as stock; retain uniform mode."""
    X=np.asarray(snapshots,dtype=float); norms=np.linalg.norm(X,axis=0)
    if np.any(norms<=0) or not np.all(np.isfinite(norms)): raise ValueError('invalid snapshot')
    U,s,_=la.svd(X/norms,full_matrices=False,check_finite=False)
    V=U[:,s>=tolerance*s[0]]
    q=np.ones(len(X))/np.sqrt(len(X))
    for _ in range(2): q-=V@(V.T@q)
    if np.linalg.norm(q)>np.finfo(float).eps*len(X): V=np.column_stack((V,q/np.linalg.norm(q)))
    return np.ascontiguousarray(V)
