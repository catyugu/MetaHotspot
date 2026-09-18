"""Research extractors. The production FANTASTIC implementation is not modified.

Known ingredients: adaptive tangential rational Krylov and reduced CG bases.
The tested question is their joint, multi-parameter BCI construction, not a
claim that either ingredient or the finite random acceptance test is new.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla


def orthogonal_block(V, X):
    X=np.asarray(X,dtype=float).copy()
    scale=float(la.norm(X))
    if scale==0: return np.empty((V.shape[0],0))
    for _ in range(2):
        if V.shape[1]: X-=V@(V.T@X)
    Q,R,_=la.qr(X,mode='economic',pivoting=True,check_finite=False)
    keep=np.abs(np.diag(R))>np.finfo(float).eps*max(X.shape)*scale
    return np.ascontiguousarray(Q[:,keep])


def direction(R, kind):
    if kind=='column':
        d=np.zeros(R.shape[1]); d[np.argmax(la.norm(R,axis=0))]=1
        return d
    if kind!='tangent': raise ValueError(kind)
    _,_,Z=la.svd(R,full_matrices=False,check_finite=False)
    return Z[0]


def cg_directions(A, b, precondition, steps):
    """Retain PCG search directions, not a falsely converged full solution."""
    r=np.asarray(b,dtype=float).copy(); norm0=float(la.norm(r))
    cols=[]; z=None; p=None; old_rz=None
    if norm0==0: return np.empty((b.size,0)),{'iterations':0,'relative_residual':0.}
    for _ in range(steps):
        z=np.asarray(precondition(r)); rz=float(r@z)
        if rz<=0: raise ValueError('nonpositive preconditioned residual')
        p=z.copy() if p is None else z+(rz/old_rz)*p
        Ap=A@p; curvature=float(p@Ap)
        if curvature<=0: raise ValueError('nonpositive CG curvature')
        cols.append(p.copy()); r-=(rz/curvature)*Ap; old_rz=rz
        if la.norm(r)<=1e-12*norm0: break
    return np.column_stack(cols),{'iterations':len(cols),'relative_residual':float(la.norm(r)/norm0)}


def dense_be(C, K, G, powers, dt):
    """One common factored BDF1 backend for every ROM in timing comparisons."""
    L=la.cho_factor(C/dt+K,check_finite=False)
    X=np.zeros((len(powers),C.shape[0]))
    for i in range(1,len(powers)):
        X[i]=la.cho_solve(L,C@X[i-1]/dt+G@powers[i],check_finite=False)
    return X


def response_compress(V, reduced_ops, G, points, tolerance):
    """Small-coordinate normalized response POD over CONSTRUCTION points only."""
    F=V.T@G; covariance=np.zeros((V.shape[1],V.shape[1]))
    for coeff in points:
        A=sum(c*M for c,M in zip(coeff,reduced_ops))
        Y=la.solve(A,F,assume_a='pos',check_finite=False)
        norms=la.norm(Y,axis=0)
        Y=Y/np.maximum(norms,np.finfo(float).tiny)
        covariance+=Y@Y.T
    vals,U=la.eigh(covariance,check_finite=False)
    keep=vals>=tolerance*tolerance*vals[-1]
    Q=np.ascontiguousarray(V@U[:,keep])
    c=np.ones((V.shape[0],1))/np.sqrt(V.shape[0])
    add=orthogonal_block(Q,c)
    if add.shape[1]: Q=np.column_stack((Q,add))
    return Q,{'pre_compression_order':V.shape[1], 'basis_order':Q.shape[1],
              'compression_points':len(points),'covariance_eigenvalues':vals.tolist()}


class SharedSpace:
    def __init__(self, operators, G):
        self.ops=operators; self.G=G
        self.V=np.empty((G.shape[0],0))
        self.images=[np.empty((G.shape[0],0)) for _ in operators]
        self.small=[np.empty((0,0)) for _ in operators]
        self.F=np.empty((0,G.shape[1]))
        self.append(np.ones((G.shape[0],1))/np.sqrt(G.shape[0]))

    def append(self, X):
        Q=orthogonal_block(self.V,X)
        if not Q.shape[1]: return 0
        for j,A in enumerate(self.ops):
            AQ=A@Q; cross=self.V.T@AQ; diag=Q.T@AQ
            self.small[j]=np.block([[self.small[j],cross],[cross.T,diag]])
            self.images[j]=np.column_stack((self.images[j],AQ))
        self.V=np.column_stack((self.V,Q)); self.F=np.vstack((self.F,Q.T@self.G))
        return Q.shape[1]

    def response(self, coeff):
        Ar=sum(c*M for c,M in zip(coeff,self.small))
        Y=la.solve(Ar,self.F,assume_a='pos',check_finite=False)
        AV=sum(c*M for c,M in zip(coeff,self.images))
        return Y,self.G-AV@Y


def extract(core, G, boundary_terms, ranges, *, tolerance, seed,
            method, probe_rounds=10, max_order=1024, krylov_steps=4):
    """All-source residual acceptance; exact-column/tangent and Krylov variants.

    No source trajectory is used. All variants use the same union spectral
    interval and random-boundary state machine. Only enrichment differs.
    A random test is not a certificate on a continuous parameter domain.
    """
    from metahotspot.macromodel import utils as stock
    started=time.perf_counter()
    bounds=[]
    for j in range(G.shape[1]):
        bounds.append(stock.port_eigenvalue_bounds(core.K,core.C,G[:,j]))
    lmin=min(p[0] for p in bounds); lmax=max(p[1] for p in bounds)
    count=stock.mpmm_elliptic_shift_count(tolerance,lmin,lmax)
    shifts=stock.mpmm_elliptic_shifts(count,lmax,lmax/lmin)
    planning=time.perf_counter()-started
    norms=la.norm(G,axis=0); Gn=G/norms
    operators=[core.K,core.C,*boundary_terms]
    space=SharedSpace(operators,Gn)
    rng=np.random.default_rng(seed); points=[]; history=[]
    full_solves=0; preconditioners=0; cycles=0; checks=0
    for shift in shifts:
        accepted=0
        while accepted<probe_rounds:
            h=stock._draw_h(ranges,rng); coeff=(1.,float(shift),*h)
            points.append(coeff); M=None
            while True:
                Y,R=space.response(coeff); checks+=1
                score=float(np.max(la.norm(R,axis=0)))
                if score<=tolerance:
                    accepted+=1
                    break
                accepted=0
                if space.V.shape[1]>=max_order or checks>10000:
                    raise RuntimeError('research extraction reached its declared cap')
                d=direction(R,'column' if method=='shared_column' else 'tangent')
                A=sum(c*M0 for c,M0 in zip(coeff,operators)).tocsc()
                before=space.V.shape[1]
                if method=='krylov_tangent':
                    if M is None:
                        M=stock._rs_preconditioner(A.tocsr()); preconditioners+=1
                    X,inner=cg_directions(A,R@d,lambda z:M@z,krylov_steps)
                    cycles+=inner['iterations']
                elif method=='shared_tangent_cached':
                    if M is None:
                        M=stock._rs_preconditioner(A.tocsr()); preconditioners+=1
                    count_inner=[0]
                    def count_iteration(_x): count_inner[0]+=1
                    solution,flag=sla.cg(A,Gn@d,x0=space.V@(Y@d),M=M,rtol=stock.ENRICH_RTOL,atol=0.,maxiter=2000,callback=count_iteration)
                    if flag!=0: raise RuntimeError('cached exact CG did not converge')
                    X=solution[:,None]; full_solves+=1; cycles+=count_inner[0]
                    inner={'iterations':count_inner[0]}
                else:
                    X=stock.spd_solve(A,Gn@d,x0=space.V@(Y@d))[:,None]
                    full_solves+=1; preconditioners+=1
                    inner={}
                added=space.append(X)
                if added==0: raise RuntimeError('linearly dependent enrichment before acceptance')
                history.append({'shift':float(shift),'effective_h':list(h),
                                'residual_before':score,'order_before':before,
                                'added':added,**inner})
    t=time.perf_counter()
    V,compression=response_compress(space.V,space.small,G,points,tolerance)
    compression_s=time.perf_counter()-t
    return V,{'method':method,'seconds':time.perf_counter()-started,
              'planning_seconds':planning,'compression_seconds':compression_s,
              'per_port_bounds':bounds,'shifts_per_s':shifts.tolist(),
              'full_solves':full_solves,'preconditioners':preconditioners,
              'krylov_cycles':cycles,'checks':checks,'history':history,
              'basis_bytes':V.nbytes,
              'peak_explicit_space_bytes':space.V.nbytes+sum(M.nbytes for M in space.images),
              'orthogonality_error':float(la.norm(V.T@V-np.eye(V.shape[1]),ord=2)),
              **compression}
