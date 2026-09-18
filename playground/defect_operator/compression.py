"""Optimal finite-matrix gates and source-independent spectral correction.

Spectral low-rank updates and stationary refinement are established operations.
The research question concerns the complexity remaining after coefficient
normalization and exact retention of aligned jumps, NOT invention of Lanczos.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, eigsh


def exact_oracle(a,backbone):
    q=backbone.q(np.eye(a.shape[0]))
    h=q.T@(a@q)
    return la.eigh((h+h.T)*.5,check_finite=False)


def spectral_inverse(a,backbone,rank:int,tolerance:float=1e-9):
    """Build inverse update using A applications only, no full-order solves.

    Rayleigh-Ritz re-evaluation keeps the small inverse positive; off-invariant
    errors are recorded, not silently assumed zero. No optimal-tail certificate
    is claimed for approximate Lanczos output at large N.
    """
    n=a.shape[0]
    if rank<0 or rank>=n-1: raise ValueError('invalid correction rank')
    started=time.perf_counter(); calls=0
    if rank==0:
        return backbone.apply,{'rank':0,'operator_columns':0,'setup_s':0.,'update_bytes':0,'ritz_values':[],'invariance_residual':0.}
    def action(x):
        nonlocal calls
        calls+=1 if x.ndim==1 else x.shape[1]
        return backbone.qt(a@backbone.q(x))-x
    op=LinearOperator(a.shape,matvec=action,matmat=action,dtype=float)
    v0=np.cos(np.arange(n)*.731)+.2
    vals,u=eigsh(op,k=rank,which='LM',tol=tolerance,v0=v0,ncv=min(n,max(2*rank+8,24)),maxiter=5000)
    hu=action(u)+u
    hc=(u.T@hu); hc=(hc+hc.T)*.5
    ev=la.eigvalsh(hc)
    if ev[0]<=0: raise RuntimeError('nonpositive projected operator')
    correction=la.solve(hc,np.eye(rank),assume_a='pos')-np.eye(rank)
    v=backbone.q(u)
    defect=float(la.norm(hu-u@hc,2))
    def inverse(b):
        return backbone.apply(b)+v@(correction@(v.T@b))
    return inverse,{'rank':rank,'operator_columns':calls,'setup_s':time.perf_counter()-started,
                    'update_bytes':v.nbytes+correction.nbytes,'ritz_values':ev.tolist(),
                    'invariance_residual':defect}


def refine(a,inverse,b,degree:int):
    """Apply a fixed-degree inverse polynomial; not tolerance-driven CG.

    For symmetric R, the energy-error matrix of m steps is (I-A^.5 R A^.5)^m.
    This identity holds without exact Ritz vectors. If its norm exceeds one,
    extra steps may worsen the approximation, and are still reported.
    """
    if degree<1: raise ValueError('positive degree required')
    x=inverse(b)
    for _ in range(degree-1):
        x=x+inverse(b-a@x)
    return x


def adaptive_inverse(a,backbone,tail_target=.09,cap=64):
    """Lanczos outlier deflation with a fixed spectral target and explicit cap.

    Numerical tail observation is not a formally verified completeness bound.
    No physical source or validation solution enters the construction.
    Every repeat and postprocessing operator action is charged.
    """
    if tail_target<=0 or cap<2: raise ValueError('invalid adaptive target/cap')
    start=time.perf_counter(); calls=0; n=a.shape[0]; cap=min(cap,n-2); stages=[]
    def action(x):
        nonlocal calls
        calls+=1 if x.ndim==1 else x.shape[1]
        return backbone.qt(a@backbone.q(x))-x
    op=LinearOperator(a.shape,matvec=action,matmat=action,dtype=float)
    v0=np.cos(np.arange(n)*.731)+.2
    k=min(8,cap)
    while True:
        ev,u=eigsh(op,k=k,which='LM',tol=1e-9,v0=v0,ncv=min(n,max(2*k+8,24)),maxiter=5000)
        stages.append(k)
        if np.min(abs(ev))<=tail_target or k==cap: break
        k=min(2*k,cap)
    seen=bool(np.min(abs(ev))<=tail_target)
    mask=abs(ev)>tail_target; u=u[:,mask]; rank=u.shape[1]
    if rank:
        hu=action(u)+u; hc=u.T@hu; hc=(hc+hc.T)*.5
        eig=la.eigvalsh(hc)
        if eig[0]<=0: raise RuntimeError('nonpositive projected matrix')
        v=backbone.q(u); c=la.solve(hc,np.eye(rank),assume_a='pos')-np.eye(rank)
        defect=float(la.norm(hu-u@hc,2))
        def inverse(b): return backbone.apply(b)+v@(c@(v.T@b))
        storage=v.nbytes+c.nbytes
    else:
        eig=np.array([]); defect=0.; storage=0; inverse=backbone.apply
    return inverse,{'rank':rank,'operator_columns':calls,'setup_s':time.perf_counter()-start,
        'update_bytes':storage,'ritz_values':eig.tolist(),'invariance_residual':defect,
        'stages':stages,'tail_observed':seen,'last_ritz_magnitude':float(np.min(abs(ev))),
        'tail_target':tail_target,'cap':cap}
