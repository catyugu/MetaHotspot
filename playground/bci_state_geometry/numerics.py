"""Physical contracts reused from bci_comparison; same native temperature RISE."""
import numpy as np


def effective_h(h, half_over_k):
    h=np.asarray(h,dtype=float)
    return h/(1.+h*np.asarray(half_over_k))


def split_ports(G, power, centers):
    columns=[]; powers=[]
    for j in range(G.shape[1]):
        ids=np.flatnonzero(G[:,j])
        x,y=centers[ids,:2].T
        mx=(x.min()+x.max())/2; my=(y.min()+y.max())/2
        for sx,sy in ((False,False),(False,True),(True,False),(True,True)):
            chosen=ids[((x>mx)==sx)&((y>my)==sy)]
            if not len(chosen): raise ValueError('mesh cannot resolve source subdivision')
            col=np.zeros(len(G)); col[chosen]=G[chosen,j]
            fraction=col.sum()
            columns.append(col/fraction); powers.append(power[j]*fraction)
    B=np.column_stack(columns); p=np.array(powers)
    np.testing.assert_allclose(B@p,G@power,rtol=1e-13,atol=1e-15)
    return B,p


def errors(reference, approximation, G, capacity, steady_reference, *, trajectory_normalization=False):
    """Inputs (time,cell,independent_experiment), no ambient offset.

    Normalize each source experiment by its own steady maximum rise, then
    take the worst; weak source responses cannot be hidden by a strong source.
    """
    ref=np.asarray(reference); app=np.asarray(approximation)
    denom=np.maximum(np.max(np.abs(steady_reference),axis=0),1e-14)
    delta=app-ref
    field=float(np.max(np.max(np.abs(delta),axis=(0,1))/denom))
    ports=np.einsum('np,tnm->tpm',G,ref)
    portdiff=np.einsum('np,tnm->tpm',G,delta)
    portden=np.maximum(np.max(np.abs(ports),axis=(0,1)) if trajectory_normalization else np.max(np.abs(G.T@steady_reference),axis=0),1e-14)
    junction=float(np.max(np.max(np.abs(portdiff),axis=(0,1))/portden))
    pe=np.abs(app.max(axis=1)-ref.max(axis=1))
    c=np.asarray(capacity)[None,:,None]
    return {'field_relative':field,'junction_relative':junction,
            'peak_absolute_K':float(pe.max()),
            'peak_relative':float(np.max(pe/denom)),
            'capacity_l2_relative':float(np.sqrt(np.sum(c*delta**2)/max(np.sum(c*ref**2),1e-30)))}

import time
import scipy.linalg as la
from reference_backend import LinearSolver


def timed(fn,repeats=3):
    fn()  # warmup; not counted as an online execution
    times=[]; value=None
    for _ in range(repeats):
        start=time.perf_counter(); value=fn(); times.append(time.perf_counter()-start)
    return value,float(np.median(times))


def dense_be(C,K,F,powers,dt,*,steps=None):
    """Same BDF1 equations as stock solve_rom_transient, with a dense factor."""
    n=len(C); nr=F.shape[1] if powers is None else None
    nt=steps+1 if powers is None else len(powers)
    X=np.zeros((nt,n,nr)) if powers is None else np.zeros((nt,n))
    L=la.cho_factor(C/dt+K,check_finite=False)
    for i in range(1,nt):
        rhs=C@X[i-1]/dt+(F if powers is None else F@powers[i])
        X[i]=la.cho_solve(L,rhs,check_finite=False)
    return X


def modal_be(C,K,F,powers,dt,*,steps=None):
    """Strong common baseline: diagonalize each fixed-HTC ROM once.

    Eigen-decomposition and input rotation are included in this call. Field
    decoder rotation U is returned and charged separately by the benchmark.
    This changes neither time discretization nor the physical power sequence.
    """
    lam,U=la.eigh(K,C,check_finite=False); B=U.T@F
    nt=steps+1 if powers is None else len(powers)
    X=np.zeros((nt,len(C),F.shape[1])) if powers is None else np.zeros((nt,len(C)))
    d=1/(1+dt*lam)
    for i in range(1,nt):
        rhs=X[i-1]+dt*(B if powers is None else B@powers[i])
        X[i]=d[:,None]*rhs if powers is None else d*rhs
    return X,U,B


def recover(V,Z):
    if Z.ndim==2:return (Z@V.T)[:,:,None]
    R=V@Z.transpose(1,0,2).reshape(V.shape[1],-1)
    return R.reshape(V.shape[0],Z.shape[0],Z.shape[2]).transpose(1,0,2)


def output(F,Z):
    return np.einsum('rp,trm->tpm',F,Z) if Z.ndim==3 else Z@F


def reference(K,C,G,dt,steps,powers=None):
    start=time.perf_counter(); A=(K+C/dt).tocsc(); factor=LinearSolver(A)
    setup=time.perf_counter()-start; c=C.diagonal(); m=G.shape[1] if powers is None else 1
    X=np.zeros((steps+1,K.shape[0],m)); start=time.perf_counter()
    for i in range(1,steps+1):
        rhs=c[:,None]*X[i-1]/dt+(G if powers is None else (G@powers[i])[:,None])
        X[i]=factor.solve(rhs,x0=X[i-1])
    solve=time.perf_counter()-start; residual=0.
    for i in (1,steps//2,steps):
        rhs=c[:,None]*X[i-1]/dt+(G if powers is None else (G@powers[i])[:,None])
        residual=max(residual,float(la.norm(A@X[i]-rhs)/max(la.norm(rhs),1e-30)))
    if residual>1e-9:raise RuntimeError(f'reference residual {residual}')
    return X,{'setup_seconds':setup,'solve_seconds':solve,'residual':max(residual,factor.residual),
              'backend':factor.backend,'iterations':factor.iterations}
