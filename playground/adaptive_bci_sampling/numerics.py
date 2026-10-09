"""Shared numerical primitives; no experiment or certificate driver imports."""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import pyamg

def f(t,l):
    return -np.expm1(-np.asarray(l)*t)/l

class Solver:
    def __init__(self,A,rtol=1e-11):
        self.A=sp.csr_matrix(A);self.rhs=0;self.iterations=0
        start=time.perf_counter()
        self.amg=pyamg.ruge_stuben_solver(self.A,interpolation='direct')
        self.M=self.amg.aspreconditioner(cycle='V')
        self.setup_seconds=time.perf_counter()-start;self.rtol=rtol
        self.solve_seconds=0.
    def solve(self,B):
        B=np.asarray(B);one=B.ndim==1
        if one:B=B[:,None]
        out=[];start=time.perf_counter()
        for b in B.T:
            self.rhs+=1
            if not np.any(b):out.append(np.zeros_like(b));continue
            def callback(x):self.iterations+=1
            x,info=sla.cg(self.A,b,M=self.M,rtol=self.rtol,atol=0.,maxiter=2000,callback=callback)
            if info:raise RuntimeError(f'CG failure {info}')
            out.append(x)
        self.solve_seconds+=time.perf_counter()-start
        X=np.column_stack(out)
        return X[:,0] if one else X
    def counts(self):
        return dict(rhs=self.rhs,cg_iterations=self.iterations,amg_setups=1,
                    setup_seconds=self.setup_seconds,solve_seconds=self.solve_seconds)

def operator(K,H,h):return K+sum((x*J for x,J in zip(h,H)),sp.csc_matrix(K.shape))

def c_basis(S,c):
    A=np.sqrt(c)[:,None]*S
    A=A/np.maximum(la.norm(A,axis=0),np.finfo(float).tiny)
    Q,R,p=la.qr(A,mode='economic',pivoting=True)
    diagonal=np.abs(np.diag(R))
    rank=int(np.sum(diagonal>np.finfo(float).eps*max(A.shape)*diagonal[0]))
    return Q[:,:rank]/np.sqrt(c)[:,None]

def decay_lower(A,c,solver):
    """Collatz lower bound, verified using actual product, NOT a Ritz minimum."""
    w=solver.solve(c)
    if np.any(w<=0):raise ValueError('positive supersolution not obtained')
    lower=float(np.min((A@w)/(c*w)))
    if lower<=0:raise ValueError('positive decay certificate not obtained')
    return lower

def sym(a):
    return (a+a.T)/2

def invroot(q):
    l,v=la.eigh(sym(q))
    if l[0]<=0:
        raise ValueError('dependent source response')
    return (v/np.sqrt(l))@v.T

def relative(e,x,metric=None):
    xx=x if metric is None else metric@x
    ee=e if metric is None else metric@e
    return float(np.sqrt(max(0.,la.eigvalsh(sym(e.T@ee),sym(x.T@xx))[-1])))

def counts(rows):
    if not rows:
        return {}
    return {k:sum(r[k] for r in rows) for k in rows[0]}
