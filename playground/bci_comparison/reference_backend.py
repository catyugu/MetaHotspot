"""Memory-bounded reference solves; never an alternative ROM speed baseline."""
import numpy as np
import scipy.sparse.linalg as sla


def cg_solve(A,B,M,*,x0=None,rtol=1e-10):
    B=np.asarray(B,dtype=float); vector=B.ndim==1
    if vector: B=B[:,None]
    X=np.zeros_like(B); iterations=[0]; worst=0.
    if x0 is not None:
        x0=np.asarray(x0).reshape(B.shape)
    def count(_x): iterations[0]+=1
    for j in range(B.shape[1]):
        norm=np.linalg.norm(B[:,j])
        if norm==0.: continue
        X[:,j],flag=sla.cg(A,B[:,j],M=M,x0=None if x0 is None else x0[:,j],rtol=rtol,atol=0.,maxiter=3000,callback=count)
        residual=float(np.linalg.norm(A@X[:,j]-B[:,j])/norm)
        worst=max(worst,residual)
        if flag!=0 or residual>1e-9:
            raise RuntimeError(f'fine-grid reference CG failed: flag={flag}, residual={residual}')
    return (X[:,0] if vector else X),{'iterations':iterations[0],'relative_residual':worst}


class LinearSolver:
    def __init__(self,A):
        self.A=A.tocsc(); self.calls=0; self.iterations=0; self.residual=0.
        self.backend='sparse_LU' if A.shape[0]<=50000 else 'AMG_CG_rtol_1e-10'
        if self.backend=='sparse_LU':
            self.factor=sla.splu(self.A)
        else:
            from metahotspot.macromodel import utils as stock
            self.M=stock._rs_preconditioner(self.A.tocsr())

    def solve(self,B,x0=None):
        self.calls+=1
        if self.backend=='sparse_LU': return self.factor.solve(B)
        X,stats=cg_solve(self.A,B,self.M,x0=x0)
        self.iterations+=stats['iterations']; self.residual=max(self.residual,stats['relative_residual'])
        return X
