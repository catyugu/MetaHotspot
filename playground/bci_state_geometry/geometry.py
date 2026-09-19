"""State-space geometry experiments, not changes to FANTASTIC.

All matrix solves here are in an already extracted parent ROM. The parent
extraction remains a charged prerequisite. Positive-real-frequency sampled
losses are not H-infinity norms or continuous-parameter error certificates.
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as la
from scipy.special import logsumexp


def orth(X):
    X=np.asarray(X,dtype=float)
    if not X.size: return np.empty((X.shape[0],0))
    U,s,_=la.svd(X,full_matrices=False,check_finite=False)
    return U[:,s>max(X.shape)*np.finfo(float).eps*s[0]]


def whiten(C):
    L=la.cholesky(C,lower=True,check_finite=False)
    return la.solve_triangular(L.T,np.eye(len(C)),lower=False,check_finite=False)


def project(C,K,F,W):
    return W.T@C@W,W.T@K@W,W.T@F


def make_bank(stiffnesses,F,shifts):
    bank=[]
    for ih,K in enumerate(stiffnesses):
        for shift in shifts:
            A=K+float(shift)*np.eye(len(K))
            X=la.solve(A,F,assume_a='pos',check_finite=False)
            Z=F.T@X; scale=1./np.sqrt(np.maximum(np.diag(Z),1e-300))
            B=F*scale; X=X*scale; Z=B.T@X
            bank.append({'A':A,'B':B,'X':X,'Z':(Z+Z.T)/2,
                         'h_id':ih,'shift':float(shift)})
    return bank


def pod_basis(bank,fixed):
    """Conventional normalized-response POD control; keep the constant mode."""
    fixed=np.asarray(fixed).reshape(-1,1); fixed=fixed/la.norm(fixed)
    X=np.column_stack([b['X']/np.maximum(la.norm(b['X'],axis=0),1e-300) for b in bank])
    X=X-fixed@(fixed.T@X)
    U=orth(X)[:,:len(fixed)-1]
    # Unpivoted QR preserves every leading POD span; re-SVD of orthogonal U
    # would scramble its equal singular values and destroy rank ordering.
    U,_=la.qr(U-fixed@(fixed.T@U),mode='economic',check_finite=False)
    return np.column_stack((fixed,U))


def loss_gradient(W,bank,tau):
    """Smooth maximum normalized port-impedance deficit and exact gradient.

    Z - B^T W(W^T A W)^-1 W^T B = E^T A E >= 0.
    Each port is normalized by its parent self impedance at that sample.
    tau*log(sum(exp(lambda/tau))) smooths across every sample and port
    eigendirection. This is a finite-bank objective, not a global theorem.
    """
    if tau<=0: raise ValueError('tau must be positive')
    records=[]; values=[]
    for b in bank:
        A,B=b['A'],b['B']; AW=A@W
        Y=la.solve(W.T@AW,W.T@B,assume_a='pos',check_finite=False)
        D=b['Z']-(B.T@W)@Y; D=(D+D.T)/2
        lam,U=la.eigh(D,check_finite=False)
        records.append((B-AW@Y,Y,U)); values.extend(lam)
    values=np.asarray(values); logden=logsumexp(values/tau)
    weights=np.exp(values/tau-logden)
    grad=np.zeros_like(W); offset=0
    for R,Y,U in records:
        m=U.shape[1]; S=(U*weights[offset:offset+m])@U.T; offset+=m
        grad-=2*(R@S)@Y.T
    return float(tau*logden),grad,float(np.max(values))


def optimize_space(W,bank,fixed,*,iterations=10,temperatures=(.05,.01,.002)):
    """Fixed-rank Grassmann descent with constant-mode constraint and Armijo.

    A conventional POD initialization and the same sample bank are explicit
    controls. No method receives validation solutions or parameter cases.
    """
    W=np.asarray(W).copy(); fixed=np.asarray(fixed).reshape(-1,1)
    fixed=fixed/la.norm(fixed); W[:,0]=fixed[:,0]; history=[]
    for tau in temperatures:
        f,g,worst=loss_gradient(W,bank,tau)
        history.append({'tau':tau,'iteration':0,'loss':f,'worst':worst})
        for step in range(iterations):
            tangent=g-W@(W.T@g); tangent[:,0]=0.
            norm2=float(np.sum(tangent*tangent))
            if norm2<1e-22: break
            alpha=min(1./np.sqrt(norm2),100.); accepted=False
            for backtrack in range(14):
                tail=W[:,1:]-alpha*tangent[:,1:]
                tail-=fixed@(fixed.T@tail)
                Q,_=la.qr(tail,mode='economic',check_finite=False)
                candidate=np.column_stack((fixed,Q))
                fc,gc,wc=loss_gradient(candidate,bank,tau)
                if fc<=f-1e-4*alpha*norm2:
                    W,f,g,worst=candidate,fc,gc,wc; accepted=True; break
                alpha*=.5
            history.append({'tau':tau,'iteration':step+1,'loss':f,'worst':worst,
                            'backtracks':backtrack,'accepted':accepted})
            if not accepted: break
    return W,history


def _with_fixed(X,fixed,rank):
    fixed=fixed.reshape(-1,1)/la.norm(fixed)
    U=orth(X-fixed@(fixed.T@X))
    return np.column_stack((fixed,U[:,:rank-1]))


def query_space(K,F,POD,rank,fixed,kind):
    """Parameter-local spaces with DC-exactness in the parent, no feedthrough.

    Harmonic trial columns T=V-Z D^-1 Z^T K V and L=Z D^-1 Z^T F
    span the parent static solution. The final Galerkin system retains mass
    and zero initial response; unlike algebraic fast-state elimination it
    does not invent an instantaneous temperature jump.

    The matched simple control adds K^-1 F to untransported POD columns.
    Both pay for construction and full-field decoder multiplication per HTC.
    """
    n,m=F.shape
    if rank>=n: return np.eye(n),{'construction':'identity','rank':n,'dc_defect':0.}
    if rank<=m+1: raise ValueError('rank must leave room for ports and constant')
    k=rank-m-1; V=POD[:,1:k+1]
    if kind=='harmonic':
        Q,_=la.qr(V,mode='full',check_finite=False); Z=Q[:,k:]
        D=Z.T@K@Z
        rhs=np.column_stack((Z.T@K@V,Z.T@F))
        S=la.solve(D,rhs,assume_a='pos',check_finite=False)
        T=V-Z@S[:,:k]; L=Z@S[:,k:]
        W=_with_fixed(np.column_stack((T,L)),fixed,rank)
    elif kind=='dc_augment':
        X=la.solve(K,F,assume_a='pos',check_finite=False)
        W=_with_fixed(np.column_stack((V,X)),fixed,rank)
    else: raise ValueError('unknown query-space construction')
    return W,{'construction':kind,'rank':W.shape[1]}


def balanced_space(K,F,rank,fixed):
    """Conventional local symmetric-system controllability Gramian control.

    In capacity-whitened collocated coordinates the observability and
    controllability Gramians coincide. Preserve the constant explicitly.
    """
    lam,U=la.eigh(K,check_finite=False); B=U.T@F
    gram=U@((B@B.T)/(lam[:,None]+lam[None,:]))@U.T
    c=fixed.reshape(-1,1)/la.norm(fixed); P=np.eye(len(K))-c@c.T
    d,Q=la.eigh(P@gram@P,check_finite=False)
    return _with_fixed(Q[:,np.argsort(d)[::-1]][:,:rank-1],c,rank)
