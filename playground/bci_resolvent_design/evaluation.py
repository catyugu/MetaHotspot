"""Common backward-Euler equations and physical rise-error normalization."""
from __future__ import annotations
import numpy as np
import scipy.linalg as la


def march_dense(C,K,F,powers,dt):
    powers=np.asarray(powers); r=len(C); experiments=powers.shape[2]
    factor=la.cho_factor(K+C/dt,check_finite=False)
    Z=np.zeros((len(powers)+1,r,experiments))
    for t,p in enumerate(powers):
        Z[t+1]=la.cho_solve(factor,C@Z[t]/dt+F@p,check_finite=False)
    return Z


def march_modal(C,K,F,powers,dt):
    lam,Q=la.eigh(K,C,check_finite=False)
    if np.min(lam)<=0: raise ValueError('closed ROM is not SPD')
    Fr=Q.T@F; inv=1/(1+dt*lam)
    Z=np.zeros((len(powers)+1,len(C),powers.shape[2]))
    for t,p in enumerate(powers): Z[t+1]=inv[:,None]*(Z[t]+dt*Fr@p)
    return Q,Z


def errors(reference,approximation,G,capacity,steady,mixed):
    """Time x cell x input. Each independent input has its OWN rise scale."""
    X=np.asarray(reference); Y=np.asarray(approximation); diff=Y-X
    denom=np.maximum(np.max(np.abs(X),axis=(0,1)) if mixed else np.max(np.abs(steady),axis=0),1e-14)
    field=float(np.max(np.max(np.abs(diff),axis=(0,1))/denom))
    port=np.einsum('np,tnm->tpm',G,X); dp=np.einsum('np,tnm->tpm',G,diff)
    pden=np.maximum(np.max(np.abs(port),axis=(0,1)) if mixed else np.max(np.abs(G.T@steady),axis=0),1e-14)
    peak=np.abs(Y.max(axis=1)-X.max(axis=1))
    c=np.asarray(capacity)[None,:,None]
    l2den=np.sqrt(np.sum(c*X*X,axis=(0,1)))
    l2num=np.sqrt(np.sum(c*diff*diff,axis=(0,1)))
    return {'field_relative':field,'junction_relative':float(np.max(np.max(np.abs(dp),axis=(0,1))/pden)),
            'peak_error_K':float(peak.max()),'peak_relative':float(np.max(peak/denom)),
            'capacity_L2_relative':float(np.max(l2num/np.maximum(l2den,1e-14)))}
