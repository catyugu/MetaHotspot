"""Common discrete equations and temperature-rise error contracts."""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


def reduced_history(C,K,F,dt,steps,powers=None):
    """BDF1 with zero initial rise, identical for every reduced model."""
    L=la.cho_factor(C/dt+K,check_finite=False)
    m=F.shape[1] if powers is None else 1
    Z=np.zeros((steps+1,C.shape[0],m))
    for i in range(1,steps+1):
        f=F if powers is None else (F@powers[i])[:,None]
        Z[i]=la.cho_solve(L,C@Z[i-1]/dt+f,check_finite=False)
    return Z


def from_steps(S,powers):
    """Exact BDF1 superposition at fixed h; no additional approximate model."""
    powers=np.array(powers,copy=True); powers[0]=0.
    increments=np.diff(powers,axis=0)
    X=np.zeros((len(powers),S.shape[1],1))
    for j in np.flatnonzero(np.any(increments!=0,axis=1)):
        # A change at index j+1 uses step responses at elapsed indices 1,...
        X[j+1:,:,0]+=np.einsum('tnp,p->tn',S[1:len(powers)-j],increments[j],optimize=True)
    return X


def recover(V,Z):
    R=V@Z.transpose(1,0,2).reshape(V.shape[1],-1)
    return R.reshape(V.shape[0],Z.shape[0],Z.shape[2]).transpose(1,0,2)


def ports(G,X):
    # Source supports are local; this measures exactly G.T X, without a costly
    # dense multiplication by predominantly zero full-order source columns.
    return np.stack([sp.csr_matrix(G.T)@x for x in X])


def measure(ref,app,G,c,steady=None):
    """Every independent source experiment and every junction channel matters.

    Unit-step field denominator: maximum steady reference rise of THAT input.
    Junction denominator: steady rise of THAT output/input pair, not strongest
    other port. For mixed loads both denominators use actual time-trace peaks.
    Absolute rises smaller than 1e-14 K are floored and counted explicitly.
    """
    delta=app-ref
    denom=np.max(np.abs(steady),axis=0) if steady is not None else np.max(np.abs(ref),axis=(0,1))
    denom=np.maximum(denom,1e-14)
    yr=ports(G,ref); dy=ports(G,delta)
    jd=np.abs(G.T@steady) if steady is not None else np.max(np.abs(yr),axis=0)
    peak=np.abs(app.max(axis=1)-ref.max(axis=1))
    c=np.asarray(c)
    # Matrix products avoid allocating another temperature-history-sized array.
    num=sum(float(c@np.sum(d*d,axis=1)) for d in delta)
    den=sum(float(c@np.sum(x*x,axis=1)) for x in ref)
    return {'field_relative':float(np.max(np.max(np.abs(delta),axis=(0,1))/denom)),
            'junction_channel_relative':float(np.max(np.max(np.abs(dy),axis=0)/np.maximum(jd,1e-14))),
            'junction_floored_channels':int(np.count_nonzero(jd<1e-14)),
            'junction_absolute_K':float(np.max(np.abs(dy))),
            'peak_absolute_K':float(peak.max()),
            'peak_relative':float(np.max(peak/denom)),
            'capacity_l2_relative':float(np.sqrt(num/max(den,1e-30)))}
