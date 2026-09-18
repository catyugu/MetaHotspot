"""Source and comparison contracts, always using temperature RISE."""
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
