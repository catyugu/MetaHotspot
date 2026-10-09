"""Input-driven Robin port candidate construction; no acceptance implied."""
import itertools
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

def hpoints(ranges,levels):
    for p in itertools.product(levels,repeat=len(ranges)):
        yield np.exp(np.log(ranges[:,0])+np.asarray(p)*np.log(ranges[:,1]/ranges[:,0]))

def ports(H):
    q=[];groups=[];ids=[]
    for i,h in enumerate(H):
        d=h.diagonal(); ix=np.flatnonzero(d)
        q.append(sp.csc_matrix((np.sqrt(d[ix]),(ix,np.arange(len(ix)))),shape=(len(d),len(ix))))
        groups.extend([i]*len(ix));ids.append(ix.tolist())
    return sp.hstack(q,format='csc'),np.asarray(groups),ids

def group_basis(traces,groups,rank):
    z=[];spectra=[]
    for i in np.unique(groups):
        ix=np.flatnonzero(groups==i)
        u,s,_=la.svd(traces[ix],full_matrices=False)
        r=min(rank,u.shape[1]); zz=np.zeros((len(groups),r));zz[ix]=u[:,:r]
        z.append(zz);spectra.append(s.tolist())
    return np.column_stack(z),spectra
