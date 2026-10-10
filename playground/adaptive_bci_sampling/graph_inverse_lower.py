"""Full-rank block graph lower operator, with no neglected spatial tail.

Retain within-partition edges of the exact ground-state Laplacian A_min-D.
Dropping edges subtracts a PSD matrix, so B <= A(h) throughout the Robin box.
Small block Cholesky factors whiten residuals in the rigorous B^{-1} norm.
"""
import time
import numpy as np
import scipy.linalg as la


class GraphInverseLower:
    def __init__(self, A, z, D, labels):
        start=time.perf_counter()
        z=np.asarray(z); labels=np.asarray(labels)
        if np.any(z<=0) or np.any(D<=0) or len(labels)!=len(z):
            raise ValueError('positive witness, diagonal, and full partition required')
        entries=A.tocoo()
        keep=(entries.row<entries.col)
        i,j,v=entries.row[keep],entries.col[keep],entries.data[keep]
        if np.any(v>0):
            raise ValueError('nonpositive off-diagonal entries required')
        retain=labels[i]==labels[j]
        i,j,v=i[retain],j[retain],v[retain]
        diagonal=np.asarray(D).copy()
        np.add.at(diagonal,i,-v*z[j]/z[i])
        np.add.at(diagonal,j,-v*z[i]/z[j])
        order=np.argsort(labels,kind='stable')
        sorted_labels=labels[order]
        starts=np.r_[0,np.flatnonzero(np.diff(sorted_labels))+1,len(z)]
        self.blocks=[]
        position=np.empty(len(z),int)
        block_index=np.empty(len(z),int)
        for k,(a,b) in enumerate(zip(starts[:-1],starts[1:])):
            ids=order[a:b]
            position[ids]=np.arange(len(ids));block_index[ids]=k
            self.blocks.append([ids,np.diag(diagonal[ids])])
        for left,right,value in zip(i,j,v):
            matrix=self.blocks[block_index[left]][1]
            matrix[position[left],position[right]]=value
            matrix[position[right],position[left]]=value
        self.matrices=[matrix.copy() for _,matrix in self.blocks]
        for block in self.blocks:
            block[1]=la.cholesky(block[1],lower=True,check_finite=False)
        self.statistics=dict(blocks=len(self.blocks),max_block=max(len(ids) for ids,_ in self.blocks),
                             retained_edges=len(i),factor_seconds=time.perf_counter()-start,
                             whiten_calls=0,whiten_rhs_columns=0,whiten_seconds=0.)
        self.robin=None

    def prepare_robin(self,H,G,hmin,shape,degree):
        """A projected positive Robin term is a LOWER operator, not a fitted tail."""
        start=time.perf_counter()
        ix,iy,iz=np.indices(shape)
        x=(ix+.5)/shape[0];y=(iy+.5)/shape[1]
        raw=[(np.cos(np.pi*k*x)*np.cos(np.pi*l*y)).ravel()
             for k in range(degree+1) for l in range(degree+1)]
        for g in G.T:
            footprint=np.any(g.reshape(shape)!=0,axis=2)
            raw.append(np.broadcast_to(footprint[:,:,None],shape).ravel())
        raw=np.column_stack(raw)
        factors=[];parameters=[]
        for j,A in enumerate(H):
            diag=A.diagonal();active=np.flatnonzero(diag>0)
            if not len(active):
                continue
            Q,R,p=la.qr(raw[active],mode='economic',pivoting=True,check_finite=False)
            rank=int(np.sum(np.abs(np.diag(R))>np.finfo(float).eps*max(raw[active].shape)*abs(R[0,0])))
            F=np.zeros((len(diag),rank));F[active]=np.sqrt(diag[active])[:,None]*Q[:,:rank]
            factors.append(F);parameters.extend([j]*rank)
        F=np.column_stack(factors)
        W=self.whiten(F)
        Q,T=la.qr(W,mode='economic',check_finite=False)
        self.robin=dict(Q=Q,T=T,parameters=np.asarray(parameters),hmin=np.asarray(hmin))
        self.statistics.update(robin_rank=F.shape[1],robin_prepare_seconds=time.perf_counter()-start)

    def robin_cholesky(self,h):
        info=self.robin;delta=np.asarray(h)-info['hmin']
        if np.any(delta<0):
            raise ValueError('query outside the certified Robin lower endpoints')
        T=info['T'];weights=delta[info['parameters']]
        return la.cholesky(np.eye(len(T))+(T*weights)@T.T,lower=True,check_finite=False)

    def whiten(self,R):
        start=time.perf_counter(); output=np.empty_like(R)
        for ids,L in self.blocks:
            output[ids]=la.solve_triangular(L,R[ids],lower=True,check_finite=False)
        self.statistics['whiten_calls']+=1
        self.statistics['whiten_rhs_columns']+=R.shape[1]
        self.statistics['whiten_seconds']+=time.perf_counter()-start
        return output


def column_partition(shape,width):
    if width<1:
        raise ValueError('positive horizontal block width required')
    ix,iy,iz=np.indices(shape)
    return ((ix//width)*int(np.ceil(shape[1]/width))+iy//width).ravel()
