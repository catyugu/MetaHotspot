"""Independent cell-centred diffusion and an exact separable inverse factor.

These dimensionless research matrices do not replace the native 3-D assembler.
The DST/triangular factor evaluates P=Q Q.T without a dense inverse or snapshots.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse as sp
from scipy.fft import dst, idst


def assemble(k: np.ndarray, shift: float) -> sp.csr_matrix:
    k = np.asarray(k, dtype=float)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] < 2:
        raise ValueError('k must be square, with at least two cells per axis')
    if not np.all(np.isfinite(k)) or np.any(k <= 0) or not np.isfinite(shift) or shift < 0:
        raise ValueError('positive finite conductivities and nonnegative finite shift required')
    n=k.shape[0]; indices=np.arange(n*n).reshape(n,n); diag=np.full(n*n,shift)
    rows=[]; cols=[]; data=[]
    for axis in (0,1):
        s0=[slice(None),slice(None)]; s1=s0.copy()
        s0[axis]=slice(None,-1); s1[axis]=slice(1,None)
        a=k[tuple(s0)]; b=k[tuple(s1)]
        g=(2*a*b/(a+b)*n*n).ravel()
        i=indices[tuple(s0)].ravel(); j=indices[tuple(s1)].ravel()
        np.add.at(diag,i,g); np.add.at(diag,j,g)
        rows.extend([i,j]); cols.extend([j,i]); data.extend([-g,-g])
        for boundary in (0,n-1):
            sl=[slice(None),slice(None)]; sl[axis]=boundary
            ids=indices[tuple(sl)].ravel()
            np.add.at(diag,ids,2*n*n*k[tuple(sl)].ravel())
    rows.append(indices.ravel()); cols.append(indices.ravel()); data.append(diag)
    return sp.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(n*n,n*n)).tocsr()


def coefficients(n: int, family: str, seed: int):
    if n % 3:
        raise ValueError('research meshes must align with one-third layers')
    rng=np.random.default_rng(seed); phases=rng.uniform(0,2*np.pi,2)
    x,y=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n,indexing='ij')
    smooth=np.exp(.65*np.sin(2*np.pi*x)*np.cos(np.pi*y)+.35*np.cos(np.pi*x+phases[0])*np.sin(2*np.pi*y+phases[1]))
    layer=np.where(y[0]<1/3,1.,np.where(y[0]<2/3,30.,3.))
    if family=='smooth': layer=np.ones(n)
    if family not in ('smooth','layered','inclusion','constant_layers'):
        raise ValueError('unknown family')
    if family=='constant_layers': smooth=np.ones_like(smooth)
    k=smooth*layer[None,:]
    if family=='inclusion':
        inside=(x>=.25)&(x<.5)&(y>=.25)&(y<.75)
        k=k*np.where(inside,20.,1.)
    return k,layer


class Backbone:
    """Factor of D [K_layer + shift*diag(mean_x(D^-2))] D.

    DST-II diagonalizes the x direction. Per-mode y matrices are tridiagonal.
    q = D^-1 F.T L^-T and qt = L^-1 F D^-1. Row ordering is x then y.
    """
    def __init__(self, layer: np.ndarray, scale: np.ndarray, shift: float):
        self.layer=np.asarray(layer,float).copy(); self.scale=np.asarray(scale,float).copy()
        n=self.layer.size; self.n=n
        if self.scale.shape!=(n,n) or n<2 or np.any(self.layer<=0) or np.any(self.scale<=0):
            raise ValueError('invalid positive backbone coefficients')
        if not np.all(np.isfinite(self.scale)) or not np.all(np.isfinite(self.layer)) or shift<0:
            raise ValueError('invalid backbone coefficients')
        g=2*self.layer[:-1]*self.layer[1:]/(self.layer[:-1]+self.layer[1:])*n*n
        diag=np.zeros(n); diag[:-1]+=g; diag[1:]+=g
        diag[0]+=2*n*n*self.layer[0]; diag[-1]+=2*n*n*self.layer[-1]
        lx=4*n*n*np.sin(np.pi*np.arange(1,n+1)/(2*n))**2
        d=diag[None,:]+lx[:,None]*self.layer[None,:]+shift*np.mean(self.scale**-2,axis=0)[None,:]
        self.ld=np.empty((n,n)); self.sub=np.empty((n,n-1)); self.ld[:,0]=np.sqrt(d[:,0])
        for j in range(1,n):
            self.sub[:,j-1]=-g[j-1]/self.ld[:,j-1]
            self.ld[:,j]=np.sqrt(d[:,j]-self.sub[:,j-1]**2)
        self.storage_bytes=self.scale.nbytes+self.ld.nbytes+self.sub.nbytes+self.layer.nbytes

    def _reshape(self,b):
        b=np.asarray(b,float)
        if b.ndim not in (1,2) or b.shape[0]!=self.n**2:
            raise ValueError('incompatible vector/block')
        return b.reshape(self.n,self.n,-1),b.shape

    def qt(self,b):
        b,shape=self._reshape(b)
        z=dst(b/self.scale[:,:,None],type=2,axis=0,norm='ortho')
        z[:,0,:]/=self.ld[:,0,None]
        for j in range(1,self.n):
            z[:,j,:]=(z[:,j,:]-self.sub[:,j-1,None]*z[:,j-1,:])/self.ld[:,j,None]
        return z.reshape(shape)

    def q(self,b):
        b,shape=self._reshape(b); z=b.copy()
        z[:,-1,:]/=self.ld[:,-1,None]
        for j in range(self.n-2,-1,-1):
            z[:,j,:]=(z[:,j,:]-self.sub[:,j,None]*z[:,j+1,:])/self.ld[:,j,None]
        z=idst(z,type=2,axis=0,norm='ortho')/self.scale[:,:,None]
        return z.reshape(shape)

    def apply(self,b):
        return self.q(self.qt(b))


def build_backbone(k, layer, shift, kind):
    n=k.shape[0]
    if kind=='homogeneous':
        return Backbone(np.full(n,np.exp(np.mean(np.log(k)))),np.ones((n,n)),shift)
    if kind=='raw_layer':
        return Backbone(layer,np.ones((n,n)),shift)
    if kind=='scaled_uniform':
        return Backbone(np.ones(n),np.sqrt(k),shift)
    if kind=='jump_aware':
        return Backbone(layer,np.sqrt(k/layer[None,:]),shift)
    raise ValueError('unknown backbone')
