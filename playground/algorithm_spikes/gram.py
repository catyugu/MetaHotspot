"""Data-only shared-PSD-Gram factor feasibility versus direct positive pencils.

C is whitened to I. Both methods use K0=L0 L0^T+1e-6 I and K1=L1 L1^T.
The kernel method introduces a shared latent response Y, fitting
 diag(s) Y^T Y + Y^T K0 Y + diag(mu) Y^T K1 Y = 1 h^T
and Y^T f=h. It is inverse-free during fitting. It is NOT a convex completion,
not a minimum-rank certificate, and no global optimization claim is made.
Direct fitting uses exactly the same pencil parameterization, starts and data.
"""
from __future__ import annotations
import itertools
import time
import numpy as np
from scipy import linalg as la
from scipy.optimize import least_squares
from common import fingerprint, sym

FLOOR = 1e-6


def response(k0, k1, b, s, mu, c=None):
    c = np.eye(len(b)) if c is None else c
    return np.array([b@la.solve(z*c+k0+p*k1, b, assume_a='pos') for z, p in zip(s, mu)])


def witness_grams(k0, k1, b, s, mu):
    y = np.column_stack([la.solve(z*np.eye(len(b))+k0+p*k1, b, assume_a='pos') for z,p in zip(s,mu)])
    return y.T@y, y.T@k0@y, y.T@k1@y, y.T@b


def gram_residual(e, g0, g1, s, mu, h):
    return s[:,None]*e+g0+mu[:,None]*g1-np.ones((len(s),1))*h[None,:]


def diagonal_counterexample():
    k0 = np.array([[2., -.3],[-.3, 1.5]])
    k1 = np.diag([1.3, .8]); c = np.eye(2); j = np.diag([.4, .2]); b = np.array([1., .4])
    z = np.geomspace(.1, 10., 20)
    same1 = response(k0,k1,b,z,z,c)
    same2 = response(k0,k1-j,b,z,z,c+j)
    s = np.tile(z, 3); mu = np.repeat([0.,.4,1.],len(z))
    h1 = response(k0,k1,b,s,mu,c); h2 = response(k0,k1-j,b,s,mu,c+j)
    return {'diagonal_max_difference':float(np.max(np.abs(same1-same2))),
            'off_diagonal_max_relative_difference':float(np.max(np.abs(h1-h2)/h1))}


class PencilFit:
    def __init__(self, rank, s, mu, h, method):
        if method not in ('direct','kernel'):
            raise ValueError('unknown objective')
        self.r = rank; self.s = np.asarray(s); self.mu = np.asarray(mu); self.h = np.asarray(h)
        if np.any(self.s <= 0) or np.any(self.mu < 0) or np.any(self.h <= 0):
            raise ValueError('positive real SISO samples and nonnegative parameters required')
        self.method = method; self.indices = np.tril_indices(rank); self.t = len(self.indices[0])
        self.base_size = 2*self.t+rank; self.m = len(s)
        self.hscale = np.maximum(np.abs(h), .01*np.median(np.abs(h)))
        self.denom = (s+1.+mu)[:,None]*self.hscale[None,:]
        self.fun_calls = 0; self.jac_calls = 0

    def unpack(self, x):
        l0 = np.zeros((self.r,self.r)); l1 = np.zeros_like(l0)
        l0[self.indices] = x[:self.t]; l1[self.indices] = x[self.t:2*self.t]
        b = x[2*self.t:self.base_size]
        y = x[self.base_size:].reshape(self.r,self.m) if self.method == 'kernel' else None
        return l0@l0.T+FLOOR*np.eye(self.r), l1@l1.T, b, y

    def initial(self, seed):
        rng = np.random.default_rng(seed)
        l0 = np.diag(np.sqrt(np.geomspace(.3, 8., self.r)))
        l1 = np.diag(np.sqrt(np.geomspace(.2, 4., self.r)))
        l0 += np.tril(rng.normal(scale=.15,size=l0.shape),-1)
        l1 += np.tril(rng.normal(scale=.15,size=l1.shape),-1)
        b = rng.normal(size=self.r); b /= la.norm(b)
        k0, k1 = l0@l0.T+FLOOR*np.eye(self.r), l1@l1.T
        pred = response(k0,k1,b,self.s,self.mu)
        b *= np.sqrt(np.median(self.h/pred))
        x = np.r_[l0[self.indices],l1[self.indices],b]
        if self.method == 'kernel':
            y = np.column_stack([la.solve(z*np.eye(self.r)+k0+p*k1,b,assume_a='pos') for z,p in zip(self.s,self.mu)])
            x = np.r_[x,y.ravel()]
        return x

    def fun(self,x):
        self.fun_calls += 1
        k0,k1,b,y = self.unpack(x)
        if self.method == 'direct':
            return (response(k0,k1,b,self.s,self.mu)-self.h)/self.hscale
        e,g0,g1 = y.T@y,y.T@k0@y,y.T@k1@y
        g = gram_residual(e,g0,g1,self.s,self.mu,self.h)/self.denom
        port = np.sqrt(self.m)*(y.T@b-self.h)/self.hscale
        return np.r_[g.ravel(),port]

    def jac(self,x):
        self.jac_calls += 1
        k0,k1,b,y = self.unpack(x)
        l0=np.zeros((self.r,self.r)); l1=np.zeros_like(l0)
        l0[self.indices]=x[:self.t]; l1[self.indices]=x[self.t:2*self.t]
        if self.method == 'direct':
            z=np.column_stack([la.solve(s*np.eye(self.r)+k0+p*k1,b,assume_a='pos') for s,p in zip(self.s,self.mu)])
            j=np.empty((self.m,self.base_size))
            for block,l in enumerate((l0,l1)):
                for k,(a,c) in enumerate(zip(*self.indices)):
                    dl=np.zeros_like(l); dl[a,c]=1.
                    dk=dl@l.T+l@dl.T
                    val=-np.sum(z*(dk@z),axis=0)
                    if block: val*=self.mu
                    j[:,block*self.t+k]=val/self.hscale
            j[:,2*self.t:self.base_size]=2*z.T/self.hscale[:,None]
            return j
        j=np.zeros((self.m*self.m+self.m,len(x)))
        for block,l in enumerate((l0,l1)):
            for k,(a,c) in enumerate(zip(*self.indices)):
                dl=np.zeros_like(l); dl[a,c]=1.
                dk=dl@l.T+l@dl.T
                dg=y.T@dk@y
                if block: dg=self.mu[:,None]*dg
                j[:self.m*self.m,block*self.t+k]=(dg/self.denom).ravel()
        j[self.m*self.m:,2*self.t:self.base_size]=np.sqrt(self.m)*y.T/self.hscale[:,None]
        for a in range(self.r):
            for c in range(self.m):
                dy=np.zeros_like(y); dy[a,c]=1.
                de=dy.T@y+y.T@dy
                dg0=dy.T@k0@y+y.T@k0@dy
                dg1=dy.T@k1@y+y.T@k1@dy
                dg=self.s[:,None]*de+dg0+self.mu[:,None]*dg1
                col=self.base_size+a*self.m+c
                j[:self.m*self.m,col]=(dg/self.denom).ravel()
                j[self.m*self.m:,col]=np.sqrt(self.m)*(dy.T@b)/self.hscale
        return j

    def predict(self,x,s,mu):
        k0,k1,b,_=self.unpack(x)
        return response(k0,k1,b,s,mu)

    def solve(self,seed,max_nfev=200):
        self.fun_calls=self.jac_calls=0
        start=time.perf_counter()
        result=least_squares(self.fun,self.initial(seed),jac=self.jac,method='trf',
                             x_scale='jac',max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        return {'x':result.x,'seconds':time.perf_counter()-start,'objective':float(result.cost),
                'status':int(result.status),'nfev':int(result.nfev),'njev':int(result.njev or 0),
                'actual_fun_calls':self.fun_calls,'actual_jac_calls':self.jac_calls,
                'optimality':float(result.optimality)}


def make_truth(seed,n=8):
    rng=np.random.default_rng(seed)
    q=la.qr(rng.normal(size=(n,n)))[0]
    k0=sym((q*np.geomspace(.3,12.,n))@q.T)
    q=la.qr(rng.normal(size=(n,n)))[0]
    k1=sym((q*np.geomspace(.1,8.,n))@q.T)
    b=rng.normal(size=n); b/=la.norm(b)
    return k0,k1,b


def run(seed,smoke=False):
    k0,k1,b=make_truth(seed,5 if smoke else 8)
    rng=np.random.default_rng(seed+3001)
    test_s=np.exp(rng.uniform(np.log(.07),np.log(15.),180))
    test_mu=rng.uniform(0.,1.,180)
    test_h=response(k0,k1,b,test_s,test_mu)
    rows=[]; audits=[]
    layouts=['grid'] if smoke else ['grid','diagonal']
    noises=[0.] if smoke else [0.,1e-4]
    ranks=[2] if smoke else [2,4]
    for layout,noise,rank in itertools.product(layouts,noises,ranks):
        if layout=='grid':
            s=np.tile(np.geomspace(.1,10.,6),4); mu=np.repeat([0.,1./3.,2./3.,1.],6)
        else:
            s=np.geomspace(.1,10.,24); mu=s/10.
        exact=response(k0,k1,b,s,mu)
        h=exact*(1.+noise*np.random.default_rng(seed+5001).normal(size=len(s)))
        for method in ['direct','kernel']:
            fit=PencilFit(rank,s,mu,h,method)
            starts=[fit.solve(seed+100+start,max_nfev=60 if smoke else 200) for start in range(1 if smoke else 2)]
            # Only the method's own training objective is used to pick a start.
            best=min(starts,key=lambda z:z['objective'])
            x=best['x']; kh0,kh1,bh,y=fit.unpack(x)
            pred=fit.predict(x,test_s,test_mu); train=fit.predict(x,s,mu)
            err=np.abs(pred-test_h)/test_h
            row={'layout':layout,'noise':noise,'rank':rank,'method':method,
                 'train_relative_rms':float(np.sqrt(np.mean(((train-h)/h)**2))),
                 'holdout_relative_rms':float(np.sqrt(np.mean(err**2))),
                 'holdout_relative_max':float(np.max(err)),
                 'offline_all_starts_s':float(sum(z['seconds'] for z in starts)),
                 'all_nfev':sum(z['nfev'] for z in starts),'all_njev':sum(z['njev'] for z in starts),
                 'fitted_variables':len(x),'minimum_k0_eigenvalue':float(la.eigvalsh(kh0)[0]),
                 'minimum_k1_eigenvalue':float(la.eigvalsh(kh1)[0]),
                 'training_samples':len(s),'selected_status':best['status'],
                 'selected_objective':best['objective'],'selected_optimality':best['optimality'],
                 'latent_condition':float(np.linalg.cond(y)) if y is not None else None,
                 'model_bytes':kh0.nbytes+kh1.nbytes+bh.nbytes}
            if y is not None:
                row['relative_latent_equation_defect']=float(la.norm(np.column_stack([
                    (ss*np.eye(rank)+kh0+mm*kh1)@y[:,i]-bh for i,(ss,mm) in enumerate(zip(s,mu))]))/
                    max(la.norm(bh)*np.sqrt(len(s)),1e-30))
            rows.append(row)
            audits.append({'layout':layout,'noise':noise,'rank':rank,'method':method,
                           'sample_sha256':fingerprint(s,mu,h),'s':s.tolist(),'mu':mu.tolist(),'h':h.tolist(),
                           'K0':kh0.tolist(),'K1':kh1.tolist(),'b':bh.tolist(),
                           'starts':[{k:v for k,v in z.items() if k!='x'} for z in starts]})
            print(f'gram {layout=} {noise=} {rank=} {method=} error={row["holdout_relative_max"]:.3g}',flush=True)
    return rows,{'data_only_fit':True,'true_state_count':len(b),'starts':1 if smoke else 2,
                 'per_start_nfev_cap':60 if smoke else 200,'test_sha256':fingerprint(test_s,test_mu),
                 'sample_fit_audits':audits,'diagonal_counterexample':diagonal_counterexample(),
                 'scope':'fixed-rank nonconvex feasibility, not a full positive-kernel realization algorithm',
                 'holdout_used_for_selection':False}
