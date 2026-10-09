"""Continuous-box step certificate from a lifted velocity, not a step defect.

Exact-arithmetic sufficient inequalities; floating-point evaluation is NOT
outward rounded. Small/large runs use AMG-CG, never a full eigensolve here.
See VELOCITY_DEFECT_PROOF.md. A supplied common W is only a trial Riesz space:
its discarded directions are included in the actual polynomial equation defect.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import itertools
import math
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from residual_reconstruction import Solver, operator, decay_lower, norm, f
from polynomial_trajectory import interpolation_controls, product_controls


class TaylorModalEnvelope:
    """Lazy parameter controls; certified Taylor remainder for time integration.

    Avoids a squared (nodes * modes) Gram and a controls*nodes*output*modes
    tensor. Temporal quadrature integrates a polynomial exactly; exponentials
    enter only through Taylor coefficients and an explicit remainder bound.
    """
    def __init__(self,maps,coeff,vectors,rates,inputs,order=12):
        self.maps=maps;self.coeff=coeff;self.vectors=vectors
        self.rates=rates;self.inputs=inputs;self.order=order
        self.weights=np.zeros((len(coeff),*rates.shape))
        for j,M in enumerate(maps):
            projected=np.array([M@u if M.ndim==2 else M[n]@u for n,u in enumerate(vectors)])
            weights=la.norm(projected,axis=1)*la.norm(inputs,axis=2)
            self.weights+=np.abs(coeff[:,:,j,None])*weights[None,:,:]
        self.gauss,self.gauss_weights=np.polynomial.legendre.leggauss(order+1)

    def matrices(self,t,derivative=0):
        factors=f(t,self.rates) if derivative==0 else (-self.rates)**(derivative-1)*np.exp(-t*self.rates)
        base=np.einsum('npr,nr,nri->npi',self.vectors,factors,self.inputs,optimize=True)
        out=None
        for j,M in enumerate(self.maps):
            value=np.einsum('op,npi->noi',M,base,optimize=True) if M.ndim==2 else np.einsum('nop,npi->noi',M,base,optimize=True)
            term=np.einsum('cn,noi->coi',self.coeff[:,:,j],value,optimize=True)
            out=term if out is None else out+term
        return out

    def values(self,t,derivative=0):
        M=self.matrices(t,derivative)
        return np.array([norm(x) for x in M])

    def polynomial_integral(self,a,b):
        h=(b-a)/2;mid=(a+b)/2;P=[];scaled=1.
        for k in range(self.order+1):
            if k:scaled*=h/k
            P.append(self.matrices(mid,k+1)*scaled)
        poly=np.zeros((len(self.coeff),len(self.gauss),*P[0].shape[1:]))
        for x in P[::-1]:poly=poly*self.gauss[None,:,None,None]+x[:,None,:,:]
        gram=np.einsum('cqoi,cqoj,q->cij',poly,poly,self.gauss_weights/2,optimize=True)
        # v^(p+1) uses rates^(p+1); multiply by h before taking powers.
        kernel=np.exp(-a*self.rates)*(h*self.rates)**(self.order+1)/math.factorial(self.order+1)
        remainder=np.einsum('cnr,nr->c',self.weights,kernel,optimize=True)
        return gram,remainder

    def integral_controls(self,a,b):
        gram,remainder=self.polynomial_integral(a,b)
        eigen=np.array([la.eigvalsh((g+g.T)/2)[-1] for g in gram])
        return (b-a)*(np.sqrt(np.maximum(eigen,0))+remainder)

    def integral_matrix_upper(self,a,b):
        gram,remainder=self.polynomial_integral(a,b)
        out=[]
        for g,rem in zip(gram,remainder):
            size=np.sqrt(max(0.,float(la.eigvalsh((g+g.T)/2)[-1])))
            theta=max(1e-12,min(1e12,float(rem/max(size,1e-300))))
            out.append((b-a)*((1+theta)*g+(1+1/theta)*rem**2*np.eye(len(g))))
        return np.array(out)

    def tail(self,t):
        return float(np.max(np.einsum('cnr,nr->c',self.weights,np.exp(-t*self.rates)/self.rates,optimize=True)))

    def limit(self):return float(np.max(self.values(np.inf)))


def control_root(blocks, weight):
    """QR retains cross terms without forming a large squared Gram matrix."""
    joined=np.column_stack([weight[:,None]*x for x in blocks])
    return la.qr(joined,mode='economic')[1]


def scalar_controls(dim, degree, polynomial):
    controls=[]
    for k in itertools.product(range(degree+1),repeat=dim):
        controls.append(sum(x*np.prod([math.comb(k[i],p[i])/math.comb(degree,p[i])
                                     for i in range(dim)])
                            for p,x in polynomial.items() if all(p[i]<=k[i] for i in range(dim))))
    return np.asarray(controls)


def static_norm(root, polynomials, dim, r):
    coeff=np.column_stack([scalar_controls(dim,2,p) for p in polynomials])
    return max(norm(sum((x*root[:,j*r:(j+1)*r] for j,x in enumerate(row)),
                        np.zeros((root.shape[0],r)))) for row in coeff)


def modal_envelope(root, polynomials, dim, degree, target, vectors, rates, inputs):
    r=vectors.shape[-1]
    coeff=np.stack([product_controls(dim,degree,target,p) for p in polynomials],axis=2)
    return TaylorModalEnvelope([root[:,j*r:(j+1)*r] for j in range(len(polynomials))],coeff,vectors,rates,inputs)


def certify_velocity(K,C,G,H,V,ranges,degree=1,time_ratio=1.2,W=None):
    start=time.perf_counter();c=C.diagonal();dim=len(H);r=V.shape[1]
    if not 1<time_ratio<=1.3 or degree<1:raise ValueError('invalid grid/degree')
    if np.any(c<=0) or (C-sp.diags(c)).nnz or np.any(ranges[:,0]<=0):raise ValueError('positive diagonal mass/parameters required')
    if np.any(ranges[:,1]<ranges[:,0]):raise ValueError('ordered parameter box required')
    asym=K-K.T;off=K-sp.diags(K.diagonal())
    if (asym.nnz and np.max(np.abs(asym.data))>1e-12) or np.any(off.data>0):raise ValueError('symmetric M-matrix required')
    if norm((K@np.ones(len(c)))[:,None])>1e-8*max(1.,norm(K.diagonal()[:,None])):raise ValueError('conservative conduction required')
    if norm(V.T@(c[:,None]*V)-np.eye(r))>1e-8:raise ValueError('C orthonormal V required')
    for J in H:
        if np.any(J.diagonal()<0) or (J-sp.diags(J.diagonal())).nnz:
            raise ValueError('PSD Robin operators required')
    mid=np.mean(ranges,axis=1);width=ranges[:,1]-ranges[:,0]
    Ac=operator(K,H,mid);solver=Solver(Ac)
    alpha=decay_lower(Ac,c,solver)
    rho=float(np.max(width/(2*mid)));alphabox=(1-rho)*alpha
    A=(V.T@(Ac@V));A=(A+A.T)/2;Ai=[V.T@(J@V) for J in H]
    B=V.T@G;l,U=la.eigh(B.T@B)
    if l[0]<=0:raise ValueError('independent represented inputs required')
    change=(U/np.sqrt(l))@U.T;B=B@change;G=G@change
    R0=G-c[:,None]*(V@B);r0=norm(R0/np.sqrt(c)[:,None])
    D=Ac@V-c[:,None]*(V@A)
    Di=[J@V-c[:,None]*(V@a) for J,a in zip(H,Ai)]
    if W is None:
        solve=solver.solve
    else:
        if norm(W.T@(c[:,None]*W)-np.eye(W.shape[1]))>1e-8:raise ValueError('C orthonormal W required')
        aw=W.T@(Ac@W)
        solve=lambda rhs:W@la.solve(aw,W.T@rhs,assume_a='pos')
    T0=solve(D);Z=[solve(d-J@T0) if w>0 else np.zeros_like(T0) for d,J,w in zip(Di,H,width)]
    zero=(0,)*dim;one={zero:1.}
    delta=[{zero:-width[i]/2,tuple(int(j==i) for j in range(dim)):width[i]} for i in range(dim)]
    tpolys=[one,*delta];fpolys=[one,*delta]
    defects=[D-Ac@T0,*[d-J@T0-Ac@z for d,J,z in zip(Di,H,Z)]]
    for i,J in enumerate(H):
        for j,z in enumerate(Z):
            poly={}
            for p,x in delta[i].items():
                for q,y in delta[j].items():
                    k=tuple(a+b for a,b in zip(p,q));poly[k]=poly.get(k,0)-x*y
            fpolys.append(poly);defects.append(J@z)
    lifts=[x for x,p in zip([T0,*Z],tpolys) if any(p.values())]
    tpolys=[p for p in tpolys if any(p.values())]
    defects=[x for x,p in zip(defects,fpolys) if any(p.values())]
    fpolys=[p for p in fpolys if any(p.values())]
    tcroot=control_root(lifts,np.sqrt(c))
    joined=np.column_stack(lifts);gram=joined.T@(Ac@joined)
    l,U=la.eigh((gram+gram.T)/2);tkroot=np.sqrt(np.maximum(l,0))[:,None]*U.T
    froot=control_root(defects,1/np.sqrt(c))
    fsup=static_norm(froot,fpolys,dim,r)
    ysup=static_norm(tcroot,tpolys,dim,r)+fsup/alphabox
    yksup=np.sqrt(1+rho)*static_norm(tkroot,tpolys,dim,r)+fsup/np.sqrt(alphabox)
    nodes,_=interpolation_controls(dim,degree,degree+2)
    nodeA=[A+sum(((x-.5)*w*a for x,w,a in zip(node,width,Ai)),np.zeros_like(A)) for node in nodes]
    eig=[la.eigh((a+a.T)/2) for a in nodeA]
    rates=np.array([v[0] for v in eig]);vectors=np.array([v[1] for v in eig]);inputs=np.array([u.T@B for u in vectors])
    low=V.T@(operator(K,H,ranges[:,0])@V);high=V.T@(operator(K,H,ranges[:,1])@V)
    beta=float(la.eigvalsh((low+low.T)/2)[0]);bmax=float(la.eigvalsh((high+high.T)/2)[-1])
    trial=modal_envelope(tcroot,tpolys,dim,degree,degree+1,vectors,rates,inputs)
    ferr=modal_envelope(froot,fpolys,dim,degree,degree+2,vectors,rates,inputs)
    base=product_controls(dim,degree,degree+1,one)
    qcoeff=np.stack([base,*[product_controls(dim,degree,degree+1,p) for p in delta]],axis=2)
    qenv=TaylorModalEnvelope([np.array([A-a for a in nodeA]),*Ai],qcoeff,vectors,rates,inputs)
    center_rates,center_vectors=la.eigh(A);center_inputs=center_vectors.T@B
    da=max(norm(sum(((x-.5)*w*a for x,w,a in zip(v,width,Ai)),np.zeros_like(A)))
           for v in itertools.product([0.,1.],repeat=dim))
    def denom(t,end=None):
        rational=t*(B.T@la.solve(np.eye(r)+t*high,B,assume_a='pos'))
        lower=float(la.eigvalsh((rational+rational.T)/2)[0])
        center=float(la.svdvals(f(t,center_rates)[:,None]*center_inputs)[-1])
        drift=da/(beta*center_rates[0]) if end is None else da*float(f(end,beta))*float(f(end,center_rates[0]))
        return max(lower,center-drift)
    dmax=max(norm((D+sum(((x-.5)*w*d for x,w,d in zip(v,width,Di)),np.zeros_like(D)))/np.sqrt(c)[:,None])
             for v in itertools.product([0.,1.],repeat=dim))
    tiny=1e-7/bmax;end=35/min(rates[:,0])
    grid=np.r_[0.,np.geomspace(tiny,end,int(np.ceil(np.log(end/tiny)/np.log(time_ratio)))+1)]
    it=iferr=iq=0.;maximum=(tiny*dmax/2+r0)*np.exp(bmax*tiny);worst=tiny;parts={}
    for a,b in zip(grid[:-1],grid[1:]):
        ct=float(np.max(trial.integral_controls(a,b)))
        cf=float(np.max(ferr.integral_controls(a,b)))
        cq=float(np.max(qenv.integral_controls(a,b)))
        if a>0:
            p=float(np.max(trial.values(a)))+it+2*ct
            fp=2*(iferr+cf)/alphabox;qp=2*ysup*(iq+cq)/beta
            value=(p+fp+qp+min(b*r0,r0/alphabox))/denom(a,b)
            if value>maximum:
                maximum=value;worst=a;parts=dict(lift=p,lift_defect=fp,velocity_defect=qp,denominator=denom(a,b))
        it+=ct;iferr+=cf;iq+=cq
    itinf=it+trial.tail(end);ifinf=iferr+ferr.tail(end);iqinf=iq+qenv.tail(end)
    tailparts=dict(lift=trial.limit()+trial.tail(end)+itinf,
                   lift_defect=2*ifinf/alphabox,velocity_defect=2*ysup*iqinf/beta,
                   denominator=denom(end))
    tail=(sum(tailparts[k] for k in ['lift','lift_defect','velocity_defect'])+r0/alphabox)/tailparts['denominator']
    if tail>maximum:maximum=tail;worst=end;parts=tailparts
    steady=modal_envelope(tkroot,tpolys,dim,degree,degree+1,vectors,rates,inputs)
    stationary_gram=B.T@la.solve(high,B,assume_a='pos')
    sd=np.sqrt(float(la.eigvalsh((stationary_gram+stationary_gram.T)/2)[0]))
    # Galerkin K-orthogonality gives the exact error/ROM -> error/truth conversion.
    sn=(np.sqrt(1+rho)*steady.limit()+ferr.limit()/np.sqrt(alphabox)
        +yksup*iqinf/beta+r0/np.sqrt(alphabox))
    sr=sn/sd
    return dict(n=len(c),order=r,error_space_order=0 if W is None else W.shape[1],
                bound=float(maximum/(1-maximum)) if maximum<1 else None,
                error_over_rom_bound=float(maximum),steady_bound=float(sr/np.hypot(1,sr)),
                steady_error_over_rom_bound=float(sr),full_decay_lower=alphabox,reduced_decay_lower=beta,
                relative_form_radius=rho,static_lift_error_operator_bound=float(fsup/alphabox),
                inverse_residual_lift_operator_bound=float(ysup),worst_time=float(worst),worst_terms=parts,
                integrated_lift_velocity=float(itinf),integrated_lift_defect_velocity=float(ifinf),
                integrated_velocity_equation_defect=float(iqinf),infinite_tail_over_rom=float(tail),
                degree=degree,time_intervals=len(grid)-1,counts=solver.counts(),seconds=time.perf_counter()-start,
                supplied_parameter_box=np.asarray(ranges).tolist(),full_direct_factorizations=0,
                full_eigendecompositions=0,floating_point_certified=False,
                continuous_parameter_inequalities=True,passed=bool(maximum<=np.sqrt(.001)/(1+np.sqrt(.001)) and sr/np.hypot(1,sr)<=.001))
