"""Bernstein parameter trajectories with certified equation defects.

All inequalities are in exact arithmetic; no outward rounding is implemented.
See POLYNOMIAL_TRAJECTORY_PROOF.md. Full solves use the existing AMG-CG.
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
from residual_reconstruction import Solver, operator, decay_lower, norm, f, gram_root


def interpolation_controls(dim, degree, target):
    if degree<1 or target<degree:
        raise ValueError('positive interpolation degree and target >= degree required')
    axis=(1-np.cos(np.linspace(0,np.pi,degree+1)))/2
    powers=[]
    for j,x in enumerate(axis):
        other=np.delete(axis,j)
        p=np.polynomial.polynomial.polyfromroots(other)/np.prod(x-other)
        powers.append(p)
    univariate=np.zeros((target+1,degree+1))
    for k in range(target+1):
        for j in range(degree+1):
            univariate[k,j]=sum(powers[j][s]*math.comb(k,s)/math.comb(target,s)
                                 for s in range(min(k,degree)+1))
    ids=list(itertools.product(range(degree+1),repeat=dim))
    controls=np.array([[np.prod([univariate[k[i],j[i]] for i in range(dim)])
                        for j in ids] for k in itertools.product(range(target+1),repeat=dim)])
    return np.array([[axis[j] for j in index] for index in ids]),controls


def product_controls(dim, degree, target, polynomial):
    """Controls for each Lagrange polynomial times a scalar power polynomial."""
    axis=(1-np.cos(np.linspace(0,np.pi,degree+1)))/2
    powers=[]
    for j,x in enumerate(axis):
        other=np.delete(axis,j)
        powers.append(np.polynomial.polynomial.polyfromroots(other)/np.prod(x-other))
    nodes=list(itertools.product(range(degree+1),repeat=dim))
    controls=list(itertools.product(range(target+1),repeat=dim))
    out=np.zeros((len(controls),len(nodes)))
    for ni,j in enumerate(nodes):
        for exponent,coefficient in polynomial.items():
            for s in itertools.product(range(degree+1),repeat=dim):
                power=tuple(s[i]+exponent[i] for i in range(dim))
                value=coefficient*np.prod([powers[j[i]][s[i]] for i in range(dim)])
                for ci,k in enumerate(controls):
                    if all(power[i]<=k[i] for i in range(dim)):
                        out[ci,ni]+=value*np.prod([math.comb(k[i],power[i])/math.comb(target,power[i]) for i in range(dim)])
    return out


class Envelope:
    """Operator norm enclosure of a Bernstein-controlled modal trajectory.

    matrices[c,node] maps a node's modal coordinates to common Euclidean output
    coordinates. Rank-one modal sums bound time derivatives on an interval.
    """
    def __init__(self, matrices, rates, inputs):
        # Identical ROMs (zero-width axes) represent the same exponential sum.
        # Combine their coefficients before a triangle inequality is applied.
        groups=[]
        for j in range(len(rates)):
            for group in groups:
                k=group[0]
                if np.array_equal(rates[j],rates[k]) and np.array_equal(inputs[j],inputs[k]):
                    group.append(j);break
            else:groups.append([j])
        matrices=np.stack([np.sum(matrices[:,g],axis=1) for g in groups],axis=1)
        rates=np.array([rates[g[0]] for g in groups]);inputs=np.array([inputs[g[0]] for g in groups])
        self.matrices=matrices;self.rates=rates;self.inputs=inputs
        self.weights=la.norm(matrices,axis=2)*la.norm(inputs,axis=2)[None,:,:]
        self.gram=None

    def prepare_integral(self):
        matrices=self.matrices.transpose(0,2,1,3).reshape(len(self.matrices),self.matrices.shape[2],-1)
        self.gram=matrices.transpose(0,2,1)@matrices
        self.flat_rates=self.rates.ravel()
        self.flat_inputs=self.inputs.reshape(-1,self.inputs.shape[-1])
        self.rate_sums=self.flat_rates[:,None]+self.flat_rates[None,:]

    def integral_controls(self,a,b):
        if self.gram is None:self.prepare_integral()
        kernel=np.exp(-self.rate_sums*a)*(-np.expm1(-self.rate_sums*(b-a)))/self.rate_sums
        # Exact integrated derivative Gram, followed only by time Cauchy–Schwarz.
        q=self.flat_inputs.T[None,:,:]@((self.gram*kernel)@self.flat_inputs)
        eigenvalues=np.array([la.eigvalsh(matrix)[-1] for matrix in (q+q.transpose(0,2,1))/2])
        return np.sqrt((b-a)*np.maximum(eigenvalues,0))

    def gram_interval(self,a,b):
        return float(np.max(self.values(a)+self.integral_controls(a,b)))

    def values(self,t,derivative=0):
        factors=f(t,self.rates) if derivative==0 else (-self.rates)**(derivative-1)*np.exp(-t*self.rates)
        # Axis ordering: control, node, output, mode; node, mode, input.
        values=np.einsum('cnor,nr,nri->coi',self.matrices,factors,self.inputs,optimize=True)
        grams=np.einsum('coi,coj->cij',values,values,optimize=True)
        return np.sqrt(np.maximum([la.eigvalsh(matrix)[-1] for matrix in grams],0))

    def derivative_upper(self,a,derivative=0):
        factors=self.rates**derivative*np.exp(-a*self.rates)
        return np.einsum('cnr,nr->c',self.weights,factors,optimize=True)

    def interval(self,a,b,derivative=0):
        return float(np.max(self.values((a+b)/2,derivative)+(b-a)/2*self.derivative_upper(a,derivative)))

    def tail(self,t):
        decay=np.exp(-t*self.rates)/self.rates
        return float(np.max(np.einsum('cnr,nr->c',self.weights,decay,optimize=True)))

    def limit(self):
        return float(np.max(self.values(np.inf)))


def certify_trajectory(K,C,G,H,V,ranges,degree=2,time_ratio=1.1,rtol=1e-11,center_denominator=True):
    """Continuous box/all-time step bound for the original fixed Galerkin ROM.

    Same physical assumptions as residual_reconstruction.certify; independent
    small ROM eigensolves at interpolation nodes never serve as acceptance tests.
    """
    start=time.perf_counter();ranges=np.asarray(ranges);dim=len(H);r=V.shape[1]
    if not 1<time_ratio<=1.3:raise ValueError('time_ratio must be in (1,1.3]')
    if dim<1 or np.any(ranges[:,0]<=0) or np.any(ranges[:,1]<ranges[:,0]):raise ValueError('positive ordered box required')
    c=C.diagonal()
    if np.any(c<=0) or (C-sp.diags(c)).nnz:raise ValueError('positive diagonal C required')
    if norm(V.T@(C@V)-np.eye(r))>1e-8:raise ValueError('C orthonormal basis required')
    off=K-sp.diags(K.diagonal())
    if norm(np.asarray(K@np.ones(len(c)))[:,None])>1e-8*max(1,norm(K.diagonal()[:,None])) or np.any(off.data>0):raise ValueError('conservative M-matrix required')
    asym=K-K.T
    if asym.nnz and np.max(np.abs(asym.data))>1e-12:raise ValueError('symmetric conduction required')
    for J in H:
        if (J-sp.diags(J.diagonal())).nnz or np.any(J.diagonal()<0):raise ValueError('nonnegative diagonal Robin terms required')
    mid=np.mean(ranges,axis=1);width=ranges[:,1]-ranges[:,0]
    Ac=operator(K,H,mid);solver=Solver(Ac,rtol);alpha=decay_lower(Ac,c,solver)
    rho=float(np.max(width/(2*mid)));alphabox=(1-rho)*alpha
    Ar=V.T@(Ac@V);Ar=(Ar+Ar.T)/2;Ai=[V.T@(J@V) for J in H]
    B=V.T@G;l,U=la.eigh(B.T@B)
    if l[0]<=0:raise ValueError('independent represented inputs required')
    change=(U/np.sqrt(l))@U.T;B=B@change;G=G@change
    R0=G-(C@V)@B;r0=norm(R0/np.sqrt(c)[:,None])
    D=Ac@V-(C@V)@Ar;Y=solver.solve(D);d0=D-Ac@Y
    Z=[];defects=[d0];zero=(0,)*dim;one={zero:1.}
    delta=[]
    for i in range(dim):
        exponent=tuple(int(j==i) for j in range(dim))
        delta.append({zero:-width[i]/2,exponent:width[i]})
    Tpolys=[one,*delta]
    Epolys=[one,*delta]
    for J,a in zip(H,Ai):
        S=J@V-(C@V)@a-J@Y;z=solver.solve(S);Z.append(z);defects.append(S-Ac@z)
    for i,J in enumerate(H):
        for j,z in enumerate(Z):
            poly={}
            for p,x in delta[i].items():
                for q,y in delta[j].items():
                    k=tuple(a+b for a,b in zip(p,q));poly[k]=poly.get(k,0)-x*y
            Epolys.append(poly);defects.append(J@z)
    nodes,_=interpolation_controls(dim,degree,degree+1)
    nodeA=[Ar+sum(((x-.5)*w*a for x,w,a in zip(node,width,Ai)),np.zeros_like(Ar)) for node in nodes]
    eig=[la.eigh((a+a.T)/2) for a in nodeA]
    rates=np.array([v[0] for v in eig]);vectors=np.array([v[1] for v in eig]);inputs=np.array([w.T@B for w in vectors])
    low=Ar-sum((w/2*a for w,a in zip(width,Ai)),np.zeros_like(Ar))
    high=Ar+sum((w/2*a for w,a in zip(width,Ai)),np.zeros_like(Ar))
    beta=float(la.eigvalsh(low)[0]);bmax=float(la.eigvalsh(high)[-1])
    center_rates,center_vectors=la.eigh(Ar);center_inputs=center_vectors.T@B
    deltaA=max(norm(sum(((x-.5)*w*a for x,w,a in zip(vertex,width,Ai)),np.zeros_like(Ar)))
               for vertex in itertools.product([0.,1.],repeat=dim))
    # Joint C Gram retains every cross term of the affine lift.
    lift=np.column_stack([Y,*Z]);root=gram_root(lift.T@(c[:,None]*lift));del lift
    coeff=np.stack([product_controls(dim,degree,degree+1,p) for p in Tpolys],axis=2)
    matrices=sum((coeff[:,:,j,None,None]*(root[:,j*r:(j+1)*r][None,None,:,:]@vectors[None,:,:,:]) for j in range(len(Tpolys))),np.zeros((len(coeff),len(nodes),root.shape[0],r)))
    trial=Envelope(matrices,rates,inputs)
    # Triangle only between polynomial defect blocks; retain all input/mode
    # directions inside each block. Actual CG defects are among these blocks.
    errors=[]
    for defect,poly in zip(defects,Epolys):
        if not any(poly.values()):continue
        coeff=product_controls(dim,degree,degree+2,poly)
        root=gram_root(defect.T@(defect/c[:,None]))
        matrices=coeff[:,:,None,None]*(root[None,None,:,:]@vectors[None,:,:,:])
        errors.append(Envelope(matrices,rates,inputs))
    # q = B - p' - A p = sum L_node (A_node-A) p_node.
    base=product_controls(dim,degree,degree+1,one)
    qmaps=np.zeros((len(base),len(nodes),r,r))
    for j,a in enumerate(nodeA):qmaps[:,j]=base[:,j,None,None]*(a-Ar)
    for poly,a in zip(delta,Ai):qmaps-=product_controls(dim,degree,degree+1,poly)[:,:,None,None]*a
    qenv=Envelope(qmaps@vectors[None,:,:,:],rates,inputs)
    def denom(t,interval_end=None):
        M=t*(B.T@la.solve(np.eye(r)+t*high,B,assume_a='pos'))
        lower=float(la.eigvalsh((M+M.T)/2)[0])
        if not center_denominator:return lower
        center=float(la.svdvals(f(t,center_rates)[:,None]*center_inputs)[-1])
        drift=(deltaA/(beta*center_rates[0]) if interval_end is None else
               deltaA*float(f(interval_end,beta))*float(f(interval_end,center_rates[0])))
        return max(lower,center-drift)
    Dmax=max(norm((D+sum(((x-.5)*w*(J@V-(C@V)@a) for x,w,J,a in zip(vertex,width,H,Ai)),np.zeros_like(D)))/np.sqrt(c)[:,None]) for vertex in itertools.product([0.,1.],repeat=dim))
    tiny=1e-7/bmax;end=35/min(rates[:,0])
    grid=np.r_[0.,np.geomspace(tiny,end,int(np.ceil(np.log(end/tiny)/np.log(time_ratio)))+1)]
    integral=0.;econv=0.;qfull=0.;qrom=0.;maximum=(tiny*Dmax/2+r0)*np.exp(bmax*tiny)
    worst=tiny;maxq=0.;maxdef=0.
    for a,b in zip(grid[:-1],grid[1:]):
        dt=b-a
        trial_integral=trial.integral_controls(a,b);integral+=float(np.max(trial_integral))
        defect=sum(env.interval(a,b) for env in errors);q=qenv.gram_interval(a,b)
        next_e=np.exp(-alphabox*dt)*econv+float(f(dt,alphabox))*defect
        next_qf=np.exp(-alphabox*dt)*qfull+float(f(dt,alphabox))*q
        next_qr=np.exp(-beta*dt)*qrom+float(f(dt,beta))*q
        if a>0:
            numerator=(float(np.max(trial.values(a)+trial_integral))+integral+max(econv,next_e)
                       +max(qfull,next_qf)+max(qrom,next_qr)+min(b*r0,r0/alphabox))
            value=numerator/denom(a,b)
            if value>maximum:maximum=value;worst=a
        econv,qfull,qrom=next_e,next_qf,next_qr
        maxq=max(maxq,q);maxdef=max(maxdef,defect)
    taildef=sum(env.limit()+env.tail(end) for env in errors)
    tailq=qenv.limit()+qenv.tail(end)
    numerator=(trial.limit()+trial.tail(end)+integral+trial.tail(end)
               +max(econv,taildef/alphabox)+max(qfull,tailq/alphabox)
               +max(qrom,tailq/beta)+r0/alphabox)
    tail_bound=numerator/denom(end);maximum=max(maximum,tail_bound)
    relative=maximum/(1-maximum) if maximum<1 else float('inf')
    # Same lift solves also certify steady K-energy; no corner inverse RHS.
    lift=np.column_stack([Y,*Z]);rootK=gram_root(lift.T@(Ac@lift));del lift
    coeff=np.stack([product_controls(dim,degree,degree+1,p) for p in Tpolys],axis=2)
    matrices=sum((coeff[:,:,j,None,None]*(rootK[:,j*r:(j+1)*r][None,None,:,:]@vectors[None,:,:,:]) for j in range(len(Tpolys))),np.zeros((len(coeff),len(nodes),rootK.shape[0],r)))
    steady_trial=Envelope(matrices,rates,inputs).limit()*np.sqrt(1+rho)
    steady_defect=sum(env.limit() for env in errors)
    steady_q=qenv.limit()
    steady_num=(steady_trial+(steady_defect+steady_q+r0)/np.sqrt(alphabox)
                +steady_q/np.sqrt(beta))
    steady_den=np.sqrt(max(0.,la.eigvalsh(B.T@la.solve(high,B,assume_a='pos'))[0]))
    steady_ratio=steady_num/steady_den
    steady_bound=steady_ratio/(1-steady_ratio) if steady_ratio<1 else float('inf')
    return dict(bound=float(relative),error_over_rom_bound=float(maximum),degree=degree,
                steady_bound=float(steady_bound),steady_error_over_rom_bound=float(steady_ratio),
                steady_lift_term=float(steady_trial),steady_defect_term=float(steady_defect),
                steady_reduced_equation_defect=float(steady_q),
                worst_time=float(worst),infinite_tail_over_rom=float(tail_bound),
                derivative_integral=float(integral),lift_defect_convolution=float(econv),
                reduced_equation_defect_convolution=float(qfull+qrom),
                maximum_reduced_equation_defect=float(maxq),maximum_lift_defect=float(maxdef),
                full_decay_lower=float(alphabox),reduced_decay_lower=beta,
                counts=solver.counts(),seconds=time.perf_counter()-start,n=len(c),order=r,
                parameter_nodes=len(nodes),time_intervals=len(grid)-1,
                denominator_method='rational_and_center' if center_denominator else 'rational_only',
                full_eigendecompositions=0,full_direct_factorizations=0,
                floating_point_certified=False,
                scope='continuous supplied parameter box, all t>0, all input combinations')
