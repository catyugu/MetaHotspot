"""Residual-only, eigen-free-in-FOM continuous step majorant.

Exact-arithmetic inequalities with explicit CG defects; ordinary floating point,
NOT an outward-rounded numerical certificate. See RESIDUAL_RECONSTRUCTION_PROOF.md.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import itertools
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import pyamg


def norm(A):
    return float(la.norm(A,2)) if np.size(A) else 0.


def f(t,l):
    return -np.expm1(-np.asarray(l)*t)/l


class Solver:
    def __init__(self,A,rtol=1e-11):
        self.A=sp.csr_matrix(A);self.rhs=0;self.iterations=0
        start=time.perf_counter()
        self.amg=pyamg.ruge_stuben_solver(self.A,interpolation='direct')
        self.M=self.amg.aspreconditioner(cycle='V')
        self.setup_seconds=time.perf_counter()-start;self.rtol=rtol
        self.solve_seconds=0.
    def solve(self,B):
        B=np.asarray(B);one=B.ndim==1
        if one:B=B[:,None]
        out=[];start=time.perf_counter()
        for b in B.T:
            self.rhs+=1
            if not np.any(b):out.append(np.zeros_like(b));continue
            def callback(x):self.iterations+=1
            x,info=sla.cg(self.A,b,M=self.M,rtol=self.rtol,atol=0.,maxiter=2000,callback=callback)
            if info:raise RuntimeError(f'CG failure {info}')
            out.append(x)
        self.solve_seconds+=time.perf_counter()-start
        X=np.column_stack(out)
        return X[:,0] if one else X
    def counts(self):
        return dict(rhs=self.rhs,cg_iterations=self.iterations,amg_setups=1,
                    setup_seconds=self.setup_seconds,solve_seconds=self.solve_seconds)


def operator(K,H,h):return K+sum((x*J for x,J in zip(h,H)),sp.csc_matrix(K.shape))


def c_basis(S,c):
    A=np.sqrt(c)[:,None]*S
    A=A/np.maximum(la.norm(A,axis=0),np.finfo(float).tiny)
    Q,R,p=la.qr(A,mode='economic',pivoting=True)
    diagonal=np.abs(np.diag(R))
    rank=int(np.sum(diagonal>np.finfo(float).eps*max(A.shape)*diagonal[0]))
    return Q[:,:rank]/np.sqrt(c)[:,None]


def build_basis(K,C,G,H,ranges,shifts,parameter_points=None):
    """Deterministic candidate only; no acceptance inferred from snapshot count."""
    c=C.diagonal();points=[np.mean(ranges,axis=1)] if parameter_points is None else parameter_points
    snapshots=[G/c[:,None],np.ones((len(c),1))]
    counters=[]
    for h in points:
        A=operator(K,H,h)
        for shift in shifts:
            solve=Solver(A+shift*C);snapshots.append(solve.solve(G));counters.append(solve.counts())
    V=c_basis(np.column_stack(snapshots),c)
    return V,{key:sum(v[key] for v in counters) for key in counters[0]}


def decay_lower(A,c,solver):
    """Collatz lower bound, verified using actual product, NOT a Ritz minimum."""
    w=solver.solve(c)
    if np.any(w<=0):raise ValueError('positive supersolution not obtained')
    lower=float(np.min((A@w)/(c*w)))
    if lower<=0:raise ValueError('positive decay certificate not obtained')
    return lower


def gram_root(M):
    l,U=la.eigh((M+M.T)/2)
    return np.sqrt(np.maximum(l,0))[:,None]*U.T


def certify(K,C,G,H,V,ranges,time_ratio=1.06,rtol=1e-11):
    """Analytic upper bound; returns unresolved if sufficient condition is large.

    Requires diagonal positive C, M-matrix K(h), PSD affine H, positive h.
    No conversion of an n by n sparse matrix to dense occurs.
    """
    if not (1<time_ratio<=1.3):raise ValueError('time_ratio must be in (1,1.3]')
    c=np.asarray(C.diagonal())
    if (C-sp.diags(c)).nnz or np.any(c<=0):raise ValueError('positive diagonal C required')
    if np.any(ranges[:,0]<=0):raise ValueError('positive parameter coordinates required')
    # Supported physical structure: nonnegative Robin diagonal, conservative conduction.
    asymmetry=K-K.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data))>1e-12:raise ValueError('symmetric conduction required')
    off=K-sp.diags(K.diagonal())
    if np.any(off.data>0) or norm(np.asarray(K@np.ones(len(c)))[:,None])>1e-8*max(1.,norm(K.diagonal()[:,None])):
        raise ValueError('conservative symmetric M-matrix conduction required')
    for J in H:
        if (J-sp.diags(J.diagonal())).nnz or np.any(J.diagonal()<0):
            raise ValueError('nonnegative diagonal Robin terms required')
    if norm(V.T@(C@V)-np.eye(V.shape[1]))>1e-8:raise ValueError('C orthonormal basis required')
    start=time.perf_counter();mid=np.mean(ranges,axis=1);Ac=operator(K,H,mid)
    solver=Solver(Ac,rtol=rtol);alpha=decay_lower(Ac,c,solver)
    rho=float(np.max((ranges[:,1]-ranges[:,0])/(2*mid))) if len(H) else 0.
    if rho>=1:raise ValueError('relative form radius >=1')
    alphabox=(1-rho)*alpha
    Ar=V.T@(Ac@V);Ar=(Ar+Ar.T)/2
    Ai=[V.T@(J@V) for J in H]
    B=V.T@G;M0=B.T@B
    l0,U0=la.eigh(M0)
    if l0[0]<=0:raise ValueError('independent represented inputs required')
    change=(U0/np.sqrt(l0))@U0.T
    B=B@change;G=G@change
    R0=G-(C@V)@B;r0=norm(R0/np.sqrt(c)[:,None])
    D=Ac@V-(C@V)@Ar
    Di=[J@V-(C@V)@a for J,a in zip(H,Ai)]
    Y=solver.solve(D);defect=D-Ac@Y
    # Directly lifted affine residual innovation, including actual center CG defect.
    Z=[];Zdef=[]
    for J,d in zip(H,Di):
        S=d-J@Y;z=solver.solve(S);Z.append(z);Zdef.append(S-Ac@z)
    def0=norm(defect/np.sqrt(c)[:,None])/alpha
    deltaL=def0*(1+rho/(1-rho))
    max_param_lift=0.;vertices=[];vertex_coordinates=[];deltaA=0.;Dmax=0.
    for h in itertools.product(*ranges):
        dh=np.asarray(h)-mid;vertex_coordinates.append(np.r_[1.,dh])
        E=sum((x*a for x,a in zip(dh,Ai)),np.zeros_like(Ar));vertices.append(E)
        deltaA=max(deltaA,norm(E))
        Dz=D+sum((x*d for x,d in zip(dh,Di)),np.zeros_like(D))
        Dmax=max(Dmax,norm(Dz/np.sqrt(c)[:,None]))
        z=sum((x*v for x,v in zip(dh,Z)),np.zeros_like(Y))
        zd=sum((x*v for x,v in zip(dh,Zdef)),np.zeros_like(Y))
        defect_norm=norm(zd/np.sqrt(c)[:,None])
        zC=norm(np.sqrt(c)[:,None]*z)+defect_norm/alpha
        zK=np.sqrt(max(0.,la.eigvalsh(z.T@(Ac@z))[-1]))+defect_norm/np.sqrt(alpha)
        # First-order directional trial is retained; only its quadratic remainder is normed.
        upper=defect_norm/alphabox+rho/(1-rho)/np.sqrt(alpha)*np.sqrt(max(0.,la.eigvalsh(z.T@(Ac@z))[-1]))
        max_param_lift=max(max_param_lift,upper)
    deltaL+=max_param_lift
    lowAr=V.T@(operator(K,H,ranges[:,0])@V)
    highAr=V.T@(operator(K,H,ranges[:,1])@V)
    beta=float(la.eigvalsh(lowAr)[0]);bmax=float(la.eigvalsh(highAr)[-1])
    rates,W=la.eigh(Ar);Bm=W.T@B
    # M is computed in full space but is only r by r; no dense FOM operators.
    M=Y.T@(c[:,None]*Y);root=gram_root(W.T@M@W)
    column_norms=la.norm(root,axis=0);bnorms=la.norm(Bm,axis=1)
    # At each vertex, row norm provides an integrable directional envelope.
    Jweights=np.zeros(len(rates))
    for E in vertices:
        Jweights=np.maximum(Jweights,column_norms*la.norm(W.T@E@W,axis=1))
    weights=column_norms*bnorms
    lifts=[Y,*Z]
    blocks=[[W.T@(a.T@(c[:,None]*b))@W for b in lifts] for a in lifts]
    Mvertices=[]
    for av in vertex_coordinates:
        row=[]
        for bv in vertex_coordinates:
            row.append(sum((av[i]*bv[j]*blocks[i][j] for i in range(len(lifts)) for j in range(len(lifts))),np.zeros_like(M)))
        Mvertices.append(row)
    binary=list(itertools.product([0,1],repeat=len(H)))
    controls={}
    for i,v in enumerate(binary):
        for j,w in enumerate(binary):
            degree=tuple(a+b for a,b in zip(v,w))
            weight=2.**(-sum(a==1 for a in degree))
            controls.setdefault(degree,[]).append((i,j,weight))
    def trial_rhonorm(t):
        F=f(t,rates)[:,None]*Bm
        return max(np.sqrt(max(0.,la.eigvalsh(F.T@row[i]@F)[-1])) for i,row in enumerate(Mvertices))

    def derivative(t):return root@(np.exp(-rates*t)[:,None]*Bm)
    def rhonorm(t):return norm(root@(f(t,rates)[:,None]*Bm))
    def denominator(t,end=None):
        center=float(la.svdvals(f(t,rates)[:,None]*Bm)[-1])
        drift=(deltaA/(beta*rates[0]) if end is None else
               deltaA*float(f(end,beta))*float(f(end,rates[0])))
        rational=t*(B.T@la.solve(np.eye(len(Ar))+t*highAr,B,assume_a="pos"))
        lower=float(la.eigvalsh((rational+rational.T)/2)[0])
        return max(lower,center-drift)
    tiny=1e-7/bmax;T=35/rates[0]
    grid=np.r_[0.,np.geomspace(tiny,T,int(np.ceil(np.log(T/tiny)/np.log(time_ratio)))+1)]
    integrals=[];Ic=[0.];Jc=[0.]
    sums=rates[:,None]+rates[None,:];Mm=root.T@root
    Ew=[W.T@E@W for E in vertices]
    Jgrams=[]
    for terms in controls.values():
        g=np.zeros_like(Mm)
        for v,w,a in terms:
            for x,y,b in terms:
                g+=a*b*Mvertices[v][x]*(Ew[w]@Ew[y].T)
        Jgrams.append(g)

    for a,b in zip(grid[:-1],grid[1:]):
        kernel=np.exp(-sums*a)*(-np.expm1(-sums*(b-a)))/sums
        upper=0.
        for i,row in enumerate(Mvertices):
            Q=Bm.T@(row[i]*kernel)@Bm
            upper=max(upper,np.sqrt((b-a)*max(0.,la.eigvalsh((Q+Q.T)/2)[-1])))
        integrals.append(upper);Ic.append(Ic[-1]+upper)
        # Frobenius integrated Gram, convex quadratic in affine parameter.
        Ju=max([np.sqrt((b-a)*max(0.,float(np.sum(g*kernel)))) for g in Jgrams],default=0.)
        Jc.append(Jc[-1]+Ju)
    Ic=np.asarray(Ic);Jc=np.asarray(Jc)
    def penalty(k):return 2*float(f(grid[k],beta))*(deltaL+Jc[k])
    early=(tiny*Dmax/2+r0)*np.exp(bmax*tiny)
    maximum=early;worst_time=tiny
    for k in range(1,len(grid)-1):
        a,b=grid[k:k+2]
        numerator=trial_rhonorm(a)+Ic[k]+2*integrals[k]+penalty(k+1)+min(b*r0,r0/alphabox)
        value=numerator/denominator(a,b)
        if value>maximum:maximum=value;worst_time=float(a)
    vertex_weights=[]
    for i,row in enumerate(Mvertices):
        vertex_weights.append(np.sqrt(np.maximum(np.diag(row[i]),0))*bnorms)
    tail=max(float(np.dot(w,np.exp(-rates*T)/rates)) for w in vertex_weights)
    Iinf=Ic[-1]+tail
    Jtail=0.
    for terms in controls.values():
        wt=np.zeros(len(rates))
        for v,w,a in terms:
            wt+=a*np.sqrt(np.maximum(np.diag(Mvertices[v][v]),0))*la.norm(Ew[w],axis=1)
        Jtail=max(Jtail,float(np.dot(wt,np.exp(-rates*T)/rates)))
    Jinf=Jc[-1]+Jtail
    numerator=trial_rhonorm(1e8/rates[0])+tail+Iinf+2*(deltaL+Jinf)/beta+r0/alphabox
    tail_bound=numerator/denominator(T)
    maximum=max(maximum,tail_bound)
    relative=maximum/(1-maximum) if maximum<1 else float('inf')
    # Diagnostic point version uses same derivative quadrature; no parameter penalty.
    pointI=[0.];pointint=[]
    for a,b in zip(grid[:-1],grid[1:]):
        kernel=np.exp(-sums*a)*(-np.expm1(-sums*(b-a)))/sums
        Q=Bm.T@(Mm*kernel)@Bm
        val=np.sqrt((b-a)*max(0.,la.eigvalsh((Q+Q.T)/2)[-1]))
        pointint.append(val);pointI.append(pointI[-1]+val)
    pointtail=float(np.dot(weights,np.exp(-rates*T)/rates))
    point_max=early
    for k in range(1,len(grid)-1):
        a=grid[k];d=float(la.svdvals(f(a,rates)[:,None]*Bm)[-1])
        point_max=max(point_max,(rhonorm(a)+pointI[k]+2*pointint[k]+min(grid[k+1]*r0,r0/alpha)+2*def0*float(f(grid[k+1],rates[0])))/d)
    point_num=norm(root@(Bm/rates[:,None]))+2*pointtail+pointI[-1]+r0/alpha+2*def0/rates[0]
    point_max=max(point_max,point_num/float(la.svdvals(f(T,rates)[:,None]*Bm)[-1]))
    point_bound=point_max/(1-point_max) if point_max<1 else float('inf')
    return dict(bound=float(relative),error_over_rom_bound=float(maximum),center_bound=float(point_bound),
                center_error_over_rom_bound=float(point_max),relative_form_radius=rho,
                parameter_lift_operator_bound=deltaL,center_CG_lift_error=def0,
                parameter_penalty_infinity=float(2*(deltaL+Jinf)/beta),
                full_decay_lower=alpha,reduced_decay_lower=beta,reduced_decay_upper=bmax,
                center_total_variation_upper=float(Iinf),worst_time=worst_time,
                time_intervals=len(integrals),infinite_tail_over_rom=float(tail_bound),
                scope='continuous supplied parameter box, all t>0, all input combinations',
                full_eigendecompositions=0,full_direct_factorizations=0,
                floating_point_certified=False,counts=solver.counts(),seconds=time.perf_counter()-start,
                order=V.shape[1],n=len(c))


def steady_chord(K,C,G,H,V,ranges,rtol=1e-11):
    """Matrix-convex full chord / reduced tangent continuous steady certificate.

    m*2**d+1 full inverse RHS, independent of ROM dimension and polynomial degree.
    The full inverse is operator convex; Galerkin error Gram is Qfull-Qrom.
    CG defects are compensated with the verified common Collatz lower bound.
    """
    start=time.perf_counter();c=C.diagonal();counts=[]
    low=operator(K,H,ranges[:,0]);base=Solver(low,rtol);alpha=decay_lower(low,c,base)
    mid=np.mean(ranges,axis=1);A=operator(K,H,mid);B=V.T@G
    Ar=V.T@(A@V);Z=la.solve(Ar,B,assume_a='pos')
    # Tangent is represented variationally, so no derivative implementation is needed.
    Qlow=B.T@la.solve(V.T@(operator(K,H,ranges[:,1])@V),B,assume_a='pos')
    bound2=0.;corners=[]
    for index,h in enumerate(itertools.product(*ranges)):
        Ah=operator(K,H,h)
        solver=base if index==0 else Solver(Ah,rtol)
        X=solver.solve(G);D=G-Ah@X
        upper=G.T@X+X.T@G-X.T@(Ah@X)+(D.T@(D/c[:,None]))/alpha
        tangent=B.T@Z+Z.T@B-Z.T@(V.T@(Ah@V))@Z
        error_upper=(upper-tangent);error_upper=(error_upper+error_upper.T)/2
        value=max(0.,float(la.eigvalsh(error_upper,Qlow)[-1]))
        bound2=max(bound2,value);corners.append(value);counts.append(solver.counts())
    aggregate={key:sum(v[key] for v in counts) for key in counts[0]}
    return dict(bound=float(np.sqrt(bound2)),full_inverse_rhs=aggregate['rhs'],counts=aggregate,
                scope='continuous supplied parameter box, all input combinations, steady K-energy',
                floating_point_certified=False,full_eigendecompositions=0,full_direct_factorizations=0,
                full_decay_lower=alpha,seconds=time.perf_counter()-start,corner_squared_bounds=corners)
