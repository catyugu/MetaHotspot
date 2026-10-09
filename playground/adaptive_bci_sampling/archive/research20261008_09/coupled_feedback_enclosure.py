"""Input-coupled rational cell enclosure and composite real-resolvent tails.

Fixed inverse residual Gram includes actual CG defects. No snapshot tail is used.
Exact-arithmetic continuous-box inequalities, evaluated in ordinary floating point.
Real resolvent scope only: NOT an all-time step certificate.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la

from boundary_feedback_gate import sym, invroot, opnorm, ports, group_basis, hpoints, relative, counts
from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower, c_basis


def root(a, inverse=False):
    l,u=la.eigh(sym(a))
    if inverse and l[0]<=0:raise ValueError('nonpositive metric')
    return (u*(1/np.sqrt(l) if inverse else np.sqrt(np.maximum(l,0))))@u.T


def corners(box):return [np.asarray(v) for v in itertools.product(*box)]


def coefficient_cell(S,F,box,group_diagonals):
    """c=(I+Delta S)^-1 Delta F, including zero Delta endpoints."""
    parameter_mid=box.mean(axis=1)
    mid=sum((h*d for h,d in zip(parameter_mid,group_diagonals)),np.zeros(len(S)))
    Sh=root(S);Sih=root(S,True)
    A=np.eye(len(S))+Sh@(mid[:,None]*Sh);Aih=root(A,True)
    c0=la.solve(np.eye(len(S))+mid[:,None]*S,mid[:,None]*F)
    innovation=F-S@c0
    # First derivative kept as an input matrix, not a state energy ball.
    Jmap=Sih@Aih
    linear=[];g=[];rho=0.
    for parameter in corners(box):
        dh=sum((h*d for h,d in zip(parameter-parameter_mid,group_diagonals)),np.zeros(len(S)))
        j=Aih@Sh@(dh[:,None]*Sh)@Aih
        rho=max(rho,opnorm(j))
        gj=Aih@Sh@(dh[:,None]*innovation);g.append(gj)
        linear.append(Sih@Aih@gj)
    rem=float('inf') if rho>=1 else rho/(1-rho)*max(map(opnorm,g))
    return c0,linear,Jmap,rem,rho


def reduced_cell(A,B,Ai,box):
    mid=box.mean(axis=1);Ac=A+sum((h*a for h,a in zip(mid,Ai)),np.zeros_like(A))
    Aih=root(Ac,True);z0=la.solve(Ac,B,assume_a='pos')
    linear=[];rho=0.;g=[]
    for h in corners(box):
        E=sum((d*a for d,a in zip(h-mid,Ai)),np.zeros_like(A))
        rho=max(rho,opnorm(Aih@E@Aih));gj=-Aih@E@z0
        g.append(gj);linear.append(Aih@gj)
    rem=float('inf') if rho>=1 else rho/(1-rho)*max(map(opnorm,g))
    return z0,linear,Aih,rem,rho


def bernstein2(values):
    # Values at 0,1/2,1 tensor nodes -> degree-two Bernstein controls.
    change=np.array([[1,0,0],[-.5,2,-.5],[0,0,1.]])
    return np.einsum('ia,jb,abkm->ijkm',change,change,values).reshape((-1,*values.shape[2:]))


class Certificate:
    def __init__(self,K,C,G,H,ranges,Z,V,s,alpha,local_metric=True):
        self.local_metric=local_metric
        self.c=C.diagonal();self.sc=np.sqrt(self.c);self.V=V;self.G=G
        self.H=H;self.ranges=ranges;self.s=s;self.alpha=alpha+s
        self.low=operator(K,H,ranges[:,0])+s*C;self.high=operator(K,H,ranges[:,1])+s*C
        Q,groups,_=ports(H);QZ=np.asarray(Q@Z);self.Z=Z;self.groups=groups
        sol=Solver(self.low);self.F=sol.solve(G);self.T=sol.solve(QZ)
        self.S=sym(QZ.T@self.T);self.Fz=QZ.T@self.F
        # Exact source innovation, with fixed CG/symmetrization defects.
        Fp=np.asarray(Q.T@self.F);Tp=np.asarray(Q.T@self.T)
        PF=Fp-Z@(Z.T@Fp);PT=Tp-Z@(Z.T@Tp)
        self.U=[np.asarray(Q@((groups==i)[:,None]*PF)) for i in range(len(H))]
        dc=[]
        for i in range(len(H)):
            dc.append(np.array([float(np.any((groups==i)&(np.abs(Z[:,j])>1e-14))) for j in range(Z.shape[1])]))
        self.W=[np.asarray(Q@((groups==i)[:,None]*PT))+QZ@(dc[i][:,None]*(Z.T@Tp-self.S)) for i in range(len(H))]
        dF=G-self.low@self.F;dT=self.low@self.T-QZ
        self.E=np.column_stack([dF,*self.U,dT,*self.W])
        Y=sol.solve(self.E);D=self.E-self.low@Y
        gram=sym(self.E.T@Y+Y.T@self.E-Y.T@(self.low@Y))+(D/self.sc[:,None]).T@(D/self.sc[:,None])/self.alpha
        self.dual=root(gram)
        self.w=sol.solve(self.c)
        self.QZ=QZ
        jointE=np.column_stack([self.E,QZ]);jointY=np.column_stack([Y,self.T])
        jointD=jointE-self.low@jointY
        self.joint_gram=sym(jointE.T@jointY+jointY.T@jointE-jointY.T@(self.low@jointY))+(jointD/self.sc[:,None]).T@(jointD/self.sc[:,None])/self.alpha
        # Negative roundoff eigenvalues are exposed, not a machine certificate.
        self.gram_min_eigenvalue=float(la.eigvalsh(sym(gram))[0])
        self.cost=sol.counts();m=G.shape[1];q=Z.shape[1];self.m=m;self.q=q
        self.dc=[np.full(q,0.) for _ in H]
        for i in range(len(H)):
            self.dc[i]=np.array([float(np.any((groups==i)&(np.abs(Z[:,j])>1e-14))) for j in range(q)])
        if not np.allclose(sum(self.dc),1):raise ValueError('non group-separated port space')
        self.Ar=sym(V.T@(self.low@V));self.Ai=[sym(V.T@(h@V)) for h in H];self.B=V.T@G
        self.Fo=self.F-V@(V.T@(self.c[:,None]*self.F))
        self.To=self.T-V@(V.T@(self.c[:,None]*self.T))
        self.qspan=np.column_stack([V.T@(dF+self.low@self.Fo),
            *[V.T@(u-h@self.Fo) for u,h in zip(self.U,H)],
            V.T@(dT-self.low@self.To),*[V.T@(w-h@self.To) for w,h in zip(self.W,H)]])
        self.out_metric=root(sym(np.column_stack([self.Fo,self.To]).T@(self.high@np.column_stack([self.Fo,self.To]))))
        hi=Solver(self.high);Xhi=hi.solve(G)
        # Certified in exact arithmetic lower bound on G^T high^-1 G.
        lower=sym(G.T@Xhi+Xhi.T@G-Xhi.T@(self.high@Xhi))
        self.dK=np.sqrt(float(la.eigvalsh(lower)[0]));self.high_cost=hi.counts()

    def coeff(self,d,c):
        m,q=self.m,self.q
        return np.vstack([np.eye(m),*[-h*np.eye(m) for h in d],c,*[h*c for h in d]])

    def coeff_map(self,d):
        return np.vstack([np.zeros((self.m*(1+len(d)),self.q)),np.eye(self.q),*[h*np.eye(self.q) for h in d]])

    def cell(self,box):
        # box expressed as Delta, not physical HTC.
        c0,cl,Jc,ec,rhoc=coefficient_cell(self.S,self.Fz,box,self.dc)
        mid=box.mean(axis=1)
        Arc=self.Ar+sum((h*a for h,a in zip(mid,self.Ai)),np.zeros_like(self.Ar))
        Aih=root(Arc,True);rhoz=max(opnorm(Aih@sum((h*a for h,a in zip(d-mid,self.Ai)),np.zeros_like(Arc))@Aih) for d in corners(box))
        if rhoc>=1 or rhoz>=1:
            return {'bound_C':None,'bound_K':None,'rho_feedback':rhoc,'rho_ROM':rhoz}
        vs=corners(box);mid=box.mean(axis=1)
        # Reconstruct affine c at arbitrary node by its two independent derivatives.
        half=(box[:,1]-box[:,0])/2
        derivatives=[]
        for i in range(2):
            dh=np.zeros(2);dh[i]=half[i]
            dp=sum((h*d for h,d in zip(dh,self.dc)),np.zeros(self.q))
            derivatives.append(la.solve(np.eye(self.q)+sum((h*d for h,d in zip(mid,self.dc)),np.zeros(self.q))[:,None]*self.S,dp[:,None]*(self.Fz-self.S@c0)))
        dual=self.dual;alpha=self.alpha
        if self.local_metric:
            dp=sum((h*d for h,d in zip(box[:,0],self.dc)),np.zeros(self.q));rr=np.sqrt(np.maximum(dp,0))
            j=self.joint_gram;e=self.E.shape[1]
            solve=rr[:,None]*la.solve(np.eye(self.q)+rr[:,None]*j[e:,e:]*rr[None,:],rr[:,None]*j[e:,:e],assume_a='pos')
            # Monotone variational Woodbury map of a joint inverse Gram upper.
            dual=root(sym(j[:e,:e]-j[:e,e:]@solve))
            wc=self.w-self.T@(rr*la.solve(np.eye(self.q)+rr[:,None]*self.S*rr[None,:],rr*(self.QZ.T@self.w),assume_a='pos'))
            if np.all(wc>0):
                product=self.low@wc+sum((h*(J@wc) for h,J in zip(box[:,0],self.H)),np.zeros(len(wc)))
                alpha=max(alpha,float(np.min(product/(self.c*wc))))
        controls=np.empty((3,3,self.E.shape[1],self.m))
        for i,j in itertools.product(range(3),repeat=2):
            ab=np.array([i-1,j-1]);d=mid+ab*half
            ca=c0+sum((h*v for h,v in zip(ab,derivatives)),np.zeros_like(c0))
            controls[i,j]=self.coeff(d,ca)
        residual=max(opnorm(dual@v) for v in bernstein2(controls))
        residual_rem=max(opnorm(dual@self.coeff_map(d)@Jc) for d in vs)*ec
        taildual=residual+residual_rem;tailC=taildual/np.sqrt(alpha)
        # Residual-driven reconstruction avoids independent c/z energy balls.
        # q=V^T r+V^T A e_out, and gap=e_out-V Ar^-1 q.
        ctr=bernstein2(controls)
        qtrial=[self.qspan@v for v in ctr]
        qrem=max(opnorm(Aih@self.qspan@self.coeff_map(d)@Jc) for d in vs)*ec
        qnorm=max(opnorm(Aih@v) for v in qtrial)
        dynamic=max(opnorm(la.solve(Arc,v,assume_a='pos')) for v in qtrial)
        dynamic+=opnorm(Aih)*(rhoz*qnorm+qrem)/(1-rhoz)
        out0=self.Fo-self.To@c0
        outC=max(opnorm(self.sc[:,None]*(out0-self.To@a)) for a in cl)+opnorm(self.sc[:,None]*self.To@Jc)*ec
        gap=outC+dynamic
        X0=self.F-self.T@c0
        variation=max(opnorm(self.sc[:,None]*self.T@a) for a in cl)+opnorm(self.sc[:,None]*self.T@Jc)*ec
        denominator=float(la.svdvals(self.sc[:,None]*X0)[-1])-variation-tailC
        upper=None if denominator<=0 else (tailC+gap)/denominator
        AhiX=self.low@X0+sum((h*(J@X0) for h,J in zip(box[:,1],self.H)),np.zeros_like(X0))
        dkcell=float(la.eigvalsh(sym(self.G.T@X0+X0.T@self.G-X0.T@AhiX))[0])
        dk=max(self.dK,np.sqrt(max(0.,dkcell)))
        # Galerkin best approximation: choose the projection of the proxy.
        outK=max(opnorm(self.out_metric@np.vstack([np.eye(self.m),-(c0+a)])) for a in cl)
        outK+=opnorm(self.out_metric@np.vstack([np.zeros((self.m,self.q)),-Jc]))*ec
        return {'bound_C':upper,'bound_K':(taildual+outK)/dk,'rho_feedback':rhoc,'rho_ROM':rhoz,
                'feedback_tail_C':tailC,'feedback_dual_tail':taildual,'dynamic_gap_C':gap,
                'denominator_C':denominator,'coefficient_remainder':ec,'projected_residual_dual':qnorm+qrem,
                'proxy_outside_V_C':outC,'local_decay_lower':alpha}

    def cover(self,level):
        # Subdivide in log(1+Delta), includes zero endpoints without singularity.
        widths=self.ranges[:,1]-self.ranges[:,0]
        axes=[np.expm1(np.linspace(0,np.log1p(w),2**level+1)) for w in widths]
        rows=[]
        for i,j in itertools.product(range(2**level),repeat=2):
            box=np.array([[axes[0][i],axes[0][i+1]],[axes[1][j],axes[1][j+1]]]);rows.append(self.cell(box))
        finite=[r['bound_C'] for r in rows if r['bound_C'] is not None]
        return {'level':level,'cells':len(rows),'unresolved_C_cells':len(rows)-len(finite),
                'max_bound_C':max(finite,default=None),'max_bound_K':max((r['bound_K'] for r in rows if r['bound_K'] is not None),default=None),
                'max_rho_feedback':max(r['rho_feedback'] for r in rows),'max_rho_ROM':max(r['rho_ROM'] for r in rows),
                'max_feedback_tail_C':max((r.get('feedback_tail_C',0) for r in rows)),
                'max_dynamic_gap_C':max((r.get('dynamic_gap_C',0) for r in rows)),
                'scope':'entire original continuous parameter box at this fixed real s',
                'all_time_step_certified':False,'floating_point_certified':False}


def run(args):
    start=time.perf_counter();K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm);c=C.diagonal();sc=np.sqrt(c)
    G=G@invroot((G/sc[:,None]).T@(G/sc[:,None]));Q,groups,_=ports(H)
    prep=Solver(operator(K,H,ranges[:,0]));alpha=decay_lower(prep.A,c,prep)
    raw=G/c[:,None];rate=float(la.eigvalsh(sym(raw.T@(operator(K,H,ranges[:,1])@raw)))[-1])
    poles=np.r_[0,np.geomspace(alpha,rate,3)];trace=[];train=[]
    for h in hpoints(ranges,[0,.5,1]):
        for s in poles:
            sol=Solver(operator(K,H,h)+s*C);X=sol.solve(G);trace.append(np.asarray(Q.T@X)@invroot((sc[:,None]*X).T@(sc[:,None]*X)));train.append(sol.counts())
    Z,_=group_basis(np.column_stack(trace),groups,args.rank);QZ=np.asarray(Q@Z)
    snaps=[G/c[:,None],np.ones((len(c),1))];basis=[]
    for s in np.r_[0,np.geomspace(alpha,rate,7)]:
        sol=Solver(prep.A+s*C);snaps.append(sol.solve(np.column_stack([G,QZ])));basis.append(sol.counts())
    V=c_basis(np.column_stack(snaps),c);out={'n':len(c),'metadata':meta,'parameters':ranges.tolist(),'port_order':Z.shape[1],'V_order':V.shape[1],
        'training_cost':counts(train),'basis_cost':counts(basis),'preparation_cost':prep.counts(),'poles':[],'local_feedback_metric':not args.global_metric,'all_time_step_certified':False,'floating_point_certified':False}
    for s in ([0.] if args.steady_only else [0.,np.sqrt(alpha*rate)/2,rate/3]):
        cert=Certificate(K,C,G,H,ranges,Z,V,s,alpha,not args.global_metric);levels=[]
        for level in args.levels:
            t=time.perf_counter();row=cert.cover(level);row['seconds']=time.perf_counter()-t;levels.append(row)
            print(json.dumps({'n':len(c),'s':s,'cover':row}),flush=True)
        checks=[];checkcost=[];violations=0
        for h in hpoints(ranges,[.125,.375,.625,.875]):
            sol=Solver(operator(K,H,h)+s*C);X=sol.solve(G);checkcost.append(sol.counts())
            xv=V@la.solve(sym(V.T@(sol.A@V)),V.T@G,assume_a='pos')
            d=h-ranges[:,0];box=np.column_stack([d,d]);b=cert.cell(box)
            obs=relative(sc[:,None]*(X-xv),sc[:,None]*X)
            if b['bound_C'] is not None and obs>b['bound_C']*(1+1e-7):violations+=1
            checks.append({'h':h.tolist(),'observed_C':obs,'point_bound':b})
        import pickle
        cache=args.output.with_name(args.output.stem+f'_s{len(out["poles"])}.pickle')
        with cache.open('wb') as handle:pickle.dump(cert,handle)
        out['poles'].append({'s':s,'covers':levels,'fixed_inverse_cost':cert.cost,'high_denominator_cost':cert.high_cost,
                            'audit_cost':counts(checkcost),'audit':checks,'audit_violations':violations,'dual_gram_min_eigenvalue':cert.gram_min_eigenvalue})
    out['seconds']=time.perf_counter()-start;args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'n':len(c),'seconds':out['seconds'],'output':str(args.output)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,default=10);p.add_argument('--rank',type=int,default=16)
    p.add_argument('--levels',type=int,nargs='+',default=[0,2,4]);p.add_argument('--steady-only',action='store_true');p.add_argument('--global-metric',action='store_true');p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
