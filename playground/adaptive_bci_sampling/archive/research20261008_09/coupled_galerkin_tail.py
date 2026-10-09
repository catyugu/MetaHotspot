"""Direct input-coupled Galerkin tail: charge actual V, not a Robin surrogate.

Residual direction matrices are orthogonal to represented state directions in
C coordinates. One joint inverse Gram is reused over all parameter cells.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import itertools
import json
import pickle
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from coupled_feedback_enclosure import Certificate, root, corners, bernstein2, reduced_cell
from boundary_feedback_gate import opnorm, sym, invroot
from residual_reconstruction import Solver


class DirectTail:
    def __init__(self,cert):
        self.cert=cert;v=cert.V;c=cert.c;self.m=cert.m;self.r=v.shape[1]
        self.D0=cert.low@v-(c[:,None]*v)@cert.Ar
        self.Di=[h@v-(c[:,None]*v)@a for h,a in zip(cert.H,cert.Ai)]
        R0=cert.G-(c[:,None]*v)@cert.B
        E=np.column_stack([R0,self.D0,*self.Di]);sol=Solver(cert.low);Y=sol.solve(E);defect=E-cert.low@Y
        gram=sym(E.T@Y+Y.T@E-Y.T@(cert.low@Y))+(defect/cert.sc[:,None]).T@(defect/cert.sc[:,None])/cert.alpha
        jointE=np.column_stack([E,cert.QZ]);jointY=np.column_stack([Y,cert.T]);jointD=jointE-cert.low@jointY
        joint=sym(jointE.T@jointY+jointY.T@jointE-jointY.T@(cert.low@jointY))+(jointD/cert.sc[:,None]).T@(jointD/cert.sc[:,None])/cert.alpha
        self.GEQ=joint[:E.shape[1],E.shape[1]:];self.GQQ=joint[E.shape[1]:,E.shape[1]:]
        self.gram=gram;self.dual=root(gram);self.cost=sol.counts();self.gram_min_eigenvalue=float(la.eigvalsh(gram)[0])

    def coeff(self,d,z,change=None):return np.vstack([np.eye(self.m) if change is None else change,-z,*[-h*z for h in d]])
    def cmap(self,d):return np.vstack([np.zeros((self.m,self.r)),-np.eye(self.r),*[-h*np.eye(self.r) for h in d]])

    def cell(self,box):
        c=self.cert;mid=box.mean(axis=1);z0,zlin,J,ez,rho=reduced_cell(c.Ar,c.B,c.Ai,box)
        if rho>=1:return {'bound_K':None,'bound_C':None,'rho':rho}
        Aupper=c.Ar+sum((h*a for h,a in zip(box[:,1],c.Ai)),np.zeros_like(c.Ar))
        represented_gram=sym(c.B.T@la.solve(Aupper,c.B,assume_a='pos'))
        change=invroot(represented_gram)
        z0=z0@change;zlin=[z@change for z in zlin]
        Arc=c.Ar+sum((h*a for h,a in zip(mid,c.Ai)),np.zeros_like(c.Ar));half=np.diff(box,axis=1).ravel()/2
        deriv=[-la.solve(Arc,half[i]*c.Ai[i]@z0,assume_a='pos') for i in range(2)]
        values=np.empty((3,3,self.dual.shape[1],self.m))
        for i,j in itertools.product(range(3),repeat=2):
            ab=np.array([i-1,j-1]);z=z0+sum((h*v for h,v in zip(ab,deriv)),np.zeros_like(z0));values[i,j]=self.coeff(mid+half*ab,z,change)
        gram=self.gram
        if hasattr(self,'GEQ'):
            dp=sum((h*d for h,d in zip(box[:,0],c.dc)),np.zeros(c.q));rr=np.sqrt(np.maximum(dp,0))
            change=rr[:,None]*la.solve(np.eye(c.q)+rr[:,None]*self.GQQ*rr[None,:],rr[:,None]*self.GEQ.T,assume_a='pos')
            gram=sym(gram-self.GEQ@change)
        dualtrial=max(np.sqrt(max(0.,float(la.eigvalsh(sym(v.T@gram@v))[-1]))) for v in bernstein2(values))
        # Directional second-order output remainder, not ||D|| times ||z||.
        # delta z = J (I+J E J)^-1 (-J E z0), with J=Arc^-1/2.
        verts=corners(box);gmax=max(opnorm(J@sum((h*a for h,a in zip(d-mid,c.Ai)),np.zeros_like(Arc))@z0) for d in verts)
        directional=0.
        variations=[J@sum((h*a for h,a in zip(v-mid,c.Ai)),np.zeros_like(Arc))@J for v in verts]
        # Contract the fixed dual Gram before any small eigensolve.
        for d in verts:
            coordinate=self.cmap(d)@J
            output_gram=sym(coordinate.T@gram@coordinate)
            for variation in variations:
                variation_gram=sym(variation.T@output_gram@variation)
                directional=max(directional,np.sqrt(max(0.,float(la.eigvalsh(variation_gram,subset_by_index=[self.r-1,self.r-1])[-1]))))
        rem=directional*gmax/(1-rho)
        tail=dualtrial+rem
        Aupper=c.Ar+sum((h*a for h,a in zip(box[:,1],c.Ai)),np.zeros_like(c.Ar))
        represented=float(la.eigvalsh(sym(c.B.T@la.solve(Aupper,c.B,assume_a='pos')))[0])
        steady=tail/np.sqrt(1+tail**2)
        # C lower denominator based on computed response plus verified variation.
        normalized_remainder=rho/(1-rho)*gmax
        variation=max(opnorm(z) for z in zlin)+opnorm(J)*normalized_remainder
        tailC=tail/np.sqrt(c.alpha);den=float(la.svdvals(z0)[-1])-variation-tailC
        return {'bound_K':steady,'bound_C':None if den<=0 else tailC/den,'rho':rho,'tail_dual':tail,
                'directional_remainder':rem,'trial_dual':dualtrial,'represented_energy_lower':represented,'input_energy_gram_normalized':True}

    def cover(self,level):
        widths=self.cert.ranges[:,1]-self.cert.ranges[:,0]
        axes=[np.expm1(np.linspace(0,np.log1p(w),2**level+1)) for w in widths];rows=[]
        for i,j in itertools.product(range(2**level),repeat=2):rows.append(self.cell(np.array([[axes[0][i],axes[0][i+1]],[axes[1][j],axes[1][j+1]]])))
        return {'level':level,'cells':len(rows),'unresolved_K':sum(r['bound_K'] is None for r in rows),
                'unresolved_C':sum(r['bound_C'] is None for r in rows),
                'max_bound_K':max((r['bound_K'] for r in rows if r['bound_K'] is not None),default=None),
                'max_bound_C':max((r['bound_C'] for r in rows if r['bound_C'] is not None),default=None),
                'max_directional_remainder':max((r.get('directional_remainder',0.) for r in rows)),
                'scope':'original whole continuous parameter box; fixed real resolvent',
                'all_time_step_certified':False,'floating_point_certified':False}


def run(args):
    start=time.perf_counter()
    if args.prepared:
        with args.prepared.open('rb') as handle:tail=pickle.load(handle)
    else:
        with args.archive.open('rb') as handle:cert=pickle.load(handle)
        tail=DirectTail(cert)
        with args.output.with_suffix('.pickle').open('wb') as handle:pickle.dump(tail,handle)
    if not hasattr(tail,'gram'):tail.gram=tail.dual.T@tail.dual
    rows=[]
    for level in args.levels:
        t=time.perf_counter();r=tail.cover(level);r['seconds']=time.perf_counter()-t;rows.append(r);print(json.dumps(r),flush=True)
    out={'n':len(tail.cert.c),'s':tail.cert.s,'V_order':tail.r,'additional_inverse_cost':tail.cost,
         'prepared_reused':args.prepared is not None,'new_inverse_RHS':0 if args.prepared else tail.cost['rhs'],
         'covers':rows,'seconds':time.perf_counter()-start,'gram_min_eigenvalue':tail.gram_min_eigenvalue,
         'all_time_step_certified':False,'floating_point_certified':False}
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);p.add_argument('--prepared',type=Path)
    p.add_argument('--levels',type=int,nargs='+',default=[0,2,4]);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
