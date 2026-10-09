"""Separate port truncation from fixed-reference dynamic reconstruction error.

Reuse exactly the port space already trained by boundary_feedback_gate. Change
only the nonparametric diffusion pole coverage; no new HTC training or parameter
polynomial. This is a sampled formal-scale diagnostic, not continuum acceptance.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la

from boundary_feedback_gate import ports, sym, invroot, opnorm, relative, hpoints, counts
from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower, c_basis


def run(args):
    start=time.perf_counter()
    K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm)
    c=C.diagonal();sc=np.sqrt(c)
    G=G@invroot((G/sc[:,None]).T@(G/sc[:,None]))
    archive=np.load(args.archive)
    if not np.array_equal(archive['ranges'],ranges) or not np.allclose(archive['G'],G,rtol=1e-13,atol=1e-15):
        raise ValueError('space archive does not match operator/input contract')
    Q,groups,_=ports(H);Z=archive[f'Z{args.rank}'];QZ=np.asarray(Q@Z)
    lo=operator(K,H,ranges[:,0]);prep=Solver(lo);alpha=decay_lower(lo,c,prep)
    raw=G/c[:,None];rate=float(la.eigvalsh(sym(raw.T@(operator(K,H,ranges[:,1])@raw)))[-1])
    poles=np.r_[0.,np.geomspace(alpha,rate,args.poles-1)]
    checks=np.r_[0.,alpha/3,np.sqrt(alpha*rate)/2,rate/3]
    snapshots=[G/c[:,None],np.ones((len(c),1))];cost=[]
    for s in poles:
        sol=Solver(lo+s*C);snapshots.append(sol.solve(np.column_stack([G,QZ])));cost.append(sol.counts())
    V=c_basis(np.column_stack(snapshots),c)
    models=[{'label':'original_4_poles','V':archive[f'V{args.rank}']},{'label':f'fixed_ports_{args.poles}_poles','V':V}]
    for m in models:
        m.update(max_resolvent_relative_C=0.,max_steady_relative_K=0.,worst=None)
    refs=[];maxcg=0.;per_parameter=[]
    point_poles=[(h,checks) for h in hpoints(ranges,[.125,.375,.625,.875])]
    point_poles.extend((h,[0.]) for h in hpoints(ranges,[0.,1.]))
    for h,shifts in point_poles:
        for s in shifts:
            A=operator(K,H,h)+s*C;sol=Solver(A);X=sol.solve(G);refs.append(sol.counts())
            x=sc[:,None]*X;den=float(la.svdvals(x)[-1])
            cg=opnorm((G-A@X)/sc[:,None])/(alpha+s)/den;maxcg=max(maxcg,cg)
            if cg>=1:
                raise ValueError('unresolved reference denominator')
            row={'h':h.tolist(),'s':float(s),'reference_CG_relative_C_correction':cg,'models':[]}
            for m in models:
                v=m['V'];ar=sym(v.T@(operator(K,H,h)@v))+s*np.eye(v.shape[1])
                xv=v@la.solve(ar,v.T@G,assume_a='pos')
                value=(relative(sc[:,None]*(X-xv),x)+cg)/(1-cg)
                if value>m['max_resolvent_relative_C']:
                    m['max_resolvent_relative_C']=value;m['worst']={'h':h.tolist(),'s':float(s)}
                steady=None
                if s==0:
                    # Reference energy residual correction includes actual CG.
                    # eK(reference,true)<=||d/C^.5||/sqrt(alpha).
                    d=opnorm((G-A@X)/sc[:,None])/np.sqrt(alpha)
                    denK=float(np.sqrt(la.eigvalsh(sym(X.T@(A@X)))[0]))
                    cgK=d/denK
                    if cgK>=1:
                        raise ValueError('unresolved reference energy denominator')
                    steady=(relative(X-xv,X,A)+cgK)/(1-cgK)
                    m['max_steady_relative_K']=max(m['max_steady_relative_K'],steady)
                row['models'].append({'label':m['label'],'relative_C_with_CG':value,'steady_K_with_CG':steady})
            per_parameter.append(row)
    out={'n':len(c),'metadata':meta,'rank_per_group':args.rank,'port_order':Z.shape[1],
         'poles':poles.tolist(),'checked_poles':checks.tolist(),
         'reused_archive':str(args.archive),'new_parameter_training_RHS':0,
         'models':[{k:v for k,v in m.items() if k!='V'}|{'order':m['V'].shape[1]} for m in models],
         'costs':{'preparation':prep.counts(),'new_dynamic_basis':counts(cost),'reference_audit':counts(refs)},
         'max_reference_CG_correction':maxcg,'sampled_checks':per_parameter,
         'full_eigendecompositions':0,'full_direct_factorizations':0,
         'continuous_parameter_certified':False,'all_time_step_certified':False,'floating_point_certified':False,
         'seconds':time.perf_counter()-start}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in {'sampled_checks'}},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,required=True)
    p.add_argument('--rank',type=int,default=16);p.add_argument('--poles',type=int,default=8)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
