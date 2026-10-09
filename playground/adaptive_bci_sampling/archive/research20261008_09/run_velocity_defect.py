"""Research driver; no unit tests and no sampled acceptance criteria."""

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
import scipy.sparse as sp

from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower, build_basis, f
from velocity_defect import certify_velocity


def prepare(path,mesh,shifts,times):
    start=time.perf_counter();K,C,G,H,ranges,meta=case1_reconstruction(mesh)
    c=C.diagonal();n=len(c)
    if n>2000:raise ValueError('full spectral snapshot oracle is restricted to small models')
    solver=Solver(operator(K,H,ranges[:,0]));alpha=decay_lower(solver.A,c,solver)
    source=G/c[:,None]
    source_rate=float(la.eigvalsh(source.T@(operator(K,H,ranges[:,1])@source),G.T@source)[-1])
    shifts=np.r_[0.,np.geomspace(alpha/10,source_rate*10,shifts-1)]
    V,counts=build_basis(K,C,G,H,ranges,shifts,[np.sqrt(ranges[:,0]*ranges[:,1]),*map(np.asarray,itertools.product(*ranges))])
    l,U=la.eigh((V.T@G).T@(V.T@G));G=G@((U/np.sqrt(l))@U.T);B=V.T@G
    grid=np.geomspace(1e-5/source_rate,40/alpha,times);inverse=sp.diags(1/np.sqrt(c));vw=np.sqrt(c)[:,None]*V
    snapshots=[];states=[]
    for levels,training in [([0.,.5,1.],True),([.125,.375,.625,.875],False)]:
        for p in itertools.product(levels,repeat=len(H)):
            h=np.exp(np.log(ranges[:,0])+np.asarray(p)*(np.log(ranges[:,1])-np.log(ranges[:,0])))
            Ah=operator(K,H,h);rates,U=la.eigh((inverse@Ah@inverse).toarray())
            bu=U.T@(G/np.sqrt(c)[:,None]);lr,Ur=la.eigh(V.T@(Ah@V));br=Ur.T@B
            for t in grid:
                x=U@(f(t,rates)[:,None]*bu);z=Ur@(f(t,lr)[:,None]*br);err=x-vw@z
                if training:snapshots.append(err/float(la.svdvals(x)[-1]))
                states.append((h,t,err,x.T@x,training))
    W,s,_=la.svd(np.column_stack(snapshots),full_matrices=False);W=W/np.sqrt(c)[:,None]
    diagnostic=[]
    for k in [16,32,64]:
        w=W[:,:k];worst_projection=worst_dynamic=0.
        grouped={}
        for h,t,e,q,training in states:
            if not training:grouped.setdefault(tuple(h),[]).append((t,e,q))
        for hs,rows in grouped.items():
            Ah=operator(K,H,np.asarray(hs));ar=V.T@(Ah@V);aw=w.T@(Ah@w)
            D=Ah@V-c[:,None]*(V@ar);r0=G-c[:,None]*(V@B)
            L=np.block([[ar,np.zeros((len(ar),k))],[w.T@D,aw]])
            rhs=np.vstack([B,w.T@r0]);yinfty=la.solve(L,rhs)
            for t,e,q in rows:
                y=yinfty-la.expm(-t*L)@yinfty;ehat=np.sqrt(c)[:,None]*(w@y[len(ar):])
                tail=e-ehat;proj=e-np.sqrt(c)[:,None]*(w@(w.T@(np.sqrt(c)[:,None]*e)))
                worst_projection=max(worst_projection,np.sqrt(max(0.,float(la.eigvalsh(proj.T@proj,q)[-1]))))
                worst_dynamic=max(worst_dynamic,np.sqrt(max(0.,float(la.eigvalsh(tail.T@tail,q)[-1]))))
        diagnostic.append(dict(rank=k,sampled_projection_tail=worst_projection,sampled_auxiliary_dynamic_tail=worst_dynamic))
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,V=V,W=W,G=G,ranges=ranges)
    result=dict(n=n,primal_order=V.shape[1],metadata=meta,counts=counts,preparation_counts=solver.counts(),
                training_points=9,held_out_points=16,times_per_point=times,full_eigendecompositions=25,
                diagnostics=diagnostic,seconds=time.perf_counter()-start,continuous_parameter_certified=False)
    path.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main(args):
    if args.prepare:prepare(args.archive,args.mesh_mm,args.shifts,args.times);return
    K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm)
    saved=np.load(args.archive);V=saved['V'] if 'V' in saved else saved['candidate']
    if 'G' in saved:G=saved['G']
    if len(V)!=K.shape[0] or ('ranges' in saved and not np.allclose(saved['ranges'],ranges)):raise ValueError('archive/model mismatch')
    report=dict(metadata=meta,method='lifted velocity with joint polynomial residual',runs=[],
                archive=str(args.archive),raw_archive_committed=False)
    for width in args.widths:
        center=np.sqrt(ranges[:,0]*ranges[:,1])
        box=np.column_stack([center-width*(center-ranges[:,0]),center+width*(ranges[:,1]-center)])
        for k in args.error_ranks:
            w=None if k==0 else saved['W'][:,:k]
            row=certify_velocity(K,C,G,H,V,box,args.degree,args.time_ratio,W=w)
            row.update(width_fraction=width,lift_space='full_Riesz_CG' if k==0 else 'common_snapshot_W')
            report['runs'].append(row)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print('VELOCITY_DEFECT',width,k,row['error_over_rom_bound'],row['steady_bound'],row['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,default=10.)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--shifts',type=int,default=8);p.add_argument('--times',type=int,default=48)
    p.add_argument('--degree',type=int,default=1);p.add_argument('--time-ratio',type=float,default=1.2)
    p.add_argument('--widths',type=float,nargs='+',default=[0.,.001,.01,.1,1.])
    p.add_argument('--error-ranks',type=int,nargs='+',default=[0])
    p.add_argument('--output',type=Path,default=Path('results/velocity_defect.json'))
    main(p.parse_args())
