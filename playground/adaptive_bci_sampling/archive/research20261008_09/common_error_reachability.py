"""Whole-domain sampled common error-space gate, with independent held-out grid.

Dense FOM spectral diagnostics restricted to small models. Singular values and
held-out errors are evidence, never a deterministic continuum tail certificate.
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
import scipy.sparse as sp

from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, build_basis, decay_lower, f, c_basis
from common_residual_gramian import certify, sym


def run(args):
    start=time.perf_counter()
    K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm)
    c=C.diagonal(); n=len(c)
    if n>2000:
        raise ValueError('Dense FOM eigendecomposition is a small-model diagnostic only')
    lo=operator(K,H,ranges[:,0]); sol=Solver(lo)
    alpha=decay_lower(lo,c,sol)
    source=G/c[:,None]
    source_rate=float(la.eigvalsh(sym(source.T@(operator(K,H,ranges[:,1])@source)),sym(G.T@source))[-1])
    shifts=np.r_[0.,np.geomspace(alpha/10,source_rate*10,args.shifts-1)]
    center=np.sqrt(ranges[:,0]*ranges[:,1])
    points=[center,*map(np.asarray,itertools.product(*ranges))]
    V,counts=build_basis(K,C,G,H,ranges,shifts,points)
    b=V.T@G; ll,uu=la.eigh(b.T@b)
    G=G@((uu/np.sqrt(ll))@uu.T)
    B=V.T@G
    times=np.geomspace(1e-5/source_rate,40/alpha,args.times)
    vw=np.sqrt(c)[:,None]*V
    inv=sp.diags(1/np.sqrt(c))
    snapshots=[]; train_states=[]; test_states=[]; eigen_count=0
    def sample(levels, target, training):
        nonlocal eigen_count
        for par in itertools.product(levels,repeat=len(H)):
            h=np.exp(np.log(ranges[:,0])+np.asarray(par)*(np.log(ranges[:,1])-np.log(ranges[:,0])))
            A=operator(K,H,h)
            af=(inv@A@inv).toarray(); rates,U=la.eigh(af); eigen_count+=1
            bu=U.T@(G/np.sqrt(c)[:,None])
            ar=sym(V.T@(A@V)); lr,Ur=la.eigh(ar); br=Ur.T@B
            for t in times:
                x=U@(f(t,rates)[:,None]*bu)
                z=Ur@(f(t,lr)[:,None]*br)
                err=x-vw@z
                gram=sym(x.T@x)
                target.append((err,gram))
                if training:
                    snapshots.append(err/float(la.svdvals(x)[-1]))
    sample([0.,.5,1.],train_states,True)
    sample([.125,.375,.625,.875],test_states,False)
    S=np.column_stack(snapshots)
    W,singular,_=la.svd(S,full_matrices=False)
    total=float(np.sum(singular**2))
    ranks={str(frac):int(np.searchsorted(np.cumsum(singular**2),frac*total)+1)
           for frac in [.99,.9999,.999999]}
    def worst(states,w=None):
        maximum=0.
        for e,q in states:
            tail=e if w is None else e-w@(w.T@e)
            maximum=max(maximum,np.sqrt(max(0.,float(la.eigvalsh(sym(tail.T@tail),q)[-1]))))
        return maximum
    report=dict(n=n,primal_order=V.shape[1],metadata=meta,extraction_counts=counts,
                preparation=sol.counts(),training_parameter_points=9,held_out_parameter_points=16,
                times_per_point=len(times),time_range=[float(times[0]),float(times[-1])],
                full_eigendecompositions=eigen_count,
                ranks_by_snapshot_energy=ranks,singular_values=singular.tolist(),
                sampled_train_true_step=worst(train_states),sampled_held_out_true_step=worst(test_states),
                common_spaces=[],continuous_parameter_certified=False,
                floating_point_certified=False,
                scope='small FOM oracle; common W trained on 3x3 parameter grid, independently checked on 4x4 grid')
    for rank in args.ranks:
        w=W[:,:rank]
        row=dict(error_space_order=rank,sampled_train_projection_tail=worst(train_states,w),
                 sampled_held_out_projection_tail=worst(test_states,w))
        report['common_spaces'].append(row)
        if args.certify_union:
            union=c_basis(np.column_stack([V,w/np.sqrt(c)[:,None]]),c)
            row['union_order']=union.shape[1]
            row['union_status']='running'
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            row['union_tail_certificate']=certify(K,C,G,H,union,ranges,args.storage)
            row['union_status']='completed'
        print('COMMON_ERROR_SPACE',n,rank,row['sampled_held_out_projection_tail'],flush=True)
    report['seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mesh-mm',type=float,default=10.)
    parser.add_argument('--shifts',type=int,default=8)
    parser.add_argument('--times',type=int,default=48)
    parser.add_argument('--ranks',type=int,nargs='+',default=[8,16,32,64])
    parser.add_argument('--certify-union',action='store_true')
    parser.add_argument('--storage',choices=['gate','span','diagonal','dense'],default='gate')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
