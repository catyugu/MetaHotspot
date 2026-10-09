"""AMG-CG common input-driven error-space diagnostic on large thermal models.

CG defects enter sampled resolvent tail bounds. Parameter/time continuum claims
are deliberately separate and use common_residual_gramian.certify only for U.
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

from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower, c_basis, norm
from common_residual_gramian import sym, certify


def run(args):
    start=time.perf_counter()
    K,C,G,H,ranges,meta=case1_reconstruction(args.mesh_mm)
    c=C.diagonal(); sc=np.sqrt(c)
    V=np.load(args.archive)['candidate']
    B=V.T@G
    lo=operator(K,H,ranges[:,0]); prep=Solver(lo)
    alpha=decay_lower(lo,c,prep)
    raw=G/c[:,None]
    source_rate=float(la.eigvalsh(sym(raw.T@(operator(K,H,ranges[:,1])@raw)),sym(G.T@raw))[-1])
    shifts=np.array([0.,alpha,np.sqrt(alpha*source_rate),source_rate])
    checks=np.array([alpha/3,np.sqrt(alpha*source_rate)/2,source_rate/3])
    snapshots=[]; train_counts=[]; check_counts=[]; max_cg=0.; training_max=0.
    def errors(levels,poles,counters):
        nonlocal max_cg
        for par in itertools.product(levels,repeat=len(H)):
            h=np.exp(np.log(ranges[:,0])+np.asarray(par)*(np.log(ranges[:,1])-np.log(ranges[:,0])))
            A=operator(K,H,h); Ar=sym(V.T@(A@V))
            for s in poles:
                solver=Solver(A+s*C)
                X=solver.solve(G)
                z=la.solve(Ar+s*np.eye(len(Ar)),B,assume_a='pos')
                x=sc[:,None]*X; error=sc[:,None]*(X-V@z)
                q=sym(x.T@x);ql,qu=la.eigh(q)
                invroot=(qu/np.sqrt(ql))@qu.T
                defect=G-(A+s*C)@X
                rel=norm((defect/sc[:,None])@invroot)/(alpha+s)
                max_cg=max(max_cg,rel)
                counters.append(solver.counts())
                yield error,q,rel,float(la.svdvals(x)[-1])
    for e,q,rel,den in errors([0.,.5,1.],shifts,train_counts):
        snapshots.append(e/den)
        val=np.sqrt(max(0.,float(la.eigvalsh(sym(e.T@e),q)[-1])))
        training_max=max(training_max,(val+rel)/(1-rel))
    S=np.column_stack(snapshots)
    W,singular,_=la.svd(S,full_matrices=False)
    del S,snapshots
    if args.save_space:
        args.save_space.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.save_space,V=V,W=W/sc[:,None],G=G,ranges=ranges)
    ranks=args.ranks; maxima={rank:0. for rank in ranks}; check_max=0.
    for e,q,rel,den in errors([.125,.375,.625,.875],checks,check_counts):
        val=np.sqrt(max(0.,float(la.eigvalsh(sym(e.T@e),q)[-1])))
        check_max=max(check_max,(val+rel)/(1-rel))
        for rank in ranks:
            w=W[:,:rank];tail=e-w@(w.T@e)
            val=np.sqrt(max(0.,float(la.eigvalsh(sym(tail.T@tail),q)[-1])))
            maxima[rank]=max(maxima[rank],(val+rel)/(1-rel))
    total=np.sum(singular**2)
    def counts(rows):
        return {k:sum(row[k] for row in rows) for k in rows[0]}
    out=dict(n=len(c),primal_order=V.shape[1],metadata=meta,
             basis_archive=str(args.archive),basis_extraction_cost='attributed in COMMON_GRAMIAN large-model record',
             alpha=alpha,source_rate=source_rate,training_shifts=shifts.tolist(),held_out_shifts=checks.tolist(),
             training_parameter_points=9,held_out_parameter_points=16,
             training_counts=counts(train_counts),held_out_counts=counts(check_counts),preparation=prep.counts(),
             training_max_resolvent_relative_C=training_max,held_out_max_resolvent_relative_C=check_max,
             max_reference_CG_relative_correction=max_cg,
             singular_values=singular.tolist(),
             ranks_by_snapshot_energy={str(frac):int(np.searchsorted(np.cumsum(singular**2),frac*total)+1)
                                       for frac in [.99,.9999,.999999]},
             common_spaces=[dict(error_space_order=r,held_out_projection_tail_C_with_CG_correction=maxima[r]) for r in ranks],
             continuous_parameter_certified=False,all_time_step_certified=False,
             floating_point_certified=False,full_eigendecompositions=0,full_direct_factorizations=0,
             scope='sampled nonnegative-real resolvents, arbitrary input combinations; not a step-error certificate')
    if args.certify_union:
        rank=args.certify_union
        union=c_basis(np.column_stack([V,W[:,:rank]/sc[:,None]]),c)
        out['union_tail_certificate']=certify(K,C,G,H,union,ranges,'gate')
        out['union_order']=union.shape[1]
    out['seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('COMMON_RESOLVENT_SPACE',len(c),V.shape[1],maxima,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mesh-mm',type=float,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--ranks',type=int,nargs='+',default=[8,16,32,64])
    parser.add_argument('--certify-union',type=int,default=0)
    parser.add_argument('--save-space',type=Path,help='ignored reusable common V/W archive; never evidence of certification')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
