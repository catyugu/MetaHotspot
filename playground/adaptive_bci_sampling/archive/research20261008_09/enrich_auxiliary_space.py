"""Input-driven pole enrichment; acceptance is delegated to deterministic tails.

Adds an independent logarithmic pole grid at the center to the archived common
error training space. This is candidate generation, not a continuum argument.
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

from case1_system import case1_reconstruction
from residual_reconstruction import Solver, operator, decay_lower


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--training-record',type=Path,required=True)
    p.add_argument('--poles',type=int,default=16);p.add_argument('--save',type=Path,required=True)
    args=p.parse_args();start=time.perf_counter()
    K,C,_,H,ranges,meta=case1_reconstruction(args.mesh_mm);saved=np.load(args.archive)
    V=saved['V'];W=saved['W'];G=saved['G'];c=C.diagonal();sc=np.sqrt(c)
    record=json.loads(args.training_record.read_text());singular=np.asarray(record['singular_values'])
    prep=Solver(operator(K,H,ranges[:,0]));alpha=decay_lower(prep.A,c,prep)
    raw=G/c[:,None];rate=float(la.eigvalsh(raw.T@(operator(K,H,ranges[:,1])@raw),G.T@raw)[-1])
    poles=np.r_[0.,np.geomspace(alpha/10,rate*10,args.poles-1)]
    center=np.sqrt(ranges[:,0]*ranges[:,1]);Ac=operator(K,H,center);Ar=V.T@(Ac@V);B=V.T@G
    snapshots=[sc[:,None]*W*singular[None,:]];counts=[]
    for s in poles:
        solver=Solver(Ac+s*C);X=solver.solve(G);z=la.solve(Ar+s*np.eye(len(Ar)),B,assume_a='pos')
        normx=float(la.svdvals(sc[:,None]*X)[-1]);snapshots.append(sc[:,None]*(X-V@z)/normx);counts.append(solver.counts())
    enriched,s,_=la.svd(np.column_stack(snapshots),full_matrices=False)
    args.save.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.save,V=V,W=enriched/sc[:,None],G=G,ranges=ranges)
    out=dict(metadata=meta,source_archive=str(args.archive),source_training_record=str(args.training_record),
             inherited_training_RHS=record['training_counts']['rhs'],new_poles=poles.tolist(),
             new_training_parameter=center.tolist(),new_training_counts={k:sum(row[k] for row in counts) for k in counts[0]},
             preparation=prep.counts(),singular_values=s.tolist(),seconds=time.perf_counter()-start,
             full_eigendecompositions=0,full_direct_factorizations=0,continuous_parameter_certified=False,
             scope='candidate enrichment; cost of inherited training is not free')
    args.save.with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('ENRICHED_AUXILIARY_SPACE',len(c),len(s),out['seconds'],flush=True)
