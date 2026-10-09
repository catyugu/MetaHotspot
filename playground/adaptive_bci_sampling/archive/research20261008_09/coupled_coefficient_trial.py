"""Compress the physical coefficient family inside the FIXED parent V.

The child is only a trial for the parent's steady error. Parent V does not change.
Snapshot singular values choose a trial; the exact full equation defect certifies
its tail. Full inverse cost is charged anew, not inferred from sample rank.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import scipy.linalg as la
from coupled_feedback_enclosure import Certificate
from coupled_galerkin_tail import DirectTail
from boundary_feedback_gate import sym, invroot, hpoints, relative


def run(args):
    with args.archive.open('rb') as handle:parent=pickle.load(handle)
    if parent.s!=0 or not 0<args.rank<=parent.V.shape[1]:raise ValueError('steady archive and valid trial rank required')
    snapshots=[];points=[]
    for h in hpoints(parent.ranges,np.linspace(0,1,9)):
        delta=h-parent.ranges[:,0];a=parent.Ar+sum((d*j for d,j in zip(delta,parent.Ai)),np.zeros_like(parent.Ar))
        z=la.solve(a,parent.B,assume_a='pos');snapshots.append(z@invroot(z.T@z));points.append((delta,a,z))
    U,s,_=la.svd(np.column_stack(snapshots),full_matrices=False);U=U[:,:args.rank]
    child=copy.copy(parent);child.V=parent.V@U;child.Ar=sym(U.T@parent.Ar@U);child.Ai=[sym(U.T@a@U) for a in parent.Ai];child.B=U.T@parent.B
    tail=DirectTail(child)
    tail.parent_order=parent.V.shape[1];tail.trial_order=args.rank;tail.parent_steady_only=True
    with args.output.with_suffix('.pickle').open('wb') as handle:pickle.dump(tail,handle)
    differences=[]
    for d,a,z in points:
        zc=U@la.solve(sym(U.T@a@U),U.T@parent.B,assume_a='pos')
        differences.append(relative(z-zc,z,a))
    out={'n':len(parent.c),'parent_V_order':parent.V.shape[1],'trial_order':args.rank,'coefficient_training_points':81,
         'new_coefficient_training_full_RHS':0,'coefficient_singular_values':s.tolist(),
         'sampled_child_parent_steady_K':max(differences),'additional_inverse_cost':tail.cost,
         'target':'steady K error of ORIGINAL parent V, via Galerkin best approximation using nested child trial',
         'all_time_step_certified':False,'floating_point_certified':False}
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='coefficient_singular_values'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--rank',type=int,default=16)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
