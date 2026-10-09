"""Small-model independent steady/step oracle for the frozen parent V.

Dense FOM oracle restricted to n<=2000. Step samples are diagnostics only.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import itertools
import json
import pickle
from pathlib import Path
import numpy as np
import scipy.linalg as la
from coupled_feedback_enclosure import Certificate
from coupled_galerkin_tail import DirectTail
from boundary_feedback_gate import sym, relative, hpoints
from residual_reconstruction import f


def run(args):
    with args.archive.open('rb') as handle:c=pickle.load(handle)
    if c.s!=0:raise ValueError('step oracle requires zero-shift archive')
    if len(c.c)>2000:raise ValueError('dense oracle restricted to small models')
    V=c.V;iv=1/c.sc;vg=c.sc[:,None]*V;maxsteady=maxstep=0.;pointcert_violations=0
    eigs=0;directs=0;tails=None
    if args.tail:
        with args.tail.open('rb') as handle:tails=pickle.load(handle)
        if not hasattr(tails,'gram'):tails.gram=tails.dual.T@tails.dual
    for h in list(hpoints(c.ranges,[.125,.375,.625,.875]))+[np.asarray(v) for v in itertools.product(*c.ranges)]:
        d=h-c.ranges[:,0];A=c.low+sum((v*H for v,H in zip(d,c.H)),c.low*0)
        full=A.toarray();l,U=la.eigh(iv[:,None]*full*iv[None,:]);eigs+=1
        B=U.T@(c.G/c.sc[:,None]);lr,Ur=la.eigh(sym(V.T@(A@V)));br=Ur.T@(V.T@c.G)
        X=la.solve(full,c.G,assume_a='pos');directs+=1;Xv=V@la.solve(sym(V.T@(A@V)),V.T@c.G,assume_a='pos')
        ek=relative(X-Xv,X,A);maxsteady=max(maxsteady,ek)
        if tails:
            upper=tails.cell(np.column_stack([d,d]))['bound_K']
            if ek>upper*(1+1e-5)+1e-10:pointcert_violations+=1
        for t in np.geomspace(1e-7/l[-1],40/l[0],64):
            x=U@(f(t,l)[:,None]*B);xv=vg@Ur@(f(t,lr)[:,None]*br)
            maxstep=max(maxstep,relative(x-xv,x))
    out={'n':len(c.c),'V_order':V.shape[1],'parameters':20,'times_per_parameter':64,
         'sampled_steady_relative_K':maxsteady,'sampled_step_relative_C':maxstep,'point_bound_violations':pointcert_violations,
         'full_eigendecompositions':eigs,'full_direct_factorizations':directs,'full_AMG_RHS':0,
         'scope':'small dense oracle, NOT continuous-parameter or all-time acceptance','floating_point_certified':False}
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--tail',type=Path)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
