"""Independent numerical audit of coupled coefficient and residual identities.

This is research evidence, not a unit test or a continuum certificate by sampling.
No full inverse is performed. The source-coupled cell theorem provides continuum.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import scipy.linalg as la
from coupled_feedback_enclosure import Certificate, coefficient_cell
from boundary_feedback_gate import opnorm


def run(args):
    with args.archive.open('rb') as handle:c=pickle.load(handle)
    rng=np.random.default_rng(20261009);worst_identity=0.;worst_enclosure=0.
    widths=c.ranges[:,1]-c.ranges[:,0];axes=[np.expm1(np.linspace(0,np.log1p(w),9)) for w in widths]
    for _ in range(args.points):
        ids=rng.integers(0,8,size=2);box=np.array([[axes[i][j],axes[i][j+1]] for i,j in enumerate(ids)])
        d=box[:,0]+rng.random(2)*(box[:,1]-box[:,0]);mid=box.mean(axis=1)
        cc,_,J,rem,rho=coefficient_cell(c.S,c.Fz,box,c.dc)
        dp=sum((h*v for h,v in zip(d,c.dc)),np.zeros(c.q));mp=sum((h*v for h,v in zip(mid,c.dc)),np.zeros(c.q))
        exact=la.solve(np.eye(c.q)+dp[:,None]*c.S,dp[:,None]*c.Fz)
        linear=la.solve(np.eye(c.q)+mp[:,None]*c.S,(dp-mp)[:,None]*(c.Fz-c.S@cc))
        actual_rem=opnorm(la.solve(J,exact-cc-linear))
        worst_enclosure=max(worst_enclosure,actual_rem/max(rem,1e-30))
        proxy=c.F-c.T@exact
        direct=c.G-c.low@proxy-sum((h*(H@proxy) for h,H in zip(d,c.H)),np.zeros_like(proxy))
        algebra=c.E@c.coeff(d,exact)
        worst_identity=max(worst_identity,opnorm((direct-algebra)/c.sc[:,None])/opnorm(c.G/c.sc[:,None]))
    out={'n':len(c.c),'s':c.s,'points':args.points,'max_residual_identity_input_normalized_defect':worst_identity,
         'max_actual_coefficient_remainder_over_enclosure':worst_enclosure,'new_full_inverse_RHS':0,
         'scope':'independent interior numerical audit, not acceptance by sampling','floating_point_certified':False}
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--points',type=int,default=128)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
