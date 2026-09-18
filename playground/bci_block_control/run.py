"""Matched-code-path attribution check; does not tune the completed screen.

Compare the already tested block_cap=4 with block_cap=1 in exactly the SAME
extractor. The legacy cached scalar control is included to expose any remaining
matrix-assembly-cache effect. Both scalar spaces must be numerically identical.
This is an offline attribution test, not an additional physical validation set.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import scipy.linalg as la

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bci_block'))
from benchmark import load_data
from block_extract import extract
from algorithms import extract as legacy


def compare_spaces(V,W):
    if V.shape!=W.shape:
        raise ValueError('scalar control rank/shape changed')
    gap=float(la.norm(W-V@(V.T@W))/max(la.norm(W),1e-30))
    if gap>1e-9:
        raise ValueError(f'scalar control space changed: {gap}')
    return gap


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    from metahotspot.macromodel import utils as u
    K,C,H,data=load_data(args.data,16); G=data['G']
    core=u.normalized_operators(K,C,G@data['power'])
    names=('legacy_cached_single','matched_single','matched_block')
    rows=[]; gaps=[]
    for rep in (0,1):
        spaces={}
        for name in (names if rep==0 else names[::-1]):
            start=time.perf_counter()
            if name=='legacy_cached_single':
                V,s=legacy(core,G,H,data['ranges'],tolerance=1e-4,seed=args.seed,
                           probe_rounds=10,max_order=u.MAX_ORDER,method='shared_tangent_cached')
            else:
                V,s=extract(core,G,H,data['ranges'],tolerance=1e-4,seed=args.seed,
                            probe_rounds=10,max_order=u.MAX_ORDER,method='full_block',
                            block_cap=1 if name=='matched_single' else 4)
            extraction=time.perf_counter()-start
            start=time.perf_counter(); u.project_bci(core,G,H,V,boundary_epsilon=1e-3)
            projection=time.perf_counter()-start
            row={'seed':args.seed,'method':name,'repeat':rep,'rank':V.shape[1],
                 'offline_seconds':extraction+projection,'extraction_seconds':extraction,
                 'projection_seconds':projection,'full_solves':s['full_solves'],
                 'checks':s['checks'],'preconditioners':s['preconditioners'],
                 'basis_sha256':hashlib.sha256(V.tobytes()).hexdigest()}
            rows.append(row); spaces[name]=V
            with (args.output/'rows.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        gaps.append(compare_spaces(spaces['legacy_cached_single'],spaces['matched_single']))
    med={name:statistics.median(r['offline_seconds'] for r in rows if r['method']==name) for name in names}
    result={'seed':args.seed,'completed':True,'scalar_space_relative_gaps':gaps,
            'median_offline_seconds':med,
            'batch_speed_same_kernel':med['matched_single']/med['matched_block'],
            'cache_speed_scalar':med['legacy_cached_single']/med['matched_single'],
            'note':'Same native 9072-cell,16-port case and epsilon1e-4; this controlled follow-up does not use new holdout information or change the primary algorithms.'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
