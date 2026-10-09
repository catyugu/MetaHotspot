"""Same archived candidate, same box: mathematical certificate ablation."""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import json
import math
from pathlib import Path
import numpy as np
from case1_system import case1_reconstruction
from polynomial_trajectory import certify_trajectory


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mesh-mm',type=float,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--basis',default='candidate')
    p.add_argument('--widths',type=float,nargs='+',default=[0,.000003,.00005,.001,1])
    p.add_argument('--degree',type=int,default=2)
    p.add_argument('--time-ratio',type=float,default=1.1)
    p.add_argument('--rational-denominator-only',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();K,C,G,H,ranges,meta=case1_reconstruction(a.mesh_mm)
    V=np.load(a.archive)[a.basis];center=np.sqrt(ranges[:,0]*ranges[:,1])
    report=dict(mesh_mm=a.mesh_mm,n=len(G),metadata=meta,archive=str(a.archive),basis=a.basis,
                full_ranges=ranges.tolist(),certificates=[],degree=a.degree,
                extraction_reused=True,extraction_free=False)
    for width in a.widths:
        box=np.column_stack([center-width*(center-ranges[:,0]),center+width*(ranges[:,1]-center)])
        row=certify_trajectory(K,C,G,H,V,box,degree=a.degree,time_ratio=a.time_ratio,
                              center_denominator=not a.rational_denominator_only)
        row.update(width_fraction=width,ranges=box.tolist(),
                   accepted_step_analytic=bool(row['bound']<=np.sqrt(.001)),
                   accepted_steady_analytic=bool(row['steady_bound']<=.001),
                   accepted_analytic=bool(row['bound']<=np.sqrt(.001) and row['steady_bound']<=.001))
        report['certificates'].append(row)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(finite_json(report),indent=2,allow_nan=False)+'\n')
        print('CERT',len(G),width,row['bound'],row['steady_bound'],row['lift_defect_convolution'],row['reduced_equation_defect_convolution'],row['seconds'],flush=True)
