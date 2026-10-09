"""Continuous centered-domain diagnostics; no additional full inverse RHS."""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from coupled_feedback_enclosure import Certificate
from coupled_galerkin_tail import DirectTail


def run(args):
    with args.archive.open('rb') as handle:tail=pickle.load(handle)
    if not hasattr(tail,'gram'):tail.gram=tail.dual.T@tail.dual
    c=tail.cert;mid=np.sqrt(c.ranges[:,0]*c.ranges[:,1]);rows=[]
    for ratio in [1.,1.22,1.82,3.,10.,30.,100.]:
        hbox=np.column_stack([np.maximum(c.ranges[:,0],mid/np.sqrt(ratio)),np.minimum(c.ranges[:,1],mid*np.sqrt(ratio))])
        box=hbox-c.ranges[:,0,None];bound=tail.cell(box)
        rows.append({'requested_axis_endpoint_ratio':ratio,'effective_h_box':hbox.tolist(),'bounds':bound,
                     'passed_steady':bool(bound['bound_K'] is not None and bound['bound_K']<=.001)})
    out={'n':len(c.c),'s':c.s,'V_order':c.V.shape[1],'new_full_inverse_RHS':0,'continuous_boxes':rows,
         'all_time_step_certified':False,'floating_point_certified':False}
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
