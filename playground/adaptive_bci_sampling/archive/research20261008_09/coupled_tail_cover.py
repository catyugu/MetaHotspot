"""Deterministic covering of the ORIGINAL box, reusing a fixed inverse Gram.

A leaf is accepted only by an analytic cell inequality. Splitting does not add
snapshots, polynomial degrees, full inverse RHS, or empirical tail assumptions.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse
import heapq
import json
import pickle
import time
from pathlib import Path
import numpy as np
from coupled_feedback_enclosure import Certificate
from coupled_galerkin_tail import DirectTail


def run(args):
    start=time.perf_counter()
    with args.archive.open('rb') as handle:obj=pickle.load(handle)
    if isinstance(obj,DirectTail) and not hasattr(obj,'gram'):obj.gram=obj.dual.T@obj.dual
    cert=obj.cert if isinstance(obj,DirectTail) else obj
    if getattr(obj,'parent_steady_only',False) and args.metric!='K':raise ValueError('nested trial only certifies parent steady K error')
    whole=np.column_stack([np.zeros(len(cert.ranges)),cert.ranges[:,1]-cert.ranges[:,0]])
    heap=[];serial=0;evaluations=0
    key='bound_K' if args.metric=='K' else 'bound_C'
    def add(box):
        nonlocal serial,evaluations
        row=obj.cell(box);value=row[key];priority=float('inf') if value is None else value
        heapq.heappush(heap,(-priority,serial,box,row));serial+=1;evaluations+=1
    add(whole)
    while len(heap)+3<=args.leaves and -heap[0][0]>args.target:
        _,_,box,_=heapq.heappop(heap)
        mid=np.expm1(np.log1p(box).mean(axis=1))
        for i in range(2):
            for j in range(2):
                add(np.array([[box[0,0] if i==0 else mid[0],mid[0] if i==0 else box[0,1]],
                              [box[1,0] if j==0 else mid[1],mid[1] if j==0 else box[1,1]]]))
    rows=[{'delta_box':b.tolist(),'bounds':r} for _,_,b,r in heap]
    valid=[r['bounds'][key] for r in rows if r['bounds'][key] is not None]
    out={'n':len(cert.c),'s':cert.s,'V_order':cert.V.shape[1],'metric':args.metric,'target_error':args.target,
         'target_parent_V_order':getattr(obj,'parent_order',cert.V.shape[1]),
         'target':'original parent steady K error' if getattr(obj,'parent_steady_only',False) else 'specified Galerkin V',
         'leaves':len(heap),'cell_evaluations':evaluations,'new_full_inverse_RHS':0,
         'unresolved':len(rows)-len(valid),'max_bound':max(valid,default=None),
         'analytic_inequality_passed':bool(len(valid)==len(rows) and max(valid)<=args.target),
         'scope':'entire original continuous effective-Robin parameter box at listed fixed real s',
         'all_time_step_certified':False,'floating_point_certified':False,'seconds':time.perf_counter()-start,'cover':rows}
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cover'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--leaves',type=int,default=512)
    p.add_argument('--metric',choices=['K','C'],default='K');p.add_argument('--target',type=float,default=.001)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
