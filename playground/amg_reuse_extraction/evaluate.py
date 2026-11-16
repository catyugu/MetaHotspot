#!/usr/bin/env python3
"""Validate every measured ROM against independent tight-tolerance affine FOMs."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'playground/bci_rom_testcase1'))
from model_case1 import Case1Config,Case1Model
from metahotspot.macromodel import utils as u


def tight_solver(A, audit):
    A=A.tocsr()
    M=u._rs_preconditioner(A)
    def solve(b,x0=None):
        x,info=sla.cg(A,b,M=M,x0=x0,rtol=1e-10,atol=0.,maxiter=2000)
        if info: raise RuntimeError(f'FOM CG info={info}')
        relative=float(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        audit.append(relative)
        return x
    return solve


def evaluate(output):
    start=time.perf_counter()
    m=Case1Model(Case1Config())
    K,C,G,H=m.core.K,m.core.C,m.source_shape,m.boundary_terms
    power=np.full(4,5.)
    b=G@power
    rng=np.random.default_rng(74021)
    points=[list(h) for h in itertools.product([1.,1e4],repeat=3)]
    points.append([100.,1000.,30.])
    points.extend((10.**rng.uniform(0,4,(4,3))).tolist())
    transient_ids={0,7,8}
    records=[]
    for path in sorted(output.glob('*_2026080[567].json')):
        d=json.loads(path.read_text())
        V=np.load(path.with_name(path.stem+'_basis.npy'))
        reduced=u.project_bci(m.core,G,H,V)
        records.append(dict(tag=path.stem,V=V,reduced=reduced,points=[],method=d['method'],seed=d['seed']))
    if len(records)!=9: raise RuntimeError(f'Expected nine full runs, found {len(records)}')
    audit=[]
    for i,h in enumerate(points):
        print('VALIDATION',i,h,flush=True)
        p=m.physical_to_effective(h)
        Kh=K.copy()
        for q,term in zip(p,H,strict=True): Kh=Kh+q*term
        full=tight_solver(Kh,audit)(b)
        scale=float(np.max(np.abs(full)))
        junction=G.T@full
        local=[]
        steady_fields={}
        for record in records:
            V=record['V']; Cr,Kr,Gr,F,A=record['reduced']
            Kh_r=u.assemble_reduced_k(Kr,F,A,p).toarray()
            theta=la.solve(Kh_r,Gr@power,assume_a='pos')
            field=V@theta
            steady_fields[(record["method"],record["seed"])]=field
            metrics=dict(h=h,steady_field_error_K=float(np.max(np.abs(field-full))),
                         steady_junction_error_K=float(np.max(np.abs(Gr.T@theta-junction))),
                         steady_field_relative=float(np.max(np.abs(field-full))/scale),
                         reference_max_rise_K=scale)
            record['points'].append(metrics)
            if i in transient_ids:
                lhs=Cr.toarray()/20.+Kh_r
                factor=la.cho_factor(lhs)
                local.append((record,np.zeros(V.shape[1]),factor,Cr.toarray(),Gr))
                metrics.update(transient_field_error_K=0.,transient_junction_error_K=0.,transient_field_relative=0.)
        for record in records:
            baseline=steady_fields[('fresh',record['seed'])]
            record['points'][-1]['steady_delta_vs_fresh_K']=float(np.max(np.abs(steady_fields[(record['method'],record['seed'])]-baseline)))
        if i in transient_ids:
            full_step=tight_solver(Kh+C/20.,audit)
            x=np.zeros(K.shape[0])
            for step in range(1,51):
                x=full_step(C@x/20.+b,x0=x)
                j=G.T@x
                for k,(record,theta,factor,Cr,Gr) in enumerate(local):
                    theta=la.cho_solve(factor,Cr@theta/20.+Gr@power)
                    field=record['V']@theta
                    point=record['points'][-1]
                    error=float(np.max(np.abs(field-x)))
                    point['transient_field_error_K']=max(point['transient_field_error_K'],error)
                    point['transient_field_relative']=max(point['transient_field_relative'],error/scale)
                    point['transient_junction_error_K']=max(point['transient_junction_error_K'],float(np.max(np.abs(Gr.T@theta-j))))
                    local[k]=(record,theta,factor,Cr,Gr)
    result=dict(reference_max_true_residual=max(audit),reference_residuals=audit,seconds=time.perf_counter()-start,runs=[])
    for record in records:
        record.pop('V');record.pop('reduced')
        record['worst']={key:max(p.get(key,0.) for p in record['points']) for key in ('steady_field_error_K','steady_junction_error_K','steady_field_relative','transient_field_error_K','transient_junction_error_K','transient_field_relative','steady_delta_vs_fresh_K')}
        result['runs'].append(record)
    (output/'validation.json').write_text(json.dumps(result,indent=2))
    print('VALIDATED',result['seconds'],'reference residual',result['reference_max_true_residual'],flush=True)
    for record in result['runs']: print(record['tag'],record['worst'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results');args=parser.parse_args()
    with threadpool_limits(limits=1):evaluate(args.output)
