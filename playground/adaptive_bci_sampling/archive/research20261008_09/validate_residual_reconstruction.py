"""Independent pointwise rational reference diagnostics, not an acceptance oracle.

Two independent richer shifted spaces are compared. Their agreement is NOT a
rigorous FOM error certificate. All full inverse actions use AMG-CG and are counted.
"""

# Archived driver: historical local modules first, shared Case1 assembly second.
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(1, str(_ArchivePath(__file__).resolve().parents[2]))

import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from case1_system import case1_reconstruction
from residual_reconstruction import operator,Solver,decay_lower,build_basis


def difference_metric(Z,F,mass,reference_mass,cross):
    """Physical C-energy metric, including non-C-orthonormal comparator bases."""
    Q=F.T@reference_mass@F;mixed=Z.T@cross@F
    error=Z.T@mass@Z+Q-mixed-mixed.T;error=(error+error.T)/2
    return float(np.sqrt(max(0.,la.eigvalsh(error,Q)[-1])))


def validate(mesh,archive,output,candidate_shifts=20,center_only=False,reference_counts=(28,40)):
    K,C,G,H,ranges,metadata=case1_reconstruction(mesh);center=np.sqrt(ranges[:,0]*ranges[:,1])
    if archive.exists():
        with np.load(archive) as a:bases={name:a[name] for name in a.files}
    else:
        # Reconstruct deterministic candidate only, keeping this cost visible.
        bases={}
    report=dict(n=len(G),mesh_mm=mesh,reference_scope='pointwise richer-rational-space diagnostics, not exact FOM reference',points=[],metrics={name:0. for name in bases})
    box=np.column_stack([center-.00005*(center-ranges[:,0]),center+.00005*(ranges[:,1]-center)])
    for h in ([center] if center_only else [center,box[:,0],box[:,1]]):
        start=time.perf_counter();A=operator(K,H,h);solver=Solver(A);lower=decay_lower(A,C.diagonal(),solver)
        s=sp.diags(1/np.sqrt(C.diagonal()));upper=float(np.max(np.asarray(abs(s@A@s).sum(axis=1))))
        refs=[];counts=[]
        for number in reference_counts:
            shifts=np.r_[0.,np.geomspace(lower/20,upper*20,number-1)]
            V,cnt=build_basis(K,C,G,H,np.column_stack([h,h]),shifts,parameter_points=[h]);refs.append(V);counts.append(cnt)
        if not bases:
            shifts=np.r_[0.,np.geomspace(lower/10,upper*10,candidate_shifts-1)]
            v,cnt=build_basis(K,C,G,H,np.column_stack([center,center]),shifts,parameter_points=[center])
            bases={'candidate':v};report['metrics']={'candidate':0.};report['reconstructed_candidate_counts']=cnt
        coarse=f'reference{reference_counts[0]}';fine=f'reference{reference_counts[1]}'
        allbases={**bases,coarse:refs[0],fine:refs[1]}
        modal={};ref=refs[1]
        for name,v in allbases.items():
            mass=v.T@(C@v)
            l,W=la.eigh(v.T@(A@v),mass);modal[name]=(l,W,W.T@(v.T@G),v.T@(C@ref),mass)
        lr,Wr,Br,_,Mr=modal[fine]
        metrics={name:0. for name in allbases if name!=fine}
        times=np.geomspace(1e-7/upper,40/lower,120)
        for t in times:
            F=Wr@((-np.expm1(-lr*t)/lr)[:,None]*Br);Q=F.T@Mr@F
            for name,(l,W,B,cross,mass) in modal.items():
                if name==fine:continue
                Z=W@((-np.expm1(-l*t)/l)[:,None]*B)
                value=difference_metric(Z,F,mass,Mr,cross)
                metrics[name]=max(metrics[name],value)
        for name in report['metrics']:report['metrics'][name]=max(report['metrics'][name],metrics[name])
        row=dict(h=h.tolist(),metrics=metrics,reference_orders=[v.shape[1] for v in refs],reference_counts=counts,
                 spectral_setup_counts=solver.counts(),times=len(times),seconds=time.perf_counter()-start)
        report['points'].append(row);output.write_text(json.dumps(report,indent=2)+'\n')
        print('VALIDATE',mesh,row,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh-mm',type=float,required=True)
    p.add_argument('--center-only',action='store_true');p.add_argument('--reference-counts',type=int,nargs=2,default=[28,40]);p.add_argument('--candidate-shifts',type=int,default=20);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();validate(a.mesh_mm,a.archive,a.output,a.candidate_shifts,a.center_only,a.reference_counts)
