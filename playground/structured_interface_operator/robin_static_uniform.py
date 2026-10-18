"""Uniform-over-all-HTC static Schur monotonicity (exact arithmetic claim).

If T=Schur_z(K0), Tr=Galerkin_Schur_z(K0), and T>0,
Tr>=T. Let delta=max generalized eigenvalue(Tr-T,T). Then for any
Robin H>=0 acting only on the retained internal boundary face b:
   0 <= Schur_b(Tr+H)-Schur_b(T+H) <= delta Schur_b(T+H).
Proof: T <= Tr <= (1+delta)T, positive H, Schur monotonicity
and positive homogeneity. This is DC ONLY; not a dynamic Hinf guarantee.
"""
import argparse
import numpy as np
from scipy import linalg as la
from robin_model import make_model, basis_for, schur, reduced_schur_on_ports, robin_eliminate


def analyze(nx=12,ny=15,ranks=(36,40,56,72)):
    model=make_model(nx,ny)
    g=model['top'];b=model['bottom'];ab=np.r_[g,b]
    u=schur(model['k'],ab)
    alpha=float(la.eigvalsh(u)[0]);m=len(g)
    print('STATIC_UNIFORM_SETUP',{'top_physical_ports':m,'robin_internal_unknowns':len(b),
           'static_intermediate_SPD_lambda_min':alpha},flush=True)
    for rank in ranks:
        try:
            project=basis_for(model,'trace_exact',rank)['p']
        except ValueError as ex:
            print('SKIP',rank,str(ex));continue
        ur=reduced_schur_on_ports(model['k'],ab,project)
        e=(ur-u+ur.T-u.T)/2
        delta=float(max(0,la.eigh(e,u,eigvals_only=True)[-1]))
        eigerr=float(la.eigvalsh(e)[0])
        worst=(0,None);maxabs=0;max_rel=0
        for h1 in (0.,.008,.04,.4,3.,100.,1e5):
            for h2 in (0.,.008,.04,.4,3.,100.,1e5):
                hv=np.r_[np.full(len(b)//2,h1),np.full(len(b)-len(b)//2,h2)]
                se=robin_eliminate(u,hv,m)
                sr=robin_eliminate(ur,hv,m)
                d=(sr-se+sr.T-se.T)/2
                evals=la.eigh(d,se,eigvals_only=True)
                rel=float(max(0,evals[-1]));negl=float(evals[0])
                if rel>worst[0]:worst=(rel,(h1,h2))
                maxabs=max(maxabs,la.norm(sr-se,2))
                max_rel=max(max_rel, max(0.,-negl))
        print('STATIC_UNIFORM',dict(budget=rank,actual=project.shape[1],
            delta_all_h_exact_arithmetic_bound=delta,
            worst_sampled_gen_eigenvalue=worst,
            max_abs_Schur_error_sampled=maxabs,
            numerical_negative_eigenvalue_defect=max_rel,
            aux_delta_psd_min_eigenvalue=eigerr),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--nx',type=int,default=12)
    p.add_argument('--ny',type=int,default=15)
    p.add_argument('--ranks',type=int,nargs='+',default=[36,40,56,72])
    a=p.parse_args();analyze(a.nx,a.ny,a.ranks)