"""Source-separated random task tournaments with in-loop joint risk acceptance.

See RANDOM_TASK_PROOF.md. No shared response space during snapshot selection.
Certificates concern steady K-energy and ALL-TIME step C-energy before SVD.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from case1_system import case1_reconstruction
from numerics import Solver, operator, c_basis, sym, counts, f as f_local
from probabilistic_extraction import stock, parameters, corners, audit_steady
from diagonal_time_certificate import DiagonalTimeCertificate
from metahotspot.macromodel import utils


class LocalTasks:
    def __init__(self,K,C,H,g,ranges,shifts):
        self.ops=[K]+H+[C]
        self.scales=np.r_[1.,np.sqrt(ranges[:,0]*ranges[:,1]),max(max(shifts),1.)]
        self.ops=[x*A for x,A in zip(self.scales,self.ops)]
        self.g=g
        self.V=np.empty((len(g),0))
        self.Q=(g/la.norm(g))[:,None]
        self.T=np.array([[la.norm(g)]])
        self.drops=[0.]
        self.reduced=[np.empty((0,0)) for _ in self.ops]
        self.F=np.empty(0)
        self.queries=0

    def add(self,x):
        block=utils.orthonormalize_block(self.V,x[:,None])
        if not block.shape[1]:
            return False
        v=block[:,0]
        new=[A@v for A in self.ops]
        # Append affine columns with two-pass QR. Source g is the first column;
        # new basis columns are interleaved across operators.
        for b in new:
            t=self.Q.T@b
            rem=b-self.Q@t
            dt=self.Q.T@rem; t+=dt; rem-=self.Q@dt
            norm=la.norm(rem)
            old=self.T
            # Near-dependent residual blocks must not manufacture inaccurate
            # QR directions. Their discarded norm remains in every query bound.
            if norm > 1e-12*la.norm(b) and self.Q.shape[1]<len(v):
                self.Q=np.column_stack([self.Q,rem/norm])
                self.T=np.zeros((old.shape[0]+1,old.shape[1]+1))
                self.T[:-1,:-1]=old; self.T[:-1,-1]=t; self.T[-1,-1]=norm
                self.drops.append(0.)
            else:
                self.T=np.column_stack([old,t])
                self.drops.append(float(norm))
        for i,b in enumerate(new):
            ar=self.reduced[i]
            cross=self.V.T@b
            self.reduced[i]=np.block([[ar,cross[:,None]],[cross[None,:],np.array([[v@b]])]])
        self.V=np.column_stack([self.V,v]); self.F=self.V.T@self.g
        return True

    def residual(self,h,s):
        h=np.atleast_2d(h); s=np.broadcast_to(s,len(h))
        self.queries+=len(h)
        if not self.V.shape[1]:
            return np.ones(len(h))
        weights=np.column_stack([np.ones(len(h)),h,s])/self.scales
        ar=np.einsum('bi,ijk->bjk',weights,np.asarray(self.reduced))
        y=np.linalg.solve(ar,np.broadcast_to(self.F[None,:,None],(len(h),len(self.F),1)))[:,:,0]
        coeff=np.column_stack([np.ones(len(h)),-(y[:,:,None]*weights[:,None,:]).reshape(len(h),-1)])
        return (la.norm(self.T@coeff.T,axis=0)+np.abs(coeff)@np.asarray(self.drops))/la.norm(self.g)


def run(a):
    start=time.perf_counter()
    K,C,G,H,ranges,meta=case1_reconstruction(a.mesh_mm)
    baseline_start=time.perf_counter()
    stock_pre,stock_post,baseline=stock(K,C,G,H,ranges,a.seed)
    baseline_seconds=time.perf_counter()-baseline_start
    print(json.dumps(dict(stage='stock',n=len(G),rhs=baseline['full_rhs'],seconds=baseline_seconds)),flush=True)
    candidate_start=time.perf_counter()
    plans=[np.r_[0.,p['shifts_per_s']] for p in baseline['per_port_plans']]
    local=[LocalTasks(K,C,H,G[:,j],ranges,plans[j]) for j in range(G.shape[1])]
    rng=np.random.default_rng(a.seed+1)
    # Every task owns its own response space. Initial slopes cost no RHS.
    for j,m in enumerate(local):
        m.add(G[:,j]/C.diagonal())
    solver=Solver(operator(K,H,ranges[:,0]))
    z=solver.solve(C.diagonal())
    if np.any(z<=0):
        raise ValueError('positive ground state required')
    D=(solver.A@z)/z
    entries=solver.A.tocoo()
    if np.any(entries.data[entries.row!=entries.col]>0):
        raise ValueError('M-matrix off-diagonal sign required')
    if np.any(D<=0):
        raise ValueError('positive diagonal certificate required')
    alpha=float(min(D/C.diagonal()))
    costs=[solver.counts()]
    history=[]; epochs=[]
    training=a.training_tolerance
    certificate=None; risk_pass=False; trigger=None; trigger_time=None; force_check=False
    feedback_streak=0
    for iteration in range(a.max_rhs):
        candidates=[]
        if trigger is not None:
            # A rejected all-time/steady check is fed back to task selection.
            # This source/shift choice is discovery only, never an acceptance.
            h=trigger
            feedback_rhs={}
            if a.feedback=='residual':
                A=operator(K,H,h)
                for j,m in enumerate(local):
                    if trigger_time:
                        lam,U=la.eigh(sym(m.V.T@(A@m.V)),sym(m.V.T@(C@m.V)))
                        W=m.V@U; F=W.T@G[:,j]
                        residual=G[:,j]-(C@W)@(np.exp(-lam*trigger_time)*F)-(A@W)@(f_local(trigger_time,lam)*F)
                        value=la.norm(residual/np.sqrt(C.diagonal()))/la.norm(G[:,j]/np.sqrt(C.diagonal()))
                        shift=1./trigger_time
                    else:
                        y=la.solve(sym(m.V.T@(A@m.V)),m.V.T@G[:,j],assume_a='pos')
                        residual=G[:,j]-A@(m.V@y)
                        value=la.norm(residual/np.sqrt(D))/np.sqrt(G[:,j]@(m.V@y))
                        shift=0.
                    feedback_rhs[j]=residual
                    candidates.append((float(value),j,h,shift))
            else:
                for j,m in enumerate(local):
                    values=m.residual(np.repeat(h[None,:],len(plans[j]),axis=0),plans[j])
                    k=int(np.argmax(values)); candidates.append((float(values[k]),j,h,float(plans[j][k])))
            feedback_h=h.copy(); feedback_time=trigger_time
            trigger=None
        else:
            feedback_rhs={}; feedback_h=None; feedback_time=None
            hp=parameters(ranges,rng,a.tournament)
            offset=int(rng.integers(len(local)))
            for ci,h in enumerate(hp):
                if rng.random()<a.corner_mixture:
                    h=ranges[np.arange(len(ranges)),rng.integers(0,2,len(ranges))]
                j=(offset+ci)%len(local)
                values=local[j].residual(np.repeat(h[None,:],len(plans[j]),axis=0),plans[j])
                k=int(np.argmax(values))
                candidates.append((float(values[k]),j,h,float(plans[j][k])))
        value,j,h,s=max(candidates,key=lambda item:item[0])
        if force_check or (not feedback_rhs and value<=training) or (iteration and iteration%a.checkpoint==0):
            force_check=False
            # Snapshot spaces are frozen for this entire acceptance attempt.
            V=c_basis(np.column_stack([m.V for m in local]+[np.ones((len(G),1))]),C.diagonal())
            if len(G)<=2000:
                a.output.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(a.output.with_name(a.output.stem+'_epoch'+str(len(epochs)+1)+'.npz'),V=V)
            certificate=DiagonalTimeCertificate(K,C,H,G,V,D,alpha,a.time_ratio)
            evaluate=(certificate.evaluate_matrix if a.certificate_mode=='matrix' else certificate.evaluate)
            epoch=len(epochs)+1
            delta_epoch=a.delta/(epoch*(epoch+1))
            n=math.ceil(math.log(delta_epoch)/math.log1p(-a.risk))
            vrng=np.random.default_rng(np.random.SeedSequence([a.seed,20261010,epoch]))
            rejected=None; diagnostics=[]; certification_start=time.perf_counter()
            # Corner checks are extra deterministic rejection opportunities;
            # only the following fresh iid log-uniform stream gives risk mass.
            corner_checks=((corners(ranges) if len(ranges)<=4 else
                            np.asarray([ranges[np.arange(len(ranges)),rng.integers(0,2,len(ranges))] for _ in range(16)]))
                           if a.accept_corners else [])
            for hp in corner_checks:
                d=evaluate(hp,a.tolerance)
                diagnostics.append(d)
                if not d['passed']:
                    rejected=hp; break
            checked=0
            if rejected is None:
                for hp in parameters(ranges,vrng,n):
                    d=evaluate(hp,a.tolerance)
                    diagnostics.append(d); checked+=1
                    if not d['passed']:
                        rejected=hp; break
            ep=dict(epoch=epoch,delta=delta_epoch,required=n,checked=checked,
                    accepted=rejected is None,rhs=len(history),order=V.shape[1],
                    certificate_build_seconds=certificate.build_seconds,
                    check_seconds=time.perf_counter()-certification_start,
                    max_steady_bound=max(d.get('steady_bound',float('inf')) for d in diagnostics),
                    max_step_bound=(max(d['step_bound'] for d in diagnostics if d.get('step_bound') is not None)
                                    if any(d.get('step_bound') is not None for d in diagnostics) else None),
                    reject_h=None if rejected is None else rejected.tolist(),
                    last_diagnostic=diagnostics[-1])
            epochs.append(ep)
            print(json.dumps(dict(stage='epoch',n=len(G),**ep)),flush=True)
            if rejected is None:
                risk_pass=True; break
            trigger=rejected
            failure_time=(diagnostics[-1].get('forcing_time') if a.time_feedback=='dominant' else
                          diagnostics[-1].get('failure_time'))
            if failure_time is None:
                failure_time=diagnostics[-1].get('failure_time')
            trigger_time=failure_time
            if failure_time and a.feedback=='input':
                # A temporal rejection proposes new real matching points. The
                # original finite pole list is discovery support, not a theorem
                # that certifies all time or restricts later enrichment.
                extra=np.array([.1,1.,10.])/failure_time
                for jj in range(len(plans)):
                    plans[jj]=np.unique(np.r_[plans[jj],extra])
            # If finite-task residuals are already small but all-time/steady
            # enclosure still fails, tighten discovery, preserving all data.
            scores=[]
            for jj,m in enumerate(local):
                scores.extend(m.residual(np.repeat(rejected[None,:],len(plans[jj]),axis=0),plans[jj]))
            training=min(training,float(max(scores))*.8)
            continue
        so=Solver(operator(K,H,h)+s*C,rtol=1e-10)
        x=so.solve(feedback_rhs.get(j,G[:,j])); costs.append(so.counts())
        if not local[j].add(x/max(la.norm(x),np.finfo(float).tiny)):
            print(json.dumps(dict(stage='dependent_enrichment',port=j,residual=value)),flush=True)
            V=c_basis(np.column_stack([m.V for m in local]+[np.ones((len(G),1))]),C.diagonal())
            break
        history.append(dict(port=j,h=h.tolist(),shift=s,residual=value,
                            kind=('temporal_residual_correction' if feedback_time else 'steady_residual_correction') if feedback_rhs else 'input_response',time=feedback_time))
        if feedback_rhs:
            feedback_streak+=1
            if feedback_streak%a.feedback_batch:
                trigger=feedback_h; trigger_time=feedback_time
            else:
                force_check=True
    else:
        V=c_basis(np.column_stack([m.V for m in local]+[np.ones((len(G),1))]),C.diagonal())
    candidate_seconds=time.perf_counter()-candidate_start
    audit=audit_steady(K,C,H,G,ranges,dict(stock_pre=stock_pre,stock_post=stock_post,task_pre=V),
                       np.random.default_rng(a.seed+100),a.audit)
    out=dict(n=len(G),metadata=meta,seed=a.seed,ranges=ranges.tolist(),stock=baseline,
             stock_wall_seconds=baseline_seconds,candidate_rhs=len(history),total_rhs=sum(c['rhs'] for c in costs),
             costs=counts(costs),candidate_seconds=candidate_seconds,
             common_spectral_plan_seconds=baseline['common_spectral_plan_seconds'],
             risk_accepted=risk_pass,risk=a.risk,delta=a.delta,alpha=alpha,
             epochs=epochs,history=history,local_orders=[m.V.shape[1] for m in local],order=V.shape[1],
             residual_queries=sum(m.queries for m in local),steady_audit=audit,
             configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
             seconds=time.perf_counter()-start,
             scope='joint continuous log-uniform HTC risk of steady and ALL-TIME step, arbitrary signed inputs; pre-SVD, exact-arithmetic theorem; no outward rounding')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(a.output.with_suffix('.npz'),V=V,G=G,ranges=ranges,stock_pre=stock_pre,stock_post=stock_post,alpha=alpha,
                        K0_reduced=sym(V.T@(K@V)),C_reduced=sym(V.T@(C@V)),
                        H_reduced=np.asarray([sym(V.T@(A@V)) for A in H]),F_reduced=V.T@G)
    print(json.dumps(dict(stage='complete',n=len(G),rhs=out['total_rhs'],accepted=risk_pass,
                         stock_rhs=baseline['full_rhs'],seconds=candidate_seconds,order=V.shape[1])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mesh-mm',type=float,default=10)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--tournament',type=int,default=4)
    p.add_argument('--corner-mixture',type=float,default=.5)
    p.add_argument('--checkpoint',type=int,default=32)
    p.add_argument('--feedback',choices=['residual','input'],default='residual')
    p.add_argument('--time-feedback',choices=['dominant','failure'],default='dominant')
    p.add_argument('--accept-corners',action='store_true',help='additional box-style rejection checks; not required for distribution risk')
    p.add_argument('--feedback-batch',type=int,default=4)
    p.add_argument('--training-tolerance',type=float,default=.001)
    p.add_argument('--tolerance',type=float,default=.001)
    p.add_argument('--time-ratio',type=float,default=1.1)
    p.add_argument('--certificate-mode',choices=['matrix','scalar'],default='matrix')
    p.add_argument('--risk',type=float,default=.01)
    p.add_argument('--delta',type=float,default=1e-6)
    p.add_argument('--max-rounds','--max-rhs',dest='max_rhs',type=int,default=180,
                   help='outer-loop round budget including certificate attempts; --max-rhs is a legacy alias')
    p.add_argument('--audit',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not (0<a.risk<1 and 0<a.delta<1 and a.tournament>0 and a.checkpoint>0 and a.time_ratio>1
            and a.max_rhs>0 and a.feedback_batch>0 and a.training_tolerance>0 and a.tolerance>0 and a.audit>=0 and 0<=a.corner_mixture<1):
        p.error('invalid budget, probability or tolerance')
    run(a)
