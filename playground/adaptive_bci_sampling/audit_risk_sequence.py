"""Finite-horizon Bernoulli dynamic-programming audit, not Monte Carlo.

Enumerates probability mass by (sample count, rejection count), absorbs the
actual anytime acceptance/rejection boundaries, and reports type-I probability.
An experimental audit, not a unit-test suite or a floating-point proof.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from risk_sequence import RiskEvidence


def audit(rho,delta,cap,p):
    evidence=RiskEvidence(rho,delta,cap)
    mass=np.ones(1);accepted=rejected=expected=0.
    for n in range(1,cap+1):
        updated=np.zeros(n+1)
        updated[:-1]+=mass*(1-p)
        updated[1:]+=mass*p
        k=np.arange(n+1)
        fields=(n-k[:,None])*evidence.good[None,:]
        fields[:,1:]+=k[:,None]*evidence.bad[None,1:]
        fields[k>0,0]=-np.inf
        current=logsumexp(fields,axis=1)-np.log(len(evidence.good))
        future=logsumexp(fields+(cap-n)*evidence.good,axis=1)-np.log(len(evidence.good))
        accept=current>=evidence.threshold
        reject=(future<evidence.threshold)|((n==cap)&~accept)
        reject &= ~accept
        am=float(updated[accept].sum());rm=float(updated[reject].sum())
        accepted+=am;rejected+=rm;expected+=n*(am+rm)
        updated[accept|reject]=0
        mass=updated
    return dict(p=p,accept_probability=accepted,reject_probability=rejected,
                expected_samples=expected,probability_defect=abs(accepted+rejected-1))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--risk',type=float,default=.01)
    parser.add_argument('--delta',type=float,default=1e-9)
    parser.add_argument('--cap',type=int,default=4200)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    rows=[audit(a.risk,a.delta,a.cap,p) for p in [0,a.risk/10,a.risk/2,a.risk,2*a.risk]]
    a.output.write_text(json.dumps(dict(risk=a.risk,delta=a.delta,cap=a.cap,rows=rows,
        scope='ordinary floating-point finite-horizon probability enumeration; exact-arithmetic validity comes from the supermartingale proof'),indent=2)+'\n')
    for row in rows:
        print(json.dumps(row),flush=True)
