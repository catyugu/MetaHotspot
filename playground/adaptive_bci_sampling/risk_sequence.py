"""Anytime-valid test of a Bernoulli certificate rejection mass >= rho.

Fixed finite mixture of likelihood-ratio supermartingales, including q=0.
The sample cap is computational, not a relaxation of the acceptance threshold.
"""
import numpy as np
from scipy.special import logsumexp


class RiskEvidence:
    def __init__(self, rho, delta, cap):
        if not 0 < rho < 1 or not 0 < delta < 1 or cap < 1:
            raise ValueError('invalid risk, confidence, or cap')
        alternatives = rho*np.asarray([0., 1/16, 1/4, 1/2])
        self.good = np.log1p(-alternatives)-np.log1p(-rho)
        self.bad = np.full_like(alternatives, -np.inf)
        self.bad[1:] = np.log(alternatives[1:]/rho)
        self.log_capitals = np.zeros_like(alternatives)
        self.threshold = -np.log(delta)
        self.cap = cap
        self.n = self.failures = 0
        if logsumexp(cap*self.good)-np.log(len(self.good)) < self.threshold:
            raise ValueError('sample cap cannot accept even an all-good stream')

    @property
    def log_evidence(self):
        return float(logsumexp(self.log_capitals)-np.log(len(self.good)))

    def update(self, failed):
        self.n += 1
        self.failures += int(failed)
        self.log_capitals += self.bad if failed else self.good
        if self.log_evidence >= self.threshold:
            return 'accept'
        remaining = self.cap-self.n
        optimistic = logsumexp(self.log_capitals+remaining*self.good)-np.log(len(self.good))
        if remaining <= 0 or optimistic < self.threshold:
            return 'reject'
        return 'continue'
