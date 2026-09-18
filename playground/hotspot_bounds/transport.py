"""Spatial comparison supersolution repaired after approximate inverse actions.

No positivity, convergence, or accuracy assumption is made about the approximate
inverse. The final row inequality is repaired using the positive comparison
vectors already checked by Certificate. The result is intersected with the
cheap original bound. This remains guarded floating point, not interval code.
"""
from __future__ import annotations

import numpy as np

from bounds import Certificate


class TransportCertificate(Certificate):
    """Retain spatial shape when propagating the nonnegative error drive.

    For q >= D b_previous + abs(r), take any z approximating A^-1 q. Then
        gamma_j = max(0, max_i (q - A z)_i / (A w_j)_i)
        A (z + gamma_j w_j) >= q.
    Thus z + gamma_j w_j is an upper bound for abs(error), regardless of how z
    was computed. This tests whether one or two existing AMG cycles can buy a
    tighter enclosure more cheaply than additional full-accuracy time steps.
    """

    def __init__(self, A, D, weights, inverse_action, cycles=1):
        super().__init__(A, D, weights)
        if cycles < 1:
            raise ValueError('cycles must be positive')
        self.inverse_action = inverse_action
        self.cycles = cycles
        self.inverse_calls = 0

    def bound(self, estimate, previous, prior, forcing):
        cheap = super().bound(estimate, previous, prior, forcing)
        residual = forcing + self.D * previous - self.A @ estimate
        scale = np.abs(forcing) + self.D * np.abs(previous) + self.abs_A @ np.abs(estimate)
        q = self.D * prior + np.abs(residual) + self.guard * scale
        q = np.nextafter(q * (1 + self.guard), np.inf)
        z = np.zeros_like(q)
        for _ in range(self.cycles):
            z += self.inverse_action(q - self.A @ z)
            self.inverse_calls += 1
        if not np.isfinite(z).all():
            raise RuntimeError('nonfinite approximate inverse output')
        defect = q - self.A @ z
        defect += self.guard * (np.abs(q) + self.abs_A @ np.abs(z))
        gamma = np.maximum(0., np.max(defect[:, None] / self.AW_lower, axis=0))
        repaired = z[:, None] + self.W * gamma[None, :]
        repaired += self.guard * (np.abs(z[:, None]) + self.W * gamma[None, :])
        b = np.maximum(0., np.min(repaired, axis=1))
        return np.nextafter(np.minimum(cheap, b) * (1 + self.guard), np.inf)
