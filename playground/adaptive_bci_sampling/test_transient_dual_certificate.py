#!/usr/bin/env python3
"""Check primal/dual BDF1 step bounds on a coupled SPD thermal system."""

import unittest

import numpy as np
import scipy.sparse as sp

from compare_transient import step_transfer
from transient_dual_certificate import prepare_step_certificate


class TransientDualCertificateTests(unittest.TestCase):
    def test_different_dual_space_bounds_every_transfer_and_time(self):
        K = sp.csc_matrix([
            [4.0, -0.8, 0.0, 0.0], [-0.8, 3.5, -0.7, 0.0],
            [0.0, -0.7, 3.0, -0.5], [0.0, 0.0, -0.5, 2.8],
        ])
        C = sp.diags([2.0, 1.2, 0.7, 1.5], format="csc")
        H = [sp.diags([1.0, 0.0, 0.3, 0.0], format="csc"),
             sp.diags([0.0, 0.2, 0.0, 1.0], format="csc")]
        G = np.array([[1.0, 0.0], [0.0, 0.0],
                      [0.0, 0.8], [0.0, 0.0]])
        V = np.ones((4, 1)) / 2.0
        W = np.column_stack((V, np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2)))
        ranges = np.array([[0.2, 5.0], [0.2, 5.0]])
        for p in (np.array([0.2, 5.0]), np.array([3.0, 0.2])):
            A = K + p[0] * H[0] + p[1] * H[1]
            exact = step_transfer(A, C, G, dt=0.5, duration=3.0)
            enriched = prepare_step_certificate(K, C, H, G, ranges, V, W, dt=0.5)
            rom, correction, bound = enriched.evaluate(p, steps=6)
            self.assertEqual(bound.shape, exact.shape)
            self.assertTrue(np.all(np.abs(exact - rom) <= bound + 1e-12))
            same = prepare_step_certificate(K, C, H, G, ranges, V, V, dt=0.5)
            _rom, zero_correction, same_bound = same.evaluate(p, steps=6)
            self.assertLess(np.max(np.abs(zero_correction)), 1e-12)
            self.assertTrue(np.all(np.abs(exact - rom) <= same_bound + 1e-12))
            full = prepare_step_certificate(
                K, C, H, G, ranges, V, np.eye(4), dt=0.5
            )
            _rom, exact_correction, full_bound = full.evaluate(p, steps=6)
            self.assertLess(np.max(np.abs((exact - rom) - exact_correction)), 1e-12)
            self.assertLess(np.max(np.abs(full_bound - np.abs(exact_correction))), 1e-8)


if __name__ == "__main__":
    unittest.main()
