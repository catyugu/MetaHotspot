"""Tests for new block kernels; these do not require local AMG installation."""
import unittest
import numpy as np
import scipy.linalg as la
from block_extract import choose_rank, residual_tangents, ritz_corrections


class BlockTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        L = rng.normal(size=(12, 12))
        self.A = L @ L.T + np.eye(12)
        self.V = la.qr(rng.normal(size=(12, 3)), mode='economic')[0]
        B = rng.normal(size=(12, 4))
        self.AV = self.A @ self.V
        self.Ar = self.V.T @ self.AV
        self.R = B - self.AV @ la.solve(self.Ar, self.V.T @ B)

    def test_rank_fraction_and_cap(self):
        self.assertEqual(choose_rank(np.array([9., 1., 0.]), .9, 4), 1)
        self.assertEqual(choose_rank(np.ones(8), .9, 4), 4)
        self.assertEqual(choose_rank(np.zeros(2), .9, 4), 0)

    def test_rank_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            choose_rank(np.ones(3), 1.2, 4)

    def test_tangents_have_orthonormal_columns(self):
        D, _ = residual_tangents(self.R, 4, 1.)
        np.testing.assert_allclose(D.T @ D, np.eye(4), atol=1e-12)

    def test_exact_preconditioner_recovers_entire_missing_response(self):
        Z = la.solve(self.A, self.R)
        W, info = ritz_corrections(self.A, self.R, Z, self.V, self.AV, self.Ar, 4, 1.)
        np.testing.assert_allclose(W @ (W.T @ self.R), Z, rtol=1e-10, atol=1e-10)
        self.assertEqual(info['retained'], 4)

    def test_ritz_is_energy_orthonormal_and_deflated(self):
        Z = self.R / self.A.diagonal()[:, None]
        W, _ = ritz_corrections(self.A, self.R, Z, self.V, self.AV, self.Ar, 3, 1.)
        np.testing.assert_allclose(W.T @ self.A @ W, np.eye(3), atol=2e-12)
        np.testing.assert_allclose(self.V.T @ self.A @ W, 0., atol=2e-12)

    def test_reported_gain_equals_true_energy_error_reduction(self):
        Z = self.R / self.A.diagonal()[:, None]
        W, info = ritz_corrections(self.A, self.R, Z, self.V, self.AV, self.Ar, 2, 1.)
        E = la.solve(self.A, self.R)
        Ep = E - W @ (W.T @ self.R)
        gain = np.sum(E * (self.A @ E)) - np.sum(Ep * (self.A @ Ep))
        self.assertAlmostEqual(gain, info['retained_gain'], places=10)

    def test_best_rank_one_beats_random_trial_directions(self):
        Z = self.R / self.A.diagonal()[:, None]
        W, info = ritz_corrections(self.A, self.R, Z, self.V, self.AV, self.Ar, 1, 1.)
        Z -= self.V @ la.solve(self.Ar, self.AV.T @ Z)
        rng = np.random.default_rng(14)
        for _ in range(30):
            z = Z @ rng.normal(size=4)
            gain = np.sum((z @ self.R)**2) / (z @ self.A @ z)
            self.assertGreaterEqual(info['retained_gain'] + 1e-10, gain)

    def test_dependent_source_columns_do_not_create_spurious_rank(self):
        R = np.tile(self.R[:, :1], (1, 4))
        W, _ = ritz_corrections(self.A, R, la.solve(self.A, R), self.V, self.AV, self.Ar, 4, 1.)
        self.assertEqual(W.shape[1], 1)

    def test_zero_residual_returns_empty(self):
        R = np.zeros_like(self.R)
        W, _ = ritz_corrections(self.A, R, R, self.V, self.AV, self.Ar, 4, .9)
        self.assertEqual(W.shape[1], 0)

    def test_indefinite_trial_energy_is_rejected(self):
        with self.assertRaises(ValueError):
            ritz_corrections(-np.eye(3), np.eye(3), np.eye(3), np.empty((3,0)),
                             np.empty((3,0)), np.empty((0,0)), 2, .9)

    def test_orthogonal_input_rotation_preserves_gains(self):
        D = la.qr(np.random.default_rng(8).normal(size=(4,4)))[0]
        Z = self.R / self.A.diagonal()[:, None]
        _, i1 = ritz_corrections(self.A,self.R,Z,self.V,self.AV,self.Ar,2,1.)
        _, i2 = ritz_corrections(self.A,self.R@D,Z@D,self.V,self.AV,self.Ar,2,1.)
        self.assertAlmostEqual(i1['retained_gain'],i2['retained_gain'],places=10)

    def test_invalid_method_cannot_silently_select_another_algorithm(self):
        from block_extract import extract
        with self.assertRaises(ValueError):
            extract(None,None,None,None,tolerance=1e-3,seed=1,method='unknown')

if __name__ == '__main__':
    unittest.main()
