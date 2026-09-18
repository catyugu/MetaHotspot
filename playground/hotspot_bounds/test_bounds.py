"""Algebraic and temporal-history tests independent of MetaHotspot/AMG."""
import unittest

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from bounds import Certificate, decide, solve_checked, replay


def problem(n=17):
    rng = np.random.default_rng(42)
    edges = rng.uniform(0.1, 5.0, n - 1)
    K = sp.diags([-edges, np.r_[edges, 0] + np.r_[0, edges] + 0.1, -edges],
                 [-1, 0, 1], format="csr")
    D = rng.uniform(0.2, 2.0, n)
    A = K + sp.diags(D)
    W = np.column_stack([np.ones(n), sla.spsolve(K, D)])
    return A, D, W


class BoundsTests(unittest.TestCase):
    def test_reject_positive_offdiagonal(self):
        with self.assertRaises(ValueError):
            Certificate(sp.csr_matrix([[3., 1.], [1., 3.]]), np.ones(2))

    def test_reject_nonpositive_capacity(self):
        with self.assertRaises(ValueError):
            Certificate(sp.eye(2, format="csr"), np.array([1., 0.]))

    def test_reject_nondominant_operator(self):
        with self.assertRaises(ValueError):
            Certificate(sp.csr_matrix([[1., -2.], [-2., 1.]]), np.ones(2))

    def test_reject_invalid_weights(self):
        A, D, _ = problem()
        with self.assertRaises(ValueError):
            Certificate(A, D, -np.ones((len(D), 1)))

    def test_comparison_bound_against_dense_inverse(self):
        A, D, W = problem()
        cert = Certificate(A, D, W)
        rng = np.random.default_rng(1)
        for _ in range(20):
            previous = rng.normal(size=len(D))
            prior = rng.uniform(0.0, 0.3, len(D))
            true_previous = previous + rng.uniform(-1, 1, len(D)) * prior
            forcing = rng.normal(size=len(D))
            estimate = rng.normal(size=len(D))
            exact = np.linalg.solve(A.toarray(), D * true_previous + forcing)
            b = cert.bound(estimate, previous, prior, forcing)
            self.assertTrue(np.all(np.abs(exact - estimate) <= b + 1e-12))

    def test_history_must_not_be_reset_after_current_step_solve(self):
        cert = Certificate(sp.csr_matrix([[2.]]), np.ones(1))
        previous = np.array([0.])  # Actual preceding state is 1, within prior=1.
        estimate = np.array([0.])  # Exact solve of the APPROXIMATE local RHS.
        correct = cert.bound(estimate, previous, np.array([1.]), np.zeros(1))
        incorrect = cert.bound(estimate, previous, np.zeros(1), np.zeros(1))
        self.assertGreaterEqual(correct[0], 0.5)
        self.assertLess(incorrect[0], 0.5)

    def test_peak_location_may_differ_between_endpoints(self):
        lo, hi = Certificate.peak(np.array([10., 9.]), np.array([0.1, 5.]))
        self.assertLessEqual(lo, 9.9)
        self.assertGreaterEqual(hi, 14.)

    def test_intersection_of_barriers_remains_valid(self):
        A, D, W = problem()
        rng = np.random.default_rng(3)
        x = rng.normal(size=len(D))
        args = (x, np.zeros(len(D)), np.ones(len(D)) * .1, rng.normal(size=len(D)))
        both = Certificate(A, D, W).bound(*args)
        first = Certificate(A, D, W[:, :1]).bound(*args)
        second = Certificate(A, D, W[:, 1:]).bound(*args)
        np.testing.assert_allclose(both, np.minimum(first, second), rtol=1e-12)

    def test_many_steps_with_nonuniform_capacity_and_signed_load(self):
        A, D, W = problem()
        cert = Certificate(A, D, W)
        rng = np.random.default_rng(9)
        true = np.zeros(len(D))
        previous = true.copy()
        prior = true.copy()
        for _ in range(30):
            f = rng.normal(size=len(D))
            true = sla.spsolve(A, D * true + f)
            estimate = sla.spsolve(A, D * previous + f) + rng.normal(0, .001, len(D))
            prior = cert.bound(estimate, previous, prior, f)
            self.assertTrue(np.all(np.abs(true - estimate) <= prior + 1e-12))
            previous = estimate

    def test_unknown_is_not_safe(self):
        self.assertEqual(decide(9., 11., 10.), "unknown")
        self.assertEqual(decide(9., 10., 10.), "unknown")
        self.assertEqual(decide(9., 9.9, 10.), "safe")
        self.assertEqual(decide(10.1, 11., 10.), "unsafe")

    def test_checked_solver_accepts_without_iteration_only_if_decisive(self):
        A, D, W = problem()
        cert = Certificate(A, D, W)
        n = len(D)
        result = solve_checked(cert, np.zeros(n), np.zeros(n), np.zeros(n),
                               np.zeros(n), lambda x: x, threshold=10., width=.1)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(decide(result.lower, result.upper, 10.), "safe")

    def test_checked_solver_resolves_nontrivial_rhs(self):
        A, D, W = problem()
        cert = Certificate(A, D, W)
        n = len(D)
        f = np.linspace(1., 3., n)
        result = solve_checked(cert, np.zeros(n), np.zeros(n), f,
                               np.zeros(n), lambda x: x / A.diagonal(), width=1e-5)
        exact = sla.spsolve(A, f)
        self.assertGreater(result.iterations, 0)
        self.assertTrue(result.accepted)
        self.assertTrue(np.all(np.abs(result.state - exact) <= result.radius + 1e-12))
        self.assertLessEqual(result.upper - result.lower, 1e-5)

    def test_inherited_uncertainty_cannot_be_removed_by_local_iterations(self):
        cert = Certificate(sp.csr_matrix([[2.]]), np.ones(1))
        result = solve_checked(cert, np.array([0.]), np.array([1.]), np.array([0.]),
                               np.array([0.]), lambda x: x / 2, width=.01)
        self.assertFalse(result.accepted)
        self.assertGreater(result.upper - result.lower, .99)

    def test_replay_restarts_from_checkpoint_not_uncertain_recent_state(self):
        cert = Certificate(sp.csr_matrix([[2.]]), np.ones(1))
        fs = [np.array([1.]), np.array([0.]), np.array([2.])]
        x, b, steps, iterations = replay(cert, fs, 0, 3, np.zeros(1), np.zeros(1),
                                         lambda x: x / 2)
        self.assertAlmostEqual(x[0], 1.125, places=11)
        self.assertLess(b[0], 1e-9)
        self.assertEqual(steps, 3)
        self.assertGreater(iterations, 0)


if __name__ == "__main__":
    unittest.main()
