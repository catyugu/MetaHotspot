"""Small algebraic regressions; research seeds and holdouts are reserved for CI."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg as la


class CommutatorTests(unittest.TestCase):
    def test_projected_defect_identity(self):
        from commutator import projected_defect
        rng = np.random.default_rng(5)
        a, b = [rng.normal(size=(9, 9)) for _ in range(2)]
        a, b = a + a.T, b + b.T
        v = la.qr(rng.normal(size=(9, 3)), mode='economic')[0]
        lhs, rhs, _ = projected_defect(a, b, v)
        assert_allclose(lhs, rhs, atol=3e-13)

    def test_rank_one_small_defect_does_not_imply_no_leakage(self):
        from commutator import projected_defect
        a = np.diag([1., 3., 8.])
        b = np.array([[2., 1., 0.], [1., 3., 1.], [0., 1., 5.]])
        v = np.array([[1.], [0.], [0.]])
        lhs, _, leak = projected_defect(a, b, v)
        assert_allclose(lhs, 0., atol=1e-14)
        self.assertGreater(la.norm(leak), 1.)

    def test_commuting_control_and_preserved_spectrum(self):
        from commutator import make_family
        x = make_family(20, 42, 'low_rank', 0.)
        y = make_family(20, 42, 'low_rank', 0.8)
        for a in x['operators']:
            for b in x['operators']:
                assert_allclose(a @ b, b @ a, atol=1e-11)
        for a, b in zip(x['operators'], y['operators']):
            assert_allclose(la.eigvalsh(a), la.eigvalsh(b), atol=1e-11)
            self.assertGreater(la.eigvalsh(b)[0], 0.)

    def test_low_rank_commutator_with_full_rank_parameter_change(self):
        from commutator import make_family
        a, b, c = make_family(24, 42, 'low_rank', 0.8)['operators']
        self.assertGreater(np.linalg.matrix_rank(c - b), 18)
        self.assertLessEqual(np.linalg.matrix_rank(a @ c - c @ a, tol=1e-9), 8)

    def test_exact_constant_input_step(self):
        from commutator import exact_step
        a = np.diag([1., 3.]); x = np.array([2., -1.]); f = np.array([4., 1.])
        dt = 0.13
        expected = np.exp(-np.diag(a)*dt)*x + (-np.expm1(-np.diag(a)*dt))/np.diag(a)*f
        assert_allclose(exact_step(a, x, f, dt), expected, atol=1e-13)

    def test_commuting_homogeneous_order_control(self):
        from commutator import homogeneous_terminal, make_family
        fam = make_family(16, 42, 'low_rank', 0.)
        path = np.random.default_rng(8).uniform(0., 1., (12, 2))
        x = fam['B'][:, 0]
        first = homogeneous_terminal(fam['operators'], x, path, .05)
        second = homogeneous_terminal(fam['operators'], x, path[::-1], .05)
        assert_allclose(first, second, atol=3e-13)

    def test_generalized_gramian_residual(self):
        from commutator import bilinear_gramian
        rng = np.random.default_rng(4)
        a = np.diag(np.arange(1., 9.))
        ns = [np.diag(np.linspace(0.1, 0.5, 8))]
        b = rng.normal(size=(8, 2))
        p, info = bilinear_gramian(a, ns, b)
        res = -a @ p-p @ a + info['gamma']**2*sum(n @ p @ n for n in ns)+b @ b.T
        self.assertLess(la.norm(res)/la.norm(b @ b.T), 1e-9)
        self.assertGreater(la.eigvalsh(p)[0], -1e-12)


class GramTests(unittest.TestCase):
    def test_full_gram_identity(self):
        from gram import witness_grams, gram_residual
        rng = np.random.default_rng(8)
        z = rng.normal(size=(6, 6)); k0 = z @ z.T + np.eye(6)
        z = rng.normal(size=(6, 6)); k1 = z @ z.T
        b = rng.normal(size=6)
        s = np.array([0.2, 1., 3., 8.]); mu = np.array([0., 0.8, 0.4, 1.])
        e, g0, g1, h = witness_grams(k0, k1, b, s, mu)
        assert_allclose(gram_residual(e, g0, g1, s, mu, h), 0., atol=5e-13)
        for g in [e, g0, g1]:
            self.assertGreater(la.eigvalsh(g)[0], -1e-12)

    def test_diagonal_data_are_not_identifiable(self):
        from gram import diagonal_counterexample
        d = diagonal_counterexample()
        self.assertLess(d['diagonal_max_difference'], 1e-12)
        self.assertGreater(d['off_diagonal_max_relative_difference'], 1e-3)

    def test_direct_analytic_jacobian(self):
        from gram import PencilFit
        fit = PencilFit(2, np.linspace(0.2, 2., 8), np.linspace(1., 0., 8), np.linspace(.8,.2,8), 'direct')
        x = fit.initial(42)
        j = fit.jac(x)
        for k in range(len(x)):
            d = np.zeros_like(x); d[k] = 1e-6
            assert_allclose(j[:, k], (fit.fun(x+d)-fit.fun(x-d))/(2e-6), atol=2e-6, rtol=2e-5)

    def test_kernel_analytic_jacobian(self):
        from gram import PencilFit
        fit = PencilFit(2, np.linspace(0.2, 2., 7), np.linspace(1., 0., 7), np.linspace(.8,.2,7), 'kernel')
        x = fit.initial(42)
        j = fit.jac(x)
        for k in range(len(x)):
            d = np.zeros_like(x); d[k] = 1e-6
            assert_allclose(j[:, k], (fit.fun(x+d)-fit.fun(x-d))/(2e-6), atol=2e-6, rtol=2e-5)

    def test_parameterization_is_positive(self):
        from gram import PencilFit
        fit = PencilFit(3, np.linspace(.2, 3., 8), np.linspace(0.,1.,8), np.ones(8), 'kernel')
        k0, k1, b, y = fit.unpack(fit.initial(1))
        self.assertGreater(la.eigvalsh(k0)[0], 0.)
        self.assertGreater(la.eigvalsh(k1)[0], 0.)
        for g in [y.T@y, y.T@k0@y, y.T@k1@y]:
            self.assertLessEqual(np.linalg.matrix_rank(g, tol=1e-9), 3)
            self.assertGreater(la.eigvalsh(g)[0], -1e-10)

    def test_scalar_recovery_both_objectives(self):
        from gram import PencilFit
        s = np.geomspace(.1, 10., 12); mu = np.tile([0., .5, 1.], 4)
        h = 4./(s+2.+3.*mu)
        for method in ['direct', 'kernel']:
            fit = PencilFit(1, s, mu, h, method)
            result = fit.solve(42, max_nfev=150)
            pred = fit.predict(result['x'], s, mu)
            assert_allclose(pred, h, rtol=2e-5, atol=1e-6)


class SpectralTests(unittest.TestCase):
    def test_positive_nonaffine_conductance_and_subset(self):
        from spectral import EdgeFamily
        family = EdgeFamily(8, 3, 42, 'strong')
        q = np.array([1., -.7, .5])
        g = family.conductance(q)
        self.assertTrue(np.all(g > 0.))
        ids = np.array([0, 4, len(g)-1])
        assert_allclose(family.conductance(q, ids), g[ids])
        self.assertGreater(la.eigvalsh(family.matrix(q))[0], 0.)

    def test_spectral_relative_error_and_direction(self):
        from spectral import worst_direction
        a = np.array([[2., .4], [.4, 1.]])
        error, v = worst_direction(a, 1.2*a)
        self.assertAlmostEqual(error, .2)
        self.assertAlmostEqual(float(v @ a @ v), 1.)

    def test_all_edges_are_exact(self):
        from spectral import EdgeFamily
        family = EdgeFamily(7, 3, 42, 'mild')
        q = np.array([.2, .4, -.1])
        ids = np.arange(family.edge_count)
        assert_allclose(family.matrix(q, ids, np.ones(len(ids))), family.matrix(q), atol=1e-12)

    def test_nonnegative_greedy_budget(self):
        from spectral import positive_greedy
        rng = np.random.default_rng(42); a = rng.random((12, 30)); b = a @ np.ones(30)
        ids, w = positive_greedy(a, b, 8)
        self.assertLessEqual(len(ids), 8)
        self.assertTrue(np.all(w >= 0.))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertLess(la.norm(a[:, ids]@w-b), la.norm(b))

    def test_minimax_refit_does_not_worsen_active_constraints(self):
        from spectral import minimax_weights
        rng = np.random.default_rng(42); a = rng.random((15, 6)); old = np.ones(6)
        w, t = minimax_weights(a)
        self.assertTrue(np.all(w >= 0.))
        self.assertLessEqual(np.max(np.abs(a@w-1.)), np.max(np.abs(a@old-1.))+1e-8)
        self.assertLessEqual(np.max(np.abs(a@w-1.)), t+1e-7)

    def test_exchange_preserves_budget_and_positive_definiteness(self):
        from spectral import EdgeFamily, spectral_exchange
        fam = EdgeFamily(6, 3, 42, 'mild')
        q = np.random.default_rng(2).uniform(-1.,1.,(8,3))
        ids, w, info = spectral_exchange(fam, q, 8, rounds=2)
        self.assertLessEqual(len(ids), 8)
        self.assertTrue(np.all(w >= 0.))
        self.assertGreater(la.eigvalsh(fam.matrix(q[0], ids, w))[0], 0.)
        self.assertEqual(info['pool_states'], 8)

if __name__ == '__main__':
    unittest.main()
