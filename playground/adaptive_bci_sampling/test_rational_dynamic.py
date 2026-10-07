"""Independent checks for the rational-time continuous-cell prototype."""
import unittest

import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate

from rational_dynamic import (
    AffineSystem, cell_certificate, coefficients, exact_impulse_gram,
    polynomial_controls, shift_sequence,
)


class RationalDynamicTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        rotation = la.qr(rng.normal(size=(7, 7)))[0]
        self.C = np.diag(np.geomspace(.2, 4., 7))
        self.K = rotation @ np.diag(np.geomspace(.01, 20., 7)) @ rotation.T
        self.G = rng.normal(size=(7, 2))
        self.H = [np.diag(rng.uniform(.1, .7, 7)),
                  np.diag(rng.uniform(.1, .4, 7))]
        self.V = la.qr(rng.normal(size=(7, 4)))[0][:, :4]
        self.system = AffineSystem(self.K, self.C, self.G, self.H, self.V)

    def test_parseval_and_exact_tail(self):
        K = self.system.operator([.2, .7])
        shifts = shift_sequence(self.system, [.2, .7], 64)
        full, q = coefficients(K, self.C, self.G, shifts)
        reduced, qr = coefficients(self.V.T @ K @ self.V,
                                    self.V.T @ self.C @ self.V,
                                    self.V.T @ self.G, shifts)
        gram = sum((x - self.V @ y).T @ self.C @ (x - self.V @ y)
                   for x, y in zip(full, reduced))
        exact, Q = exact_impulse_gram(K, self.C, self.G, self.V)
        # Full-response Parseval identity includes an exact, nonzero tail.
        full_energy = sum(x.T @ self.C @ x for x in full)
        np.testing.assert_allclose(full_energy + .5*q.T @ la.solve(K, q),
                                   Q, atol=2e-11, rtol=2e-10)
        np.testing.assert_allclose(gram, exact, atol=1e-9, rtol=2e-7)

    def test_bernstein_controls_are_not_nodal_values(self):
        # p(x)=4*x*(1-x) has zero endpoints but maximum one.
        power = np.array([0., 4., -4.])[:, None, None]
        controls = polynomial_controls(power, 1)
        np.testing.assert_allclose(controls[:, 0, 0], [0., 2., 0.])

    def test_continuous_cell_dominates_independent_points(self):
        low, high = np.array([.2, .6]), np.array([.3, .8])
        shifts = shift_sequence(self.system, (low+high)/2, 40)
        cert = cell_certificate(self.system, low, high, shifts, degree=2)
        rng = np.random.default_rng(301)
        for h in [low, high, *rng.uniform(low, high, (12, 2))]:
            J, Q = exact_impulse_gram(self.system.operator(h), self.C,
                                      self.G, self.V)
            value = np.sqrt(max(0., la.eigvalsh(J, Q)[-1]))
            self.assertLessEqual(value, cert['bound']*(1+1e-9))

    def test_complete_space_tightens_to_zero(self):
        system = AffineSystem(self.K, self.C, self.G, self.H, np.eye(7))
        h = np.array([.2, .6])
        shifts = shift_sequence(system, h, 80)
        cert = cell_certificate(system, h, h, shifts, degree=0)
        self.assertLess(cert['bound'], 1e-7)

    def test_rejects_rank_deficient_denominator(self):
        system = AffineSystem(self.K, self.C, np.column_stack([self.G[:, 0]]*2),
                              self.H, self.V)
        with self.assertRaises(ValueError):
            cell_certificate(system, [.2, .6], [.3, .8], [1.], degree=0)

    def test_exact_reference_matches_direct_time_quadrature(self):
        K = self.system.operator([.2, .7])
        rates, U = la.eigh(K, self.C)
        rr, Ur = la.eigh(self.V.T @ K @ self.V, self.V.T @ self.C @ self.V)
        W = self.V @ Ur
        b, br = U.T @ self.G, W.T @ self.G
        def integrand(t):
            e = U @ (np.exp(-rates[:, None]*t)*b) - W @ (np.exp(-rr[:, None]*t)*br)
            return e.T @ self.C @ e
        independent, _ = integrate.quad_vec(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10)
        J, _ = exact_impulse_gram(K, self.C, self.G, self.V)
        np.testing.assert_allclose(J, independent, rtol=2e-10, atol=2e-10)

    def test_worst_input_combination_is_not_columnwise_error(self):
        K, C = np.diag([1., 2.]), np.eye(2)
        G = np.array([[1., 1.], [1e-4, -1e-4]])
        V = np.array([[1.], [0.]])
        system = AffineSystem(K, C, G, [np.diag([1., 1.])], V)
        J, Q = exact_impulse_gram(system.operator([1.]), C, G, V)
        self.assertLess(np.max(np.diag(J)/np.diag(Q)), 1e-7)
        self.assertGreater(la.eigvalsh(J, Q)[-1], .999)
        cert = cell_certificate(system, [1.], [1.], np.geomspace(.1, 10., 40), 0)
        self.assertGreaterEqual(cert['bound'], .999)

    def test_collatz_interval_contains_full_and_reduced_spectra(self):
        K = np.diag([2., 3., 2.])+np.diag([-1., -1.], 1)+np.diag([-1., -1.], -1)
        C = np.diag([.2, 1., 3.])
        V = np.eye(3)[:, :2]
        system = AffineSystem(K, C, np.ones((3, 1)), [np.eye(3)], V)
        cert = cell_certificate(system, [.1], [1.], [1., 2.], 1)
        alpha, beta = cert['spectral_interval']
        for h in [.1, .5, 1.]:
            eigs = la.eigvalsh(system.operator([h]), C)
            self.assertLessEqual(alpha, eigs[0])
            self.assertGreaterEqual(beta, eigs[-1])

    def test_certificate_is_invariant_under_state_coordinate_change(self):
        rng = np.random.default_rng(71)
        S = np.eye(7)+.1*rng.normal(size=(7, 7))
        transformed = AffineSystem(S.T @ self.K @ S, S.T @ self.C @ S,
            S.T @ self.G, [S.T @ H @ S for H in self.H], la.solve(S, self.V))
        args = ([.2, .6], [.3, .8], [1., .1, 4., 1.], 2)
        original = cell_certificate(self.system, *args, spectral_contraction=False)
        changed = cell_certificate(transformed, *args, spectral_contraction=False)
        np.testing.assert_allclose(changed['bound'], original['bound'], rtol=2e-10)

    def test_exact_single_shift_matching_does_not_certify_impulse(self):
        K, C, G = np.diag([1., 2.]), np.eye(2), np.ones((2, 1))
        V = la.solve(K+C, G)
        V /= la.norm(V)
        exact = la.solve(K+C, G)
        approx = V @ la.solve(V.T @ (K+C) @ V, V.T @ G)
        np.testing.assert_allclose(exact, approx, atol=1e-15)
        system = AffineSystem(K, C, G, [np.zeros((2, 2))], V)
        cert = cell_certificate(system, [0.], [0.], np.geomspace(.1, 10., 40), 0)
        self.assertGreater(cert['point_lower_bound'], .01)

    def test_case1_reconstruction_has_conservative_faces_and_correct_ports(self):
        from run_rational_dynamic import case1_reconstruction
        K, C, G, H, _, meta = case1_reconstruction(10.)
        np.testing.assert_allclose(K @ np.ones(K.shape[0]), 0., atol=2e-14)
        np.testing.assert_allclose(G.sum(axis=0), 1., atol=1e-14)
        np.testing.assert_allclose([float(term.diagonal().sum()) for term in H],
                                   [4e-4, 6e-3], atol=1e-15)
        self.assertTrue(np.all(C.diagonal() > 0))
        self.assertFalse(meta['native_validated'])

    def test_bernstein_half_split_preserves_polynomial(self):
        from rational_dynamic import split_bernstein
        from math import comb
        controls = np.array([1., -3., 2., 4.])[:, None, None]
        left, right = split_bernstein(controls, 0)
        def evaluate(c, t):
            p = len(c)-1
            return sum(comb(p, j)*t**j*(1-t)**(p-j)*c[j] for j in range(p+1))
        for t in np.linspace(0., 1., 13):
            np.testing.assert_allclose(evaluate(left, t), evaluate(controls, t/2), atol=1e-14)
            np.testing.assert_allclose(evaluate(right, t), evaluate(controls, .5+t/2), atol=1e-14)

    def test_refined_envelope_covers_reference_without_new_trial_solves(self):
        low, high = [.2, .6], [.3, .8]
        shifts = [1., .1, 4., 1.]*10
        old = cell_certificate(self.system, low, high, shifts, 2, propagation_model='energy')
        new = cell_certificate(self.system, low, high, shifts, 2,
                               propagation_model='energy', envelope_depth=2)
        self.assertLessEqual(new['bound'], old['bound']*(1+1e-12))
        self.assertEqual(new['counts']['trial_rhs'], old['counts']['trial_rhs'])
        self.assertEqual(new['counts']['riesz_rhs'], old['counts']['riesz_rhs'])
        self.assertEqual(new['envelope_cells'], 16)
        for h in np.random.default_rng(107).uniform(low, high, (12, 2)):
            J, Q = exact_impulse_gram(self.system.operator(h), self.C, self.G, self.V)
            self.assertLessEqual(np.sqrt(max(0., la.eigvalsh(J, Q)[-1])), new['bound'])

    def test_innovation_cross_gram_identity_including_signed_cayley_factors(self):
        shifts = np.array([.02, 4., .3, 1., .07, 8.])
        for rate in [.001, .05, .7, 3., 50.]:
            q = np.zeros(len(shifts))
            rows = []
            b = 2*shifts/(rate+shifts)
            t = (rate-shifts)/(rate+shifts)
            for k, sigma in enumerate(shifts):
                forcing = np.eye(len(shifts))[k]
                rows.append(np.sqrt(2*sigma*rate)/(rate+sigma)*(q+forcing))
                q = t[k]*q-b[k]*forcing
            rows.append(q/np.sqrt(2))
            exact = np.asarray(rows).T @ np.asarray(rows)
            formula = np.diag(b)
            for j in range(len(shifts)):
                for k in range(j+1, len(shifts)):
                    formula[j, k] = formula[k, j] = -b[j]*b[k]*np.prod(t[j+1:k])/2
            np.testing.assert_allclose(exact, formula, rtol=2e-12, atol=2e-14)

    def test_global_energy_majorant_covers_arbitrary_forced_recurrence(self):
        from rational_dynamic import innovation_energy_bound
        rng = np.random.default_rng(93)
        shifts = np.array([.02, 4., .3, 1., .07, 8.])
        for rate in [.001, .05, .7, 3., 50.]:
            forces = rng.normal(size=(len(shifts), 3))
            q = np.zeros(3)
            energy = 0.
            for sigma, f in zip(shifts, forces):
                x = np.sqrt(2*sigma*rate)/(rate+sigma)*(q+f)
                energy += x @ x
                q = (rate-sigma)/(rate+sigma)*q-2*sigma/(rate+sigma)*f
            energy += .5*(q @ q)
            bound = innovation_energy_bound(np.linalg.norm(forces, axis=1),
                                             shifts, (rate, rate))
            self.assertLessEqual(np.sqrt(energy), bound*(1+1e-12))

    def test_global_energy_cell_is_valid_and_never_worse_than_independent(self):
        low, high = [.2, .6], [.3, .8]
        shifts = [1., .1, 4., 1.]*10
        old = cell_certificate(self.system, low, high, shifts, 2)
        new = cell_certificate(self.system, low, high, shifts, 2,
                               propagation_model='energy')
        self.assertLessEqual(new['bound'], old['bound']*(1+1e-12))
        for h in np.random.default_rng(105).uniform(low, high, (12, 2)):
            J, Q = exact_impulse_gram(self.system.operator(h), self.C, self.G, self.V)
            self.assertLessEqual(np.sqrt(max(0., la.eigvalsh(J, Q)[-1])), new['bound'])

    def test_coupled_error_recurrence_covers_independent_dynamic_reference(self):
        low, high = np.array([.2, .6]), np.array([.3, .8])
        cert = cell_certificate(self.system, low, high, [1., .1, 4., 1.]*5, 2,
                                propagation_model='coupled')
        rng = np.random.default_rng(802)
        for h in rng.uniform(low, high, (12, 2)):
            J, Q = exact_impulse_gram(self.system.operator(h), self.C, self.G, self.V)
            self.assertLessEqual(np.sqrt(max(0., la.eigvalsh(J, Q)[-1])), cert['bound'])


if __name__ == '__main__':
    unittest.main()
