import unittest
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from energy_residual import (  # noqa: E402
    certify_energy_ratio,
    exact_energy_ratio,
    mass_upper_bound,
    residual_and_base_energy,
)


class EnergyResidualTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(12, 12))
        self.cdiag = 0.5 + rng.random(12)
        self.shift = 0.4
        K = X.T @ X + 0.2 * np.eye(12)
        self.A = sp.csc_matrix(K + self.shift * np.diag(self.cdiag))
        self.b = rng.normal(size=12)
        Q, _ = np.linalg.qr(rng.normal(size=(12, 4)))
        Ar = Q.T @ (self.A @ Q)
        self.estimate = Q @ np.linalg.solve(Ar, Q.T @ self.b)

    def test_energy_decomposition_is_exact(self):
        residual, base = residual_and_base_energy(self.A, self.b, self.estimate)
        ratio, _, q, _ = exact_energy_ratio(
            self.A, self.b, self.estimate, residual=residual
        )
        exact = float(self.b @ np.linalg.solve(self.A.toarray(), self.b))
        self.assertAlmostEqual(base + q, exact, places=11)
        self.assertAlmostEqual(ratio, q / exact, places=12)

    def test_mass_bound_contains_exact_inverse_quadratic_form(self):
        residual, _ = residual_and_base_energy(self.A, self.b, self.estimate)
        _, _, q, _ = exact_energy_ratio(
            self.A, self.b, self.estimate, residual=residual
        )
        upper = mass_upper_bound(residual, self.cdiag, self.shift)
        self.assertLessEqual(q, upper * (1.0 + 1e-12))

    def test_certificate_matches_exact_accept_and_reject(self):
        ratio, _, q, _ = exact_energy_ratio(self.A, self.b, self.estimate)
        for tolerance in ((1.0 + ratio) * 0.5, ratio * 0.5):
            result = certify_energy_ratio(
                self.A,
                self.cdiag,
                self.b,
                self.estimate,
                self.shift,
                tolerance,
                max_iterations=12,
            )
            expected = "accept" if ratio <= tolerance else "reject"
            self.assertEqual(result.decision, expected)
            self.assertLessEqual(result.lower_q, q * (1.0 + 1e-10) + 1e-14)
            self.assertGreaterEqual(
                result.upper_q * (1.0 + 1e-10) + 1e-14, q
            )

    def test_zero_iteration_accept_is_certified(self):
        A = sp.eye(4, format="csc") * 2.0
        cdiag = np.ones(4)
        b = np.ones(4)
        estimate = 0.99 * np.linalg.solve(A.toarray(), b)
        ratio, *_ = exact_energy_ratio(A, b, estimate)
        result = certify_energy_ratio(
            A, cdiag, b, estimate, 1.9, 1.0e-3, max_iterations=4
        )
        self.assertLessEqual(ratio, 1.0e-3)
        self.assertEqual(result.decision, "accept")
        self.assertEqual(result.iterations, 0)


if __name__ == "__main__":
    unittest.main()
