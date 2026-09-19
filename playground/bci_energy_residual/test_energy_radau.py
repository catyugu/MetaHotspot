import unittest
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from energy_radau import certify_energy_ratio_radau  # noqa: E402
from energy_residual import exact_energy_ratio  # noqa: E402


class GaussRadauCertificateTests(unittest.TestCase):
    def test_matches_exact_inverse_form_on_random_spd_problems(self):
        rng = np.random.default_rng(17)
        for n in (8, 20):
            for _ in range(8):
                X = rng.normal(size=(n, n))
                K = X.T @ X
                anchor = 0.2 + rng.random(n)
                A = sp.csc_matrix(K + np.diag(anchor))
                b = rng.normal(size=n)

                Q, _ = np.linalg.qr(
                    rng.normal(size=(n, min(5, n - 1)))
                )
                Ar = Q.T @ (A @ Q)
                estimate = Q @ np.linalg.solve(Ar, Q.T @ b)
                exact, *_ = exact_energy_ratio(A, b, estimate)

                for tolerance in (
                    max(exact * 0.5, 1.0e-12),
                    min(0.9, 0.5 * (1.0 + exact)),
                    1.0e-3,
                ):
                    result = certify_energy_ratio_radau(
                        A,
                        anchor,
                        b,
                        estimate,
                        tolerance,
                        max_iterations=n,
                    )
                    expected = (
                        "accept" if exact <= tolerance else "reject"
                    )
                    self.assertEqual(result.decision, expected)

    def test_zero_step_anchor_bound_can_accept(self):
        anchor = np.array([2.0, 3.0, 4.0])
        A = sp.diags(anchor, format="csc")
        b = np.ones(3)
        exact_solution = b / anchor
        estimate = 0.99 * exact_solution
        exact, *_ = exact_energy_ratio(A, b, estimate)
        result = certify_energy_ratio_radau(
            A, anchor, b, estimate, 1.0e-3
        )
        self.assertLessEqual(exact, 1.0e-3)
        self.assertEqual(result.decision, "accept")
        self.assertEqual(result.iterations, 0)


if __name__ == "__main__":
    unittest.main()
