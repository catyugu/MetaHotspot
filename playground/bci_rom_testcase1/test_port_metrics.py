"""Regression checks for port-response metrics used by the reproduction."""

import unittest

import numpy as np

from reproduce_case1 import relative_port_error


class PortMetricsTests(unittest.TestCase):
    def test_entrywise_transfer_exposes_error_hidden_by_nominal_powers(self):
        exact = np.array([[2.0, 1.0], [1.0, 2.0]])
        approximate = exact + np.array([[0.1, -0.1], [0.0, 0.0]])
        power = np.ones(2)
        self.assertTrue(np.allclose((approximate - exact) @ power, 0.0))
        self.assertAlmostEqual(relative_port_error(exact, approximate, exact), 0.1)

    def test_port_error_uses_steady_transfer_to_scale_transient(self):
        steady = np.array([[2.0, 1.0], [1.0, 2.0]])
        exact = np.zeros((2, 2, 2))
        estimate = exact.copy()
        estimate[1, 0, 1] = 0.02
        self.assertAlmostEqual(relative_port_error(exact, estimate, steady), 0.02)

    def test_incompatible_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            relative_port_error(np.eye(2), np.eye(2), np.eye(3))


if __name__ == "__main__":
    unittest.main()
