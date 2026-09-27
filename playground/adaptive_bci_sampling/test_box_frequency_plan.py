"""Semantic checks for the box-wide spectral enclosure of the HTC family.

The production frequency plan is built from the bare conduction kernel, whose
estimator skips the Neumann constant mode.  These tests pin down what that
costs: on ``K(h) = K + sum_i h_i H_i`` with ``H_i >= 0`` the Loewner order
gives a deterministic enclosure

    K- = K + sum_i h_i- H_i  <=  K(h)  <=  K+ = K + sum_i h_i+ H_i,

and the smallest eigenvalue of ``K-`` is the lifted constant mode, which lies
below the first positive eigenvalue of ``K`` that the plan uses as its lower
endpoint.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bci_rom_testcase1"))

from model_case1 import Case1Config, Case1Model  # noqa: E402

from metahotspot.macromodel.utils import (  # noqa: E402
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
)


def smallest(kernel, mass):
    return float(spla.eigsh(kernel, k=1, M=mass, sigma=0.0, which="LM",
                            return_eigenvectors=False)[0])


def largest(kernel, mass):
    return float(spla.eigsh(kernel, k=1, M=mass, which="LM",
                            return_eigenvectors=False)[0])


class BoxSpectralEnclosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = Case1Model(Case1Config(max_xy_cell_mm=20.0,
                                       max_z_cell_mm=20.0))
        cls.kernel = model.core.K.tocsc()
        cls.mass = model.core.C.tocsc()
        cls.terms = [term.tocsc() for term in model.boundary_terms]
        cls.ranges = np.asarray(model.h_ranges(), dtype=float)
        cls.lower = cls.robin(cls.ranges[:, 0])
        cls.upper = cls.robin(cls.ranges[:, 1])
        cls.lam_lower = smallest(cls.lower, cls.mass)
        cls.lam_upper = largest(cls.upper, cls.mass)

    @classmethod
    def robin(cls, parameter):
        operator = cls.kernel.copy()
        for value, term in zip(parameter, cls.terms):
            operator = operator + float(value) * term
        return operator.tocsc()

    def test_boundary_terms_are_positive_semidefinite(self):
        for index, term in enumerate(self.terms):
            with self.subTest(term=index):
                self.assertLess(float(abs(term - term.T).max()), 1e-12)
                self.assertGreater(
                    float(spla.eigsh(term, k=1, which="SA",
                                     return_eigenvectors=False)[0]), -1e-9)

    def test_enclosure_holds_for_sampled_parameters(self):
        interior = [self.ranges[:, 0], self.ranges[:, 1]]
        for first in (self.ranges[0, 0], 0.5 * sum(self.ranges[0]),
                      self.ranges[0, 1]):
            for second in (self.ranges[1, 0], 0.5 * sum(self.ranges[1]),
                           self.ranges[1, 1]):
                interior.append([first, second])
        for parameter in interior:
            operator = self.robin(parameter)
            with self.subTest(parameter=list(np.round(parameter, 6))):
                self.assertGreaterEqual(
                    smallest(operator, self.mass), self.lam_lower * (1 - 1e-9))
                self.assertLessEqual(
                    largest(operator, self.mass), self.lam_upper * (1 + 1e-9))

    def test_loewner_order_of_the_corner_operators(self):
        for parameter in ([self.ranges[0, 0], self.ranges[1, 1]],
                          [self.ranges[0, 1], self.ranges[1, 0]]):
            operator = self.robin(parameter)
            low = operator - self.lower
            high = self.upper - operator
            with self.subTest(parameter=list(np.round(parameter, 6))):
                if low.nnz:
                    self.assertGreater(
                        float(spla.eigsh(low, k=1, which="SA",
                                         return_eigenvectors=False)[0]), -1e-9)
                if high.nnz:
                    self.assertGreater(
                        float(spla.eigsh(high, k=1, which="SA",
                                         return_eigenvectors=False)[0]), -1e-9)

    def test_lifted_constant_mode_sits_below_the_bare_first_positive(self):
        """The plan's lower endpoint cannot see the lifted Neumann mode."""
        shift = float(np.median(np.asarray(self.mass.diagonal()))) * 1e-6
        values = spla.eigsh(self.kernel + shift * self.mass, k=4, M=self.mass,
                            which="LM", sigma=0.0,
                            return_eigenvectors=False) - shift
        first_positive = float(np.sort(values[values > 1e-9])[0])
        self.assertLess(self.lam_lower, first_positive)
        uniform = np.ones(self.kernel.shape[0]) / math.sqrt(
            self.kernel.shape[0])
        quotient = float(uniform @ (self.lower @ uniform)) / float(
            uniform @ (self.mass @ uniform))
        self.assertGreaterEqual(quotient, self.lam_lower * (1 - 1e-9))
        self.assertLess(quotient, first_positive)

    def test_box_plan_reaches_below_the_bare_lower_endpoint(self):
        shift = float(np.median(np.asarray(self.mass.diagonal()))) * 1e-6
        values = spla.eigsh(self.kernel + shift * self.mass, k=4, M=self.mass,
                            which="LM", sigma=0.0,
                            return_eigenvectors=False) - shift
        first_positive = float(np.sort(values[values > 1e-9])[0])
        count = mpmm_elliptic_shift_count(1e-3, self.lam_lower, self.lam_upper)
        shifts = mpmm_elliptic_shifts(count, self.lam_upper,
                                      self.lam_upper / self.lam_lower)
        self.assertLess(float(shifts.min()), first_positive)
        self.assertLessEqual(float(shifts.min()), self.lam_lower * 1.5)
        self.assertGreaterEqual(float(shifts.max()), self.lam_lower)


if __name__ == "__main__":
    unittest.main()
