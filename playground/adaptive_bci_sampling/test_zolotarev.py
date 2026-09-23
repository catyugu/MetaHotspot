#!/usr/bin/env python3
"""Focused tests for the finite-interval Zolotarev sampling utilities."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from zolotarev import (
    coordinate_spectral_enclosures,
    zolotarev_count,
    zolotarev_rule,
)
from certified_greedy import (
    prepare_residual_certificate,
    select_worst_certificate,
)


class ZolotarevRuleTests(unittest.TestCase):
    def test_infinite_relative_certificate_ties_use_absolute_bound(self):
        relative = np.asarray([2.0, np.inf, np.inf, 4.0])
        absolute = np.asarray([100.0, 0.2, 0.7, 200.0])
        self.assertEqual(select_worst_certificate(relative, absolute), 2)

    def test_nodes_stay_inside_the_parameter_and_spectral_intervals(self):
        rule = zolotarev_rule(
            spectral_interval=(2.0, 300.0),
            parameter_interval=(0.5, 9000.0),
            count=6,
        )

        self.assertTrue(np.all(rule.parameter_nodes > 0.5))
        self.assertTrue(np.all(rule.parameter_nodes < 9000.0))
        self.assertTrue(np.all(rule.spectral_nodes > 2.0))
        self.assertTrue(np.all(rule.spectral_nodes < 300.0))
        self.assertTrue(np.all(np.diff(rule.parameter_nodes) > 0.0))
        self.assertTrue(np.all(np.diff(rule.spectral_nodes) > 0.0))

    def test_skeleton_relative_error_obeys_the_zolotarev_bound(self):
        rule = zolotarev_rule(
            spectral_interval=(1.0, 100.0),
            parameter_interval=(1.0, 100.0),
            count=5,
        )
        spectral_grid = np.geomspace(1.0, 100.0, 151)
        parameter_grid = np.geomspace(1.0, 100.0, 151)
        interpolation_matrix = 1.0 / (
            rule.spectral_nodes[:, None] + rule.parameter_nodes[None, :]
        )

        worst = 0.0
        for parameter in parameter_grid:
            coefficients = np.linalg.solve(
                interpolation_matrix,
                1.0 / (rule.spectral_nodes + parameter),
            )
            approximation = (
                1.0 / (spectral_grid[:, None] + rule.parameter_nodes[None, :])
            ) @ coefficients
            relative = np.abs(1.0 - (spectral_grid + parameter) * approximation)
            worst = max(worst, float(relative.max()))

        self.assertLessEqual(worst, rule.error_bound * (1.0 + 1.0e-10))

    def test_count_is_the_smallest_one_satisfying_the_closed_form_bound(self):
        tolerance = 1.0e-3
        count = zolotarev_count(
            spectral_interval=(13.0, 3.0e5),
            parameter_interval=(1.0, 9.0e3),
            tolerance=tolerance,
        )
        current = zolotarev_rule((13.0, 3.0e5), (1.0, 9.0e3), count)
        previous = zolotarev_rule((13.0, 3.0e5), (1.0, 9.0e3), count - 1)

        self.assertLessEqual(current.error_bound, tolerance)
        self.assertGreater(previous.error_bound, tolerance)

    def test_coordinate_spectrum_encloses_the_exact_schur_spectrum(self):
        base = sp.csc_matrix(
            np.asarray(
                [
                    [4.0, -1.0, -0.3, 0.0],
                    [-1.0, 3.0, -0.2, -0.1],
                    [-0.3, -0.2, 2.5, -0.4],
                    [0.0, -0.1, -0.4, 2.0],
                ]
            )
        )
        terms = [
            sp.diags([0.7, 0.9, 0.0, 0.0], format="csc"),
            sp.diags([0.0, 0.0, 1.1, 0.8], format="csc"),
        ]
        ranges = np.asarray([[0.5, 20.0], [0.25, 8.0]])
        enclosures = coordinate_spectral_enclosures(base, terms, ranges)

        for coordinate, enclosure in enumerate(enclosures):
            active = np.flatnonzero(terms[coordinate].diagonal() > 0.0)
            inactive = np.setdiff1d(np.arange(base.shape[0]), active)
            exact_endpoints = []
            for other_value in (ranges[1 - coordinate, 0], ranges[1 - coordinate, 1]):
                matrix = base + other_value * terms[1 - coordinate]
                aa = matrix[active, :][:, active].toarray()
                ai = matrix[active, :][:, inactive].toarray()
                ii = matrix[inactive, :][:, inactive].toarray()
                schur = aa - ai @ scipy.linalg.solve(ii, ai.T, assume_a="pos")
                exact_endpoints.append(
                    scipy.linalg.eigvalsh(
                        schur,
                        np.diag(terms[coordinate].diagonal()[active]),
                    )
                )

            exact_lower = float(exact_endpoints[0].min())
            exact_upper = float(exact_endpoints[1].max())
            self.assertLessEqual(enclosure.lower, exact_lower * (1.0 + 1.0e-7))
            self.assertGreaterEqual(enclosure.upper, exact_upper * (1.0 - 1.0e-7))

    def test_primal_dual_residual_certificate_bounds_junction_error(self):
        base = sp.csc_matrix(
            np.asarray(
                [
                    [4.0, -1.0, -0.2],
                    [-1.0, 3.0, -0.4],
                    [-0.2, -0.4, 2.0],
                ]
            )
        )
        terms = [
            sp.diags([0.7, 0.0, 0.2], format="csc"),
            sp.diags([0.0, 0.5, 0.3], format="csc"),
        ]
        ranges = np.asarray([[0.5, 5.0], [0.25, 3.0]])
        source = np.asarray([[1.0, 0.0], [0.2, 0.4], [0.0, 1.0]])
        power = np.asarray([0.3, 0.7])
        sample_operator = base + 1.5 * terms[0] + 1.0 * terms[1]
        snapshots = scipy.linalg.solve(
            sample_operator.toarray(), source, assume_a="pos"
        )
        basis, _ = np.linalg.qr(snapshots, mode="reduced")
        certificate = prepare_residual_certificate(
            base, terms, source, ranges, basis
        )

        for point in (np.asarray([0.5, 0.25]), np.asarray([4.0, 2.5])):
            operator = base + point[0] * terms[0] + point[1] * terms[1]
            exact_responses = scipy.linalg.solve(
                operator.toarray(), source, assume_a="pos"
            )
            exact_transfer = source.T @ exact_responses
            exact = exact_transfer @ power
            approximate, absolute_bound, relative_bound = certificate.evaluate(
                point, power
            )
            error = np.abs(approximate - exact)
            relative = error / np.abs(exact)

            self.assertTrue(np.all(error <= absolute_bound * (1.0 + 1.0e-10)))
            self.assertTrue(np.all(relative <= relative_bound * (1.0 + 1.0e-10)))

            transfer, entry_bound, entry_relative = certificate.evaluate_entrywise(
                point
            )
            entry_error = np.abs(transfer - exact_transfer)
            entry_exact_relative = entry_error / np.abs(exact_transfer)
            self.assertTrue(
                np.all(entry_error <= entry_bound * (1.0 + 1.0e-10))
            )
            self.assertTrue(
                np.all(
                    entry_exact_relative
                    <= entry_relative * (1.0 + 1.0e-10)
                )
            )


if __name__ == "__main__":
    unittest.main()
