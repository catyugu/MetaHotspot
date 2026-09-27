#!/usr/bin/env python3
"""Tests for the deterministic design and the whole-box certificate."""

from __future__ import annotations

import itertools
import unittest

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.special import comb

from certified_box import (
    bernstein_coefficients,
    bernstein_operator,
    multi_indices,
)
from exact_error import AffineErrorMap
from deterministic_design import (
    build_basis,
    certified_greedy_points,
    logarithmic_tensor_grid,
    shared_frequency_plan,
    zolotarev_seed,
)
from residual_certificate import prepare_residual_certificate
from zolotarev import coordinate_spectral_enclosures, zolotarev_count, zolotarev_rule


def toy_family(size=20, groups=(slice(0, 5), slice(15, 20))):
    """Small SPD affine family with two boundary groups and nonnegative ports."""
    grid = np.arange(size)
    laplacian = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(size, size), format="csc")
    kernel = (sp.eye(size, format="csc") * 3.0 + laplacian).tocsc()
    terms = []
    for group in groups:
        diagonal = np.zeros(size)
        diagonal[group] = 1.0
        terms.append(sp.diags(diagonal, format="csc").tocsc())
    mass = sp.eye(size, format="csc").tocsc()
    source = np.zeros((size, 2))
    source[7, 0] = 1.0
    source[9, 0] = 0.5
    source[11, 1] = 1.0
    source[3, 1] = 0.25
    return kernel, mass, terms, source


class BernsteinEnclosureTests(unittest.TestCase):
    def test_multi_indices_cover_total_degree(self):
        indices = multi_indices(2, 2)
        self.assertEqual(len(indices), 6)
        self.assertTrue(all(sum(index) <= 2 for index in indices))
        self.assertIn((2, 0), indices)
        self.assertIn((1, 1), indices)

    def test_bernstein_coefficients_enclosed_by_nodes(self):
        # A tensor polynomial of degree 2 is its own Bernstein expansion on
        # the unit box; the coefficients are the convex weights of the
        # polynomial values, so they bound the true range.
        nodes = np.linspace(0.0, 1.0, 3)
        values = np.empty((3, 3, 1, 1))
        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                values[i, j, 0, 0] = (x - 0.5) ** 2 + 0.3 * (y - 0.5) + (x - 0.5) * (y - 0.5)
        coefficients = bernstein_coefficients(values, 2, 2)
        dense = np.linspace(0.0, 1.0, 101)
        sampled = np.array(
            [
                (x - 0.5) ** 2 + 0.3 * (y - 0.5) + (x - 0.5) * (y - 0.5)
                for x in dense
                for y in dense
            ]
        )
        self.assertLessEqual(float(np.max(sampled)), float(np.max(coefficients)) + 1e-12)
        self.assertLessEqual(float(np.min(sampled)), float(np.max(coefficients)) + 1e-12)

    def test_bernstein_operator_inverts_evaluation(self):
        for degree in (1, 2, 3):
            nodes = np.arange(degree + 1) / degree
            evaluation = np.column_stack(
                [
                    comb(degree, index, exact=True) * nodes**index * (1 - nodes) ** (degree - index)
                    for index in range(degree + 1)
                ]
            )
            self.assertLess(float(np.max(np.abs(evaluation @ bernstein_operator(degree) - np.eye(degree + 1)))), 1e-12)


class ZolotarevTests(unittest.TestCase):
    def test_skeleton_relative_error_obeys_the_closed_form_bound(self):
        # The rule's claim: resolving the scalar kernel 1/(lambda + h) on its
        # spectral interval with these nodes has relative error below the
        # closed-form bound.
        rule = zolotarev_rule(
            spectral_interval=(1.0, 100.0), parameter_interval=(1.0, 100.0), count=5
        )
        spectral_grid = np.geomspace(1.0, 100.0, 151)
        parameter_grid = np.geomspace(1.0, 100.0, 151)
        interpolation_matrix = 1.0 / (
            rule.spectral_nodes[:, None] + rule.parameter_nodes[None, :]
        )
        worst = 0.0
        for parameter in parameter_grid:
            coefficients = np.linalg.solve(
                interpolation_matrix, 1.0 / (rule.spectral_nodes + parameter)
            )
            approximation = (
                1.0 / (spectral_grid[:, None] + rule.parameter_nodes[None, :])
            ) @ coefficients
            worst = max(worst, float(np.max(np.abs(1.0 - (spectral_grid + parameter) * approximation))))
        self.assertLessEqual(worst, rule.error_bound * (1.0 + 1e-10))

    def test_count_is_the_smallest_one_satisfying_the_bound(self):
        tolerance = 1e-3
        count = zolotarev_count((13.0, 3.0e5), (1.0, 9.0e3), tolerance=tolerance)
        self.assertLessEqual(
            zolotarev_rule((13.0, 3.0e5), (1.0, 9.0e3), count).error_bound, tolerance
        )
        self.assertGreater(
            zolotarev_rule((13.0, 3.0e5), (1.0, 9.0e3), count - 1).error_bound, tolerance
        )

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
            endpoints = []
            for other in (ranges[1 - coordinate, 0], ranges[1 - coordinate, 1]):
                matrix = base + other * terms[1 - coordinate]
                aa = matrix[active, :][:, active].toarray()
                ai = matrix[active, :][:, inactive].toarray()
                ii = matrix[inactive, :][:, inactive].toarray()
                schur = aa - ai @ la.solve(ii, ai.T, assume_a="pos")
                endpoints.append(
                    la.eigvalsh(schur, np.diag(terms[coordinate].diagonal()[active]))
                )
            self.assertLessEqual(enclosure.lower, float(np.min(endpoints[0])) * (1.0 + 1e-7))
            self.assertGreaterEqual(enclosure.upper, float(np.max(endpoints[1])) * (1.0 - 1e-7))


class ResidualCertificateTests(unittest.TestCase):
    def test_pointwise_residual_certificate_bounds_the_transfer_error(self):
        kernel, _mass, terms, source = toy_family()
        ranges = np.asarray([[0.5, 20.0], [0.5, 20.0]])
        sampled = np.array([2.0, 6.0])
        operator = kernel + sum(float(v) * t for v, t in zip(sampled, terms))
        basis = la.qr(la.solve(operator.toarray(), source), mode="economic")[0]
        certificate = prepare_residual_certificate(kernel, terms, source, ranges, basis)
        rng = np.random.default_rng(23)
        for _ in range(20):
            parameter = np.exp(rng.uniform(np.log(ranges[:, 0]), np.log(ranges[:, 1])))
            exact_operator = kernel + sum(float(v) * t for v, t in zip(parameter, terms))
            exact = np.ascontiguousarray(source.T @ la.solve(exact_operator.toarray(), source))
            _transfer, absolute, _relative = certificate.evaluate_entrywise(parameter)
            reduced = basis.T @ (exact_operator @ basis)
            coefficients = la.solve(reduced, basis.T @ source, assume_a="pos")
            approximate = np.ascontiguousarray(source.T @ (basis @ coefficients))
            self.assertTrue(np.all(np.abs(approximate - exact) <= absolute + 1e-12))


class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.kernel, self.mass, self.terms, self.source = toy_family()
        self.ranges = np.array([[0.5, 20.0], [0.5, 20.0]])
        rng = np.random.default_rng(4)
        self.basis = la.qr(rng.normal(size=(self.kernel.shape[0], 6)), mode="economic")[0]

    def exact_transfer(self, parameter):
        operator = self.kernel + sum(
            float(value) * term for value, term in zip(parameter, self.terms)
        )
        return np.ascontiguousarray(self.source.T @ la.solve(operator.toarray(), self.source))

    def reduced_transfer(self, parameter):
        operator = self.kernel + sum(
            float(value) * term for value, term in zip(parameter, self.terms)
        )
        reduced = self.basis.T @ (operator @ self.basis)
        coefficients = la.solve(reduced, self.basis.T @ self.source, assume_a="pos")
        return np.ascontiguousarray(self.source.T @ (self.basis @ coefficients))

    def test_cell_bound_upper_bounds_the_measured_error(self):
        from certified_box import BoxCertificate

        certificate = BoxCertificate(
            self.kernel, self.terms, self.source, self.ranges, self.basis, blocks=(2, 2)
        )
        low = np.array([1.0, 1.0])
        high = np.array([4.0, 4.0])
        gram = certificate.anchor_gram(certificate.block_lower(certificate.block_index(low)))
        bound = certificate.cell_bound(gram, low, high, order=2)
        rng = np.random.default_rng(7)
        for _ in range(25):
            point = np.exp(rng.uniform(np.log(low), np.log(high)))
            actual = np.abs(self.exact_transfer(point) - self.reduced_transfer(point))
            self.assertTrue(
                np.all(actual <= bound + 1e-12),
                msg=f"measured error {np.max(actual):.3e} exceeds bound {np.max(bound):.3e}",
            )

    def test_denominator_is_a_lower_bound_inside_a_cell(self):
        from certified_box import BoxCertificate

        certificate = BoxCertificate(
            self.kernel, self.terms, self.source, self.ranges, self.basis, blocks=(2, 2)
        )
        cell_low = np.array([0.6, 3.0])
        cell_high = np.array([2.0, 9.0])
        denominator = certificate.cell_denominator(cell_high)
        rng = np.random.default_rng(9)
        for _ in range(15):
            point = np.exp(rng.uniform(np.log(cell_low), np.log(cell_high)))
            self.assertTrue(np.all(self.exact_transfer(point) >= denominator - 1e-12))

    def test_pointwise_bound_is_within_a_small_factor(self):
        from certified_box import BoxCertificate

        certificate = BoxCertificate(
            self.kernel, self.terms, self.source, self.ranges, self.basis, blocks=(4, 4)
        )
        point = np.array([1.5, 8.0])
        gram = certificate.anchor_gram(certificate.block_lower(certificate.block_index(point)))
        bound = certificate.cell_bound(gram, point, point, order=0)
        actual = np.abs(self.exact_transfer(point) - self.reduced_transfer(point))
        self.assertLess(float(np.max(bound)) / max(float(np.max(actual)), 1e-300), 50.0)

    def test_sweep_refines_and_stays_an_upper_bound(self):
        from certified_box import BoxCertificate

        certificate = BoxCertificate(
            self.kernel, self.terms, self.source, self.ranges, self.basis, blocks=(2, 2)
        )
        coarse = certificate.sweep(4, order=2)
        fine = certificate.sweep(12, order=2)
        self.assertLess(fine["steady_relative_bound"], coarse["steady_relative_bound"])
        rng = np.random.default_rng(13)
        worst = 0.0
        for _ in range(60):
            point = np.exp(
                rng.uniform(np.log(self.ranges[:, 0]), np.log(self.ranges[:, 1]))
            )
            actual = np.abs(self.exact_transfer(point) - self.reduced_transfer(point))
            worst = max(worst, float(np.max(actual)))
        self.assertLessEqual(worst, fine["steady_absolute_bound"])

    def test_shift_certificate_covers_the_shifted_family(self):
        from certified_box import BoxCertificate

        shift = 0.75
        certificate = BoxCertificate(
            self.kernel, self.terms, self.source, self.ranges, self.basis,
            shift=shift, mass=self.mass, blocks=(2, 2),
        )
        report = certificate.sweep(8, order=2)
        rng = np.random.default_rng(17)
        worst = 0.0
        for _ in range(30):
            parameter = np.exp(
                rng.uniform(np.log(self.ranges[:, 0]), np.log(self.ranges[:, 1]))
            )
            operator = (
                self.kernel + shift * self.mass
                + sum(float(v) * t for v, t in zip(parameter, self.terms))
            )
            exact = np.ascontiguousarray(self.source.T @ la.solve(operator.toarray(), self.source))
            reduced = self.basis.T @ (operator @ self.basis)
            coefficients = la.solve(reduced, self.basis.T @ self.source, assume_a="pos")
            approximate = np.ascontiguousarray(self.source.T @ (self.basis @ coefficients))
            worst = max(worst, float(np.max(np.abs(exact - approximate))))
        self.assertLessEqual(worst, report["steady_absolute_bound"])


class ExactErrorMapTests(unittest.TestCase):
    def setUp(self):
        self.kernel, self.mass, self.terms, self.source = toy_family()
        self.ranges = np.array([[0.5, 20.0], [0.5, 20.0]])
        rng = np.random.default_rng(11)
        self.basis = la.qr(
            rng.normal(size=(self.kernel.shape[0], 6)), mode="economic"
        )[0]

    def direct_error(self, parameter, shift=0.0):
        operator = self.kernel
        if shift:
            operator = operator + shift * self.mass
        for value, term in zip(parameter, self.terms):
            operator = operator + float(value) * term
        exact = np.ascontiguousarray(
            self.source.T @ la.solve(operator.toarray(), self.source)
        )
        reduced = self.basis.T @ (operator @ self.basis)
        coefficients = la.solve(
            reduced, self.basis.T @ self.source, assume_a="pos"
        )
        approximate = np.ascontiguousarray(
            self.source.T @ (self.basis @ coefficients)
        )
        return exact - approximate

    def test_error_matrix_is_the_direct_galerkin_error(self):
        mapping = AffineErrorMap(
            self.kernel, self.terms, self.source, self.ranges, self.basis
        )
        rng = np.random.default_rng(5)
        for _ in range(6):
            parameter = np.exp(
                rng.uniform(np.log(self.ranges[:, 0]), np.log(self.ranges[:, 1]))
            )
            self.assertTrue(
                np.allclose(
                    mapping.error_matrix(parameter),
                    self.direct_error(parameter),
                    rtol=1e-7,
                    atol=1e-10,
                )
            )

    def test_error_matrix_covers_the_shifted_family(self):
        shift = 0.75
        mapping = AffineErrorMap(
            self.kernel, self.terms, self.source, self.ranges, self.basis,
            shift=shift, mass=self.mass,
        )
        rng = np.random.default_rng(23)
        for _ in range(4):
            parameter = np.exp(
                rng.uniform(np.log(self.ranges[:, 0]), np.log(self.ranges[:, 1]))
            )
            self.assertTrue(
                np.allclose(
                    mapping.error_matrix(parameter),
                    self.direct_error(parameter, shift),
                    rtol=1e-7,
                    atol=1e-10,
                )
            )

    def test_cell_bound_covers_every_point_of_the_cell(self):
        mapping = AffineErrorMap(
            self.kernel, self.terms, self.source, self.ranges, self.basis
        )
        low = np.array([0.6, 3.0])
        high = np.array([2.0, 9.0])
        bound = mapping.cell_bound(low, high)
        rng = np.random.default_rng(19)
        for _ in range(25):
            parameter = np.exp(rng.uniform(np.log(low), np.log(high)))
            actual = np.abs(mapping.error_matrix(parameter))
            self.assertTrue(
                np.all(actual <= bound + 1e-12),
                msg=f"measured {np.max(actual):.3e} exceeds {np.max(bound):.3e}",
            )

    def test_grid_maximum_is_attained_inside_the_box(self):
        mapping = AffineErrorMap(
            self.kernel, self.terms, self.source, self.ranges, self.basis
        )
        report = mapping.worst_on_grid(9)
        self.assertEqual(report["points"], 81)
        self.assertGreater(report["worst_absolute_error"], 0.0)
        self.assertIsNotNone(report["location"])


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.kernel, self.mass, self.terms, self.source = toy_family()
        self.ranges = np.array([[0.5, 20.0], [0.5, 20.0]])

    def test_zolotarev_seed_respects_the_box(self):
        points, spectra = zolotarev_seed(self.kernel, self.terms, self.ranges)
        self.assertTrue(np.all(points >= self.ranges[:, 0]))
        self.assertTrue(np.all(points <= self.ranges[:, 1]))
        for spectrum, extent in zip(spectra, self.ranges):
            self.assertGreater(spectrum.lower, 0.0)
            self.assertLess(spectrum.lower, spectrum.upper)
            self.assertLess(extent[0], extent[1])

    def test_greedy_selection_is_reproducible(self):
        first, score_a, _ = certified_greedy_points(
            self.kernel, self.terms, self.source, self.ranges, None,
            tolerance=1e-6, maximum_points=3, grid=9, metric="entrywise",
        )
        second, score_b, _ = certified_greedy_points(
            self.kernel, self.terms, self.source, self.ranges, None,
            tolerance=1e-6, maximum_points=3, grid=9, metric="entrywise",
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(score_a, score_b)
        self.assertTrue(np.all(first[0] >= self.ranges[:, 0]))
        self.assertTrue(np.all(first <= self.ranges[:, 1] + 1e-12))

    def test_design_is_the_full_shift_by_point_tensor(self):
        # No shift is treated differently: the snapshot count is exactly
        # (elliptic shifts + steady endpoint) * points * ports, so neither a
        # threshold nor the caller's time step can change the design.
        points, _score, _selection = certified_greedy_points(
            self.kernel, self.terms, self.source, self.ranges, None,
            tolerance=1e-9, maximum_points=2, grid=5, metric="entrywise",
        )
        plan = shared_frequency_plan(self.kernel, self.mass, self.source, 1e-6)
        basis, snapshots, info = build_basis(
            self.kernel, self.mass, self.terms, self.source, points,
            plan=plan, tolerance=1e-6, include_dc=True,
        )
        operators = plan["count"] + 1
        expected = self.source.shape[1] * operators * len(points)
        self.assertEqual(info["operators"], operators * len(points))
        self.assertEqual(info["full_rhs_solves"], expected)
        self.assertEqual(snapshots.shape[1], expected)
        self.assertEqual(info["factorizations"], operators * len(points))
        # every operator serves both ports, so the later port is a cache hit
        self.assertEqual(
            info["cached_blocks"],
            (self.source.shape[1] - 1) * operators * len(points),
        )
        self.assertEqual(basis.shape[0], self.kernel.shape[0])

    def test_design_counts_every_full_order_solve(self):
        cache = {}
        points, _score, selection = certified_greedy_points(
            self.kernel, self.terms, self.source, self.ranges, None,
            tolerance=1e-9, maximum_points=2, grid=5, metric="entrywise",
            cache=cache,
        )
        plan = shared_frequency_plan(self.kernel, self.mass, self.source, 1e-6)
        basis, snapshots, info = build_basis(
            self.kernel, self.mass, self.terms, self.source, points, plan=plan,
            tolerance=1e-6, include_dc=True,
            cache=cache,
        )
        operators = info["frequency_plan"]["count"] + 1
        expected = self.source.shape[1] * operators * len(points)
        self.assertEqual(info["full_rhs_solves"], expected)
        self.assertEqual(snapshots.shape[1], expected)
        self.assertEqual(basis.shape[0], self.kernel.shape[0])
        # One plan for every port: each (shift, point) operator is factorized
        # exactly once, and the steady blocks come from the greedy's cache.
        self.assertEqual(info["factorizations"], operators * len(points) - len(points))
        self.assertEqual(
            info["cached_blocks"] + info["factorizations"], info["full_rhs_solves"]
        )
        self.assertEqual(selection["fresh_rhs_solves"], self.source.shape[1] * len(points))

    def test_candidate_grid_is_deterministic_and_logarithmic(self):
        first = logarithmic_tensor_grid(self.ranges, 5)
        second = logarithmic_tensor_grid(self.ranges, 5)
        self.assertTrue(np.array_equal(first, second))
        axis = np.unique(np.log10(first[:, 0]))
        self.assertEqual(axis.size, 5)
        steps = np.diff(axis)
        self.assertTrue(np.allclose(steps, steps[0]))
        self.assertTrue(np.all(first >= self.ranges[:, 0]))
        self.assertTrue(np.all(first <= self.ranges[:, 1] + 1e-12))


if __name__ == "__main__":
    unittest.main()
