#!/usr/bin/env python3
"""Verify the output tangency identity behind parameter sample placement."""

import unittest

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from probe_tangent_corners import seed_corner_rule
from probe_conditional_zolotarev import conditional_edge_rule
from model_case1 import Case1Config, Case1Model


class ParametricTangencyTests(unittest.TestCase):
    def test_conditional_rational_nodes_stay_inside_the_selected_edge(self):
        model = Case1Model(Case1Config(
            max_xy_cell_mm=2.5, max_z_cell_mm=2.5,
        ))
        ranges = np.asarray(model.h_ranges(), dtype=float)
        points, info = conditional_edge_rule(
            model.core.K, model.boundary_terms, model.source_shape, ranges,
        )
        self.assertEqual(points.shape, (3, 2))
        np.testing.assert_allclose(points[1:, 0], ranges[0, 1])
        self.assertTrue(np.all(np.diff(points[1:, 1]) > 0))
        self.assertTrue(np.all(points[1:, 1] > ranges[1, 0]))
        self.assertTrue(np.all(points[1:, 1] < ranges[1, 1]))
        self.assertTrue(0 < info['conditional_zolotarev_bound'] < 1)

    def test_seed_quadratic_majorant_is_largest_at_a_box_vertex(self):
        K = sp.csc_matrix([[3.0, -0.4], [-0.4, 2.0]])
        H = [sp.diags([1.0, 0.0]), sp.diags([0.0, 0.5])]
        G = np.eye(2)
        ranges = np.array([[0.5, 5.0], [0.3, 3.0]])
        seed = np.array([1.4, 0.9])
        corners, scores, geometry = seed_corner_rule(
            K, H, G, ranges, seed
        )
        self.assertEqual(corners.shape, (4, 2))
        self.assertTrue(np.all(scores[:-1] >= scores[1:]))
        for p in ([1.0, 1.2], [3.0, 2.0], [0.9, 0.5]):
            offset = np.asarray(p) - seed
            bounds = np.einsum('i,klij,j->kl', offset, geometry, offset)
            normalized = np.max(bounds)
            self.assertLessEqual(normalized, scores[0] + 1e-12)

    def test_collocated_transfer_is_hermite_at_full_snapshot_point(self):
        K = np.array([
            [4.0, -0.7, 0.0, 0.0], [-0.7, 3.0, -0.4, 0.0],
            [0.0, -0.4, 2.5, -0.6], [0.0, 0.0, -0.6, 2.0],
        ])
        H = [np.diag([1.0, 0.0, 0.5, 0.0]),
             np.diag([0.0, 0.6, 0.0, 1.0])]
        G = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        seed = np.array([1.0, 2.0])

        def operator(p):
            return K + p[0] * H[0] + p[1] * H[1]

        V, _ = np.linalg.qr(scipy.linalg.solve(operator(seed), G))

        def difference(p):
            A = operator(p)
            X = scipy.linalg.solve(A, G)
            coefficients = scipy.linalg.solve(V.T @ A @ V, V.T @ G)
            reduced = V @ coefficients
            residual = G - A @ reduced
            error = G.T @ X - G.T @ reduced
            identity = residual.T @ scipy.linalg.solve(A, residual)
            self.assertLess(np.max(np.abs(error - identity)), 2e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(error).min(), -2e-14)
            return error

        self.assertLess(np.max(np.abs(difference(seed))), 2e-14)
        delta = np.array([0.3, -0.2])
        small = np.max(np.abs(difference(seed + 1e-3 * delta)))
        half = np.max(np.abs(difference(seed + 5e-4 * delta)))
        self.assertAlmostEqual(small / half, 4.0, delta=0.02)

        _corners, _scores, geometry = seed_corner_rule(
            sp.csc_matrix(K), [sp.csc_matrix(term) for term in H],
            G, np.array([[0.2, 5.0], [0.2, 5.0]]), seed,
        )
        for p in (np.array([0.2, 5.0]), np.array([3.0, 0.2])):
            displacement = p - seed
            majorant = np.einsum(
                'a,ijab,b->ij', displacement, geometry, displacement
            )
            exact = G.T @ scipy.linalg.solve(operator(p), G)
            self.assertTrue(np.all(
                np.abs(difference(p)) / exact <= majorant + 1e-12
            ))


if __name__ == "__main__":
    unittest.main()
