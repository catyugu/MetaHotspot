#!/usr/bin/env python3
"""Check deterministic greedy residuals against unreduced sparse operators."""

import unittest

import numpy as np
import scipy.sparse as sp

from adaptive_zolotarev import greedy_shift, residual_scores


class AdaptiveZolotarevTests(unittest.TestCase):
    def test_residual_scores_match_full_vectors_and_enrichment_reaches_tolerance(self):
        K = sp.csc_matrix([
            [4.0, -0.9, 0.0, 0.0],
            [-0.9, 3.0, -0.7, 0.0],
            [0.0, -0.7, 2.7, -0.5],
            [0.0, 0.0, -0.5, 2.2],
        ])
        C = sp.diags([1.0, 1.4, 0.8, 0.6], format="csc")
        terms = [
            sp.diags([1.0, 0.0, 0.2, 0.0], format="csc"),
            sp.diags([0.0, 0.3, 0.0, 1.0], format="csc"),
        ]
        g = np.array([1.0, 0.0, 0.5, 0.2])
        shift = 0.15
        seed = np.array([[1.0, 1.0]])
        candidates = np.array([
            [0.2, 0.2], [0.2, 8.0], [8.0, 0.2], [8.0, 8.0], [1.0, 1.0]
        ])
        result = greedy_shift(
            K, C, terms, g, shift, seed, candidates,
            tolerance=1e-5, max_extra=4, basis=np.empty((4, 0)),
        )
        self.assertLessEqual(result["maximum_residual"], 1e-5)
        self.assertGreater(result["extra_solves"], 0)
        scores, coefficients = residual_scores(
            K, C, terms, g, shift, result["basis"], candidates
        )
        for point, predicted, column in zip(candidates, scores, coefficients):
            A = K + shift * C + point[0] * terms[0] + point[1] * terms[1]
            explicit = np.linalg.norm(g - A @ (result["basis"] @ column))
            self.assertAlmostEqual(predicted, explicit / np.linalg.norm(g), places=7)
        self.assertLessEqual(np.max(scores), 1e-5)


if __name__ == "__main__":
    unittest.main()
