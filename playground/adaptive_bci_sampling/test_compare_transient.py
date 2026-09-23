#!/usr/bin/env python3
"""Numerical checks for like-for-like dynamic extractor evaluation."""

from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from compare_transient import evaluate_step_case, step_transfer


class TransientComparisonTests(unittest.TestCase):
    def test_selected_parameters_reproduce_tensor_frequency_space(self):
        from model_case1 import Case1Config, Case1Model
        from compare_transient import fixed_tensor_basis, selected_frequency_basis
        from compare_zolotarev import tensor_product

        model = Case1Model(Case1Config(max_xy_cell_mm=5, max_z_cell_mm=5))
        ranges = np.asarray(model.h_ranges())
        axes = [ranges[0], ranges[1]]
        points = tensor_product(axes)
        selected, summary = selected_frequency_basis(
            model, points, 1e-3, additional_cutoffs=(1e-4,)
        )
        tensor, other = fixed_tensor_basis(model, axes, 1e-3)
        self.assertEqual(summary["parameter_points"], 4)
        self.assertEqual(summary["full_rhs_solves"], other["full_rhs_solves"])
        self.assertEqual(summary["basis_order"], other["basis_order"])
        self.assertLess(np.linalg.norm(selected @ selected.T - tensor @ tensor.T), 1e-7)
        richer = summary["additional_bases"][1e-4]
        self.assertGreaterEqual(richer.shape[1], selected.shape[1])
        self.assertLess(np.linalg.norm(selected - richer @ (richer.T @ selected)), 1e-7)

    def test_two_tolerances_reuse_stock_seed_after_greedy_extraction(self):
        script = Path(__file__).resolve().with_name("compare_transient.py")
        completed = subprocess.run(
            [sys.executable, str(script), "5", "--tolerances", "1e-3", "1e-4",
             "--counts", "--seeds", "7", "--greedy-seed-counts", "1",
             "--greedy-grid", "5", "--skip-validation"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("extracted reproduction-seed-7"), 2)

    def test_full_basis_reproduces_coupled_transient_and_zero_initial_state(self):
        K = sp.csc_matrix([[3.0, -0.3], [-0.3, 1.8]])
        C = sp.diags([2.0, 1.0], format="csc")
        G = np.eye(2)
        power = np.array([0.2, 0.7])
        full = step_transfer(K, C, G, dt=0.25, duration=1.0)
        case = evaluate_step_case(
            K, C, G, power, [np.eye(2)], dt=0.25, duration=1.0
        )
        self.assertTrue(np.all(full[0] == 0.0))
        self.assertTrue(np.allclose(full, case["reference_step"]))
        self.assertLess(case["roms"][0]["worst_nominal_step"], 1.0e-12)
        self.assertLess(case["roms"][0]["worst_entrywise_step"], 1.0e-12)

    def test_step_error_is_normalized_by_same_parameter_steady_rise(self):
        K = sp.csc_matrix([[3.0, -0.3], [-0.3, 1.8]])
        C = sp.diags([2.0, 1.0], format="csc")
        G = np.eye(2)
        power = np.array([0.2, 0.7])
        V = np.array([[1.0], [1.0]]) / np.sqrt(2.0)
        case = evaluate_step_case(K, C, G, power, [V], dt=0.25, duration=1.0)
        reference = case["reference_step"]
        estimated = case["roms"][0]["step_transfer"]
        exact_steady = case["reference_steady"]
        expected = np.max(
            np.abs((estimated - reference) @ power)
            / np.abs(exact_steady @ power)
        )
        entrywise = np.max(
            np.abs(estimated - reference) / np.abs(exact_steady)
        )
        self.assertAlmostEqual(case["roms"][0]["worst_nominal_step"], expected)
        self.assertAlmostEqual(case["roms"][0]["worst_entrywise_step"], entrywise)


if __name__ == "__main__":
    unittest.main()
