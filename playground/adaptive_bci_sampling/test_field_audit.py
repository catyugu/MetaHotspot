"""Regression checks for the experimental full-field audit, not extraction."""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from certify_extraction import field_errors


def test_all_input_energy_matches_direct_field_gram():
    operator = sp.diags([1.0, 3.0, 7.0, 11.0]).tocsr()
    source = np.array([[1., 0.], [0., 1.], [1., 2.], [2., -1.]])
    basis = np.eye(4)[:, :2]
    reference = la.solve(operator.toarray(), source)
    approximation = basis @ la.solve(basis.T @ operator @ basis, basis.T @ source)
    error = reference - approximation
    expected = np.sqrt(la.eigvalsh(error.T @ operator @ error,
                                  reference.T @ operator @ reference)[-1])
    measured = field_errors(operator, source, basis)
    np.testing.assert_allclose(measured['relative_energy_error'], expected, rtol=1e-9)
    assert measured['galerkin_identity_relative_gap'] < 1e-9


def test_field_error_is_visible_when_junction_defect_is_tiny():
    # The poorly cooled second cell is invisible to an almost collocated sensor,
    # but has a large full-field error. A port-only report would conceal it.
    operator = sp.diags([1.0, 1e-8]).tocsr()
    source = np.array([[1.0], [1e-6]])
    basis = np.array([[1.0], [0.0]])
    measured = field_errors(operator, source, basis)
    assert measured['max_absolute_field_error_per_unit_input'] > 99.0
    assert measured['relative_energy_error'] > 0.009


def test_full_space_has_zero_field_error():
    operator = sp.diags([1.0, 2.0, 4.0]).tocsr()
    measured = field_errors(operator, np.eye(3), np.eye(3))
    assert measured['relative_energy_error'] < 1e-12
    assert measured['max_absolute_field_error_per_unit_input'] < 1e-12
