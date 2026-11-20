"""Numerical contracts independent of the native model builder."""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from metahotspot._compiled_data import Operators
from metahotspot.macromodel.utils import (
    build_parametric_basis,
    mpmm_elliptic_shift_count,
)


@pytest.mark.parametrize("epsilon, hi", [(1e-3, 10.0), (1e-100, 1e12)])
def test_shift_count_satisfies_bound_without_an_iteration_cap(epsilon, hi):
    lo = 1.0
    count = mpmm_elliptic_shift_count(epsilon, lo, hi)
    log_term = np.log(4 * hi / lo)
    assert 4 * np.exp(-count * np.pi**2 / log_term) <= epsilon
    assert 4 * np.exp(-(count - 1) * np.pi**2 / log_term) > epsilon


def test_single_cell_training_preserves_only_uniform_mode():
    K = sp.csc_matrix((1, 1))
    C = sp.diags([2.0], format="csc")
    boundary = sp.diags([4.0], format="csc")
    basis, stats = build_parametric_basis(
        Operators(K, C, np.zeros(1)), np.ones((1, 1)), [boundary], [(1.0, 10.0)]
    )
    assert_allclose(basis, [[1]])
    assert stats["basis_order"] == 1
    assert stats["singular_value_ratios"] == []
