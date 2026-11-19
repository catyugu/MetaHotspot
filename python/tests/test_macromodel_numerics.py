"""Numerical contracts independent of the native model builder."""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from metahotspot._compiled_data import Operators
from metahotspot.macromodel.affine import AffineParametricModel, SourcePort
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


def test_boundary_projection_accumulates_patches_in_cell_order():
    from metahotspot.macromodel.affine import BoundaryGroup
    from metahotspot.macromodel.geometry import BoundarySurface

    class Model(AffineParametricModel):
        core = Operators(sp.eye(2), sp.eye(2), np.zeros(2))

        def boundary_groups(self):
            return (
                BoundaryGroup(
                    BoundarySurface(
                        np.array([0, 0, 1]), np.array([0.1, 0.2, 0.4]), np.arange(3)
                    )
                ),
            )

    assert_allclose(Model().boundary_terms[0].diagonal(), [0.3, 0.4])


def test_source_projection_preserves_signed_powers():
    class Model(AffineParametricModel):
        ambient_K = 300.0
        core = Operators(sp.eye(2), sp.eye(2), np.array([-2.0, -6.0]))

        def source_ports(self):
            return [
                SourcePort(cells=np.array([0]), power_W=-2.0),
                SourcePort(cells=np.array([1]), power_W=-6.0),
            ]

    model = Model()
    assert_allclose(model.source_shape, [[1, 0], [0, 1]])
    assert_allclose(model.source_shape @ model.nominal_power(), model.core.f)
    assert_allclose(
        Model().junction_temperature(np.array([310.0, 320.0])), [310.0, 320.0]
    )
