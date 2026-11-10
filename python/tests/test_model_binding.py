"""Smoke tests for the public model-to-native binding path."""

from __future__ import annotations

import numpy as np
import pytest

import metahotspot
from metahotspot import SurfaceRegion
from metahotspot._error import MetaHotspotError
from metahotspot.enums import Axis, StudyType


def _model(*, transient: bool = False) -> metahotspot.Model:
    model = metahotspot.Model()
    model.set_cartesian_mesh(
        np.array([0.0, 0.001]),
        np.array([0.0, 0.001]),
        np.array([0.0, 0.001]),
    )
    material = model.add_material("solid", "1", "1", "1", "1000", "500")
    body = model.add_body(material)
    model.body_add_box(body, [0, 0, 0], [0.001, 0.001, 0.001])
    source = model.add_volume_source("1e6")
    model.volume_source_add_box(source, [0, 0, 0], [0.001, 0.001, 0.001])
    return model


def test_model_native_calls_compile_and_assemble():
    model = _model()
    model.set_default_neumann("0")

    compiled = model.compile()
    assert compiled.cell_count == 1
    compiled.default_state()
    assert "fields" not in compiled.__dict__
    operators = compiled.assemble()

    assert compiled.cell_count == 1
    assert compiled.info.facet_count == 6
    assert operators.K.format == operators.C.format == "csr"
    assert operators.K.shape == (1, 1)
    assert operators.C.shape == (1, 1)
    assert operators.f.shape == (1,)
    assert operators.f[0] == pytest.approx(1.0e-3)
    assert compiled.fields is compiled.fields
    fields = compiled.fields
    assert fields.facet_coordinate.shape == (compiled.info.facet_count,)
    for axis in range(3):
        ids = np.flatnonzero(fields.facet_axis == axis)
        owners = np.where(
            fields.facet_minus[ids] == np.iinfo(np.uint32).max,
            fields.facet_plus[ids],
            fields.facet_minus[ids],
        )
        expected = np.where(
            fields.facet_minus[ids] == np.iinfo(np.uint32).max,
            fields.cell_min[owners, axis],
            fields.cell_max[owners, axis],
        )
        np.testing.assert_allclose(fields.facet_coordinate[ids], expected)


def test_model_native_boundary_and_function_calls():
    model = _model()
    region = [SurfaceRegion(Axis.Z, 0.0, 0.0, 0.001, 0.0, 0.001)]
    model.add_dirichlet(region, "300")
    model.add_neumann(region, "0")
    model.add_convection(region, "10", "300")
    model.set_default_dirichlet("300")
    model.set_default_neumann("0")
    model.set_default_convection("10", "300")
    model.add_function_expr("constant", "1")
    model.add_function_gauss("gauss", 1.0, 1.0, 0.0)
    model.add_function_sine("sine", 1.0, 1.0, 0.0)
    model.add_function_double_exponential("double_exp", 1.0, 1.0, 2.0)
    model.add_function_piecewise("piecewise", np.array([[0.0, 0.0], [1.0, 1.0]]))
    model.add_function_periodic_piecewise_constant(
        "periodic", np.array([0.0, 0.001]), 1.0
    )
    model.add_point_probe("center", (0.0005, 0.0005, 0.0005))
    with pytest.raises(MetaHotspotError, match="invalid fluid boundary kind"):
        model.add_fluid_boundary(
            [SurfaceRegion(Axis.Z, 0.0, 0.0, 1.0, 0.0, 1.0)], 0, 0.0
        )

    compiled = model.compile()
    assert compiled.cell_count == 1


def test_solution_views_derive_from_one_native_snapshot():
    model = _model(transient=True)
    model.set_default_neumann("0")
    model.add_point_probe("center", (0.0005, 0.0005, 0.0005))

    study = metahotspot.Study(StudyType.TRANSIENT, 300.0, 1.0, 0.5)
    solution = model.compile().solve(
        study=study, state=np.array([300, -1], dtype=np.float32)[::2]
    )

    assert solution.converged
    assert solution.state.size == 1
    assert solution.state_history.shape == (solution.history_times.size, 1)
    assert solution.state_history[0][0] == pytest.approx(300.0)
    assert solution.history_times[0] == pytest.approx(0.0)
    assert solution.history_times[-1] == pytest.approx(1.0)
    assert [trace.name for trace in solution.probes] == ["center"]
    assert solution.probes[0].values.size == solution.history_times.size
