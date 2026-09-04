"""Smoke tests for the public model-to-native binding path."""

from __future__ import annotations

import numpy as np
import pytest

import metahotspot
from metahotspot.enums import Axis, GeometryOp, LengthUnit, Study


def _model(*, transient: bool = False) -> metahotspot.Model:
    model = metahotspot.Model()
    model.set_settings(
        study=Study.TRANSIENT if transient else Study.STEADY,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=300.0,
        duration=1.0 if transient else 0.0,
        output_interval=0.5 if transient else 0.0,
    )
    model.set_mesh(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    model.add_material("solid", "1", "1", "1", "1000", "500")
    layer = model.add_layer("1")
    block = model.add_block(layer, "solid", heat_source="1e6")
    model.add_rect(block, GeometryOp.ADD, "0", "0", "1", "1")
    return model


def test_model_native_calls_compile_and_assemble():
    model = _model()
    model.set_default_neumann("0")

    compiled = model.compile()
    operators = compiled.assemble()

    assert compiled.cell_count == 1
    assert (compiled.nx, compiled.ny, compiled.nz) == (1, 1, 1)
    assert operators.K.shape == (1, 1)
    assert operators.C.shape == (1, 1)
    assert operators.f.shape == (1,)
    assert operators.f[0] == pytest.approx(1.0e-3)


def test_model_native_boundary_and_function_calls():
    model = _model()
    region = [(Axis.Z, 0.0, 0.0, 1.0, 0.0, 1.0)]
    model.add_dirichlet("300", region)
    model.add_neumann("0", region)
    model.add_convection("10", "300", region)
    model.set_default_dirichlet("300")
    model.set_default_neumann("0")
    model.set_default_convection("10", "300")
    model.add_variable("x0", "0")
    model.add_function_expr("constant", "1")
    model.add_function_gauss("gauss", 1.0, 1.0, 0.0)
    model.add_function_sine("sine", 1.0, 1.0, 0.0)
    model.add_function_double_exponential("double_exp", 1.0, 1.0, 2.0)
    model.add_function_piecewise("piecewise", np.array([[0.0, 0.0], [1.0, 1.0]]))
    model.add_function_periodic_piecewise_constant(
        "periodic", np.array([0.0, 1.0]), 1.0
    )
    model.add_probe("center", 0.5, 0.5, 0.5)
    model.add_fluid_boundary(Axis.Z, 0.0, 0.0, 1.0, 0.0, 1.0, 0, 0.0)

    compiled = model.compile()
    assert compiled.cell_count == 1


def test_solution_views_derive_from_one_native_snapshot():
    model = _model(transient=True)
    model.set_default_neumann("0")
    model.add_probe("center", 0.5, 0.5, 0.5)

    solution = model.compile().solve()

    assert solution.fvm_count == 1
    assert solution.temperature.base is solution.state
    assert np.shares_memory(solution.temperature_history, solution.state_history)
    assert solution.state_history.shape[1] == solution.state.shape[0]
    assert solution.history_times[0] == pytest.approx(0.0)
    assert solution.history_times[-1] == pytest.approx(1.0)
    assert [trace.name for trace in solution.probes] == ["center"]
