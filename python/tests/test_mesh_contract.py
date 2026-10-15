"""Native topology, geometry and algebraic ordering are separate contracts."""

from pathlib import Path

import numpy as np
import pytest
from test_model_binding import _model


def test_compiled_snapshot_exposes_explicit_faces_and_cell_bounds():
    compiled = _model().compile()
    assert compiled.metadata.face_count == 6
    assert compiled.metadata.state_count == 1
    cells = compiled.cells
    assert cells.cell_min.shape == (1, 3)
    assert cells.cell_max.shape == (1, 3)
    np.testing.assert_allclose(cells.cell_max - cells.cell_min, 1e-3)
    assert cells.face_owner.shape == (6,)
    assert cells.face_min.shape == (6, 3)
    np.testing.assert_array_equal(cells.cell_to_dof, [0])
    assert not hasattr(cells, "grid_to_cell")
    assert not hasattr(cells, "cell_to_grid")


def test_external_mesh_roundtrips_without_cartesian_input():
    original = _model().compile()
    f = original.cells
    model = _model()
    model.set_discrete_mesh(
        **{
            name: getattr(f, name)
            for name in (
                "cell_min",
                "cell_max",
                "face_owner",
                "face_neighbor",
                "face_axis",
                "face_sign",
                "face_min",
                "face_max",
            )
        }
    )
    imported = model.compile()
    np.testing.assert_allclose(imported.assemble().f, original.assemble().f)
    np.testing.assert_array_equal(imported.cells.face_owner, f.face_owner)


def test_native_dof_permutation_preserves_physical_results():
    model = _model()
    model.set_mesh(
        np.array([0.0, 0.25, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    model.set_default_dirichlet("300+x")
    original = model.compile()
    reference = original.assemble(np.array([301.0, 307.0]))
    physical = original.solve().temperature
    model.set_dof_order(np.array([1, 0]))
    reordered = model.compile()
    np.testing.assert_array_equal(reordered.cells.cell_to_dof, [1, 0])
    result = reordered.assemble(np.array([307.0, 301.0]))
    np.testing.assert_allclose(result.C.toarray(), reference.C.toarray()[::-1, ::-1])
    np.testing.assert_allclose(result.f, reference.f[::-1])
    np.testing.assert_allclose(reordered.solve().temperature, physical)


def test_subcell_sources_keep_independent_power_columns():
    from metahotspot.enums import GeometryOp
    from metahotspot.macromodel.affine import AffineParametricModel, SourcePort

    model = _model()
    second = model.add_block(0, "solid", heat_source="2e6")
    model.add_rect(second, GeometryOp.ADD, "0", "0", "0.25", "1")
    compiled = model.compile()
    q = compiled.source_operator()
    np.testing.assert_allclose(q.toarray(), [[0.00075, 0.0005]])
    np.testing.assert_allclose(q.sum(axis=1).A.ravel(), compiled.assemble().f)
    np.testing.assert_allclose(compiled.cells.projection_volume.sum(), 1e-9)

    class Ports(AffineParametricModel):
        _full = compiled
        core = compiled.assemble()

        def source_ports(self):
            return [
                SourcePort(source_ids=np.array([0]), power_W=0.00075),
                SourcePort(source_ids=np.array([1]), power_W=0.0005),
            ]

    ports = Ports()
    np.testing.assert_allclose(ports.source_shape, [[1, 1]])
    np.testing.assert_allclose(
        ports.source_shape @ ports.nominal_power(), compiled.assemble().f
    )


@pytest.mark.parametrize("case", [1, 2])
def test_production_external_mesh_decouples_cell_and_state_order(case):
    import metahotspot
    import scipy.sparse.linalg as spla

    path = (
        Path(__file__).resolve().parents[2]
        / "cases"
        / "microfluid_cases"
        / f"steady_case{case}.xml"
    )
    original_model = metahotspot.Model()
    original_model.read_xml(str(path))
    original = original_model.compile()
    reference = original.assemble()
    fields = original.cells
    order = np.random.default_rng(44).permutation(original.cell_count)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    neighbors = fields.face_neighbor.copy()
    internal = neighbors != np.iinfo(neighbors.dtype).max
    neighbors[internal] = inverse[neighbors[internal]]
    model = metahotspot.Model()
    model.read_xml(str(path))
    model.set_discrete_mesh(
        cell_min=fields.cell_min[order],
        cell_max=fields.cell_max[order],
        face_owner=inverse[fields.face_owner],
        face_neighbor=neighbors,
        face_axis=fields.face_axis,
        face_sign=fields.face_sign,
        face_min=fields.face_min,
        face_max=fields.face_max,
    )
    model.set_dof_order(order)
    imported = model.compile()
    result = imported.assemble()
    k_error = spla.norm(result.K - reference.K) / spla.norm(reference.K)
    c_error = spla.norm(result.C - reference.C) / spla.norm(reference.C)
    f_error = np.linalg.norm(result.f - reference.f) / np.linalg.norm(reference.f)
    # Hydraulic preprocessing solves pressure iteratively at relative 1e-8.
    # Reordering fluid cells changes its finite-tolerance pressure/flux result.
    # Pure geometry/capacity and source projections have no such solve error.
    assert k_error < 1e-8
    assert c_error < 1e-13
    assert f_error < 1e-8
    q_reference = original.source_operator()
    q_result = imported.source_operator()
    assert spla.norm(q_result - q_reference) <= 1e-13 * max(
        spla.norm(q_reference), 1e-300
    )
    np.testing.assert_array_equal(imported.cells.cell_min, fields.cell_min[order])
    expected_temperature = original.solve().temperature[order]
    actual_temperature = imported.solve().temperature
    temperature_error = np.max(np.abs(actual_temperature - expected_temperature))
    np.testing.assert_allclose(
        actual_temperature, expected_temperature, rtol=0, atol=1e-5
    )
    print(
        f"case={case} states={order.size} K_rel={k_error:.3e} C_rel={c_error:.3e} f_rel={f_error:.3e} T_max={temperature_error:.3e} K"
    )


def test_ctm_robin_projection_uses_native_face_normal_and_distance():
    from metahotspot.enums import Face
    from metahotspot.macromodel.affine import AffineParametricModel, BoundaryGroup
    from metahotspot.macromodel.geometry import CellGeometry

    model = _model()
    model.set_mesh(
        np.array([0.0, 0.25, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    compiled = model.compile()
    boundary = BoundaryGroup(CellGeometry(compiled.cells).surface(Face.XP))

    class Ports(AffineParametricModel):
        _full = compiled

        def boundary_groups(self):
            return (boundary,)

    fields = compiled.cells
    half = fields.owner_distance[boundary.face_ids]
    expected = np.average(1000 / (1 + 1000 * half), weights=boundary.areas)
    np.testing.assert_allclose(Ports().physical_to_effective([1000]), [expected])


def test_dof_order_rejects_fractional_and_out_of_range_values():
    model = _model()
    for order in ([0.5], [-1], [2**64], [np.nan]):
        with pytest.raises(ValueError):
            model.set_dof_order(order)


def test_external_mesh_rejects_fractional_connectivity_before_casting():
    fields = _model().compile().cells
    inputs = {
        name: getattr(fields, name).copy()
        for name in (
            "cell_min",
            "cell_max",
            "face_owner",
            "face_neighbor",
            "face_axis",
            "face_sign",
            "face_min",
            "face_max",
        )
    }
    for name in ("face_owner", "face_neighbor", "face_axis", "face_sign"):
        invalid = inputs[name].astype(float)
        invalid[0] = 0.5
        with pytest.raises(ValueError):
            _model().set_discrete_mesh(**{**inputs, name: invalid})
