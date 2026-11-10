"""Cartesian mesh input, source projection and operator export contracts."""

import numpy as np
import pytest
from test_model_binding import _model

from metahotspot import SurfaceRegion
from metahotspot.enums import MaterialKind


def test_conformal_sources_keep_independent_power_columns():
    from metahotspot.macromodel.affine import AffineParametricModel, SourcePort

    model = _model()
    model.set_cartesian_mesh([0, 0.00025, 0.001], [0, 0.001], [0, 0.001])
    second = model.add_volume(0, model.add_source("2e6"))
    model.add_box(second, [0, 0, 0], [0.00025, 0.001, 0.001])
    compiled = model.compile()
    q = compiled.source_operator()
    np.testing.assert_allclose(q.toarray(), [[0, 0.0005], [0.00075, 0]])
    np.testing.assert_allclose(q.sum(axis=1).A.ravel(), compiled.assemble().f)
    np.testing.assert_allclose(compiled.fields.projection_volume.sum(), 1e-9)

    class Ports(AffineParametricModel):
        _full = compiled
        core = compiled.assemble()

        def source_ports(self):
            return [
                SourcePort(source_ids=np.array([0]), power_W=0.00075),
                SourcePort(source_ids=np.array([1]), power_W=0.0005),
            ]

    ports = Ports()
    np.testing.assert_allclose(ports.source_shape, [[0, 1], [1, 0]])
    np.testing.assert_allclose(
        ports.source_shape @ ports.nominal_power(), compiled.assemble().f
    )


@pytest.mark.parametrize("case", [1, 2])
def test_cartesian_fluid_model_recompiles_consistently(case):
    import scipy.sparse.linalg as spla

    def fluid_model():
        model = _model()
        material = model.add_material(
            "fluid",
            "0.6",
            "0.6",
            "0.6",
            "1000",
            "4200",
            kind=MaterialKind.FLUID,
            dynamic_viscosity="0.001",
        )
        volume = model.add_volume(material, 0)
        model.add_box(volume, [0, 0, 0], [0.001, 0.001, 0.001])
        model.set_cartesian_mesh(*(np.linspace(0, 0.001, 4) for _ in range(3)))
        model.set_default_convection("10", "300")
        model.add_fluid_boundary(
            [SurfaceRegion(0, 0, 0, 0.001, 0, 0.001)], 1, 100 if case == 1 else 200, 300
        )
        model.add_fluid_boundary([SurfaceRegion(0, 0.001, 0, 0.001, 0, 0.001)], 3, 0)
        return model

    original = fluid_model().compile()
    reference = original.assemble()
    repeated = fluid_model().compile()
    result = repeated.assemble()
    # Hydraulic preprocessing is iterative; the exported CSR must also retain
    # the nonsymmetric upwind operator used by the native thermal solve.
    assert spla.norm(reference.K - reference.K.T) > 0
    assert spla.norm(result.K - reference.K) <= 1e-8 * spla.norm(reference.K)
    assert spla.norm(result.C - reference.C) <= 1e-13 * spla.norm(reference.C)
    np.testing.assert_allclose(result.f, reference.f, rtol=1e-8)
    q_reference = original.source_operator()
    assert spla.norm(repeated.source_operator() - q_reference) <= 1e-13 * max(
        spla.norm(q_reference), 1e-300
    )
    np.testing.assert_array_equal(repeated.fields.cell_min, original.fields.cell_min)
    expected = spla.spsolve(reference.K, reference.f)
    solution = repeated.solve()
    assert solution.converged
    np.testing.assert_allclose(solution.state, expected, rtol=0, atol=1e-5)


def test_ctm_robin_projection_uses_native_face_normal_and_distance():
    from metahotspot.enums import Face
    from metahotspot.macromodel.affine import AffineParametricModel, BoundaryGroup
    from metahotspot.macromodel.geometry import CellGeometry

    model = _model()
    model.set_cartesian_mesh(
        np.array([0.0, 0.00025, 0.001]), np.array([0.0, 0.001]), np.array([0.0, 0.001])
    )
    compiled = model.compile()
    boundary = BoundaryGroup(CellGeometry(compiled.fields).surface(Face.XP))

    class Ports(AffineParametricModel):
        _full = compiled

        def boundary_groups(self):
            return (boundary,)

    fields = compiled.fields
    axes = fields.facet_axis[boundary.surface.facet_ids]
    half = CellGeometry(fields).half_sizes[boundary.surface.cell_ids, axes]
    expected = np.average(1000 / (1 + 1000 * half), weights=boundary.surface.areas)
    np.testing.assert_allclose(Ports().physical_to_effective([1000]), [expected])


@pytest.mark.parametrize(
    "axis", [[0.0], [0.0, 0.0], [1.0, 0.0], [0.0, float("nan")], [0.0, float("inf")]]
)
def test_native_mesh_input_rejects_invalid_vertices(axis):
    from metahotspot._error import MetaHotspotError

    model = _model()
    if len(axis) < 2:
        with pytest.raises((ValueError, MetaHotspotError)):
            model.set_cartesian_mesh(axis, [0.0, 0.001], [0.0, 0.001])
    else:
        model.set_cartesian_mesh(axis, [0.0, 0.001], [0.0, 0.001])
        with pytest.raises((ValueError, MetaHotspotError)):
            model.compile()
