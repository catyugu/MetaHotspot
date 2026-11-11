"""Cartesian mesh input, source projection and operator export contracts."""

import numpy as np
import pytest
from test_model_binding import _model

from metahotspot import SurfaceRegion
from metahotspot.enums import MaterialKind


def test_conformal_sources_keep_independent_power_columns():
    from metahotspot.macromodel.affine import AffineParametricModel, SourcePort

    model = _model()
    model.set_conformal_mesh([0, 0.00025, 0.001], [0, 0.001], [0, 0.001])
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
        model.set_conformal_mesh(*(np.linspace(0, 0.001, 4) for _ in range(3)))
        model.set_default_convection("10", "300")
        model.add_pressure_inlet(
            [SurfaceRegion(0, 0, 0, 0.001, 0, 0.001)],
            100 if case == 1 else 200,
            inlet_temperature=300,
            pump_resistance=0 if case == 1 else 1e7,
        )
        model.add_pressure_outlet([SurfaceRegion(0, 0.001, 0, 0.001, 0, 0.001)], 0)
        return model

    original = fluid_model().compile()
    reference = original.assemble()
    explicit = fluid_model()
    _explicit_cartesian(explicit, tuple(np.linspace(0, 0.001, 4) for _ in range(3)))
    repeated = explicit.compile()
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
    model.set_conformal_mesh(
        np.array([0.0, 0.00025, 0.001]), np.array([0.0, 0.001]), np.array([0.0, 0.001])
    )
    compiled = model.compile()
    boundary = BoundaryGroup(CellGeometry(compiled.fields).surface(Face.XP))

    class Ports(AffineParametricModel):
        _full = compiled

        def boundary_groups(self):
            return (boundary,)

    fields = compiled.fields
    axes = fields.boundary_axis[boundary.surface.facet_ids]
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
            model.set_conformal_mesh(axis, [0.0, 0.001], [0.0, 0.001])
    else:
        model.set_conformal_mesh(axis, [0.0, 0.001], [0.0, 0.001])
        with pytest.raises((ValueError, MetaHotspotError)):
            model.compile()


def _explicit_cartesian(model, axes):
    shape = tuple(len(a) - 1 for a in axes)
    indices = list(np.ndindex(shape))
    lower = [[axes[a][i[a]] for a in range(3)] for i in indices]
    upper = [[axes[a][i[a] + 1] for a in range(3)] for i in indices]
    pairs = []
    for i in indices:
        for a in range(3):
            other = list(i)
            other[a] += 1
            if other[a] < shape[a]:
                pairs.append(
                    [
                        np.ravel_multi_index(i, shape),
                        np.ravel_multi_index(tuple(other), shape),
                    ]
                )
    model.set_nonconformal_mesh(lower, upper, pairs)


@pytest.mark.parametrize("transient", [False, True])
def test_equivalent_meshes_solve_steady_and_transient_with_probes(transient):
    from metahotspot import Study
    from metahotspot.enums import StudyType

    axes = ([0, 0.00025, 0.001], [0, 0.0006, 0.001], [0, 0.001])
    original, explicit = _model(), _model()
    original.set_conformal_mesh(*axes)
    _explicit_cartesian(explicit, axes)
    for model in (original, explicit):
        model.set_default_convection("10", "300")
        model.add_point_probe("interior", [0.0002, 0.0004, 0.0005])
        model.add_point_probe("surface", [0, 0.0003, 0.0005])
    original, explicit = original.compile(), explicit.compile()
    a, b = original.assemble(), explicit.assemble()
    np.testing.assert_allclose(a.K.toarray(), b.K.toarray(), rtol=1e-14, atol=1e-18)
    np.testing.assert_allclose(a.C.toarray(), b.C.toarray(), rtol=1e-14, atol=0)
    np.testing.assert_allclose(a.f, b.f, rtol=1e-14)
    np.testing.assert_array_equal(original.connections(), explicit.connections())
    study = (
        Study(type=StudyType.TRANSIENT, duration=0.2, output_interval=0.05)
        if transient
        else Study()
    )
    a, b = original.solve(study), explicit.solve(study)
    assert a.converged and b.converged
    np.testing.assert_allclose(a.state, b.state, rtol=0, atol=1e-8)
    np.testing.assert_allclose(a.state_history, b.state_history, rtol=0, atol=1e-8)
    for first, second in zip(a.probes, b.probes, strict=True):
        assert first.name == second.name
        np.testing.assert_allclose(first.values, second.values, rtol=0, atol=1e-8)


def test_nonconformal_coarse_fine_steady_transient_and_cut_ports():
    from metahotspot import Model, Study
    from metahotspot.enums import Axis, StudyType
    from metahotspot.macromodel.affine import AffineParametricModel
    from metahotspot.macromodel.embeddable import enumerate_interface_ports

    def make_model(*, steady):
        model = Model()
        material = model.add_material("solid", "1", "1", "1", "1", "1")
        source = model.add_source("0" if steady else "2")
        volume = model.add_volume(material, source)
        model.add_box(volume, [0, 0, 0], [2, 2, 1])
        model.set_nonconformal_mesh(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[1, 2, 1], [2, 1, 1], [2, 2, 1]],
            [[0, 1], [2, 0], [1, 2]],
        )
        if steady:
            model.add_dirichlet([SurfaceRegion(Axis.X, 0, 0, 2, 0, 1)], "300")
            model.add_dirichlet([SurfaceRegion(Axis.X, 2, 0, 2, 0, 1)], "400")
        return model.compile()

    steady = make_model(steady=True)
    result = steady.solve()
    assert result.converged
    np.testing.assert_allclose(result.state, [325, 375, 375], atol=1e-6)
    fields = steady.fields
    area = (fields.boundary_bounds[:, 1] - fields.boundary_bounds[:, 0]) * (
        fields.boundary_bounds[:, 3] - fields.boundary_bounds[:, 2]
    )
    assert area.sum() == pytest.approx(16)

    class Ports(AffineParametricModel):
        _full = steady

        def boundary_groups(self):
            return ()

    ports = enumerate_interface_ports(Ports(), [0])
    cut = next(p for p in ports if p.axis == 0 and p.direction == 1)
    np.testing.assert_array_equal(cut.cells, [0, 0])
    np.testing.assert_allclose(cut.areas, [1, 1])
    transient = make_model(steady=False)
    solution = transient.solve(
        Study(type=StudyType.TRANSIENT, duration=0.5, output_interval=0.1)
    )
    assert solution.converged
    expected = 300 + 2 * solution.history_times
    np.testing.assert_allclose(
        solution.state_history, np.repeat(expected[:, None], 3, axis=1), atol=1e-5
    )
    energy = solution.state_history @ np.array([2, 1, 1])
    np.testing.assert_allclose(energy - 1200, 8 * solution.history_times, atol=1e-5)
