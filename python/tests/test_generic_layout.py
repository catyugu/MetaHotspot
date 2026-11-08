"""GL-01 A / GL-02 / GL-09 through the public Python and C ABI paths."""

import numpy as np
import pytest

import metahotspot
from metahotspot import Study, SurfaceRegion
from metahotspot.enums import Axis, EntitySupport, FieldQuantity, FluidBC, StudyType


def package(*, fluid=True, anisotropic=False):
    m = metahotspot.Model()
    m.set_cartesian_mesh(
        np.arange(13) * 0.001, np.arange(9) * 0.001, np.arange(11) * 0.0005
    )
    materials = [
        m.add_material(
            "silicon",
            "130",
            "90" if anisotropic else "130",
            "150" if anisotropic else "130",
            "2330",
            "700",
        ),
        m.add_material("TIM", "4", "4", "4", "2500", "1000"),
        m.add_material("copper", "390", "390", "390", "8960", "385"),
    ]
    if fluid:
        materials.append(
            m.add_material("liquid", "0.6", "0.6", "0.6", "997", "4180", "8.9e-4")
        )
    source = m.add_source("1.25e8")
    volumes = [
        (0, None, [0, 0, 0], [0.012, 0.008, 0.001]),
        (1, None, [0, 0, 0.001], [0.012, 0.008, 0.0015]),
        (2, None, [0, 0, 0.0015], [0.012, 0.008, 0.005]),
        (0, source, [0.004, 0.002, 0], [0.008, 0.006, 0.001]),
    ]
    if fluid:
        volumes.append((3, None, [0, 0.003, 0.0025], [0.012, 0.005, 0.0035]))
    for material, power, lo, hi in volumes:
        v = m.add_volume(materials[material], 0xFFFFFFFF if power is None else power)
        m.add_box(v, lo, hi)
    m.add_convection([SurfaceRegion(Axis.Z, 0.005, 0, 0.012, 0, 0.008)], "400", "300")
    if fluid:
        m.add_fluid_boundary(
            [SurfaceRegion(Axis.X, 0, 0.003, 0.005, 0.0025, 0.0035)],
            FluidBC.MASS_FLOW_INLET,
            2e-4,
            300,
        )
        m.add_fluid_boundary(
            [SurfaceRegion(Axis.X, 0.012, 0.003, 0.005, 0.0025, 0.0035)],
            FluidBC.OUTLET,
            0,
        )
    for name, xyz in [
        ("hotspot", [0.006, 0.004, 0.00025]),
        ("upstream", [0.0015, 0.0035, 0.00275]),
        ("downstream", [0.0105, 0.0035, 0.00275]),
        ("copper", [0.006, 0.004, 0.00475]),
    ]:
        m.add_point_probe(name, xyz)
    return m


@pytest.mark.parametrize("fluid", [False, True])
def test_package_layout_recompile_and_snapshot_lifetime(fluid):
    with package(fluid=fluid) as m:
        c = m.compile()
        again = m.compile()
        np.testing.assert_array_equal(
            c.field_layouts[0].dofs, again.field_layouts[0].dofs
        )
        again.close()
    assert c.info.state_count == c.info.cell_count == 960
    assert c.info.domain_count == c.info.field_count == 1
    assert c.domains[0].cell_count == 960
    field = c.field_layouts[0]
    assert (field.domain, field.quantity, field.support, field.component) == (
        0,
        FieldQuantity.TEMPERATURE,
        EntitySupport.CELL,
        0,
    )
    np.testing.assert_array_equal(field.entities, np.arange(960))
    np.testing.assert_array_equal(field.dofs, np.arange(960))
    counts = np.bincount(c.fields.material_id)
    np.testing.assert_array_equal(
        counts, [192, 96, 624, 48] if fluid else [192, 96, 672]
    )
    assert c.source_operator().sum() == pytest.approx(2)
    steady = c.solve()
    assert steady.converged
    assert steady.state.max() > 300
    assert np.isfinite(steady.state).all()
    ops = c.assemble(steady.state)
    # Independent aggregate energy residual, relative to the prescribed 2 W source.
    assert abs((ops.K @ steady.state - ops.f).sum()) < 2e-6
    transient = c.solve(Study(StudyType.TRANSIENT, 300, 2, 0.1))
    assert transient.converged
    assert transient.state_history.shape == (21, 960)
    repeated = c.solve()
    np.testing.assert_allclose(repeated.state, steady.state, rtol=1e-12, atol=1e-10)
    c.close()
    assert transient.state.max() > 300


def test_default_study_belongs_to_definition_and_compiled_snapshot():
    with package(fluid=False) as m:
        m.set_study(Study(StudyType.TRANSIENT, 310, 0.1, 0.05))
        compiled = m.compile()
        m.set_study(Study())
    solution = compiled.solve()
    assert solution.time == pytest.approx(0.1)
    assert solution.state_history.shape == (3, 960)
    np.testing.assert_array_equal(solution.state_history[0], 310)
    override = compiled.solve(Study())
    assert override.time == 0
    compiled.close()


def test_gl02_anisotropic_solid_and_fluid_share_one_temperature_field():
    with package(anisotropic=True) as m:
        c = m.compile()
    values = c.eval_materials()
    solid = c.fields.material_id == 0
    np.testing.assert_array_equal(values.conductivity_x[solid], 130)
    np.testing.assert_array_equal(values.conductivity_y[solid], 90)
    np.testing.assert_array_equal(values.conductivity_z[solid], 150)
    assert len(c.field_layouts) == 1
    assert c.field_layouts[0].dofs.size == 960
    solution = c.solve()
    assert solution.converged
    assert len(solution.probes) == 4
    c.close()
