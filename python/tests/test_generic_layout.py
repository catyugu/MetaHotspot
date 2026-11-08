"""GL-01 A / GL-02 / GL-09 through the public Python and C ABI paths."""

import numpy as np
import pytest

import metahotspot
from metahotspot import Convection, SolveOptions, Study, SurfaceRegion
from metahotspot.enums import Axis, EntitySupport, FieldQuantity, FluidBC, StudyType


def package(*, fluid=True, anisotropic=False, split=False, coarse=False, refinement=1):
    m = metahotspot.Model()
    m.add_cartesian_component()
    m.set_cartesian_mesh(
        np.arange(12 * refinement + 1) * 0.001 / refinement,
        np.arange(8 * refinement + 1) * 0.001 / refinement,
        np.arange(10 * refinement + 1) * 0.0005 / refinement,
    )
    if split:
        m.set_cartesian_mesh(
            np.arange(13) * 0.001, np.arange(9) * 0.001, np.arange(9) * 0.0005
        )
        upper = m.add_cartesian_component()
        m.set_cartesian_mesh(
            np.arange(7 if coarse else 13) * (0.002 if coarse else 0.001),
            np.arange(5 if coarse else 9) * (0.002 if coarse else 0.001),
            [0.004, 0.0045, 0.005],
            component=upper,
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
    if split:
        volume = m.add_volume(materials[2], component=upper)
        m.add_box(volume, [0, 0, 0.004], [0.012, 0.008, 0.005], component=upper)
        region = SurfaceRegion(Axis.Z, 0.004, 0, 0.012, 0, 0.008)
        a = m.add_port(0, "top", region)
        b = m.add_port(upper, "bottom", region)
        m.connect(0, a, upper, b)
    m.add_thermal_boundary(
        [SurfaceRegion(Axis.Z, 0.005, 0, 0.012, 0, 0.008)],
        Convection("400", "300"),
        component=upper if split else 0,
    )
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
        m.add_observation(upper if split and xyz[2] >= 0.004 else 0, name, xyz)
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
    assert c.info.component_count == c.info.field_count == 1
    assert c.components[0].cell_count == 960
    field = c.field_layouts[0]
    assert (field.component, field.quantity, field.support, field.offset) == (
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


def test_study_is_solve_time_configuration():
    with package(fluid=False) as m:
        compiled = m.compile()
    steady = compiled.solve()
    assert steady.time == 0
    transient = compiled.solve(Study(StudyType.TRANSIENT, 310, 0.1, 0.05))
    assert transient.time == pytest.approx(0.1)
    assert transient.state_history.shape == (3, 960)
    np.testing.assert_array_equal(transient.state_history[0], 310)
    repeated = compiled.solve()
    np.testing.assert_allclose(repeated.state, steady.state, rtol=1e-12, atol=1e-10)
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


def test_gl04_matched_foms_equal_monolith_operators_states_and_probes():
    with package() as m, package(split=True) as split:
        for definition in (m, split):
            definition.add_observation(
                0, "near_lower_interface", [0.00625, 0.00425, 0.0039]
            )
            definition.add_observation(
                1 if definition is split else 0,
                "near_upper_interface", [0.00625, 0.00425, 0.0041]
            )
        a, b = m.compile(), split.compile()
    assert [d.cell_count for d in b.components] == [768, 192]
    xa = (a.fields.cell_min + a.fields.cell_max) / 2
    xb = (b.fields.cell_min + b.fields.cell_max) / 2
    permutation = np.array(
        [
            np.flatnonzero(np.all(np.isclose(xa, x, rtol=0, atol=1e-14), axis=1))[0]
            for x in xb
        ]
    )
    rng = np.random.default_rng(42)
    state = 300 + rng.random(960)
    ao, bo = a.assemble(state), b.assemble(state[permutation])
    np.testing.assert_allclose(
        bo.K.toarray(), ao.K[permutation][:, permutation].toarray(), atol=1e-14
    )
    np.testing.assert_allclose(
        bo.C.toarray(), ao.C[permutation][:, permutation].toarray(), atol=1e-14
    )
    np.testing.assert_allclose(bo.f, ao.f[permutation], atol=1e-14)
    assert b.source_operator().sum() == pytest.approx(2)
    np.testing.assert_array_equal(
        b.eval_materials().density, a.eval_materials().density[permutation]
    )
    for study in [Study(), Study(StudyType.TRANSIENT, 300, 2, 0.1)]:
        sa, sb = a.solve(study), b.solve(study)
        assert sa.converged and sb.converged
        np.testing.assert_allclose(
            sb.state_history, sa.state_history[:, permutation], atol=2e-5, rtol=1e-8
        )
        traces_a = {probe.name: probe.values for probe in sa.probes}
        traces_b = {probe.name: probe.values for probe in sb.probes}
        assert traces_a.keys() == traces_b.keys()
        for name in traces_a:
            np.testing.assert_allclose(traces_a[name], traces_b[name], atol=2e-5, rtol=1e-8)
    flux = b.connection_flux(0, sb.state)
    assert flux.size == 96 and np.isfinite(flux).all()
    repeated = b.solve()
    assert repeated.converged
    a.close()
    b.close()


def test_gl05_coarse_upper_fom_ncc_reaches_conservative_steady_and_transient():
    with package(split=True, coarse=True) as m:
        compiled = m.compile()
    assert [d.cell_count for d in compiled.components] == [768, 48]
    for study in [Study(), Study(StudyType.TRANSIENT, 300, 2, 0.1)]:
        solution = compiled.solve(
            study,
            opts=SolveOptions(
                linear_tolerance=1e-12, nonlinear_relative_tolerance=1e-11
            ),
        )
        assert solution.converged
        ops = compiled.assemble(solution.state, solution.time)
        if study.type == StudyType.STEADY:
            assert np.max(np.abs(ops.K @ solution.state - ops.f)) < 2e-6
            # Independent upper-domain energy balance against its Robin loss.
            flux = compiled.connection_flux(0, solution.state)
            centers = (compiled.fields.cell_min + compiled.fields.cell_max) / 2
            upper = np.arange(768, 816)
            surface = upper[np.isclose(centers[upper, 2], 0.00475)]
            conductance = 0.002**2 / (0.00025 / 390 + 1 / 400)
            robin = conductance * (solution.state[surface] - 300)
            assert flux.sum() == pytest.approx(robin.sum(), abs=2e-6)
            corrected = solution.state.copy()
            # Constant properties: at uniform 300 K every NCC gradient is zero.
            # This standalone linear solve is the zero-corrector TPFA control.
            from scipy.sparse.linalg import spsolve

            zero = compiled.assemble(compiled.default_state())
            baseline = spsolve(zero.K, zero.f)
            with package(refinement=2) as definition:
                reference = definition.compile()
            fine = reference.solve(
                opts=SolveOptions(
                    linear_tolerance=1e-12, nonlinear_relative_tolerance=1e-11
                )
            )
            assert fine.converged
            fine_centers = (reference.fields.cell_min + reference.fields.cell_max) / 2
            average = []
            for lo, hi in zip(
                compiled.fields.cell_min, compiled.fields.cell_max, strict=True
            ):
                inside = np.all((fine_centers > lo) & (fine_centers < hi), axis=1)
                average.append(fine.state[inside].mean())
            average = np.array(average)
            corrected_rms = np.sqrt(np.mean((corrected - average) ** 2))
            baseline_rms = np.sqrt(np.mean((baseline - average) ** 2))
            assert corrected_rms < baseline_rms
            # 0.5 K accommodates the independent spatial discretization error;
            # conservation and nonlinear convergence have separate tight checks.
            assert np.max(np.abs(corrected - average)) < 0.5
            reference.close()
    compiled.close()
