"""Public C/Python path for meshless reduced domains and physical connections."""

import numpy as np
import pytest

from metahotspot import Model, RomPortPatch, Study, SurfaceRegion
from metahotspot.enums import Axis, StudyType


def test_rom_only_state_layout_initialization_and_steady_solve():
    model = Model()
    rom = model.add_rom_domain(
        np.array([[2.0]]),
        np.array([[1.0]]),
        np.array([600.0]),
        initial_state=np.array([290.0]),
    )
    assert rom == 0
    compiled = model.compile()
    assert compiled.info.domain_count == 1
    assert compiled.info.cell_count == 0
    assert compiled.info.state_count == 1
    np.testing.assert_allclose(compiled.default_state(), [290.0])
    assert compiled.domains[0].cell_count == 0
    assert compiled.field_layouts[0].dofs.tolist() == [0]
    np.testing.assert_allclose(compiled.assemble().K.toarray(), [[2.0]])
    solution = compiled.solve()
    assert solution.converged
    np.testing.assert_allclose(solution.state, [300.0], atol=1e-8)


def test_fom_rom_connection_conserves_flux_and_reconstructs_probe():
    model = Model()
    model.set_cartesian_mesh([0, 1], [0, 1], [0, 1])
    solid = model.add_material("solid", "1", "1", "1", "1", "1")
    volume = model.add_volume(solid)
    model.add_box(volume, (0, 0, 0), (1, 1, 1))
    model.set_default_neumann("0")
    model.add_dirichlet(
        [SurfaceRegion(Axis.X, 0, 0, 1, 0, 1)],
        "300",
    )
    a = model.add_port(0, "fom-join", SurfaceRegion(Axis.X, 1, 0, 1, 0, 1))
    rom = model.add_rom_domain(
        np.array([[2.0]]),
        np.array([[1.0]]),
        np.array([560.0]),
        initial_state=np.array([290.0]),
    )
    b = model.add_rom_port(
        rom,
        "rom-join",
        SurfaceRegion(Axis.X, 1, 0, 1, 0, 1),
        outward_sign=-1,
        patches=[RomPortPatch((0, 1, 0, 1), 0.5, (0.5, 0.5), [1.0])],
    )
    connection = model.connect(0, a, rom, b)
    model.add_point_probe("rom-probe", (2.5, 0.5, 0.5))
    model.add_rom_observation(rom, "rom-probe", (2.5, 0.5, 0.5), [1.0])

    compiled = model.compile()
    assert compiled.info.domain_count == 2
    assert compiled.info.cell_count == 1
    assert compiled.info.state_count == 2
    np.testing.assert_allclose(compiled.default_state(), [300.0, 290.0])
    operators = compiled.assemble()
    np.testing.assert_allclose(
        operators.K.toarray(), [[3.0, -1.0], [-1.0, 3.0]], atol=1e-12
    )
    solution = compiled.solve()
    assert solution.converged
    np.testing.assert_allclose(solution.state, [295.0, 285.0], atol=1e-7)
    np.testing.assert_allclose(
        compiled.connection_flux(connection, solution.state), [10.0], atol=1e-7
    )
    assert solution.probes[0].name == "rom-probe"
    assert solution.probes[0].values[-1] == pytest.approx(285.0)

    transient = compiled.solve(study=Study(StudyType.TRANSIENT, 300.0, 0.5, 0.25))
    assert transient.converged
    assert transient.state_history.shape[1] == 2
    assert transient.probes[0].values.size == transient.history_times.size


@pytest.mark.parametrize("htc, expected", [(100.0, 100.0), (200.0, 150.0)])
def test_rom_affine_convection_reuses_modal_operator_family(htc, expected):
    model = Model()
    domain = model.add_rom_domain(
        np.array([[2.0]]), np.array([[1.0]]), np.array([0.0])
    )
    # A=0.01 m², T_inf=300 K; additive term is h*(A*T - A*T_inf).
    model.add_rom_affine_term(
        domain, "h", htc, np.array([[0.01]]), np.array([3.0])
    )
    compiled = model.compile()
    np.testing.assert_allclose(compiled.assemble().K.toarray(), [[2 + 0.01 * htc]])
    result = compiled.solve()
    assert result.converged
    np.testing.assert_allclose(result.state, [expected], atol=1e-8)
