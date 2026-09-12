"""Unit tests for the model-agnostic ``metahotspot.macromodel`` package.

These tests exercise the generic algorithms without any named case: face-port
conductance, explicit-face exclusion, common-patch area fractions, symmetry of
the coupled system, identity-basis coupling agreement with the monolithic split
reference, and the Extended-FANTASTIC extraction state machine.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import metahotspot
from metahotspot import macromodel as mm
from metahotspot.enums import Face, LengthUnit, Study
from metahotspot.macromodel.geometry import CellGeometry


def _build_operators(nx=2, ny=2, nz=3):
    model = metahotspot.Model()
    model.set_settings(study=Study.STEADY, length_unit=LengthUnit.MILLIMETER)
    model.set_mesh(
        np.linspace(0.0, float(nx), nx + 1),
        np.linspace(0.0, float(ny), ny + 1),
        np.linspace(0.0, float(nz), nz + 1),
    )
    model.add_material("solid", "1", "1", "1", "1000", "500")
    layer = model.add_layer(str(float(nz)))
    block = model.add_block(layer, "solid", heat_source="1e6")
    model.add_rect(block, 0, "0", "0", f"{nx}", f"{ny}")
    model.set_default_neumann("0")
    compiled = model.compile()
    ops = compiled.assemble()
    return model, compiled, ops


def _cell_split(cells, half_z):
    geometry = CellGeometry(cells)
    zc = geometry.centers[:, 2]
    lower = np.flatnonzero(zc < half_z)
    upper = np.flatnonzero(zc >= half_z)
    return lower, upper


class _FakePortsModel:
    def __init__(self, compiled, ops):
        self._full = compiled
        self.geometry = CellGeometry(compiled.cells)
        self._ops = ops
        self.cell_layout = mm.CellLayout(
            centers=self.geometry.centers,
            half_sizes=self.geometry.half_sizes,
            conductivity=np.column_stack(
                (
                    np.full(compiled.cell_count, 1.0),
                    np.full(compiled.cell_count, 1.0),
                    np.full(compiled.cell_count, 1.0),
                )
            ),
        )

    @property
    def full_cell_count(self):
        return self._full.cell_count

    @property
    def core(self):
        return self._ops

    def boundary_groups(self):
        return ()

    @property
    def boundary_terms(self):
        return []

    def h_ranges(self):
        return np.empty((0, 2))

    def physical_to_effective(self, h):
        return np.asarray(h, dtype=np.float64)

    @property
    def source_shape(self):
        return self._ops.f.reshape(-1, 1)

    def build_geometry(self, study, *, detail, macro):
        raise NotImplementedError


def test_face_port_conductance_uses_face_area_and_half_distance():
    _model, compiled, ops = _build_operators(1, 1, 1)
    model = _FakePortsModel(compiled, ops)
    port = mm.FacePort(
        label="z+",
        axis=2,
        direction=1,
        cells=np.array([0]),
        areas=np.array([1.0e-6]),
        k=np.array([1.0]),
        half=np.array([0.5e-3]),
        t1=0,
        t2=1,
        rects=np.array([[0.0, 1.0e-3, 0.0, 1.0e-3]]),
    )
    assert port.g[0] == pytest.approx(1.0 * 1.0e-6 / 0.5e-3)
    assert port.normal == "z+"


def test_enumerate_ports_excludes_declared_ambient_faces():
    _model, compiled, ops = _build_operators(2, 2, 3)
    model = _FakePortsModel(compiled, ops)
    ports = mm.enumerate_interface_ports(model, np.arange(compiled.cell_count))
    assert {p.normal for p in ports} == {"x-", "x+", "y-", "y+", "z-", "z+"}
    for p in ports:
        assert np.all(p.g > 0.0)

    top = CellGeometry(compiled.cells).indices[:, 2] == compiled.metadata.nz - 1
    top_cells = np.flatnonzero(top)
    top_areas = np.full(top_cells.size, 1.0e-6)
    model.boundary_groups = lambda: (
        mm.BoundaryGroup(cells=top_cells, areas=top_areas),
    )
    ports = mm.enumerate_interface_ports(model, np.arange(compiled.cell_count))
    labels = {p.normal for p in ports}
    assert "z+" not in labels
    assert "z-" in labels


def _rect_port(n, lo, hi, axis=2, direction=1):
    t1, t2 = [a for a in range(3) if a != axis]
    xs = np.linspace(lo, hi, n + 1)
    cells = np.arange(n * n)
    rects = np.column_stack(
        (
            np.repeat(xs[:-1], n),
            np.repeat(xs[1:], n),
            np.tile(xs[:-1], n),
            np.tile(xs[1:], n),
        )
    )
    area = ((hi - lo) / n) ** 2
    return mm.FacePort(
        label="z+",
        axis=axis,
        direction=direction,
        cells=cells,
        areas=np.full(n * n, area),
        k=np.full(n * n, 1.0),
        half=np.full(n * n, 0.5e-3),
        t1=t1,
        t2=t2,
        rects=rects,
    )


def test_nonconforming_ports_are_split_into_area_weighted_common_patches():
    coarse = _rect_port(1, 0.0, 1.0e-3)
    fine = _rect_port(2, 0.0, 1.0e-3)
    areas, E_l, E_r, xi_l, xi_r, li, ri = mm.common_patches(coarse, fine)

    assert areas.size == 4
    assert np.allclose(areas, 0.25e-6)
    assert np.allclose(xi_l, 0.25)
    assert np.allclose(xi_r, 1.0)
    assert set(li.tolist()) == {0}
    assert set(ri.tolist()) == {0, 1, 2, 3}
    assert E_l.shape == (4, 1) and E_r.shape == (4, 4)
    assert np.allclose(np.asarray(E_l.sum(axis=1)).ravel(), 1.0)
    assert np.allclose(np.asarray(E_r.sum(axis=1)).ravel(), 1.0)


def test_identity_coupling_reproduces_monolithic_steady():
    _model, compiled, ops = _build_operators(2, 2, 3)
    model = _FakePortsModel(compiled, ops)
    lower, upper = _cell_split(compiled.cells, 1.5e-3)

    K = ops.K.tocsc()
    pin = 0
    K = K + sp.diags(np.eye(compiled.cell_count)[pin] * 1.0e3)
    source = ops.f.ravel()
    steady_mono = sp.linalg.spsolve(K, source)

    left = mm.build_subdomain(model, lower, name="lower")
    right = mm.build_subdomain(model, upper, name="upper")
    assert {p.normal for p in left.ports} == {"x-", "x+", "y-", "y+", "z-", "z+"}
    assert {p.normal for p in right.ports} == {"x-", "x+", "y-", "y+", "z-", "z+"}

    Kc, Cc, rhsc, lo, ro, npatch = mm.connect(
        left, right, left.port("z+"), right.port("z-"), power=np.array([1.0])
    )
    Kc = sp.csc_matrix(Kc)
    Kc[pin, pin] += 1.0e3
    steady_c = mm.solve_system(Kc, Cc, rhsc, dt=1.0, duration=0.0)[0]

    full = np.zeros(compiled.cell_count)
    full[left.cells] = steady_c[:lo]
    full[right.cells] = steady_c[lo + npatch :]
    assert np.max(np.abs(full - steady_mono)) < 1.0e-8


def test_coupled_system_is_symmetric_psd():
    _model, compiled, ops = _build_operators(2, 2, 3)
    model = _FakePortsModel(compiled, ops)
    lower, upper = _cell_split(compiled.cells, 1.5e-3)
    left = mm.build_subdomain(model, lower, name="lower")
    right = mm.build_subdomain(model, upper, name="upper")
    Kc, Cc, rhsc, lo, ro, npatch = mm.connect(
        left, right, left.port("z+"), right.port("z-"), power=np.array([1.0])
    )
    Kd = Kc.toarray()
    assert np.max(np.abs(Kd - Kd.T)) < 1.0e-10
    eig = np.linalg.eigvalsh(Kd)
    assert eig.min() >= -1.0e-9
    assert Cc.shape == Kc.shape


def test_steady_solver_returns_a_converged_solution_for_ill_conditioned_spd():
    n = 256
    diagonal = np.geomspace(1.0e-8, 1.0, n)
    matrix = sp.diags(diagonal, format="csc")
    rhs = np.ones(n)

    solution = mm.utils.solve_rom_steady(matrix, rhs.reshape(-1, 1), np.array([1.0]))

    relative_residual = np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs)
    assert relative_residual < 1.0e-8


def test_closing_svd_uses_largest_singular_value_ratio():
    angle = 0.18
    snapshots = np.array(
        [
            [1.0, np.cos(angle)],
            [0.0, np.sin(angle)],
        ]
    )
    basis, singular_values = mm.utils._snapshot_svd_basis(snapshots, 0.1)

    assert basis.shape == (2, 1)
    assert singular_values[1] / singular_values[0] < 0.1


def test_closing_svd_compares_response_directions_not_column_amplitudes():
    snapshots = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scaled = snapshots * np.array([100.0, 1.0e-3, 7.0])

    basis, singular_values = mm.utils._snapshot_svd_basis(snapshots, 1.0e-3)
    scaled_basis, scaled_singular_values = mm.utils._snapshot_svd_basis(
        scaled, 1.0e-3
    )

    assert basis.shape == scaled_basis.shape
    assert np.allclose(singular_values, scaled_singular_values)
    assert np.allclose(basis @ basis.T, scaled_basis @ scaled_basis.T)


def test_each_matching_point_starts_with_an_empty_local_response_space(monkeypatch):
    monkeypatch.setattr(mm.utils, "port_eigenvalue_bounds", lambda *_: (1.0, 2.0))
    monkeypatch.setattr(mm.utils, "mpmm_elliptic_shift_count", lambda *_: 2)
    monkeypatch.setattr(
        mm.utils,
        "mpmm_elliptic_shifts",
        lambda *_: np.array([2.0, 1.0]),
    )

    full_solves = []

    def direct_solve(matrix, rhs, x0=None, rtol=None):
        full_solves.append(float(matrix.diagonal()[0]))
        return sp.linalg.spsolve(matrix, rhs)

    monkeypatch.setattr(mm.utils, "spd_solve", direct_solve)
    operators = mm.utils.normalized_operators(
        sp.diags([1.0, 3.0, 9.0]),
        sp.eye(3),
        np.zeros(3),
    )
    source = np.array([[1.0], [1.0], [0.2]])

    _basis, summary = mm.utils.build_parametric_basis(
        operators,
        source,
        [],
        np.empty((0, 2)),
        tolerance=1.0e-8,
        probe_rounds=1,
        max_order=20,
        seed=1,
    )

    assert summary["processed_candidate_count"] == 2
    assert len(full_solves) == 2


def test_failed_parameter_probe_warm_starts_the_same_point_full_solve(monkeypatch):
    monkeypatch.setattr(mm.utils, "port_eigenvalue_bounds", lambda *_: (1.0, 2.0))
    monkeypatch.setattr(mm.utils, "mpmm_elliptic_shift_count", lambda *_: 1)
    monkeypatch.setattr(mm.utils, "mpmm_elliptic_shifts", lambda *_: np.array([0.0]))

    samples = iter([(1.0,), (10.0,), (10.0,)])
    monkeypatch.setattr(mm.utils, "_draw_h", lambda _ranges, _rng: next(samples))

    initial_guesses = []

    def direct_solve(matrix, rhs, x0=None, rtol=None):
        initial_guesses.append(np.asarray(x0, dtype=np.float64).copy())
        return sp.linalg.spsolve(matrix, rhs)

    monkeypatch.setattr(mm.utils, "spd_solve", direct_solve)
    K = sp.diags([1.0, 10.0], format="csc")
    C = sp.eye(2, format="csc")
    H = sp.diags([1.0, 0.0], format="csc")
    operators = mm.utils.normalized_operators(K, C, np.zeros(2))
    source = np.ones((2, 1))

    _basis, summary = mm.utils.build_parametric_basis(
        operators,
        source,
        [H],
        np.array([[1.0, 10.0]]),
        tolerance=1.0e-6,
        probe_rounds=1,
        max_order=20,
        seed=1,
    )

    first_response = sp.linalg.spsolve(K + H, source[:, 0])
    v = first_response / np.linalg.norm(first_response)
    A_failed = (K + 10.0 * H).toarray()
    expected = v * ((v @ source[:, 0]) / (v @ A_failed @ v))

    assert len(initial_guesses) == 2
    assert np.allclose(initial_guesses[1], expected)
    assert summary["history"][0]["full_solves"] == 2
