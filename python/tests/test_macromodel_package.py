"""Unit tests for the model-agnostic ``metahotspot.macromodel`` package.

These tests exercise the generic algorithms without any named case: face-port
conductance, explicit-face exclusion, common-patch area fractions, symmetry of
the coupled system, and identity-basis coupling agreement with the monolithic
split reference (on a small synthetic grid built through the public
``metahotspot`` bindings).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import metahotspot
from metahotspot import macromodel as mm
from metahotspot.enums import Face, LengthUnit, Study


# ---------------------------------------------------------------------------
# tiny synthetic model: 2x2x3 solid block, no BCs, one volumetric source
# ---------------------------------------------------------------------------


def _build_operators(nx=2, ny=2, nz=3):
    """Compile a small uniform block; return (model, compiled, operators).

    The mesh is ``nx`` x ``ny`` cells of 1 mm in x/y and ``nz`` layers of
    1 mm in z (block height ``nz`` mm), one volumetric source over the whole
    block, no boundary conditions (adiabatic; callers pin a cell when they
    need a well-posed system).
    """
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
    model.add_rect(block, 0, "0", "0", f"{nx}", f"{ny}")  # ADD
    model.set_default_neumann("0")
    compiled = model.compile()
    ops = compiled.assemble()
    return model, compiled, ops


def _cell_split(cells, half_z):
    """Split compact cell indices by z-centre below/above ``half_z``."""
    ijk = cells.ijk
    zc = cells.cz[ijk[:, 2]]
    lower = np.flatnonzero(zc < half_z)
    upper = np.flatnonzero(zc >= half_z)
    return lower, upper


class _FakePortsModel:
    """Minimal model-shaped adapter over a compiled model for port enumeration."""

    def __init__(self, compiled, ops):
        self._full = compiled
        self._ops = ops
        self.cell_layout = mm.CellLayout(
            centers=compiled.cells.centers,
            half_sizes=compiled.cells.half_sizes,
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

    def core_operators(self):
        return self._ops

    def boundary_groups(self):
        return ()

    def boundary_terms(self):
        return []

    def h_ranges(self):
        return np.empty((0, 2))

    def physical_to_effective(self, h):
        return np.asarray(h, dtype=np.float64)

    def source_shape(self):
        return self._ops.f.reshape(-1, 1)

    def build_geometry(self, study, *, detail, macro):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# face-port conductance
# ---------------------------------------------------------------------------


def test_face_port_conductance_uses_face_area_and_half_distance():
    """g = k·A/half per face; a 1x1x1 cell face reads k·A/half exactly."""
    _model, _compiled, ops = _build_operators(1, 1, 1)
    model = _FakePortsModel(_compiled, ops)
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
    """Only explicitly declared BC faces are excluded; others stay ports."""
    _model, compiled, ops = _build_operators(2, 2, 3)
    model = _FakePortsModel(compiled, ops)
    ports = mm.enumerate_interface_ports(model, np.arange(compiled.cell_count))
    labels = {p.normal for p in ports}
    # no BC declared anywhere -> every exterior face of the block is a port
    assert labels == {"x-", "x+", "y-", "y+", "z-", "z+"}
    for p in ports:
        assert np.all(p.g > 0.0)

    # declare the top face (z+) as an ambient group -> it must disappear
    top = compiled.cells.ijk[:, 2] == compiled.nz - 1
    top_cells = np.flatnonzero(top)
    top_areas = np.full(top_cells.size, 1.0e-6)
    model.boundary_groups = lambda: (
        mm.BoundaryGroup(cells=top_cells, areas=top_areas),
    )
    ports2 = mm.enumerate_interface_ports(model, np.arange(compiled.cell_count))
    labels2 = {p.normal for p in ports2}
    assert "z+" not in labels2
    assert "z-" in labels2


# ---------------------------------------------------------------------------
# subdomain + common-patch area weighting (non-conforming)
# ---------------------------------------------------------------------------


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
    """A 1x1 coarse face vs a 2x2 fine face -> 4 common patches of equal area."""
    coarse = _rect_port(1, 0.0, 1.0e-3)  # one 1mm x 1mm face
    fine = _rect_port(2, 0.0, 1.0e-3)  # four 0.5mm x 0.5mm faces
    areas, E_l, E_r, xi_l, xi_r, li, ri = mm.common_patches(coarse, fine)

    assert areas.size == 4
    assert np.allclose(areas, 0.25e-6)
    assert np.allclose(xi_l, 0.25)  # coarse owner face split into quarters
    assert np.allclose(xi_r, 1.0)  # fine faces each fully covered
    # each coarse patch is owned by the single coarse face (index 0)
    assert set(li.tolist()) == {0}
    # fine owner indices cover all four faces
    assert set(ri.tolist()) == {0, 1, 2, 3}
    assert E_l.shape == (4, 1) and E_r.shape == (4, 4)
    assert np.allclose(np.asarray(E_l.sum(axis=1)).ravel(), 1.0)
    assert np.allclose(np.asarray(E_r.sum(axis=1)).ravel(), 1.0)


# ---------------------------------------------------------------------------
# identity coupling reproduces the monolithic solve
# ---------------------------------------------------------------------------


def test_identity_coupling_reproduces_monolithic_steady():
    """Split the block at z=1.5mm, couple the two halves as full-FVM sides,
    and check the coupled steady state equals the monolithic solve."""
    _model, compiled, ops = _build_operators(2, 2, 3)
    model = _FakePortsModel(compiled, ops)
    lower, upper = _cell_split(compiled.cells, 1.5e-3)

    K = ops.K.tocsc()
    # the bare model is adiabatic (singular kernel): pin one cell so both the
    # monolithic reference and the coupled solve have a well-posed system
    pin = 0
    K = K + sp.diags(np.eye(compiled.cell_count)[pin] * 1.0e3)
    source = ops.f.ravel()
    steady_mono = sp.linalg.spsolve(K, source)

    left = mm.build_subdomain(model, lower, name="lower")
    right = mm.build_subdomain(model, upper, name="upper")
    # lower = z layers {0,1}: exposed on all 6 faces (z- at 0, z+ at 2mm cut)
    assert {p.normal for p in left.ports} == {"x-", "x+", "y-", "y+", "z-", "z+"}
    # upper = z layer {2}: top z+ exterior, z- at the cut
    assert {p.normal for p in right.ports} == {"x-", "x+", "y-", "y+", "z-", "z+"}

    Kc, Cc, rhsc, lo, ro, npatch = mm.connect(
        left, right, left.port("z+"), right.port("z-"), power=np.array([1.0])
    )
    # pin the same cell through the interface node column of the left block
    Kc = sp.csc_matrix(Kc)
    Kc[pin, pin] += 1.0e3
    steady_c = mm.solve_system(Kc, Cc, rhsc, dt=1.0, duration=0.0)[0]

    # map coupled state back to full FVM order and compare
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
    assert eig.min() >= -1.0e-9  # positive semi-definite (singular kernel)
    assert Cc.shape == Kc.shape


def test_embeddable_rom_reduces_cells_and_exposes_interface_trace():
    """Embeddable ROM reduces all cells and couples through its physical trace."""
    _model, compiled, ops = _build_operators(4, 4, 6)
    model = _FakePortsModel(compiled, ops)
    # declare the top crown (z+) as an ambient boundary group so the BCI
    # projection has boundary content (real models always declare ambient BCs)
    top = np.flatnonzero(compiled.cells.ijk[:, 2] == compiled.nz - 1)
    top_areas = np.full(top.size, 1.0e-6)
    model.boundary_groups = lambda: (
        mm.BoundaryGroup(cells=top, areas=top_areas, h_range=(1.0, 1.0e4)),
    )
    model.boundary_terms = lambda: [
        sp.diags(np.where(np.isin(np.arange(compiled.cell_count), top), 1.0e-6, 0.0))
    ]
    model.h_ranges = lambda: np.array([[1.0, 1.0e4]])
    model.physical_to_effective = lambda h: np.asarray(h, dtype=np.float64)
    # cut at z = 3 mm: upper = z layers {3,4,5} (48 cells); the interface
    # band covers the z- cut face, the z+ crown and all lateral faces,
    # leaving a genuine interior to reduce
    lower, upper = _cell_split(compiled.cells, 3.0e-3)
    left = mm.build_subdomain(model, lower, name="lower")
    right = mm.build_subdomain(model, upper, name="upper")

    # reduce the source-bearing upper side
    rom = mm.extract_rom(right, tolerance=1.0e-2, max_order=64, probe_rounds=1, seed=7)
    assert rom.basis.shape[0] == right.cells.size
    assert rom.m >= 1
    assert np.max(np.abs(rom.C_hat.toarray() - np.eye(rom.m))) < 1.0e-10
    assert (
        np.max(np.abs(rom.K0_hat.toarray() - np.diag(rom.K0_hat.diagonal()))) < 1.0e-10
    )
    expected_ambient = rom.basis.T @ right.ambient_terms[0] @ rom.basis
    assert np.max(np.abs(rom.ambient_hat[0].toarray() - expected_ambient)) < 1.0e-10
    assert rom.summary["bilinear_closure_iterations"] == 2
    assert rom.summary["bilinear_closure_nme"] == 2
    assert rom.summary["bilinear_closure_columns"] > 0
    assert len(rom.summary["bilinear_closure_round_columns"]) == 2
    # every subdomain boundary face is exported as a physical trace V_b = A_b.T@V
    assert set(rom.boundary_traces) == {p.label for p in right.boundary_ports}
    assert set(rom.boundary_conductances) == set(rom.boundary_traces)
    for port in right.boundary_ports:
        trace = rom.boundary_traces[port.label]
        assert trace.shape == (port.cells.size, rom.m)
        assert np.allclose(rom.boundary_conductances[port.label], port.g)

    Kc, Cc, rhsc, lo, ro, npatch = mm.connect(
        rom, left, rom.port("z-"), left.port("z+"), power=np.array([1.0])
    )
    # identity coupling of the same two subdomains (exact reference)
    Ki, Ci, rhsi, lo_i, ro_i, np_i = mm.connect(
        right, left, right.port("z-"), left.port("z+"), power=np.array([1.0])
    )
    assert npatch == np_i
    assert lo == rom.m
    assert Kc.shape == (lo + npatch + ro, lo + npatch + ro)
