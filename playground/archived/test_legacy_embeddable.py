"""Archived tests for the former Python subdomain and FacePort API.

The tests cover face-port conductance, ambient-face selection and adiabatic
subdomain cuts. Project contact is covered by the native C++ tests.
"""

from __future__ import annotations

from legacy_embeddable import FacePort, Subdomain, extract_rom
from legacy_interface_enrichment import (
    clone_embeddable_rom,
    extract_interface_enriched_rom,
)
import scipy.sparse as sp
from numpy.testing import assert_allclose

import numpy as np
import pytest

import metahotspot
import legacy_embeddable as mm
from metahotspot.macromodel.affine import BoundaryGroup, CellLayout
from metahotspot.enums import Face
from metahotspot.macromodel.geometry import CellGeometry


def _build_operators(nx=2, ny=2, nz=3):
    model = metahotspot.Fom()
    model.set_cartesian_mesh(
        np.linspace(0.0, float(nx) * 1e-3, nx + 1),
        np.linspace(0.0, float(ny) * 1e-3, ny + 1),
        np.linspace(0.0, float(nz) * 1e-3, nz + 1),
    )
    material = model.add_material("solid", "1", "1", "1", "1000", "500")
    source = model.add_source("1e6")
    volume = model.add_volume(material, source)
    model.add_box(volume, [0, 0, 0], [nx * 1e-3, ny * 1e-3, nz * 1e-3])
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
        self.geometry = CellGeometry(compiled.fields)
        self._ops = ops
        self.cell_layout = CellLayout(
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


def test_face_port_conductance_uses_face_area_and_half_distance():
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

    model.boundary_groups = lambda: (BoundaryGroup(model.geometry.surface(Face.ZP)),)
    ports = mm.enumerate_interface_ports(model, np.arange(compiled.cell_count))
    labels = {p.normal for p in ports}
    assert "z+" not in labels
    assert "z-" in labels
    assert {"x-", "x+", "y-", "y+"} <= labels


def test_subdomain_cuts_remove_only_exterior_conductance():
    _model, compiled, ops = _build_operators(2, 2, 3)
    model = _FakePortsModel(compiled, ops)
    lower, upper = _cell_split(compiled.fields, 1.5e-3)

    for name, cells in (("lower", lower), ("upper", upper)):
        part = mm.build_subdomain(model, cells, name=name)
        assert {p.normal for p in part.ports} == {"x-", "x+", "y-", "y+", "z-", "z+"}
        expected = ops.K.tocsr()[cells][:, cells].toarray()
        expected -= np.diag(expected.sum(axis=1))
        np.testing.assert_allclose(part.K.toarray(), expected, atol=1e-14)
        np.testing.assert_allclose(part.C.diagonal(), ops.C.diagonal()[cells])
        np.testing.assert_allclose(part.source[:, 0], ops.f[cells])

@pytest.mark.parametrize("extract", [extract_rom, extract_interface_enriched_rom])
def test_extracted_modal_rom_preserves_projected_operators(extract):
    n = 24
    K = sp.diags(
        [-np.ones(n - 1), np.r_[1, np.full(n - 2, 2), 1], -np.ones(n - 1)],
        [-1, 0, 1],
        format="csc",
    )
    C = sp.diags(np.linspace(1, 2, n), format="csc")
    H = sp.diags(np.r_[1.0, np.zeros(n - 1)], format="csc")
    source = np.zeros((n, 1))
    source[n // 2] = 1.0
    port = FacePort(
        label="z+",
        axis=2,
        direction=1,
        cells=np.array([n - 1]),
        areas=np.ones(1),
        k=np.ones(1),
        half=np.ones(1),
        t1=0,
        t2=1,
        rects=np.array([[0.0, 1.0, 0.0, 1.0]]),
    )
    domain = Subdomain(
        name="chain",
        cells=np.arange(n),
        K=K,
        C=C,
        source=source,
        ports=[port],
        boundary_ports=[port],
        ambient_terms=[H],
        ambient_ranges=np.array([[1.0, 10.0]]),
        effective_p=np.array([2.0]),
    )
    rom = extract(domain, tolerance=1e-2, max_order=64, probe_rounds=2, seed=7)
    V = rom.basis
    assert_allclose(rom.C_hat.toarray(), V.T @ C @ V, atol=1e-12)
    assert_allclose(rom.K0_hat.toarray(), V.T @ K @ V, atol=1e-12)
    assert_allclose(rom.F_hat, V.T @ source, atol=1e-12)
    assert_allclose(rom.ambient_hat[0].toarray(), V.T @ H @ V, atol=1e-12)
    assert_allclose(rom.boundary_trace("z+"), V[port.cells])
    assert_allclose(
        rom.internal_operator().toarray(),
        V.T @ (K + 2 * H) @ V,
        atol=1e-12,
    )

    clone = clone_embeddable_rom(rom)
    clone.F_hat[:] = 0
    clone.ports[0].areas[:] = 3
    clone.summary["per_port_plans"][0]["shifts_per_s"].append(-1)
    assert_allclose(rom.F_hat, V.T @ source, atol=1e-12)
    assert_allclose(rom.ports[0].areas, 1)
    assert rom.summary["per_port_plans"][0]["shifts_per_s"][-1] > 0
