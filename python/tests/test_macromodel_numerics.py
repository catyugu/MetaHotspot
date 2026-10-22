"""Numerical contracts independent of the native model builder."""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from metahotspot._compiled_data import Operators
from metahotspot.macromodel.affine import AffineParametricModel, SourcePort
from metahotspot.macromodel.embeddable import FacePort, Subdomain, extract_rom
from metahotspot.macromodel.interface_enrichment import (
    clone_embeddable_rom,
    extract_interface_enriched_rom,
)
from metahotspot.macromodel.utils import mpmm_elliptic_shift_count


@pytest.mark.parametrize("epsilon, hi", [(1e-3, 10.0), (1e-100, 1e12)])
def test_shift_count_satisfies_bound_without_an_iteration_cap(epsilon, hi):
    lo = 1.0
    count = mpmm_elliptic_shift_count(epsilon, lo, hi)
    log_term = np.log(4 * hi / lo)
    assert 4 * np.exp(-count * np.pi**2 / log_term) <= epsilon
    assert 4 * np.exp(-(count - 1) * np.pi**2 / log_term) > epsilon


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


def test_boundary_projection_accumulates_patches_in_cell_order():
    from metahotspot.macromodel.affine import BoundaryGroup
    from metahotspot.macromodel.geometry import BoundarySurface

    class Model(AffineParametricModel):
        core = Operators(sp.eye(2), sp.eye(2), np.zeros(2))

        def boundary_groups(self):
            return (
                BoundaryGroup(
                    BoundarySurface(
                        np.array([0, 0, 1]), np.array([0.1, 0.2, 0.4]), np.arange(3)
                    )
                ),
            )

    assert_allclose(Model().boundary_terms[0].diagonal(), [0.3, 0.4])


def test_source_projection_preserves_signed_powers():
    class Model(AffineParametricModel):
        ambient_K = 300.0
        core = Operators(sp.eye(2), sp.eye(2), np.array([-2.0, -6.0]))

        def source_ports(self):
            return [
                SourcePort(cells=np.array([0]), power_W=-2.0),
                SourcePort(cells=np.array([1]), power_W=-6.0),
            ]

    model = Model()
    assert_allclose(model.source_shape, [[1, 0], [0, 1]])
    assert_allclose(model.source_shape @ model.nominal_power(), model.core.f)
    assert_allclose(
        Model().junction_temperature(np.array([310.0, 320.0])), [310.0, 320.0]
    )
