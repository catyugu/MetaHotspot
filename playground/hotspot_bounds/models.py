"""Controlled native FVM test devices and held-out power histories.

Material values are synthetic research inputs, not a calibrated real package.
No connecting-ROM code is imported. Native MetaHotspot assembly is mandatory
except for the explicitly labelled local smoke test.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass
class Device:
    K: sp.csr_matrix
    C: np.ndarray
    B: np.ndarray
    metadata: dict


class Forcings:
    """Generate N-vector loads on demand rather than caching a full trajectory."""
    def __init__(self, B, powers):
        self.B = B
        self.powers = powers

    def __len__(self):
        return len(self.powers)

    def __getitem__(self, index):
        return self.B @ self.powers[index]


def native_device(nx: int, nz_per_layer: int, bottleneck: bool) -> Device:
    import metahotspot
    from metahotspot.enums import Axis, GeometryOp, LengthUnit, Study

    model = metahotspot.Model()
    model.set_settings(study=Study.TRANSIENT, length_unit=LengthUnit.METER,
                       initial_temperature_K=300., duration=3., output_interval=.025)
    xy = np.linspace(0., .020, nx + 1)
    # Bottom to top: copper, substrate, TIM, mould containing the two dies.
    edges = [0., .0005, .0011, .0012, .0016]
    z = np.concatenate([np.linspace(a, b, nz_per_layer + 1)[:-1]
                        for a, b in zip(edges[:-1], edges[1:])] + [np.array([edges[-1]])])
    model.set_mesh(xy, xy, z)
    materials = {
        'copper': (385., 385., 385., 8930., 385.),
        'substrate': (18., 10., 1.5, 1900., 1000.),
        'tim': ((.2 if bottleneck else 2.),) * 3 + (2200., 800.),
        'mould': (.6, .6, .6, 1800., 1000.),
        'silicon': (130., 95., 80., 2330., 700.),
    }
    for name, values in materials.items():
        model.add_material(name, *map(str, values))
    for material, thickness in [('mould', .0004), ('tim', .0001),
                                ('substrate', .0006), ('copper', .0005)]:
        layer = model.add_layer(str(thickness))
        block = model.add_block(layer, material, heat_source='0')
        model.add_rect(block, GeometryOp.ADD, '0', '0', '.020', '.020')
        if material == 'mould':
            for x0, y0, x1, y1 in [(.002, .003, .008, .009), (.012, .011, .018, .017)]:
                die = model.add_block(layer, 'silicon', heat_source='0')
                model.add_rect(die, GeometryOp.ADD, *map(str, (x0, y0, x1 - x0, y1 - y0)))
    model.set_default_neumann('0')
    model.add_convection('2000', '300', [(Axis.Z, .0016, 0., .020, 0., .020)])
    model.add_convection('1000', '300', [(Axis.Z, 0., 0., .020, 0., .020)])
    compiled = model.compile()
    ops = compiled.assemble()
    cells = compiled.cells
    n = compiled.cell_count
    if n != nx * nx * 4 * nz_per_layer:
        raise RuntimeError(f'unexpected active cell count {n}')
    grid = np.asarray(cells.cell_to_grid, dtype=np.int64)
    iz = grid % (4 * nz_per_layer)
    iy = (grid // (4 * nz_per_layer)) % nx
    ix = grid // (nx * 4 * nz_per_layer)
    x, y, zz = cells.cx[ix], cells.cy[iy], cells.cz[iz]
    volume = cells.dx[ix] * cells.dy[iy] * cells.dz[iz]
    K = ops.K.tocsr()
    C = np.asarray(ops.C.diagonal())
    if (ops.C - sp.diags(C)).nnz:
        raise RuntimeError('lumped capacity is required')
    ambient_defect = float(np.max(np.abs(ops.f - K @ np.full(n, 300.))))
    if ambient_defect > 1e-7:
        raise RuntimeError(f'ambient equilibrium failed: {ambient_defect}')
    masks = [
        (x > .002) & (x < .008) & (y > .003) & (y < .009) & (zz > .0012),
        (x > .012) & (x < .018) & (y > .011) & (y < .017) & (zz > .0012),
        (x > .002) & (x < .0045) & (y > .003) & (y < .0055) & (zz > .0014),
    ]
    native_silicon = np.isclose(compiled.eval_materials().conductivity_x, 130.)
    if not np.array_equal(native_silicon, masks[0] | masks[1]):
        raise RuntimeError('native material mapping disagrees with intended die rectangles')
    B = np.column_stack([np.where(mask, volume, 0.) for mask in masks])
    if np.any(B.sum(axis=0) == 0):
        raise RuntimeError('mesh did not resolve a source')
    B /= B.sum(axis=0)
    compiled.close()
    model.close()
    return Device(K, C, B, {
        'assembly': 'native MetaHotspot C++ FVM', 'nx': nx, 'ny': nx,
        'nz': 4 * nz_per_layer, 'dof': n, 'bottleneck': bottleneck,
        'ambient_equilibrium_defect_W': ambient_defect,
        'source_cell_counts': [int(mask.sum()) for mask in masks],
        'training_sources': [0, 1], 'held_out_source': 2,
        'ambient_K': 300., 'material_values': materials,
    })


def smoke_device() -> Device:
    """Small algebraic graph only, not reported as a native device experiment."""
    n = 80
    edges = np.linspace(.1, .8, n - 1)
    K = sp.diags([-edges, np.r_[edges, 0] + np.r_[0, edges] + .02, -edges],
                 [-1, 0, 1], format='csr')
    C = np.linspace(.001, .005, n)
    B = np.zeros((n, 3))
    B[5:20, 0] = 1 / 15
    B[50:65, 1] = 1 / 15
    B[7:9, 2] = .5
    return Device(K, C, B, {'assembly': 'SMOKE ONLY: synthetic chain', 'dof': n})


def powers(steps: int, seed: int, held_out: bool) -> np.ndarray:
    """Predetermined switching workload; the localized source is never trained."""
    rng = np.random.default_rng(seed)
    P = np.zeros((steps, 3))
    split = steps // 4
    P[:split, 0] = 14.
    P[split:2*split, 1] = 24.
    P[2*split:3*split, :2] = [10., 16.]
    P[3*split:, :2] = [5., 9.]
    if held_out:
        P[2*split:3*split, 2] = 6.
        P[3*split:, 2] = 10.
    for j in range(4):
        P[j*split:(j+1)*split] *= rng.uniform(.9, 1.1)
    return P
