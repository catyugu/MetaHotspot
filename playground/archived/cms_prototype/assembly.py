"""Edge-fitted, active-cell thermal mesh; empty space has no cells or DOFs.

The extractable component ends at the silicon top. TIM and lid are assembled
separately in external.py. SI units; all material data are benchmark assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

MM = 1.0e-3
# name, thickness [m], k_xy, k_z [W/(m K)], rho*c [J/(m^3 K)]
LAYER_SPECS = (
    ("organic_substrate", 0.60 * MM, 12.0, 0.6, 1.70e6),
    ("bump_underfill", 0.05 * MM, 5.0, 3.0, 1.80e6),
    ("silicon_die", 0.30 * MM, 100.0, 100.0, 1.63e6),
)
DIE_LAYER = 2
JUNCTION_Z_MM = 0.65
SOURCE_THICKNESS = 0.0375 * MM  # fixed physical active zone, independent of mesh
COMPONENT_TOP = sum(s[1] for s in LAYER_SPECS)
CHIP_RECTS_MM = ((3.0, 3.0, 10.5, 15.0), (13.5, 3.0, 21.0, 15.0))
SOURCE_NAMES = ("logic_A", "logic_B")
MATERIAL_SCALING = {
    "isotropic": {},
    "weak_lateral": {"silicon_die": 0.02, "bump_underfill": 0.02},
    "extreme": {spec[0]: 0.02 for spec in LAYER_SPECS},
}
LATERAL_SIZE = 24.0 * MM
SPLIT_X = LATERAL_SIZE / 2
FIXED_HTC = (8000.0, 12000.0)  # lid-to-coolant, not bare-air convection
BOTTOM_HTC = 250.0  # substrate underside to ambient; fixed during extraction
AMBIENT_C = 25.0
HEX_OFFSETS = np.array(
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
)


@dataclass
class Model:
    max_cell_size: tuple[float, float, float]
    material: str
    capacity: np.ndarray
    K: sp.csc_matrix
    B_P: np.ndarray
    D_gamma: np.ndarray
    points: np.ndarray  # active vertices only, interface first
    cells: np.ndarray  # active VTK hexahedra only
    cell_layer: np.ndarray
    cell_volume: np.ndarray
    cell_rhoc: np.ndarray
    cell_conductivity: np.ndarray
    chip_fraction: np.ndarray
    source_density: np.ndarray
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    node_grid: np.ndarray  # [z,y,x] -> compact DOF, -1 in empty space
    interface_faces: np.ndarray
    robin: np.ndarray
    layer_specs: tuple
    layer_cells: tuple[int, ...]

    @property
    def N(self) -> int:
        return self.capacity.size

    @property
    def m(self) -> int:
        return self.D_gamma.size

    @property
    def n(self) -> int:
        return self.N - self.m


DEFAULT_MAX_CELL_SIZE = (0.75 * MM, 0.75 * MM, 0.1 * MM)


def cell_count(length, maximum):
    """Round up, suppressing only floating-point noise at an integer ratio."""
    ratio = length / maximum
    return max(1, int(np.ceil(ratio - 1e-12 * max(1.0, ratio))))


def lateral_axes(max_cell_size):
    """Insert exact chip edges without exceeding either lateral size limit."""
    axes = []
    for d in (0, 1):
        base = np.linspace(
            0, LATERAL_SIZE, cell_count(LATERAL_SIZE, max_cell_size[d]) + 1
        )
        edges = np.array(CHIP_RECTS_MM)[:, [d, d + 2]].ravel() * MM
        axes.append(np.unique(np.round(np.r_[base, edges], 14)))
    return tuple(axes)


def build_mesh(
    max_cell_size,
    material,
    specs,
    chip_layers,
    interface_side,
    z0=0.0,
    source_layer=None,
    bottom_htc=0.0,
    top_htc=(0.0, 0.0),
):
    """Assemble only occupied cells and compact their vertices before algebra.

    Layers in chip_layers exist only under the two chip rectangles. Other
    layers cover the footprint. Shared material faces share vertices; exposed
    cavity faces are adiabatic. No air/mold surrogate or zero-capacity nodes.
    """
    if len(max_cell_size) != 3 or any(
        not np.isfinite(v) or v <= 0 for v in max_cell_size
    ):
        raise ValueError("maximum x/y/z cell sizes must be finite and positive")
    x, y = lateral_axes(max_cell_size)
    layer_steps = []
    for i, spec in enumerate(specs):
        boundaries = [0.0, spec[1]]
        if i == source_layer:
            boundaries.insert(1, SOURCE_THICKNESS)
        layer_steps.append(
            np.concatenate(
                [
                    np.full(
                        cell_count(right - left, max_cell_size[2]),
                        (right - left) / cell_count(right - left, max_cell_size[2]),
                    )
                    for left, right in zip(boundaries[:-1], boundaries[1:])
                ]
            )
        )
    counts = tuple(len(steps) for steps in layer_steps)
    dz = np.concatenate(layer_steps)
    z = z0 + np.r_[0.0, np.cumsum(dz)]
    zlayer = np.repeat(np.arange(len(specs)), counts)
    nz, ny, nx_actual = len(dz), len(y) - 1, len(x) - 1
    ck, cj, ci = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx_actual), indexing="ij"
    )
    ci, cj, ck = ci.ravel(), cj.ravel(), ck.ravel()
    cx, cy = (x[ci] + x[ci + 1]) / 2 / MM, (y[cj] + y[cj + 1]) / 2 / MM
    chip = np.column_stack(
        [(cx > a) & (cx < c) & (cy > b) & (cy < d) for a, b, c, d in CHIP_RECTS_MM]
    )
    active = ~np.isin(zlayer[ck], list(chip_layers)) | chip.any(axis=1)
    ci, cj, ck, chip = ci[active], cj[active], ck[active], chip[active]
    layer = zlayer[ck]
    dx, dy = np.diff(x)[ci], np.diff(y)[cj]
    volume = dx * dy * dz[ck]

    def full_id(i, j, k):
        return i + len(x) * (j + len(y) * k)

    full_cells = np.column_stack(
        [full_id(ci + di, cj + dj, ck + dk) for di, dj, dk in HEX_OFFSETS]
    )
    top, bottom = ck == nz - 1, ck == 0
    side, corners = (
        (top, [4, 5, 6, 7]) if interface_side == "top" else (bottom, [0, 1, 2, 3])
    )
    faces = full_cells[side][:, corners]
    ports = np.unique(faces)
    used = np.unique(full_cells)
    ordered = np.r_[ports, np.setdiff1d(used, ports)]
    lookup = np.full(len(x) * len(y) * len(z), -1, dtype=int)
    lookup[ordered] = np.arange(len(ordered))
    cells, faces = lookup[full_cells], lookup[faces]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[ordered]
    n, m = len(ordered), len(ports)
    kxy = np.array([specs[i][2] for i in layer])
    kz = np.array([specs[i][3] for i in layer])
    rhoc = np.array([specs[i][4] for i in layer])
    kxy *= [MATERIAL_SCALING.get(material, {}).get(specs[i][0], 1.0) for i in layer]
    density = np.zeros((len(cells), len(SOURCE_NAMES)))
    if source_layer is not None:
        source_bottom = z0 + sum(s[1] for s in specs[:source_layer])
        depth = np.maximum(
            0.0,
            np.minimum(z[ck + 1], source_bottom + SOURCE_THICKNESS)
            - np.maximum(z[ck], source_bottom),
        )
        density = chip * (depth / dz[ck])[:, None]
        density = density / (density * volume[:, None]).sum(axis=0)
    cap, bp = np.zeros(n), np.zeros((n, len(SOURCE_NAMES)))
    for v in cells.T:
        np.add.at(cap, v, rhoc * volume / 8)
        for q in range(len(SOURCE_NAMES)):
            np.add.at(bp[:, q], v, density[:, q] * volume / 8)

    rows, cols, vals = [], [], []
    for direction, delta, cond in ((0, dx, kxy), (1, dy, kxy), (2, dz[ck], kz)):
        for a, offset in enumerate(HEX_OFFSETS):
            if offset[direction] != 0:
                continue
            other = offset.copy()
            other[direction] = 1
            b = np.flatnonzero(np.all(other == HEX_OFFSETS, axis=1))[0]
            va, vb = cells[:, a], cells[:, b]
            g = cond * volume / (4 * delta**2)
            rows.append(np.r_[va, va, vb, vb])
            cols.append(np.r_[va, vb, va, vb])
            vals.append(np.r_[g, -g, -g, g])
    k = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsc()
    area, robin = np.zeros(m), np.zeros(n)
    for v in faces.T:
        np.add.at(area, v, dx[side] * dy[side] / 4)
    left_width = np.clip(SPLIT_X - x[ci[top]], 0, dx[top])
    loss = (top_htc[0] * left_width + top_htc[1] * (dx[top] - left_width)) * dy[top]
    for v in cells[top][:, [4, 5, 6, 7]].T:
        np.add.at(robin, v, loss / 4)
    for v in cells[bottom][:, :4].T:
        np.add.at(robin, v, bottom_htc * dx[bottom] * dy[bottom] / 4)
    return Model(
        tuple(max_cell_size),
        material,
        cap,
        k + sp.diags(robin, format="csc"),
        bp,
        area,
        points,
        cells,
        layer,
        volume,
        rhoc,
        np.column_stack([kxy, kxy, kz]),
        chip.any(axis=1).astype(float),
        density,
        (x, y, z),
        lookup.reshape(len(z), len(y), len(x)),
        faces,
        robin,
        specs,
        counts,
    )


def assemble(max_cell_size=DEFAULT_MAX_CELL_SIZE, material: str = "isotropic") -> Model:
    """Substrate + die-local bump/underfill + silicon; two complete top ports."""
    if material not in MATERIAL_SCALING:
        raise ValueError(f"unknown material case: {material}")
    return build_mesh(
        tuple(max_cell_size),
        material,
        LAYER_SPECS,
        {1, 2},
        "top",
        source_layer=DIE_LAYER,
        bottom_htc=BOTTOM_HTC,
    )


def sampling_matrix(model, coordinates):
    """Trilinear samples in active cells; reject probes in cavities/outside."""
    rows, cols, vals = [], [], []
    for q, xyz in enumerate(coordinates):
        indices, fractions = [], []
        for axis, value in zip(model.axes, xyz):
            if value < axis[0] - 1e-14 or value > axis[-1] + 1e-14:
                raise ValueError(f"probe outside mesh: {xyz}")
            i = int(
                np.clip(
                    np.searchsorted(axis, value, side="right") - 1, 0, len(axis) - 2
                )
            )
            indices.append(i)
            fractions.append(np.clip((value - axis[i]) / (axis[i + 1] - axis[i]), 0, 1))
        i, j, k = indices
        # An active cell must exist here; eight individually present vertices
        # alone do not prove occupancy (e.g. a narrow cavity between solids).
        layer = np.searchsorted(np.cumsum(model.layer_cells), k, side="right")
        cell_nodes = model.node_grid[
            k + HEX_OFFSETS[:, 2], j + HEX_OFFSETS[:, 1], i + HEX_OFFSETS[:, 0]
        ]
        if np.any(cell_nodes < 0) or not np.any(
            np.all(model.cells == cell_nodes, axis=1)
        ):
            raise ValueError(f"probe in empty cell (layer {layer}): {xyz}")
        for node, offset in zip(cell_nodes, HEX_OFFSETS):
            weight = np.prod([f if o else 1 - f for f, o in zip(fractions, offset)])
            rows.append(node)
            cols.append(q)
            vals.append(weight)
    return sp.csc_matrix((vals, (rows, cols)), shape=(model.N, len(coordinates)))
