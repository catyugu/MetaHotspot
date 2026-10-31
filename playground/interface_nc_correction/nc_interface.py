#!/usr/bin/env python3
"""Non-conforming interface: eliminated patch conductance vs non-orthogonal correction.

Sandbox for the interface discretisation the repository's embeddable ROM uses
(``metahotspot.macromodel.embeddable.connect``, i.e. the weekly-report condensation
``H_Gamma = (H_L^-1 + H_R^-1)^-1``), evaluated against the standard finite-volume
treatment of a skewed cell-to-cell connection.

Repository scheme, per common patch ``p`` (area ``A_p``, each side's half distance
``half_S`` and normal conductivity ``k_S``):

    q_p = g_p (T_L - T_R),
    g_p = 1 / ( half_L/(k_L A_p) + half_R/(k_R A_p) ) = A_p / (half_L/k_L + half_R/k_R)

``g_p`` is the over-relaxed two-point coefficient ``|S_p|^2/(S_p.d_p)`` of the
orthogonal decomposition; the scheme carries **zero** non-orthogonal correction.  The
shared patch temperature only has to satisfy algebraic flux balance, so eliminating it
does not change ``q_p``.

The standard FVM counterpart (Jasak's skewness / non-orthogonality treatment) keeps
the same implicit coefficient and adds the deferred correction term

    q_p = g_p [ (T_R + delta_R . grad_t T_R) - (T_L + delta_L . grad_t T_L) ]

with ``delta_S`` the in-plane offset of the patch centroid from the normal projection
of side ``S``'s cell centre and ``grad_t`` the in-plane cell gradient.  ``delta_S`` is
exactly zero on a conforming interface, so there the two schemes are identical; the
correction appears only when the two interface grids do not match.  The term is
nonlinear in the unknowns, so its standard solution procedure is a deferred-correction
(Picard / "non-orthogonal corrector") iteration whose iteration-0 state **is** the
repository scheme.

Units are SI everywhere (m, W, K); temperatures are rises above a 0 K ambient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from metahotspot.macromodel.embeddable import FacePort, common_patches, connect


# ---------------------------------------------------------------------------
# structured blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layer:
    """One z-layer: thickness, isotropic conductivity, volumetric heat capacity."""

    dz: float
    k: float
    rho_cp: float


@dataclass(frozen=True)
class Box:
    """A rectangular heat source with total power ``power`` (W)."""

    name: str
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    power: float


@dataclass(frozen=True)
class Grid:
    """A structured block, separably graded in x/y and layered in z."""

    name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    k: np.ndarray  # (nx*ny*nz,) per cell, cell (i,j,k) -> (i*ny + j)*nz + k
    rho_cp: np.ndarray

    @property
    def n(self) -> tuple[int, int, int]:
        return (self.x.size - 1, self.y.size - 1, self.z.size - 1)

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            0.5 * (self.x[:-1] + self.x[1:]),
            0.5 * (self.y[:-1] + self.y[1:]),
            0.5 * (self.z[:-1] + self.z[1:]),
        )

    def spacings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.diff(self.x), np.diff(self.y), np.diff(self.z)


def make_grid(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    layers: Sequence[Layer],
    *,
    z0: float = 0.0,
) -> Grid:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = z0 + np.r_[0.0, np.cumsum([lay.dz for lay in layers])]
    nx, ny = x.size - 1, y.size - 1
    k = np.tile(np.asarray([lay.k for lay in layers], dtype=np.float64), nx * ny)
    rho_cp = np.tile(
        np.asarray([lay.rho_cp for lay in layers], dtype=np.float64), nx * ny
    )
    return Grid(name, x, y, z, np.ascontiguousarray(k), np.ascontiguousarray(rho_cp))


@dataclass
class System:
    """One assembled FVM block, compressed over its active cells."""

    name: str
    grid: Grid
    ids: np.ndarray  # (nx, ny, nz) -> compressed id, -1 when inactive
    cells_ijk: np.ndarray  # (n, 3)
    K: sp.csc_matrix
    C: sp.csc_matrix
    source: np.ndarray  # (n, n_src) unit-power shapes
    ambient_g: np.ndarray  # (n,) ambient conductance per cell, W/K
    source_names: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return int(self.cells_ijk.shape[0])

    def centers(self) -> np.ndarray:
        xc, yc, zc = self.grid.centers()
        return np.column_stack(
            (xc[self.cells_ijk[:, 0]], yc[self.cells_ijk[:, 1]], zc[self.cells_ijk[:, 2]])
        )

    def halves(self) -> np.ndarray:
        dx, dy, dz = self.grid.spacings()
        return 0.5 * np.column_stack(
            (dx[self.cells_ijk[:, 0]], dy[self.cells_ijk[:, 1]], dz[self.cells_ijk[:, 2]])
        )

    def conductivity(self) -> np.ndarray:
        return self.grid.k[np.ravel_multi_index(self.cells_ijk.T, self.grid.n)]

    def volumes(self) -> np.ndarray:
        dx, dy, dz = self.grid.spacings()
        return (
            dx[self.cells_ijk[:, 0]]
            * dy[self.cells_ijk[:, 1]]
            * dz[self.cells_ijk[:, 2]]
        )


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _face_table(
    grid: Grid, ids: np.ndarray, k_local: np.ndarray, axis: int
) -> dict[str, np.ndarray]:
    """Flatten every face plane along ``axis`` into arrays.

    Keys: ``lo``/``hi`` compressed cell ids (-1 when absent), ``area``, ``hl``/``hh``
    half distances of the lo/hi cell along ``axis``, ``kl``/``kh`` conductivities,
    and the face centre coordinates ``x``/``y``/``z``.
    """
    nx, ny, nz = grid.n
    n_axis = (nx, ny, nz)[axis]
    xc, yc, zc = grid.centers()
    spacing = grid.spacings()
    half = tuple(0.5 * s for s in spacing)
    levels = (grid.x, grid.y, grid.z)[axis]
    tr = [a for a in range(3) if a != axis]
    m0, m1 = np.meshgrid(np.arange(grid.n[tr[0]]), np.arange(grid.n[tr[1]]), indexing="ij")
    area = spacing[tr[0]][m0] * spacing[tr[1]][m1]
    coords = [None, None, None]
    coords[tr[0]] = (xc, yc, zc)[tr[0]][m0]
    coords[tr[1]] = (xc, yc, zc)[tr[1]][m1]
    idx = [None, None, None]
    idx[tr[0]], idx[tr[1]] = m0, m1

    table: dict[str, list] = {key: [] for key in ("lo", "hi", "area", "hl", "hh", "kl", "kh", "x", "y", "z")}
    for f in range(n_axis + 1):
        idx[axis] = f - 1
        lo = ids[tuple(idx)] if f > 0 else np.full(area.shape, -1, np.int64)
        idx[axis] = f
        hi = ids[tuple(idx)] if f < n_axis else np.full(area.shape, -1, np.int64)
        lo_safe = np.where(lo >= 0, lo, 0)
        hi_safe = np.where(hi >= 0, hi, 0)
        coords[axis] = np.full(area.shape, levels[f])
        table["lo"].append(lo)
        table["hi"].append(hi)
        table["area"].append(area)
        table["hl"].append(
            np.full(area.shape, half[axis][f - 1]) if f > 0 else np.zeros(area.shape)
        )
        table["hh"].append(
            np.full(area.shape, half[axis][f]) if f < n_axis else np.zeros(area.shape)
        )
        table["kl"].append(k_local[lo_safe])
        table["kh"].append(k_local[hi_safe])
        for name, values in zip(("x", "y", "z"), coords):
            table[name].append(values)
    return {key: np.concatenate(values) for key, values in table.items()}


def assemble(
    grid: Grid,
    *,
    active: np.ndarray | None = None,
    ambient: Callable[..., np.ndarray] | None = None,
    sources: Sequence[Box] = (),
    name: str | None = None,
) -> System:
    """Assemble the stiffness and capacity of one block.

    ``ambient(axis, direction, x, y, z)`` returns the physical HTC on exposed faces
    (vectorized over face centres; ``direction`` is +1 on a cell's high face).  Faces
    between two active cells are interior conductors, faces to an inactive or
    non-existent cell are ambient faces.  Every conductance is the two-point series
    form ``A/(half_a/k_a + half_b/k_b)``; an ambient face adds ``A/(half/k + 1/h)``,
    which is the repository's affine boundary term at its effective coefficient.
    """
    nx, ny, nz = grid.n
    ids = -np.ones((nx, ny, nz), dtype=np.int64)
    if active is None:
        active = np.ones((nx, ny, nz), dtype=bool)
    active = np.asarray(active, dtype=bool)
    ids[active] = np.arange(int(active.sum()))
    n = int(active.sum())
    cells_ijk = np.argwhere(ids >= 0)
    k_local = grid.k[np.ravel_multi_index(cells_ijk.T, (nx, ny, nz))]
    if ambient is None:
        ambient = lambda axis, direction, x, y, z: np.zeros_like(x)

    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    ambient_g = np.zeros(n)

    for axis in range(3):
        face = _face_table(grid, ids, k_local, axis)
        area, lo, hi = face["area"], face["lo"], face["hi"]

        both = (lo >= 0) & (hi >= 0)
        g = area[both] / (
            face["hl"][both] / face["kl"][both] + face["hh"][both] / face["kh"][both]
        )
        diag[lo[both]] += g
        diag[hi[both]] += g
        rows.append(lo[both])
        rows.append(hi[both])
        cols.append(hi[both])
        cols.append(lo[both])
        vals.append(-g)
        vals.append(-g)

        for cells, direction, has_cell, other_absent, half, kk in (
            (lo, +1, lo >= 0, hi < 0, face["hl"], face["kl"]),
            (hi, -1, hi >= 0, lo < 0, face["hh"], face["kh"]),
        ):
            select = has_cell & other_absent
            if not select.any():
                continue
            h = np.asarray(
                ambient(
                    axis,
                    direction,
                    face["x"][select],
                    face["y"][select],
                    face["z"][select],
                ),
                dtype=np.float64,
            )
            g = np.zeros(int(select.sum()))
            positive = h > 0.0
            g[positive] = area[select][positive] / (
                half[select][positive] / kk[select][positive] + 1.0 / h[positive]
            )
            diag[cells[select]] += g
            ambient_g[cells[select]] += g

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsc()

    linear = np.ravel_multi_index(cells_ijk.T, (nx, ny, nz))
    spacing = grid.spacings()
    volume = (
        spacing[0][cells_ijk[:, 0]]
        * spacing[1][cells_ijk[:, 1]]
        * spacing[2][cells_ijk[:, 2]]
    )
    C = sp.diags(grid.rho_cp[linear] * volume, format="csc")

    xc, yc, zc = grid.centers()
    centers = np.column_stack(
        (xc[cells_ijk[:, 0]], yc[cells_ijk[:, 1]], zc[cells_ijk[:, 2]])
    )
    source = np.zeros((n, len(sources)))
    for col, box in enumerate(sources):
        inside = (
            (centers[:, 0] >= box.x0)
            & (centers[:, 0] < box.x1)
            & (centers[:, 1] >= box.y0)
            & (centers[:, 1] < box.y1)
            & (centers[:, 2] >= box.z0)
            & (centers[:, 2] < box.z1)
        )
        if not inside.any():
            raise ValueError(f"source {box.name!r} does not intersect the block")
        source[inside, col] = volume[inside] / volume[inside].sum()

    return System(
        name=name or grid.name,
        grid=grid,
        ids=ids,
        cells_ijk=cells_ijk,
        K=K,
        C=C,
        source=source,
        ambient_g=ambient_g,
        source_names=tuple(box.name for box in sources),
    )


# ---------------------------------------------------------------------------
# interface ports and common patches
# ---------------------------------------------------------------------------

_TANGENT = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def window_port(
    system: System,
    window: tuple[float, float, float, float] | None = None,
    *,
    axis: int = 2,
    direction: int = 1,
    label: str | None = None,
) -> FacePort:
    """One block face as a connectable port, clipped to the contact ``window``.

    ``window`` is ``(t1lo, t1hi, t2lo, t2hi)`` in the port's tangent coordinates;
    ``None`` keeps the whole face.  ``areas`` stay the *full* face areas so the
    covered fraction ``xi`` reproduces the library patch conductance
    ``k*A_patch/half``.
    """
    nx, ny, nz = system.grid.n
    n_axis = (nx, ny, nz)[axis]
    level = n_axis - 1 if direction > 0 else 0
    t1, t2 = _TANGENT[axis]
    index = [slice(None)] * 3
    index[axis] = level
    plane = system.ids[tuple(index)]
    edges = (system.grid.x, system.grid.y, system.grid.z)
    spacing = system.grid.spacings()
    m0, m1 = np.meshgrid(
        np.arange(system.grid.n[t1]), np.arange(system.grid.n[t2]), indexing="ij"
    )
    rects = np.column_stack(
        (
            edges[t1][m0].ravel(),
            edges[t1][m0 + 1].ravel(),
            edges[t2][m1].ravel(),
            edges[t2][m1 + 1].ravel(),
        )
    )
    area = (spacing[t1][m0] * spacing[t2][m1]).ravel()
    cells = plane.ravel()

    keep = cells >= 0
    if window is not None:
        lo1, hi1, lo2, hi2 = window
        keep &= (rects[:, 1] > lo1) & (rects[:, 0] < hi1)
        keep &= (rects[:, 3] > lo2) & (rects[:, 2] < hi2)
        rects = np.column_stack(
            (
                np.maximum(rects[:, 0], lo1),
                np.minimum(rects[:, 1], hi1),
                np.maximum(rects[:, 2], lo2),
                np.minimum(rects[:, 3], hi2),
            )
        )
    cells, rects, area = cells[keep], rects[keep], area[keep]

    half = system.halves()
    k = system.conductivity()
    order = np.argsort(cells)
    cells, rects, area = cells[order], rects[order], area[order]
    return FacePort(
        label=label or ("z+" if direction > 0 else "z-"),
        axis=axis,
        direction=direction,
        cells=np.ascontiguousarray(cells),
        areas=np.ascontiguousarray(area),
        k=np.ascontiguousarray(k[cells]),
        half=np.ascontiguousarray(half[cells, axis]),
        t1=t1,
        t2=t2,
        rects=np.ascontiguousarray(rects),
    )


@dataclass(frozen=True)
class PatchSet:
    """The conservative common-patch set of two ports, with patch rectangles."""

    area: np.ndarray
    rects: np.ndarray
    li: np.ndarray
    ri: np.ndarray
    E_l: sp.csc_matrix
    E_r: sp.csc_matrix
    xi_l: np.ndarray
    xi_r: np.ndarray

    @property
    def count(self) -> int:
        return int(self.area.size)

    def centroids(self) -> np.ndarray:
        return 0.5 * np.column_stack(
            (self.rects[:, 0] + self.rects[:, 1], self.rects[:, 2] + self.rects[:, 3])
        )


def common_patches_with_rects(port_l: FacePort, port_r: FacePort) -> PatchSet:
    """``common_patches`` plus the patch rectangles (same edge/containment rule)."""
    from metahotspot.macromodel.embeddable import _EDGE_TOL, _contains

    rl, rr = port_l.rects, port_r.rects
    x_edges = np.unique(np.r_[rl[:, (0, 1)], rr[:, (0, 1)]])
    y_edges = np.unique(np.r_[rl[:, (2, 3)], rr[:, (2, 3)]])
    areas, rects, li, ri = [], [], [], []
    for xl, xr in zip(x_edges[:-1], x_edges[1:]):
        for yl, yr in zip(y_edges[:-1], y_edges[1:]):
            if (xr - xl) <= 2.0 * _EDGE_TOL or (yr - yl) <= 2.0 * _EDGE_TOL:
                continue
            lm = _contains(rl, xl, xr, yl, yr)
            rm = _contains(rr, xl, xr, yl, yr)
            if lm.size and rm.size:
                areas.append((xr - xl) * (yr - yl))
                rects.append((xl, xr, yl, yr))
                li.append(int(lm[0]))
                ri.append(int(rm[0]))
    if not areas:
        raise RuntimeError("interface ports do not overlap")
    area = np.asarray(areas, dtype=np.float64)
    li = np.asarray(li, dtype=np.int64)
    ri = np.asarray(ri, dtype=np.int64)
    n = area.size

    def incidence(idx, n_face):
        return sp.coo_matrix(
            (np.ones(n), (np.arange(n), idx)), shape=(n, n_face)
        ).tocsc()

    def fraction(idx, face_areas):
        return area / np.asarray(face_areas, dtype=np.float64)[idx]

    return PatchSet(
        area=area,
        rects=np.asarray(rects, dtype=np.float64),
        li=li,
        ri=ri,
        E_l=incidence(li, port_l.areas.size),
        E_r=incidence(ri, port_r.areas.size),
        xi_l=fraction(li, port_l.areas),
        xi_r=fraction(ri, port_r.areas),
    )


# ---------------------------------------------------------------------------
# the two sides (library "side" contract) + patch skew offsets
# ---------------------------------------------------------------------------


def pad_sources(*systems: System) -> None:
    """Give every block the same number of source columns (zero-padded).

    ``connect`` applies one power vector to both sides' ``source_matrix``, so a side
    without a physical source must still expose an all-zero column of that index.
    """
    n_src = max(max(system.source.shape[1] for system in systems), 1)
    for system in systems:
        missing = n_src - system.source.shape[1]
        if missing:
            system.source = np.hstack(
                [system.source, np.zeros((system.n, missing))]
            )


class FvmSide:
    """One assembled FVM block exposed through the embeddable-ROM side contract."""

    def __init__(self, system: System, ports: dict[str, FacePort]):
        self.system = system
        self.ports = ports

    @property
    def name(self) -> str:
        return self.system.name

    @property
    def dof_order(self) -> int:
        return self.system.n

    @property
    def capacity(self) -> sp.csc_matrix:
        return self.system.C

    @property
    def source_matrix(self) -> np.ndarray:
        return self.system.source

    def internal_operator(self) -> sp.csc_matrix:
        return self.system.K

    def port(self, label: str) -> FacePort:
        return self.ports[label]

    def boundary_trace(self, label: str) -> np.ndarray:
        return self.ports[label].cells

    def boundary_conductance(self, label: str) -> np.ndarray:
        return self.ports[label].g

    def interface_trace(self, port: FacePort, incidence, xi):
        incidence = sp.csr_matrix(incidence)
        h_b = np.asarray(self.boundary_conductance(port.label), dtype=np.float64)
        h_if = np.asarray(xi, dtype=np.float64) * np.asarray(incidence @ h_b).ravel()
        face_cells = np.asarray(self.boundary_trace(port.label), dtype=np.int64)
        n_face = face_cells.size
        V_face = sp.coo_matrix(
            (np.ones(n_face), (np.arange(n_face), face_cells)),
            shape=(n_face, self.dof_order),
        ).tocsr()
        return incidence @ V_face, h_if

    def cell_centers(self, port: FacePort) -> np.ndarray:
        return self.system.centers()[np.asarray(port.cells, dtype=np.int64)]

    def skew_term(self, port, owners, delta, dofs: slice, state) -> np.ndarray:
        """``delta . grad_t T`` at each patch owner, from the block field.

        The in-plane gradient is taken from the *whole* block field, not from the port's
        sub-rectangle: for the lid, cells just outside the die footprint still carry the
        true in-plane variation, and one-sided differences at the contact edge are exactly
        where the skew term is largest.  Blocks with inactive cells use whichever
        neighbouring cell in that direction is active (central when both are).
        """
        grid = self.system.grid
        nx, ny, nz = grid.n
        field = np.full((nx, ny, nz), np.nan)
        ijk_all = self.system.cells_ijk
        field[ijk_all[:, 0], ijk_all[:, 1], ijk_all[:, 2]] = np.asarray(
            state[dofs], dtype=np.float64
        )
        active = self.system.ids >= 0
        ijk = ijk_all[np.asarray(port.cells, dtype=np.int64)]
        centres = grid.centers()
        t1, t2 = _TANGENT[port.axis]
        components = []
        for axis in (t1, t2):
            coord = centres[axis]
            step = [(0, 0, 0), (0, 0, 0)]
            step[0] = tuple(int(axis == a) for a in range(3))
            step[1] = tuple(-int(axis == a) for a in range(3))
            fwd = ijk + np.array(step[0])
            back = ijk + np.array(step[1])
            ok_f = np.all((fwd >= 0) & (fwd < np.array((nx, ny, nz))), axis=1)
            ok_b = np.all((back >= 0) & (back < np.array((nx, ny, nz))), axis=1)
            have_f = np.zeros(ijk.shape[0], dtype=bool)
            have_b = np.zeros(ijk.shape[0], dtype=bool)
            have_f[ok_f] = active[fwd[ok_f, 0], fwd[ok_f, 1], fwd[ok_f, 2]]
            have_b[ok_b] = active[back[ok_b, 0], back[ok_b, 1], back[ok_b, 2]]
            here = field[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            value_f = np.full(ijk.shape[0], np.nan)
            value_b = np.full(ijk.shape[0], np.nan)
            value_f[have_f] = field[fwd[have_f, 0], fwd[have_f, 1], fwd[have_f, 2]]
            value_b[have_b] = field[back[have_b, 0], back[have_b, 1], back[have_b, 2]]
            dist_f = np.zeros(ijk.shape[0])
            dist_b = np.zeros(ijk.shape[0])
            dist_f[have_f] = coord[fwd[have_f, axis]] - coord[ijk[have_f, axis]]
            dist_b[have_b] = coord[ijk[have_b, axis]] - coord[back[have_b, axis]]
            grad = np.zeros(ijk.shape[0])
            both = have_f & have_b
            grad[both] = (value_f[both] - value_b[both]) / (dist_f[both] + dist_b[both])
            only_f = have_f & ~have_b
            grad[only_f] = (value_f[only_f] - here[only_f]) / dist_f[only_f]
            only_b = have_b & ~have_f
            grad[only_b] = (here[only_b] - value_b[only_b]) / dist_b[only_b]
            components.append(grad)
        grad = np.column_stack(components)
        return np.einsum("ij,ij->i", delta, grad[owners])


def patch_skew(
    port: FacePort, cells_centers: np.ndarray, patches: PatchSet, owners
) -> np.ndarray:
    """In-plane patch-centroid offset from the owner cell centre, per patch."""
    t1, t2 = _TANGENT[port.axis]
    centroids = patches.centroids()
    owner_centers = cells_centers[owners]
    return np.column_stack(
        (centroids[:, 0] - owner_centers[:, t1], centroids[:, 1] - owner_centers[:, t2])
    )


# ---------------------------------------------------------------------------
# coupled interface: repository condensation vs non-orthogonal correctors
# ---------------------------------------------------------------------------

COUPLED_RTOL = 1.0e-10
COUPLED_MAXITER = 2000


class AmgCg:
    """One reusable AMG-preconditioned CG solve for a fixed SPD matrix.

    The coupled matrix does not change under the deferred-correction iteration (only
    the right-hand side does), so one hierarchy serves every corrector and the previous
    state is the warm start -- the same contract as
    ``metahotspot.macromodel.embeddable.solve_system``.
    """

    def __init__(self, matrix, *, rtol: float = COUPLED_RTOL):
        self.A = sp.csr_matrix(matrix)
        self.rtol = float(rtol)
        hierarchy = pyamg.ruge_stuben_solver(self.A, interpolation="direct")
        self.M = hierarchy.aspreconditioner(cycle="V")
        self.iterations = 0
        self.solves = 0

    def solve(self, rhs, x0=None, label: str = "solve") -> np.ndarray:
        counter = [0]

        def count(_xk):
            counter[0] += 1

        x, info = spla.cg(
            self.A,
            np.asarray(rhs, dtype=np.float64).ravel(),
            x0=None if x0 is None else np.asarray(x0, dtype=np.float64).ravel(),
            rtol=self.rtol,
            atol=0.0,
            maxiter=COUPLED_MAXITER,
            M=self.M,
            callback=count,
        )
        if info != 0:
            raise RuntimeError(f"AMG-PCG failed for {label}: info={info}")
        self.iterations += counter[0]
        self.solves += 1
        return np.asarray(x, dtype=np.float64).ravel()


@dataclass(frozen=True)
class CorrectionStep:
    """One non-orthogonal corrector iteration."""

    index: int
    max_dT: float
    max_dq: float
    net_flux: float
    correction_scale: float


class InterfaceProblem:
    """Two sides joined on one interface; solves either interface scheme.

    ``left`` and ``right`` implement the library side contract plus
    ``cell_centers(port)``.  Correctors = 0 is the repository scheme (same matrix,
    zero correction RHS); each further corrector adds the deferred skew flux of the
    current iterate, so the limit is the fully non-orthogonally corrected scheme.
    """

    def __init__(self, left, right, port_l: FacePort, port_r: FacePort, *, power=None):
        self.left, self.right = left, right
        self.port_l, self.port_r = port_l, port_r
        self.patches = common_patches_with_rects(port_l, port_r)
        self.power = power
        self.K, self.C, self.rhs, self.ldof, self.rdof, self.n_patch = connect(
            left, right, port_l, port_r, power=power
        )
        self.K = self.K.tocsc()
        self.Vl, self.hl = left.interface_trace(
            port_l, self.patches.E_l, self.patches.xi_l
        )
        self.Vr, self.hr = right.interface_trace(
            port_r, self.patches.E_r, self.patches.xi_r
        )
        self.Vl = sp.csr_matrix(self.Vl)
        self.Vr = sp.csr_matrix(self.Vr)
        self.delta_l = patch_skew(
            port_l, left.cell_centers(port_l), self.patches, self.patches.li
        )
        self.delta_r = patch_skew(
            port_r, right.cell_centers(port_r), self.patches, self.patches.ri
        )
        self._solver = AmgCg(self.K)

    # -- state slices ------------------------------------------------------
    @property
    def node_slice(self) -> slice:
        return slice(self.ldof, self.ldof + self.n_patch)

    @property
    def right_slice(self) -> slice:
        return slice(self.ldof + self.n_patch, self.ldof + self.n_patch + self.rdof)

    def trace_values(self, state) -> tuple[np.ndarray, np.ndarray]:
        left = np.asarray(self.Vl @ state[: self.ldof]).ravel()
        right = np.asarray(self.Vr @ state[self.right_slice]).ravel()
        return left, right

    def patch_flux(self, state, *, corrected=True) -> np.ndarray:
        """Heat flux through every patch (W, positive from the left side to the right)."""
        tl, tr = self.trace_values(state)
        g_eff = self.hl * self.hr / (self.hl + self.hr)
        skew = self._skew(state) if corrected else np.zeros(self.n_patch)
        return g_eff * (tl - tr - skew)

    def interface_conductance_series(self) -> np.ndarray:
        return self.hl * self.hr / (self.hl + self.hr)

    # -- the deferred correction -------------------------------------------
    def _side_skew(self, side, port, owners, delta, dofs: slice, state) -> np.ndarray:
        """``delta . grad_t T`` at the owner face of every patch, for one side."""
        return side.skew_term(port, owners, delta, dofs, state)

    def skew_lr(self, state) -> tuple[np.ndarray, np.ndarray]:
        left = self._side_skew(
            self.left, self.port_l, self.patches.li, self.delta_l,
            slice(0, self.ldof), state,
        )
        right = self._side_skew(
            self.right, self.port_r, self.patches.ri, self.delta_r,
            self.right_slice, state,
        )
        return left, right

    def _skew(self, state) -> np.ndarray:
        left, right = self.skew_lr(state)
        return right - left

    def _correction_rhs(self, state) -> np.ndarray:
        left, right = self.skew_lr(state)
        h_cl, h_cr = self.hl * left, self.hr * right
        rhs = np.zeros(self.K.shape[0])
        rhs[self.node_slice] += h_cl + h_cr
        rhs[: self.ldof] += -np.asarray(self.Vl.T @ h_cl)
        rhs[self.right_slice] += -np.asarray(self.Vr.T @ h_cr)
        return rhs

    # -- solves ------------------------------------------------------------
    def solve(self, correctors: int = 0, *, tol: float | None = None,
              max_correctors: int = 60, relaxation: float = 1.0) -> dict:
        """Solve with ``correctors`` deferred-correction iterations.

        ``correctors=0`` is the repository scheme.  ``tol`` (relative to the first
        correction) instead iterates until the correction stops changing, capped at
        ``max_correctors``; a conforming interface gives a zero correction and stops
        after one.
        """
        base = np.asarray(self.rhs, dtype=np.float64)
        state = self._solver.solve(base, label="repo solve")
        states = [state]
        steps = [CorrectionStep(0, 0.0, 0.0, float(self.patch_flux(state).sum()), 0.0)]
        correction = np.zeros_like(base)
        reference = None
        last = correctors if tol is None else max_correctors
        for index in range(1, last + 1):
            new = self._correction_rhs(state)
            if reference is None:
                reference = max(float(np.max(np.abs(new))), 1.0e-300)
                if np.max(np.abs(new)) == 0.0:
                    break
            if tol is not None and index > 1:
                if float(np.max(np.abs(new - correction))) <= tol * reference:
                    break
            correction = (
                new
                if relaxation >= 1.0
                else correction + relaxation * (new - correction)
            )
            previous = state
            state = self._solver.solve(
                base + correction, x0=previous, label=f"corrector {index}"
            )
            states.append(state)
            change = float(np.max(np.abs(state - previous)))
            if change == 0.0:
                break
            steps.append(
                CorrectionStep(
                    index=index,
                    max_dT=change,
                    max_dq=float(
                        np.max(np.abs(self.patch_flux(state) - self.patch_flux(previous)))
                    ),
                    net_flux=float(self.patch_flux(state).sum()),
                    correction_scale=float(np.max(np.abs(new))) / reference,
                )
            )
        return {
            "state": state,
            "states": states,
            "steps": steps,
            "correctors": len(states) - 1,
            "flux": self.patch_flux(state),
        }
