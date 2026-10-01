"""Shared FVM, conservative coupling, projection, and contour-element utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pyamg
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial.legendre import Legendre

MaterialFn = Callable[[float, float, float], tuple[float, float, float, int]]
PCG_RTOL = 1.0e-10


@dataclass(frozen=True)
class SourceBox:
    name: str
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float


@dataclass
class Domain:
    name: str
    K: sp.csc_matrix
    C: sp.csc_matrix
    rhs: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    k: np.ndarray
    rho_cp: np.ndarray
    material_id: np.ndarray
    source_density: np.ndarray
    source_matrix: np.ndarray | None = None
    source_names: tuple[str, ...] = ()
    source_powers: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        return self.K.shape[0]

    @property
    def source_mask(self) -> np.ndarray:
        return self.source_density != 0.0

    @property
    def volumes(self) -> np.ndarray:
        return np.asarray(self.C.diagonal()) / self.rho_cp


@dataclass(frozen=True)
class FacePort:
    ids: np.ndarray
    half_conductance: np.ndarray
    area: np.ndarray
    rects: np.ndarray


@dataclass(frozen=True)
class CommonPatches:
    area: np.ndarray
    rects: np.ndarray
    left_owner: np.ndarray
    right_owner: np.ndarray
    left_half_conductance: np.ndarray
    right_half_conductance: np.ndarray


@dataclass(frozen=True)
class ContourElements:
    """Continuous L2-orthonormal polynomial contour elements on one rectangle.

    This is the discrete FVM realization of Eqs. (13)-(17) in Codecasa et al.,
    "Boundary Condition Independent Compact Thermal Models Enhanced by Contour
    Elements", THERMINIC 2023.  Modes are products of 1-D orthogonal Legendre
    polynomials with total degree <= ``degree``; therefore their count is
    (p+1)(p+2)/2.  ``averages`` returns exact sub-rectangle averages of the
    continuous basis, rather than point samples at face centers.
    """

    degree: int
    bounds: tuple[float, float, float, float]
    orders: tuple[tuple[int, int], ...]

    @classmethod
    def from_rects(cls, rects: np.ndarray, degree: int) -> "ContourElements":
        rects = np.asarray(rects, dtype=float)
        if rects.ndim != 2 or rects.shape[1] != 4 or rects.shape[0] == 0:
            raise ValueError("rects must have shape (n, 4)")
        if degree < 0:
            raise ValueError("degree must be non-negative")
        bounds = (
            float(rects[:, 0].min()),
            float(rects[:, 1].max()),
            float(rects[:, 2].min()),
            float(rects[:, 3].max()),
        )
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise ValueError("contour surface must have positive area")
        orders = tuple(
            (i, total - i)
            for total in range(degree + 1)
            for i in range(total + 1)
        )
        return cls(degree, bounds, orders)

    @property
    def size(self) -> int:
        return len(self.orders)

    def _integrals_1d(self, intervals: np.ndarray, axis: int) -> np.ndarray:
        if axis == 0:
            lo0, hi0 = self.bounds[0], self.bounds[1]
            max_order = max(i for i, _ in self.orders)
        else:
            lo0, hi0 = self.bounds[2], self.bounds[3]
            max_order = max(j for _, j in self.orders)
        length = hi0 - lo0
        u0 = 2.0 * (intervals[:, 0] - lo0) / length - 1.0
        u1 = 2.0 * (intervals[:, 1] - lo0) / length - 1.0
        result = np.empty((intervals.shape[0], max_order + 1), dtype=float)
        for n in range(max_order + 1):
            anti = Legendre.basis(n).integ()
            integral_u = anti(u1) - anti(u0)
            result[:, n] = np.sqrt((2.0 * n + 1.0) / length) * (length / 2.0) * integral_u
        return result

    def integrals(self, rects: np.ndarray) -> np.ndarray:
        """Exact integral of each contour basis function over each rectangle."""
        rects = np.asarray(rects, dtype=float)
        ix = self._integrals_1d(rects[:, :2], axis=0)
        iy = self._integrals_1d(rects[:, 2:], axis=1)
        return np.column_stack([ix[:, i] * iy[:, j] for i, j in self.orders])

    def averages(self, rects: np.ndarray) -> np.ndarray:
        """Exact area average of each contour basis function on each rectangle."""
        rects = np.asarray(rects, dtype=float)
        area = (rects[:, 1] - rects[:, 0]) * (rects[:, 3] - rects[:, 2])
        if np.any(area <= 0.0):
            raise ValueError("all rectangles must have positive area")
        return self.integrals(rects) / area[:, None]


class AMGPCGSolver:
    """Reusable Ruge-Stuben-AMG-preconditioned CG solver for SPD systems."""

    def __init__(self, A: sp.spmatrix | np.ndarray, rtol: float = PCG_RTOL):
        self.A = sp.csr_matrix(A)
        self.rtol = float(rtol)
        self.maxiter = 2000
        hierarchy = pyamg.ruge_stuben_solver(self.A, interpolation="direct")
        self.M = hierarchy.aspreconditioner(cycle="V")

    def _solve_vector(self, rhs: np.ndarray, x0: np.ndarray | None) -> np.ndarray:
        x, info = spla.cg(
            self.A,
            np.asarray(rhs, dtype=float),
            x0=x0,
            rtol=self.rtol,
            atol=0.0,
            maxiter=self.maxiter,
            M=self.M,
        )
        if info:
            raise RuntimeError(f"AMG-PCG failed to converge: info={info}")
        return np.asarray(x, dtype=float)

    def solve(self, rhs: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim == 1:
            return self._solve_vector(rhs, x0)
        if rhs.ndim != 2:
            raise ValueError("rhs must be a vector or matrix of RHS columns")
        if x0 is not None:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != rhs.shape:
                raise ValueError("x0 must have the same shape as rhs")
        result = np.empty_like(rhs)
        for col in range(rhs.shape[1]):
            guess = None if x0 is None else x0[:, col]
            result[:, col] = self._solve_vector(rhs[:, col], guess)
        return result


def solve_spd(A: sp.spmatrix | np.ndarray, rhs: np.ndarray, x0=None) -> np.ndarray:
    return AMGPCGSolver(A).solve(rhs, x0=x0)


def centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def refined_edges(
    lo: float, hi: float, n: int, extras: Iterable[float] = ()
) -> np.ndarray:
    return np.unique(np.r_[np.linspace(lo, hi, n + 1), tuple(extras)])


def cell_index(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return (i * ny + j) * nz + k


def build_domain(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    material: MaterialFn,
    *,
    sources: Sequence[SourceBox] = (),
    source_powers: Sequence[float] | None = None,
    bottom_htc: float | None = None,
    top_htc: float | None = None,
    top_htc_mask: np.ndarray | None = None,
) -> Domain:
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    xc, yc, zc = centers(x), centers(y), centers(z)
    n = nx * ny * nz
    conductivity = np.empty(n)
    rho_cp = np.empty(n)
    material_id = np.empty(n, dtype=np.int32)
    volume = np.empty(n)
    for i, j, k in product(range(nx), range(ny), range(nz)):
        q = cell_index(i, j, k, ny, nz)
        kk, rho, cp, mat_id = material(xc[i], yc[j], zc[k])
        conductivity[q], rho_cp[q], material_id[q] = kk, rho * cp, mat_id
        volume[q] = dx[i] * dy[j] * dz[k]

    G = np.zeros((n, len(sources)))
    for col, box in enumerate(sources):
        ids = [
            cell_index(i, j, k, ny, nz)
            for i, j, k in product(range(nx), range(ny), range(nz))
            if box.x0 <= xc[i] < box.x1
            and box.y0 <= yc[j] < box.y1
            and box.z0 <= zc[k] < box.z1
        ]
        total = float(np.sum(volume[ids]))
        if total == 0.0:
            raise ValueError(f"source {box.name!r} does not intersect the grid")
        G[ids, col] = volume[ids] / total
    powers = (
        np.ones(len(sources))
        if source_powers is None
        else np.asarray(source_powers, dtype=float)
    )
    if powers.shape != (len(sources),):
        raise ValueError("source_powers must contain one value per source")
    rhs = G @ powers if len(sources) else np.zeros(n)

    rows, cols, values = [], [], []
    diagonal = np.zeros(n)

    def connect(q, r, conductance):
        diagonal[q] += conductance
        diagonal[r] += conductance
        rows.extend((q, r))
        cols.extend((r, q))
        values.extend((-conductance, -conductance))

    for i, j, k in product(range(nx), range(ny), range(nz)):
        q = cell_index(i, j, k, ny, nz)
        if i + 1 < nx:
            r = cell_index(i + 1, j, k, ny, nz)
            connect(
                q,
                r,
                dy[j]
                * dz[k]
                / (0.5 * dx[i] / conductivity[q] + 0.5 * dx[i + 1] / conductivity[r]),
            )
        if j + 1 < ny:
            r = cell_index(i, j + 1, k, ny, nz)
            connect(
                q,
                r,
                dx[i]
                * dz[k]
                / (0.5 * dy[j] / conductivity[q] + 0.5 * dy[j + 1] / conductivity[r]),
            )
        if k + 1 < nz:
            r = cell_index(i, j, k + 1, ny, nz)
            connect(
                q,
                r,
                dx[i]
                * dy[j]
                / (0.5 * dz[k] / conductivity[q] + 0.5 * dz[k + 1] / conductivity[r]),
            )

    for side, htc, mask in (
        ("bottom", bottom_htc, None),
        ("top", top_htc, top_htc_mask),
    ):
        if htc is None:
            continue
        k = 0 if side == "bottom" else nz - 1
        for i, j in product(range(nx), range(ny)):
            if mask is not None and not mask[i, j]:
                continue
            q = cell_index(i, j, k, ny, nz)
            area = dx[i] * dy[j]
            diagonal[q] += area / (0.5 * dz[k] / conductivity[q] + 1.0 / htc)

    rows.extend(range(n))
    cols.extend(range(n))
    values.extend(diagonal)
    K = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsc()
    C = sp.diags(rho_cp * volume, format="csc")
    return Domain(
        name,
        K,
        C,
        rhs,
        x,
        y,
        z,
        conductivity,
        rho_cp,
        material_id,
        rhs / volume,
        G if len(sources) else None,
        tuple(box.name for box in sources),
        powers if len(sources) else None,
    )


def boundary_port(domain: Domain, side: str, footprint=None) -> FacePort:
    nx, ny, nz = len(domain.x) - 1, len(domain.y) - 1, len(domain.z) - 1
    k = 0 if side == "bottom" else nz - 1
    dx, dy, dz = np.diff(domain.x), np.diff(domain.y), np.diff(domain.z)
    ids, area, half, rects = [], [], [], []
    for i, j in product(range(nx), range(ny)):
        xl, xr, yl, yr = domain.x[i], domain.x[i + 1], domain.y[j], domain.y[j + 1]
        if footprint is not None:
            x0, x1, y0, y1 = footprint
            if xr <= x0 or xl >= x1 or yr <= y0 or yl >= y1:
                continue
        q = cell_index(i, j, k, ny, nz)
        a = dx[i] * dy[j]
        ids.append(q)
        area.append(a)
        half.append(domain.k[q] * a / (0.5 * dz[k]))
        rects.append((xl, xr, yl, yr))
    return FacePort(
        np.asarray(ids), np.asarray(half), np.asarray(area), np.asarray(rects)
    )


def common_patches(left: FacePort, right: FacePort, tol=1e-12) -> CommonPatches:
    xs = np.unique(np.r_[left.rects[:, :2].ravel(), right.rects[:, :2].ravel()])
    ys = np.unique(np.r_[left.rects[:, 2:].ravel(), right.rects[:, 2:].ravel()])
    area, rects, left_owner, right_owner, left_half, right_half = [], [], [], [], [], []
    for xl, xr in zip(xs[:-1], xs[1:]):
        for yl, yr in zip(ys[:-1], ys[1:]):
            if xr - xl <= tol or yr - yl <= tol:
                continue
            cx, cy = 0.5 * (xl + xr), 0.5 * (yl + yr)
            li = np.flatnonzero(
                (left.rects[:, 0] <= cx)
                & (cx <= left.rects[:, 1])
                & (left.rects[:, 2] <= cy)
                & (cy <= left.rects[:, 3])
            )
            ri = np.flatnonzero(
                (right.rects[:, 0] <= cx)
                & (cx <= right.rects[:, 1])
                & (right.rects[:, 2] <= cy)
                & (cy <= right.rects[:, 3])
            )
            if not li.size or not ri.size:
                continue
            l, r = int(li[0]), int(ri[0])
            a = (xr - xl) * (yr - yl)
            area.append(a)
            rects.append((xl, xr, yl, yr))
            left_owner.append(l)
            right_owner.append(r)
            left_half.append(left.half_conductance[l] * a / left.area[l])
            right_half.append(right.half_conductance[r] * a / right.area[r])
    if not area:
        raise ValueError("ports do not overlap")
    return CommonPatches(
        *map(np.asarray, (area, rects, left_owner, right_owner, left_half, right_half))
    )


def trace_matrix(port: FacePort, owners: np.ndarray, n_cells: int) -> sp.csr_matrix:
    return sp.coo_matrix(
        (np.ones(len(owners)), (np.arange(len(owners)), port.ids[owners])),
        shape=(len(owners), n_cells),
    ).tocsr()


def condensed_interface(
    left: Domain,
    right: Domain,
    left_side: str,
    right_side: str,
    footprint=None,
    contact_rth=None,
):
    lp, rp = boundary_port(left, left_side), boundary_port(right, right_side, footprint)
    patches = common_patches(lp, rp)
    EL = trace_matrix(lp, patches.left_owner, left.n_cells)
    ER = trace_matrix(rp, patches.right_owner, right.n_cells)
    resistance = (
        1.0 / patches.left_half_conductance + 1.0 / patches.right_half_conductance
    )
    if contact_rth is not None:
        contact = patches.area / (contact_rth * lp.area.sum())
        resistance += 1.0 / contact
    return patches, lp, rp, EL, ER, sp.diags(1.0 / resistance)


def contour_trace_coefficients(
    port: FacePort, basis: np.ndarray, degree: int
) -> tuple[ContourElements, np.ndarray, float]:
    """Project a ROM boundary trace onto standard contour elements.

    For an L2-orthonormal contour basis psi, Eq. (15) reduces to
    ``Vhat = integral psi^T v dA``.  The FVM boundary trace is piecewise
    constant on each face, so the integral is evaluated exactly from face
    rectangles.  The returned error is the area-weighted relative Frobenius
    error on the original face-average trace.
    """
    elements = ContourElements.from_rects(port.rects, degree)
    face_avg = elements.averages(port.rects)
    Vface = np.asarray(basis[port.ids, :], dtype=float)
    coefficients = elements.integrals(port.rects).T @ Vface
    reconstructed = face_avg @ coefficients
    diff = reconstructed - Vface
    num = float(np.sum(port.area[:, None] * diff * diff))
    den = float(np.sum(port.area[:, None] * Vface * Vface))
    relative = np.sqrt(num / max(den, 1e-300))
    return elements, np.ascontiguousarray(coefficients), relative


def evaluate_contour_trace(
    elements: ContourElements, coefficients: np.ndarray, rects: np.ndarray
) -> np.ndarray:
    """Evaluate the contour-element trace as exact rectangle averages (Eq. 16)."""
    return np.ascontiguousarray(elements.averages(rects) @ coefficients)


def project(domain: Domain, basis: np.ndarray):
    return (
        sp.csc_matrix(basis.T @ (domain.K @ basis)),
        sp.csc_matrix(basis.T @ (domain.C @ basis)),
        basis.T @ domain.rhs,
    )


def write_domains_vtu(
    path: str | Path, domains: Sequence[Domain], cell_fields: dict[str, Sequence[np.ndarray]]
) -> None:
    points, cells, flat = [], [], {name: [] for name in cell_fields}
    point_offset = 0
    for d, domain in enumerate(domains):
        nx, ny, nz = len(domain.x) - 1, len(domain.y) - 1, len(domain.z) - 1
        points.extend(product(domain.x, domain.y, domain.z))
        pid = lambda i, j, k: point_offset + (i * (ny + 1) + j) * (nz + 1) + k
        for i, j, k in product(range(nx), range(ny), range(nz)):
            cells.append(
                (
                    pid(i, j, k),
                    pid(i + 1, j, k),
                    pid(i + 1, j + 1, k),
                    pid(i, j + 1, k),
                    pid(i, j, k + 1),
                    pid(i + 1, j, k + 1),
                    pid(i + 1, j + 1, k + 1),
                    pid(i, j + 1, k + 1),
                )
            )
            q = cell_index(i, j, k, ny, nz)
            for name, values in cell_fields.items():
                flat[name].append(values[d][q])
        point_offset += (nx + 1) * (ny + 1) * (nz + 1)
    pts, cls = np.asarray(points), np.asarray(cells, dtype=np.int64)
    fmt = lambda x: " ".join(f"{v:.10g}" for v in np.asarray(x).ravel())
    arrays = []
    for name, values in flat.items():
        data = np.asarray(values)
        vtk_type = "Int32" if np.issubdtype(data.dtype, np.integer) else "Float64"
        text = " ".join(map(str, data)) if vtk_type == "Int32" else fmt(data)
        arrays.append(
            f'<DataArray type="{vtk_type}" Name="{name}" format="ascii">{text}</DataArray>'
        )
    Path(path).write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid><Piece NumberOfPoints="{len(pts)}" NumberOfCells="{len(cls)}">
<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">{fmt(pts)}</DataArray></Points>
<Cells>
<DataArray type="Int64" Name="connectivity" format="ascii">{' '.join(map(str, cls.ravel()))}</DataArray>
<DataArray type="Int64" Name="offsets" format="ascii">{' '.join(map(str, 8*np.arange(1, len(cls)+1)))}</DataArray>
<DataArray type="UInt8" Name="types" format="ascii">{' '.join(['12']*len(cls))}</DataArray>
</Cells><CellData>{''.join(arrays)}</CellData></Piece></UnstructuredGrid></VTKFile>"""
    )
