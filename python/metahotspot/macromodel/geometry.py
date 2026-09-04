"""Derived geometry and topology views for compiled cell fields."""

from __future__ import annotations

import numpy as np

from metahotspot._compiled_data import CellFields


_FACE_STEPS = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def grid_indices(cells: CellFields) -> np.ndarray:
    """Return ``(ix, iy, iz)`` for each compact cell."""
    yz = cells.dy.size * cells.dz.size
    grid = cells.cell_to_grid
    ijk = np.empty((grid.size, 3), dtype=np.intp)
    ijk[:, 0] = grid // yz
    ijk[:, 1] = (grid % yz) // cells.dz.size
    ijk[:, 2] = grid % cells.dz.size
    return ijk


def cell_sizes(cells: CellFields) -> np.ndarray:
    """Return per-cell side lengths in compact-cell order."""
    ijk = grid_indices(cells)
    return np.column_stack(
        (cells.dx[ijk[:, 0]], cells.dy[ijk[:, 1]], cells.dz[ijk[:, 2]])
    )


def cell_centers(cells: CellFields) -> np.ndarray:
    """Return per-cell centres in compact-cell order."""
    ijk = grid_indices(cells)
    return np.column_stack(
        (cells.cx[ijk[:, 0]], cells.cy[ijk[:, 1]], cells.cz[ijk[:, 2]])
    )


def axis_vertices(cells: CellFields, axis: int) -> np.ndarray:
    """Return cell-plane coordinates along one grid axis."""
    widths = (cells.dx, cells.dy, cells.dz)[axis]
    centers = (cells.cx, cells.cy, cells.cz)[axis]
    return np.concatenate(([centers[0] - 0.5 * widths[0]], centers + 0.5 * widths))


def exposed_face_mask(cells: CellFields) -> np.ndarray:
    """Return the six-bit exposed-face mask for each compact cell."""
    grid = cells.grid_to_cell.reshape(cells.dx.size, cells.dy.size, cells.dz.size)
    invalid = np.iinfo(grid.dtype).max
    padded = np.full(tuple(size + 2 for size in grid.shape), invalid, dtype=grid.dtype)
    padded[1:-1, 1:-1, 1:-1] = grid
    exposed = np.stack(
        [
            padded[
                1 + dx : cells.dx.size + 1 + dx,
                1 + dy : cells.dy.size + 1 + dy,
                1 + dz : cells.dz.size + 1 + dz,
            ]
            == invalid
            for dx, dy, dz in _FACE_STEPS
        ],
        axis=-1,
    )
    bits = np.sum(exposed.reshape(-1, 6) * (1 << np.arange(6, dtype=np.uint8)), axis=1)
    return bits[cells.cell_to_grid].astype(np.uint8)
