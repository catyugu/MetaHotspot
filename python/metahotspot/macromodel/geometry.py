"""Structured-grid geometry views for compiled cell fields."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from metahotspot._compiled_data import CellFields
from metahotspot.enums import Axis, Face


_FACE_STEPS = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)
_FACE_AXIS = {
    Face.XM: (Axis.X, 0),
    Face.XP: (Axis.X, -1),
    Face.YM: (Axis.Y, 0),
    Face.YP: (Axis.Y, -1),
    Face.ZM: (Axis.Z, 0),
    Face.ZP: (Axis.Z, -1),
}


@dataclass(frozen=True)
class BoundarySurface:
    """Exposed faces of one structured-grid boundary plane."""

    cell_ids: np.ndarray
    areas: np.ndarray


@dataclass(frozen=True)
class CellGeometry:
    """Geometry and topology views in compact-cell order.

    ``fields`` contains the native copy-out arrays. This view owns the meaning
    of their structured-grid layout and computes derived quantities lazily.
    """

    fields: CellFields

    @property
    def nx(self) -> int:
        return self.fields.dx.size

    @property
    def ny(self) -> int:
        return self.fields.dy.size

    @property
    def nz(self) -> int:
        return self.fields.dz.size

    @property
    def widths(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.fields.dx, self.fields.dy, self.fields.dz

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.fields.cx, self.fields.cy, self.fields.cz

    @cached_property
    def indices(self) -> np.ndarray:
        grid = self.fields.cell_to_grid
        yz = self.ny * self.nz
        result = np.empty((grid.size, 3), dtype=np.intp)
        result[:, 0] = grid // yz
        result[:, 1] = (grid % yz) // self.nz
        result[:, 2] = grid % self.nz
        return result

    @cached_property
    def sizes(self) -> np.ndarray:
        ijk = self.indices
        return np.column_stack(
            (
                self.fields.dx[ijk[:, 0]],
                self.fields.dy[ijk[:, 1]],
                self.fields.dz[ijk[:, 2]],
            )
        )

    @cached_property
    def centers(self) -> np.ndarray:
        ijk = self.indices
        return np.column_stack(
            (
                self.fields.cx[ijk[:, 0]],
                self.fields.cy[ijk[:, 1]],
                self.fields.cz[ijk[:, 2]],
            )
        )

    @cached_property
    def half_sizes(self) -> np.ndarray:
        return self.sizes * 0.5

    @cached_property
    def vertices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.concatenate(
                (
                    [centers[0] - 0.5 * widths[0]],
                    centers + 0.5 * widths,
                )
            )
            for centers, widths in zip(
                (self.fields.cx, self.fields.cy, self.fields.cz),
                (self.fields.dx, self.fields.dy, self.fields.dz),
            )
        )

    def exposed(self, face: Face) -> np.ndarray:
        """Return whether each compact cell has an exposed ``face``."""
        face = Face(face)
        dx, dy, dz = _FACE_STEPS[int(face)]
        grid = self.fields.grid_to_cell.reshape(self.nx, self.ny, self.nz)
        invalid = np.iinfo(grid.dtype).max
        padded = np.full(
            tuple(size + 2 for size in grid.shape), invalid, dtype=grid.dtype
        )
        padded[1:-1, 1:-1, 1:-1] = grid
        neighbor = padded[
            1 + dx : self.nx + 1 + dx,
            1 + dy : self.ny + 1 + dy,
            1 + dz : self.nz + 1 + dz,
        ]
        return (neighbor == invalid).ravel()[self.fields.cell_to_grid]

    def surface(
        self,
        face: Face,
        coordinate: float | None = None,
        z_range: tuple[float, float] | None = None,
    ) -> BoundarySurface:
        """Return exposed cells and face areas on a boundary plane."""
        face = Face(face)
        axis, edge = _FACE_AXIS[face]
        axis_index = int(axis)
        candidates = np.flatnonzero(self.exposed(face))
        indices = self.indices
        sizes = self.sizes
        edge_index = 0 if edge == 0 else (self.nx, self.ny, self.nz)[axis_index] - 1
        candidates = candidates[indices[candidates, axis_index] == edge_index]

        if coordinate is None:
            coordinate = self.vertices[axis_index][0 if edge == 0 else -1]
        planes = self.vertices[axis_index][
            indices[candidates, axis_index] + (edge != 0)
        ]
        keep = np.isclose(planes, coordinate, atol=1.0e-9, rtol=0.0)

        if z_range is not None and axis != Axis.Z:
            z_centers = self.fields.cz[indices[candidates, 2]]
            keep &= (z_range[0] - 1.0e-9 <= z_centers) & (
                z_centers <= z_range[1] + 1.0e-9
            )

        candidates = candidates[keep]
        tangential = [a for a in range(3) if a != axis_index]
        areas = sizes[candidates, tangential[0]] * sizes[candidates, tangential[1]]
        return BoundarySurface(
            candidates.astype(np.int64), np.asarray(areas, dtype=np.float64)
        )
