"""Immutable data shared by compiled-model adapters and wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CellFields:
    grid_to_cell: np.ndarray
    cell_to_grid: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray
    layer_id: np.ndarray
    block_id: np.ndarray
    material_id: np.ndarray
    heat_source_idx: np.ndarray

    # Face directions in exposed_face_mask bit order (bit 0..5 = XM, XP, YM,
    # YP, ZM, ZP), matching metahotspot.enums.Face.
    _FACE_STEPS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))

    @staticmethod
    def _vertices(centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
        return np.concatenate(([centers[0] - 0.5 * widths[0]], centers + 0.5 * widths))

    @property
    def x_vertices(self) -> np.ndarray:
        return self._vertices(self.cx, self.dx)

    @property
    def y_vertices(self) -> np.ndarray:
        return self._vertices(self.cy, self.dy)

    @property
    def z_vertices(self) -> np.ndarray:
        return self._vertices(self.cz, self.dz)

    @property
    def ijk(self) -> np.ndarray:
        """Per-compact-cell ``(ix, iy, iz)`` grid coordinates, decoded once."""
        return self._ijk

    @property
    def exposed_face_mask(self) -> np.ndarray:
        """Per-compact-cell ``(N, 6)`` uint8 mask: bit ``Face`` set when that
        face has no active neighbour (out of bounds or empty cell)."""
        return self._exposed_face_mask

    @property
    def cell_sizes(self) -> np.ndarray:
        ijk = self._ijk
        return np.column_stack(
            (self.dx[ijk[:, 0]], self.dy[ijk[:, 1]], self.dz[ijk[:, 2]])
        )

    @property
    def centers(self) -> np.ndarray:
        ijk = self._ijk
        return np.column_stack(
            (self.cx[ijk[:, 0]], self.cy[ijk[:, 1]], self.cz[ijk[:, 2]])
        )

    @property
    def half_sizes(self) -> np.ndarray:
        return self.cell_sizes * 0.5

    @property
    def volumes(self) -> np.ndarray:
        sizes = self.cell_sizes
        return sizes[:, 0] * sizes[:, 1] * sizes[:, 2]

    @property
    def nx(self) -> int:
        return self.dx.size

    @property
    def ny(self) -> int:
        return self.dy.size

    @property
    def nz(self) -> int:
        return self.dz.size

    def _compute_exposed_face_mask(self) -> np.ndarray:
        grid = self._grid3d
        invalid = np.iinfo(self.grid_to_cell.dtype).max
        # Pad with an invalid border so out-of-bounds neighbours read as
        # inactive, then compare each of the six shifted views to `invalid`.
        padded = np.full(
            (self.nx + 2, self.ny + 2, self.nz + 2), invalid, dtype=grid.dtype
        )
        padded[1:-1, 1:-1, 1:-1] = grid
        layers = [
            padded[
                1 + dx : self.nx + 1 + dx,
                1 + dy : self.ny + 1 + dy,
                1 + dz : self.nz + 1 + dz,
            ]
            == invalid
            for dx, dy, dz in self._FACE_STEPS
        ]
        exposed = np.stack(layers, axis=-1)  # (nx, ny, nz, 6), grid order
        bits = np.sum(
            exposed.reshape(-1, 6) * (1 << np.arange(6, dtype=np.uint8)), axis=1
        )
        return bits[self.cell_to_grid].astype(np.uint8)

    def __post_init__(self) -> None:
        yz = self.ny * self.nz
        grid = self.cell_to_grid
        ijk = np.empty((grid.size, 3), dtype=np.intp)
        ijk[:, 0] = grid // yz
        ijk[:, 1] = (grid % yz) // self.nz
        ijk[:, 2] = grid % self.nz
        object.__setattr__(self, "_ijk", ijk)
        object.__setattr__(
            self, "_grid3d", self.grid_to_cell.reshape(self.nx, self.ny, self.nz)
        )
        object.__setattr__(
            self, "_exposed_face_mask", self._compute_exposed_face_mask()
        )
        for value in self.__dict__.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)


@dataclass(frozen=True)
class CompiledMetadata:
    cell_count: int
    grid_count: int
    study_type: int
    initial_temperature: float
    nx: int
    ny: int
    nz: int
    cell_fields: CellFields
