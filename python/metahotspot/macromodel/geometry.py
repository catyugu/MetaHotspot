"""Explicit cell geometry and native face topology, independent of mesh origin."""

from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from metahotspot._compiled_data import CellFields
from metahotspot.enums import Face


@dataclass(frozen=True)
class BoundarySurface:
    """A projected physical surface; several patches may share one cell."""

    cell_ids: np.ndarray
    areas: np.ndarray
    face_ids: np.ndarray


@dataclass(frozen=True)
class CellGeometry:
    fields: CellFields

    @cached_property
    def sizes(self):
        return self.fields.cell_max - self.fields.cell_min

    @cached_property
    def centers(self):
        return (self.fields.cell_min + self.fields.cell_max) * 0.5

    @cached_property
    def half_sizes(self):
        return self.sizes * 0.5

    @cached_property
    def volumes(self):
        return np.prod(self.sizes, axis=1)

    def exposed(self, face: Face):
        face = Face(face)
        f = self.fields
        selected = (
            (f.face_neighbor == np.iinfo(np.uintp).max)
            & (f.face_axis == int(face) // 2)
            & (f.face_sign == (-1 if int(face) % 2 == 0 else 1))
        )
        result = np.zeros(len(self.centers), dtype=bool)
        result[f.face_owner[selected]] = True
        return result

    def surface(self, face: Face, coordinate: float | None = None, z_range=None):
        face = Face(face)
        axis = int(face) // 2
        sign = -1 if int(face) % 2 == 0 else 1
        f = self.fields
        ids = np.flatnonzero(
            (f.face_neighbor == np.iinfo(np.uintp).max)
            & (f.face_axis == axis)
            & (f.face_sign == sign)
        )
        if coordinate is None:
            coordinate = (
                f.cell_min[:, axis].min() if sign < 0 else f.cell_max[:, axis].max()
            )
        ids = ids[np.isclose(f.face_min[ids, axis], coordinate, atol=1e-12, rtol=0)]
        areas = f.face_area[ids].copy()
        if z_range is not None and axis != 2:
            lo = np.maximum(f.face_min[ids, 2], z_range[0])
            hi = np.minimum(f.face_max[ids, 2], z_range[1])
            areas *= np.maximum(hi - lo, 0) / (f.face_max[ids, 2] - f.face_min[ids, 2])
            keep = areas > 0
            ids = ids[keep]
            areas = areas[keep]
        return BoundarySurface(
            f.face_owner[ids].astype(np.int64), areas, ids.astype(np.int64)
        )
