"""Explicit cell geometry and native face topology, independent of mesh origin."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from metahotspot._compiled_data import CompiledFields
from metahotspot.enums import Face


@dataclass(frozen=True)
class BoundarySurface:
    """A projected physical surface; several patches may share one cell."""

    cell_ids: np.ndarray
    areas: np.ndarray
    face_ids: np.ndarray


@dataclass(frozen=True)
class CellGeometry:
    fields: CompiledFields

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

    def surface(self, face: Face, coordinate: float | None = None, z_range=None):
        face = Face(face)
        axis = int(face) // 2
        sign = -1 if int(face) % 2 == 0 else 1
        f = self.fields
        owner_centers = self.centers[f.face_owner, f.face_axis]
        signs = np.where(f.face_coordinate > owner_centers, 1, -1)
        ids = np.flatnonzero(
            (f.face_neighbor == np.iinfo(np.uint32).max)
            & (f.face_axis == axis)
            & (signs == sign)
        )
        if coordinate is None:
            coordinate = (
                f.cell_min[:, axis].min() if sign < 0 else f.cell_max[:, axis].max()
            )
        ids = ids[np.isclose(f.face_coordinate[ids], coordinate, atol=1e-12, rtol=0)]
        bounds = f.face_bounds[ids]
        areas = (bounds[:, 1] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 2])
        if z_range is not None and axis != 2:
            lo = np.maximum(bounds[:, 2], z_range[0])
            hi = np.minimum(bounds[:, 3], z_range[1])
            areas *= np.maximum(hi - lo, 0) / (bounds[:, 3] - bounds[:, 2])
            keep = areas > 0
            ids = ids[keep]
            areas = areas[keep]
        return BoundarySurface(
            f.face_owner[ids].astype(np.int64), areas, ids.astype(np.int64)
        )
