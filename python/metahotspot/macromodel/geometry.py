"""Explicit cell geometry and canonical facet topology, independent of mesh origin."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from metahotspot._compiled_data import CompiledFields
from metahotspot.enums import Face


@dataclass(frozen=True)
class BoundarySurface:
    """A selected set of canonical facets; multiple facets may share one cell."""

    cell_ids: np.ndarray
    areas: np.ndarray
    facet_ids: np.ndarray


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

    def surface(self, face: Face, coordinate: float | None = None):
        """Select complete exterior facets on a requested plane."""
        face = Face(face)
        axis = int(face) // 2
        sign = -1 if int(face) % 2 == 0 else 1
        f = self.fields
        invalid = np.iinfo(f.facet_minus.dtype).max
        cells = f.facet_minus if sign > 0 else f.facet_plus
        missing = f.facet_plus if sign > 0 else f.facet_minus
        ids = np.flatnonzero((missing == invalid) & (f.facet_axis == axis))
        if coordinate is None:
            coordinate = (
                f.cell_min[:, axis].min() if sign < 0 else f.cell_max[:, axis].max()
            )
        ids = ids[np.isclose(f.facet_coordinate[ids], coordinate, atol=1e-9, rtol=0)]
        bounds = f.facet_bounds[ids]
        areas = (bounds[:, 1] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 2])
        return BoundarySurface(cells[ids].astype(np.int64), areas, ids.astype(np.int64))
