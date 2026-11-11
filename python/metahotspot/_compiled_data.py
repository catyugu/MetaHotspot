"""Stable value contracts returned by compiled-model operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Operators(NamedTuple):
    """K, C (SciPy CSR), f of the linearised system: C * dx/dt + K * x = f."""

    K: object
    C: object
    f: np.ndarray


@dataclass(frozen=True)
class CompiledFields:
    """Compiled geometry, material indices and source quadrature in CellId order."""

    cell_min: np.ndarray
    cell_max: np.ndarray
    material_id: np.ndarray
    boundary_owner: np.ndarray
    boundary_side: np.ndarray
    boundary_axis: np.ndarray
    boundary_coordinate: np.ndarray
    boundary_bounds: np.ndarray
    projection_cell: np.ndarray
    projection_source: np.ndarray
    projection_volume: np.ndarray
    projection_center: np.ndarray
