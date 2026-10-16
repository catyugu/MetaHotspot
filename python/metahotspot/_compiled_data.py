"""Stable value contracts returned by compiled-model operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Operators(NamedTuple):
    """K, C, f of the linearised system: C * dx/dt + K * x = f."""

    K: object
    C: object
    f: np.ndarray


@dataclass(frozen=True)
class CellFields:
    """Compiled geometry, material indices and source quadrature in CellId order."""

    cell_min: np.ndarray
    cell_max: np.ndarray
    material_id: np.ndarray
    face_owner: np.ndarray
    face_neighbor: np.ndarray
    face_axis: np.ndarray
    face_sign: np.ndarray
    face_min: np.ndarray
    face_max: np.ndarray
    face_area: np.ndarray
    owner_distance: np.ndarray
    neighbor_distance: np.ndarray
    projection_cell: np.ndarray
    projection_source: np.ndarray
    projection_volume: np.ndarray
    projection_center: np.ndarray
