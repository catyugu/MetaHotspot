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
    """Compiled cell and structured-grid arrays in their native order."""

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
