import numpy as np
from typing import List

from metahotspot.metahotspot_types import (
    MeshTopology,
    PhysicalFields,
    BoundaryCondition,
)


def apply_pressure_bc(
    bc: BoundaryCondition,
    fields: PhysicalFields,
    is_pressure_boundary: np.ndarray,
) -> None:
    c_ids = bc.c_ids
    fluid_mask = fields.is_fluid[c_ids]
    valid_c_ids = c_ids[fluid_mask]

    if len(valid_c_ids) > 0:
        is_pressure_boundary[valid_c_ids] = True
        fields.pressure[valid_c_ids] = bc.parameters["pressure"]


def apply_temperature_state_bc(bc: BoundaryCondition, fields: PhysicalFields) -> None:
    if len(bc.c_ids) > 0:
        fields.boundary_temperature[bc.c_ids] = bc.parameters["temperature"]


def apply_convection_matrix_bc(
    bc: BoundaryCondition,
    topo: MeshTopology,
    fields: PhysicalFields,
    rows: List[int],
    cols: List[int],
    data: List[float],
    rhs: np.ndarray,
) -> None:
    c_ids, areas = bc.c_ids, bc.areas
    if len(c_ids) == 0:
        return

    h, t_inf = bc.parameters["h"], bc.parameters["T_inf"]
    vols, k = topo.volumes[c_ids], fields.k[c_ids]

    g = areas / ((0.5 * (vols / areas) / k) + (1.0 / h))

    rows.extend(c_ids.tolist())
    cols.extend(c_ids.tolist())
    data.extend((-g).tolist())
    rhs[c_ids] += g * t_inf


def apply_temperature_matrix_bc(
    bc: BoundaryCondition,
    topo: MeshTopology,
    fields: PhysicalFields,
    rows: List[int],
    cols: List[int],
    data: List[float],
    rhs: np.ndarray,
) -> None:
    c_ids, areas = bc.c_ids, bc.areas
    if len(c_ids) == 0:
        return

    temp = bc.parameters["temperature"]
    h_inf = 1e20
    vols, k = topo.volumes[c_ids], fields.k[c_ids]

    g = areas / ((0.5 * (vols / areas) / k) + (1.0 / h_inf))

    rows.extend(c_ids.tolist())
    cols.extend(c_ids.tolist())
    data.extend((-g).tolist())
    rhs[c_ids] += g * temp
