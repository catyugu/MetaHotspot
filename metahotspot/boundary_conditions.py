import re
import numpy as np
from typing import Tuple, List

from metahotspot.metahotspot_types import (
    MeshTopology,
    PhysicalFields,
    BoundaryCondition,
)


def resolve_boundary_cells(
    topo: MeshTopology, fields: PhysicalFields, face_key: str, target_regex: str
) -> Tuple[np.ndarray, np.ndarray]:
    if face_key not in topo.boundary_faces:
        return np.array([], dtype=int), np.array([], dtype=float)

    c_ids, _, areas = topo.boundary_faces[face_key]

    if not target_regex:
        return c_ids, areas

    pattern = re.compile(target_regex)

    layer_names = [fields.layer_name_map[fields.layer_ids[cid]] for cid in c_ids]
    unit_names = [fields.unit_name_map[fields.unit_ids[cid]] for cid in c_ids]

    mask = np.array(
        [
            bool(pattern.match(l_name)) or bool(pattern.match(u_name))
            for l_name, u_name in zip(layer_names, unit_names)
        ]
    )

    return c_ids[mask], areas[mask]


def apply_pressure_bc(
    c_ids: np.ndarray,
    bc: BoundaryCondition,
    fields: PhysicalFields,
    is_pressure_boundary: np.ndarray,
) -> None:
    fluid_mask = fields.is_fluid[c_ids]
    valid_c_ids = c_ids[fluid_mask]

    if len(valid_c_ids) > 0:
        is_pressure_boundary[valid_c_ids] = True
        fields.pressure[valid_c_ids] = bc.parameters["pressure"]


def apply_temperature_state_bc(
    c_ids: np.ndarray, bc: BoundaryCondition, fields: PhysicalFields
) -> None:
    fields.boundary_temperature[c_ids] = bc.parameters["temperature"]


def apply_convection_matrix_bc(
    c_ids: np.ndarray,
    areas: np.ndarray,
    bc: BoundaryCondition,
    topo: MeshTopology,
    fields: PhysicalFields,
    rows: List[int],
    cols: List[int],
    data: List[float],
    rhs: np.ndarray,
) -> None:
    h, t_inf = bc.parameters["h"], bc.parameters["T_inf"]
    vols, k = topo.volumes[c_ids], fields.k[c_ids]

    g = areas / ((0.5 * (vols / areas) / k) + (1.0 / h))

    rows.extend(c_ids.tolist())
    cols.extend(c_ids.tolist())
    data.extend((-g).tolist())
    rhs[c_ids] += g * t_inf


def apply_temperature_matrix_bc(
    c_ids: np.ndarray,
    areas: np.ndarray,
    bc: BoundaryCondition,
    topo: MeshTopology,
    fields: PhysicalFields,
    rows: List[int],
    cols: List[int],
    data: List[float],
    rhs: np.ndarray,
) -> None:
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
