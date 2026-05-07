import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from metahotspot.metahotspot_types import MeshTopology, PhysicalFields

class FluidPreprocessor:
    """
    Specialized for calculating and solidifying fluid dynamic fields (pressure, convection coefficients)
    before thermal assembly.
    """
    def __init__(self, config: dict):
        self.config = config

    def solve_flow(self, topo: MeshTopology, fields: PhysicalFields) -> None:
        if not np.any(fields.is_fluid):
            return
        
        # Temporary state for boundary condition tracking during flow solve
        is_pressure_boundary = np.zeros(topo.n_cells, dtype=bool)
        
        self._init_cell_hydro_properties(topo, fields)
        self._apply_pressure_boundary_conditions(topo, fields, is_pressure_boundary)
        self._solve_pressure(topo, fields, is_pressure_boundary)

    def _init_cell_hydro_properties(self, topo: MeshTopology, fields: PhysicalFields) -> None:
        m = fields.is_fluid & (fields.dynamic_viscosity > 0)
        if not np.any(m):
            return
        w, L, h, v = (
            topo.dims[m, 0],
            topo.dims[m, 1],
            topo.dims[m, 2],
            fields.dynamic_viscosity[m],
        )
        hydroC = np.zeros(np.sum(m))
        cond_eq, cond_gt = np.abs(h - w) < 1e-10, h > w
        cond_lt = ~(cond_eq | cond_gt)
        hydroC[cond_eq] = (0.42229 * h[cond_eq] ** 4) / (12 * v[cond_eq] * L[cond_eq])
        hydroC[cond_gt] = (
            (1 - 0.63 * (w[cond_gt] / h[cond_gt])) * w[cond_gt] ** 3 * h[cond_gt]
        ) / (12 * v[cond_gt] * L[cond_gt])
        hydroC[cond_lt] = (
            (1 - 0.63 * (h[cond_lt] / w[cond_lt])) * h[cond_lt] ** 3 * w[cond_lt]
        ) / (12 * v[cond_lt] * L[cond_lt])
        fields.hydroC[m] = hydroC

    def _apply_pressure_boundary_conditions(
        self, topo: MeshTopology, fields: PhysicalFields, is_pressure_boundary: np.ndarray
    ) -> None:
        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") != "pressure":
                continue
            pressure, temp, face, target = (
                float(bc["pressure"]),
                float(bc.get("temperature", np.nan)),
                bc.get("face", ""),
                bc.get("target"),
            )
            for c_id, _, _ in topo.boundary_faces.get(face, []):
                if fields.is_fluid[c_id] and (
                    not target or fields.layer_names[c_id] == target
                ):
                    (
                        is_pressure_boundary[c_id],
                        fields.pressure[c_id],
                        fields.inlet_temperature[c_id],
                    ) = (True, pressure, temp)

    def _solve_pressure(
        self, topo: MeshTopology, fields: PhysicalFields, is_pressure_boundary: np.ndarray
    ) -> None:
        fluid_ids = np.where(fields.is_fluid)[0]
        if len(fluid_ids) == 0:
            return
        n_fluid, global_to_fluid = len(fluid_ids), np.full(
            topo.n_cells, -1, dtype=int
        )
        global_to_fluid[fluid_ids] = np.arange(n_fluid)
        rows, cols, data, b_p, diag_C, is_p_bound = (
            [],
            [],
            [],
            np.zeros(n_fluid),
            np.zeros(n_fluid),
            is_pressure_boundary[fluid_ids],
        )
        bound_idx = np.where(is_p_bound)[0]
        rows.extend(bound_idx)
        cols.extend(bound_idx)
        data.extend(np.ones(len(bound_idx)))
        b_p[bound_idx] = fields.pressure[fluid_ids][bound_idx]
        for c0, c1 in topo.internal_faces:
            i0, i1 = global_to_fluid[c0], global_to_fluid[c1]
            if i0 == -1 or i1 == -1:
                continue
            sum_hc = fields.hydroC[c0] + fields.hydroC[c1]
            C_eff = (
                2.0 * fields.hydroC[c0] * fields.hydroC[c1] / sum_hc
                if sum_hc > 0
                else 0.0
            )
            if not is_pressure_boundary[c0]:
                rows.append(i0)
                cols.append(i1)
                data.append(C_eff)
                diag_C[i0] += C_eff
            if not is_pressure_boundary[c1]:
                rows.append(i1)
                cols.append(i0)
                data.append(C_eff)
                diag_C[i1] += C_eff
        for i in range(n_fluid):
            if not is_p_bound[i]:
                rows.append(i)
                cols.append(i)
                data.append(-diag_C[i])
        try:
            fields.pressure[fluid_ids] = splinalg.spsolve(
                sp.csr_matrix((data, (rows, cols)), shape=(n_fluid, n_fluid)), b_p
            )
        except Exception as e:
            print(f"[WARNING] Pressure solve failed: {e}")
