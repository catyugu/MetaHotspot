import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from metahotspot.metahotspot_types import MeshTopology, PhysicalFields
from metahotspot.boundary_operators import resolve_boundary_cells, apply_pressure_bc


class FluidPreprocessor:
    def __init__(self, config: dict):
        self.config = config

    def solve_flow(self, topo: MeshTopology, fields: PhysicalFields) -> None:
        if not np.any(fields.is_fluid):
            return

        is_pressure_boundary = np.zeros(topo.n_cells, dtype=bool)
        self._init_cell_hydro_properties(topo, fields)
        self._apply_pressure_boundary_conditions(topo, fields, is_pressure_boundary)
        self._solve_pressure(topo, fields, is_pressure_boundary)

    def _init_cell_hydro_properties(
        self, topo: MeshTopology, fields: PhysicalFields
    ) -> None:
        m = fields.is_fluid & (fields.dynamic_viscosity > 0)
        if not np.any(m):
            return
        v = fields.dynamic_viscosity[m]

        for axis in range(3):
            ax_w, ax_h = (axis + 1) % 3, (axis + 2) % 3
            L, w, h = topo.dims[m, axis], topo.dims[m, ax_w], topo.dims[m, ax_h]
            hydroC_axis = np.zeros(np.sum(m))
            cond_eq, cond_gt = np.abs(h - w) < 1e-10, h > w
            cond_lt = ~(cond_eq | cond_gt)

            hydroC_axis[cond_eq] = (0.42229 * h[cond_eq] ** 4) / (
                12 * v[cond_eq] * L[cond_eq]
            )
            hydroC_axis[cond_gt] = (
                (1 - 0.63 * (w[cond_gt] / h[cond_gt])) * w[cond_gt] ** 3 * h[cond_gt]
            ) / (12 * v[cond_gt] * L[cond_gt])
            hydroC_axis[cond_lt] = (
                (1 - 0.63 * (h[cond_lt] / w[cond_lt])) * h[cond_lt] ** 3 * w[cond_lt]
            ) / (12 * v[cond_lt] * L[cond_lt])
            fields.hydroC[m, axis] = hydroC_axis

    def _apply_pressure_boundary_conditions(
        self, topo: MeshTopology, fields: PhysicalFields, is_p_bound: np.ndarray
    ) -> None:
        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") == "pressure":
                c_ids, _ = resolve_boundary_cells(
                    topo, fields, bc["face"], bc.get("target", "")
                )
                apply_pressure_bc(c_ids, bc["parameters"], fields, is_p_bound)

    def _solve_pressure(
        self, topo: MeshTopology, fields: PhysicalFields, is_p_bound: np.ndarray
    ) -> None:
        fluid_ids = np.where(fields.is_fluid)[0]
        if len(fluid_ids) == 0:
            return
        n_fluid, g2f = len(fluid_ids), np.full(topo.n_cells, -1, dtype=int)
        g2f[fluid_ids] = np.arange(n_fluid)
        rows, cols, data, b_p, diag_C, is_p_f = (
            [],
            [],
            [],
            np.zeros(n_fluid),
            np.zeros(n_fluid),
            is_p_bound[fluid_ids],
        )

        b_idx = np.where(is_p_f)[0]
        rows.extend(b_idx)
        cols.extend(b_idx)
        data.extend(np.ones(len(b_idx)))
        b_p[b_idx] = fields.pressure[fluid_ids][b_idx]

        for c0, c1 in topo.internal_faces:
            i0, i1 = g2f[c0], g2f[c1]
            if i0 == -1 or i1 == -1:
                continue
            axis = np.argmax(np.abs(topo.centers[c1] - topo.centers[c0]))
            hc0, hc1 = fields.hydroC[c0, axis], fields.hydroC[c1, axis]
            sum_hc = hc0 + hc1
            C_eff = (2.0 * hc0 * hc1 / sum_hc) if sum_hc > 0 else 0.0
            if not is_p_bound[c0]:
                rows.append(i0)
                cols.append(i1)
                data.append(C_eff)
                diag_C[i0] += C_eff
            if not is_p_bound[c1]:
                rows.append(i1)
                cols.append(i0)
                data.append(C_eff)
                diag_C[i1] += C_eff

        for i in range(n_fluid):
            if not is_p_f[i]:
                rows.append(i)
                cols.append(i)
                data.append(-diag_C[i])

        try:
            fields.pressure[fluid_ids] = splinalg.spsolve(
                sp.csr_matrix((data, (rows, cols)), shape=(n_fluid, n_fluid)), b_p
            )
        except Exception as e:
            print(f"[WARNING] Pressure solve failed: {e}")
