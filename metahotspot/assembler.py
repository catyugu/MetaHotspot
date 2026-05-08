from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from metahotspot.metahotspot_types import (
    MeshTopology,
    PhysicalFields,
    SystemMatrices,
)
from metahotspot.assembler_kernels import (
    overlap_area_kernel,
    find_adjacent_pairs_kernel,
)


class FVMAssembler:
    GEOMETRY_TOLERANCE = 1e-15

    def __init__(
        self, topo: MeshTopology, fields: PhysicalFields, config: dict, stackup: list
    ):
        self.topo, self.fields, self.config, self.stackup = (
            topo,
            fields,
            config,
            stackup,
        )
        self.flow_axes = np.zeros(self.topo.n_cells, dtype=int)

    def assemble(self) -> SystemMatrices:
        self._precompute_flow_axes()  # Compute flow axes from pressure gradient
        A_cond = self._build_conduction_matrix()
        A_bc, b_bc = self._build_boundary_terms()
        A_adv, b_adv = self._build_advection_matrix()
        power_mat, unit_names = self._build_power_matrix()
        return SystemMatrices(
            A_cond + A_bc + A_adv, b_bc + b_adv, power_mat, unit_names
        )

    def _precompute_flow_axes(self) -> None:
        """Based on solved pressure field, infer dominant flow axis for each fluid cell (axis with largest pressure drop)"""
        if not np.any(self.fields.is_fluid):
            return
        p_drops = np.zeros((self.topo.n_cells, 3))
        for c_a, c_b, axis, _ in self._find_adjacent_pairs():
            if self.fields.is_fluid[c_a] and self.fields.is_fluid[c_b]:
                dp = abs(self.fields.pressure[c_a] - self.fields.pressure[c_b])
                p_drops[c_a, axis] = max(p_drops[c_a, axis], dp)
                p_drops[c_b, axis] = max(p_drops[c_b, axis], dp)

        fluid_mask = self.fields.is_fluid
        self.flow_axes[fluid_mask] = np.argmax(p_drops[fluid_mask], axis=1)

    def _find_adjacent_pairs(self):
        """Generator that yields adjacent cell pairs with their overlap area and normal axis."""
        c_a_arr, c_b_arr, axis_arr, area_arr, count = find_adjacent_pairs_kernel(
            self.topo.boxes
        )
        for i in range(count):
            yield c_a_arr[i], c_b_arr[i], axis_arr[i], area_arr[i]

    @staticmethod
    def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
        return overlap_area_kernel(box_a, box_b, axis)

    def _build_conduction_matrix(self) -> sp.csr_matrix:
        rows, cols, data, n = [], [], [], self.topo.n_cells
        for c_a, c_b, axis, area in self._find_adjacent_pairs():
            g = 1.0 / self._calc_resistance(c_a, c_b, axis, area)
            rows.extend([c_a, c_b, c_a, c_b])
            cols.extend([c_a, c_b, c_b, c_a])
            data.extend([-g, -g, g, g])
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    def _calc_resistance(self, c_a: int, c_b: int, axis: int, area: float) -> float:
        fluid_a, fluid_b = self.fields.is_fluid[c_a], self.fields.is_fluid[c_b]
        if fluid_a != fluid_b:
            f_id, s_id = (c_a, c_b) if fluid_a else (c_b, c_a)
            flow_axis = self.flow_axes[f_id]
            ax_w = (flow_axis + 1) % 3
            ax_h = (flow_axis + 2) % 3
            Nu = self._compute_nusselt(f_id, flow_axis)
            d_h = (
                2 * self.topo.dims[f_id, ax_w] * self.topo.dims[f_id, ax_h]
                / (self.topo.dims[f_id, ax_w] + self.topo.dims[f_id, ax_h])
            )
            h_f = (Nu * self.fields.k[f_id]) / d_h if d_h > 0 else 1e-6
            return self.topo.dims[s_id, axis] / (
                2.0 * self.fields.k[s_id] * area
            ) + 1.0 / (h_f * area)
        return (self.topo.dims[c_a, axis] / (2.0 * self.fields.k[c_a] * area)) + (
            self.topo.dims[c_b, axis] / (2.0 * self.fields.k[c_b] * area)
        )

    def _compute_nusselt(self, c_id: int, flow_axis: int) -> float:
        ax_w = (flow_axis + 1) % 3
        ax_h = (flow_axis + 2) % 3
        w, h = sorted([self.topo.dims[c_id, ax_w], self.topo.dims[c_id, ax_h]])
        AR = w / h if h > 0 else 1.0
        return 8.235 * (
            1
            - 2.0421 * AR
            + 3.0853 * AR**2
            - 2.4765 * AR**3
            + 1.0578 * AR**4
            - 0.1861 * AR**5
        )

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n, rhs, rows, cols, data = (
            self.topo.n_cells,
            np.zeros(self.topo.n_cells),
            [],
            [],
            [],
        )
        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") != "convection":
                continue
            h, t_inf, target, face_key = (
                float(bc["h"]),
                float(bc["T_inf"]),
                bc.get("target"),
                bc.get("face", ""),
            )
            for c_id, _, area in self.topo.boundary_faces.get(face_key, []):
                if target and target != self.fields.layer_names[c_id]:
                    continue
                g = area / (
                    (0.5 * (self.topo.volumes[c_id] / area) / self.fields.k[c_id])
                    + (1.0 / h)
                )
                rows.append(c_id)
                cols.append(c_id)
                data.append(-g)
                rhs[c_id] += g * t_inf
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _build_advection_matrix(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n, rows, cols, data, rhs, tol = (
            self.topo.n_cells,
            [],
            [],
            [],
            np.zeros(self.topo.n_cells),
            self.GEOMETRY_TOLERANCE,
        )
        if not np.any(self.fields.is_fluid):
            return sp.csr_matrix((n, n)), rhs

        net_outflux = np.zeros(n)
        for c_a, c_b, axis, area in self._find_adjacent_pairs():
            if not (self.fields.is_fluid[c_a] and self.fields.is_fluid[c_b]):
                continue

            # Use axis-specific hydroC for fluid-fluid pairs
            hc_a = self.fields.hydroC[c_a, axis]
            hc_b = self.fields.hydroC[c_b, axis]
            sum_hc = hc_a + hc_b
            C_eff = (
                2.0 * hc_a * hc_b / sum_hc
                if sum_hc > 0
                else 0.0
            )
            mass_flux = (
                (self.fields.pressure[c_a] - self.fields.pressure[c_b])
                * C_eff
                * (self.fields.density[c_a] + self.fields.density[c_b])
                * 0.5
            )
            net_outflux[c_a], net_outflux[c_b] = (
                net_outflux[c_a] + mass_flux,
                net_outflux[c_b] - mass_flux,
            )
            if abs(mass_flux) > tol:
                up, dn = (c_a, c_b) if mass_flux > 0 else (c_b, c_a)
                adv = abs(mass_flux) * self.fields.cp[up]
                rows.extend([up, dn])
                cols.extend([up, up])
                data.extend([-adv, adv])

        fluid_ids = np.where(self.fields.is_fluid)[0]
        for c_id in fluid_ids:
            influx = net_outflux[c_id]
            if influx > tol and not np.isnan(self.fields.inlet_temperature[c_id]):
                rhs[c_id] += (
                    influx * self.fields.cp[c_id] * self.fields.inlet_temperature[c_id]
                )
            elif influx < -tol:
                rows.append(c_id)
                cols.append(c_id)
                data.append(influx * self.fields.cp[c_id])
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _build_power_matrix(self) -> Tuple[sp.csr_matrix, List[str]]:
        active_units, z_cursor = [], 0.0
        for l in self.stackup:
            if l.active:
                for u in l.units:
                    active_units.append(
                        {
                            "name": u.name,
                            "lx": u.lx,
                            "ly": u.ly,
                            "lz": z_cursor,
                            "dx": u.dx,
                            "dy": u.dy,
                            "dz": l.thickness,
                        }
                    )
            z_cursor += l.thickness
        n_cells = self.topo.n_cells
        if not active_units:
            return sp.csr_matrix((n_cells, 0)), []
        rows, cols, data, boxes = [], [], [], self.topo.boxes
        for j, u in enumerate(active_units):
            vol_u = u["dx"] * u["dy"] * u["dz"]
            if vol_u <= 0:
                continue
            u_min, u_max = np.array([u["lx"], u["ly"], u["lz"]]), np.array(
                [u["lx"], u["ly"], u["lz"]]
            ) + np.array([u["dx"], u["dy"], u["dz"]])
            intersect = np.prod(
                np.maximum(
                    0, np.minimum(boxes[:, 3:], u_max) - np.maximum(boxes[:, :3], u_min)
                ),
                axis=1,
            )
            valid = np.where(intersect > self.GEOMETRY_TOLERANCE)[0]
            rows.extend(valid)
            cols.extend([j] * len(valid))
            data.extend(intersect[valid] / vol_u)
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(n_cells, len(active_units))
        ), [u["name"] for u in active_units]
