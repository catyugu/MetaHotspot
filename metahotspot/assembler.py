from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from metahotspot.metahotspot_types import MeshTopology, PhysicalFields, SystemMatrices
from metahotspot.assembler_kernels import (
    find_adjacent_pairs_kernel,
    build_cond_coo_kernel,
    build_adv_coo_kernel,
)


class FVMAssembler:
    GEOMETRY_TOLERANCE = 1e-12

    def __init__(
        self, topo: MeshTopology, fields: PhysicalFields, config: dict, stackup: list
    ):
        self.topo, self.fields, self.config, self.stackup = (
            topo,
            fields,
            config,
            stackup,
        )
        self.flow_axes = np.zeros(self.topo.n_cells, dtype=np.int32)

        # 【关键优化】只执行一次 O(N log N) 的邻居查找并缓存 SoA 数组
        self._c_a, self._c_b, self._axes, self._areas, self._pair_count = (
            find_adjacent_pairs_kernel(self.topo.boxes)
        )

    def assemble(self) -> SystemMatrices:
        self._precompute_flow_axes()
        A_cond = self._build_conduction_matrix()
        A_bc, b_bc = self._build_boundary_terms()
        A_adv, b_adv = self._build_advection_matrix()
        power_mat, unit_names = self._build_power_matrix()

        return SystemMatrices(
            A_total=A_cond + A_bc + A_adv,
            b_total=b_bc + b_adv,
            power_matrix=power_mat,
            unit_names=unit_names,
        )

    def _precompute_flow_axes(self) -> None:
        if not np.any(self.fields.is_fluid):
            return

        p_drops = np.zeros((self.topo.n_cells, 3))
        # 向量化处理预计算的数据
        fluid_mask_a = self.fields.is_fluid[self._c_a]
        fluid_mask_b = self.fields.is_fluid[self._c_b]
        valid = fluid_mask_a & fluid_mask_b

        v_ca = self._c_a[valid]
        v_cb = self._c_b[valid]
        v_axes = self._axes[valid]
        dps = np.abs(self.fields.pressure[v_ca] - self.fields.pressure[v_cb])

        np.maximum.at(p_drops[:, 0], v_ca[v_axes == 0], dps[v_axes == 0])
        np.maximum.at(p_drops[:, 1], v_ca[v_axes == 1], dps[v_axes == 1])
        np.maximum.at(p_drops[:, 2], v_ca[v_axes == 2], dps[v_axes == 2])

        fluid_mask = self.fields.is_fluid
        self.flow_axes[fluid_mask] = np.argmax(p_drops[fluid_mask], axis=1)

    def _build_conduction_matrix(self) -> sp.csr_matrix:
        rows, cols, data = build_cond_coo_kernel(
            self._c_a,
            self._c_b,
            self._axes,
            self._areas,
            self._pair_count,
            self.topo.dims,
            self.fields.k,
            self.fields.is_fluid,
            self.flow_axes,
        )
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(self.topo.n_cells, self.topo.n_cells)
        )

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n = self.topo.n_cells
        rhs = np.zeros(n)
        rows, cols, data = [], [], []

        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") != "convection":
                continue
            h, t_inf, target, face_key = (
                float(bc["h"]),
                float(bc["T_inf"]),
                bc.get("target"),
                bc.get("face", ""),
            )

            if face_key in self.topo.boundary_faces:
                c_ids, normals, areas = self.topo.boundary_faces[face_key]
                for i, c_id in enumerate(c_ids):
                    # 检查 target
                    if (
                        target
                        and target
                        != self.fields.layer_name_map[self.fields.layer_ids[c_id]]
                    ):
                        continue
                    area = areas[i]
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
        n = self.topo.n_cells
        rhs = np.zeros(n)
        if not np.any(self.fields.is_fluid):
            return sp.csr_matrix((n, n)), rhs

        rows, cols, data, net_outflux = build_adv_coo_kernel(
            self._c_a,
            self._c_b,
            self._axes,
            self._pair_count,
            self.fields.is_fluid,
            self.fields.pressure,
            self.fields.density,
            self.fields.hydroC,
            self.fields.cp,
        )

        # 处理进出口边界条件
        fluid_ids = np.where(self.fields.is_fluid)[0]
        influxes = net_outflux[fluid_ids]

        # 边界入流 (Influx > 0)
        in_mask = influxes > self.GEOMETRY_TOLERANCE
        valid_in_ids = fluid_ids[in_mask]
        valid_influxes = influxes[in_mask]
        valid_temps = self.fields.inlet_temperature[valid_in_ids]
        temp_mask = ~np.isnan(valid_temps)
        rhs[valid_in_ids[temp_mask]] += (
            valid_influxes[temp_mask]
            * self.fields.cp[valid_in_ids[temp_mask]]
            * valid_temps[temp_mask]
        )

        # 边界出流 (Influx < 0)
        out_mask = influxes < -self.GEOMETRY_TOLERANCE
        out_ids = fluid_ids[out_mask]
        out_vals = influxes[out_mask] * self.fields.cp[out_ids]

        if len(out_ids) > 0:
            rows = np.concatenate([rows, out_ids])
            cols = np.concatenate([cols, out_ids])
            data = np.concatenate([data, out_vals])

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
