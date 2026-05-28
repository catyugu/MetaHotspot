from typing import List, Tuple
import numpy as np
import scipy.sparse as sp

from metahotspot.metahotspot_types import (
    MeshTopology,
    PhysicalFields,
    SystemMatrices,
    BoundaryCondition,
    LayerRegion,
)
from metahotspot.assembler_kernels import (
    find_adjacent_pairs_kernel,
    build_cond_coo_kernel,
    build_adv_coo_kernel,
)
from metahotspot.boundary_conditions import (
    apply_temperature_state_bc,
    apply_convection_matrix_bc,
    apply_temperature_matrix_bc,
)


class FVMAssembler:
    GEOMETRY_TOLERANCE = 1e-15

    def __init__(
        self,
        topo: MeshTopology,
        fields: PhysicalFields,
        boundary_conditions: List[BoundaryCondition],
        layer_regions: List[LayerRegion],
    ):
        self.topo = topo
        self.fields = fields
        self.boundary_conditions = boundary_conditions
        self.layer_regions = layer_regions
        self.flow_axes = np.zeros(self.topo.n_cells, dtype=np.int32)
        self._c_a, self._c_b, self._axes, self._areas, self._pair_count = (
            find_adjacent_pairs_kernel(self.topo.boxes)
        )

    def assemble(self) -> SystemMatrices:
        self._precompute_flow_axes()
        self._apply_temperature_boundaries()
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
        valid = self.fields.is_fluid[self._c_a] & self.fields.is_fluid[self._c_b]
        v_ca, v_axes = self._c_a[valid], self._axes[valid]
        dps = np.abs(
            self.fields.pressure[v_ca] - self.fields.pressure[self._c_b[valid]]
        )

        for ax in range(3):
            np.maximum.at(p_drops[:, ax], v_ca[v_axes == ax], dps[v_axes == ax])
        self.flow_axes[self.fields.is_fluid] = np.argmax(
            p_drops[self.fields.is_fluid], axis=1
        )

    def _apply_temperature_boundaries(self) -> None:
        for bc in self.boundary_conditions:
            if bc.type == "temperature":
                apply_temperature_state_bc(bc, self.fields)

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
        rhs, rows, cols, data = np.zeros(n), [], [], []

        for bc in self.boundary_conditions:
            if bc.type == "convection":
                apply_convection_matrix_bc(
                    bc, self.topo, self.fields, rows, cols, data, rhs
                )
            elif bc.type == "temperature":
                apply_temperature_matrix_bc(
                    bc, self.topo, self.fields, rows, cols, data, rhs
                )

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

        fluid_ids = np.where(self.fields.is_fluid)[0]
        influxes = net_outflux[fluid_ids]
        in_mask = influxes > self.GEOMETRY_TOLERANCE
        v_in_ids = fluid_ids[in_mask]
        v_temps = self.fields.boundary_temperature[v_in_ids]
        temp_mask = ~np.isnan(v_temps)
        rhs[v_in_ids[temp_mask]] += (
            influxes[in_mask][temp_mask]
            * self.fields.cp[v_in_ids[temp_mask]]
            * v_temps[temp_mask]
        )

        out_mask = influxes < -self.GEOMETRY_TOLERANCE
        o_ids = fluid_ids[out_mask]
        if len(o_ids) > 0:
            o_vals = influxes[out_mask] * self.fields.cp[o_ids]
            rows = np.concatenate([rows, o_ids])
            cols = np.concatenate([cols, o_ids])
            data = np.concatenate([data, o_vals])

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _build_power_matrix(self) -> Tuple[sp.csr_matrix, List[str]]:
        n = self.topo.n_cells
        active_units = [
            (u, lr.lz, lr.dz)
            for lr in self.layer_regions
            if lr.is_active
            for u in lr.units
        ]

        if not active_units:
            return sp.csr_matrix((n, 0)), []

        rows, cols, data, boxes = [], [], [], self.topo.boxes
        unit_names = []

        for j, (u, lz, dz) in enumerate(active_units):
            vol_u = u.dx * u.dy * dz
            unit_names.append(u.name)
            if vol_u <= 0:
                continue

            u_min = np.array([u.lx, u.ly, lz])
            u_max = u_min + np.array([u.dx, u.dy, dz])

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

        return (
            sp.csr_matrix((data, (rows, cols)), shape=(n, len(active_units))),
            unit_names,
        )
