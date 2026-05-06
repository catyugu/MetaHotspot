import math
import os
from typing import List, Tuple

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from metahotspot.model25d import load_config, load_stackup


def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
    axes = [(1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 3, 4)][axis]
    d1 = min(box_a[axes[2]], box_b[axes[2]]) - max(box_a[axes[0]], box_b[axes[0]])
    d2 = min(box_a[axes[3]], box_b[axes[3]]) - max(box_a[axes[1]], box_b[axes[1]])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


class FVMSolver:
    GEOMETRY_TOLERANCE = 1e-15

    def __init__(self, config_path: str) -> None:
        self.base_dir = os.path.dirname(config_path)
        self.config = load_config(config_path)
        self.mesh_path = os.path.join(self.base_dir, self.config["mesh_file_path"])
        self.mesh = meshio.read(self.mesh_path)
        self.stackup = load_stackup(self.config, self.base_dir)

        self._init_default_props()
        self._prepare_mesh()
        self._precompute_power_matrix()

    def _init_default_props(self) -> None:
        """单一的回退处理点：使用 config 中解析好的默认材料"""
        def_mat = self.config["materials"]["default_solid"]
        self.default_props = (
            "default",
            "default_layer",
            def_mat["k"],
            def_mat["cp"],
            def_mat["density"],
            def_mat.get("dynamic_viscosity", 0.0),
            def_mat.get("fluid", False),
        )

    def _find_cell_props(self, center: np.ndarray) -> tuple:
        tol = self.GEOMETRY_TOLERANCE
        z = center[2]
        for layer, z_min, z_max in self.layer_bounds:
            if not (z_min - tol <= z <= z_max + tol):
                continue
            return self._find_props_in_layer(layer, center)
        return self.default_props

    def _find_props_in_layer(self, layer, center: np.ndarray) -> tuple:
        tol = self.GEOMETRY_TOLERANCE
        x, y = center[0], center[1]
        for u in layer.units:
            if (u.lx - tol <= x <= u.lx + u.dx + tol) and (
                u.ly - tol <= y <= u.ly + u.dy + tol
            ):
                return (
                    u.name,
                    layer.name,
                    u.k,
                    u.cp,
                    u.density,
                    u.dynamic_viscosity,
                    u.is_fluid,
                )
        return (
            "",
            layer.name,
            layer.k,
            layer.cp,
            layer.density,
            layer.dynamic_viscosity,
            layer.is_fluid,
        )

    def _prepare_mesh(self) -> None:
        print("[INFO] Preparing mesh data ...")
        hex_blocks = [b.data for b in self.mesh.cells if b.type == "hexahedron"]
        if not hex_blocks:
            raise ValueError("No hexahedron cells found in mesh")

        hex_data = np.vstack(hex_blocks)
        physical_tags = self.mesh.cell_data_dict.get("gmsh:physical", {}).get(
            "hexahedron", np.full(len(hex_data), -1)
        )

        coords = self.mesh.points[hex_data]
        lowers, uppers = np.min(coords, axis=1), np.max(coords, axis=1)
        centers, dims, vols = (
            (lowers + uppers) * 0.5,
            uppers - lowers,
            np.prod(uppers - lowers, axis=1),
        )

        sorted_indices = self._compute_morton_sort(lowers, uppers, centers)
        self.n_cells = len(centers)

        # Structure of Arrays (SoA) layout initialization
        self.c_original_id = sorted_indices.astype(int)
        self.c_center = centers[sorted_indices]
        self.c_dims = dims[sorted_indices]
        self.c_box = np.hstack([lowers[sorted_indices], uppers[sorted_indices]])
        self.c_tag = physical_tags[sorted_indices].astype(int)
        self.c_vol = vols[sorted_indices]

        # Allocate property arrays
        self.c_name, self.c_layer_name = np.empty(self.n_cells, dtype=object), np.empty(
            self.n_cells, dtype=object
        )
        self.c_k, self.c_cp, self.c_density = (
            np.empty(self.n_cells),
            np.empty(self.n_cells),
            np.empty(self.n_cells),
        )
        self.c_dynamic_viscosity = np.empty(self.n_cells)
        self.c_is_fluid = np.empty(self.n_cells, dtype=bool)

        # State arrays
        self.c_hydroC = np.zeros(self.n_cells)
        self.c_pressure = np.zeros(self.n_cells)
        self.c_temperature = np.zeros(self.n_cells)
        self.c_inlet_temperature = np.full(self.n_cells, np.nan)
        self.c_is_pressure_boundary = np.zeros(self.n_cells, dtype=bool)

        z_cursor = 0.0
        self.layer_bounds = []
        for l in self.stackup:
            self.layer_bounds.append((l, z_cursor, z_cursor + l.thickness))
            z_cursor += l.thickness

        for new_id in range(self.n_cells):
            props = self._find_cell_props(self.c_center[new_id])
            self.c_name[new_id], self.c_layer_name[new_id] = props[0], props[1]
            self.c_k[new_id], self.c_cp[new_id], self.c_density[new_id] = (
                props[2],
                props[3],
                props[4],
            )
            self.c_dynamic_viscosity[new_id], self.c_is_fluid[new_id] = (
                props[5],
                props[6],
            )

        self.orig_to_new_id = np.empty(self.n_cells, dtype=int)
        self.orig_to_new_id[self.c_original_id] = np.arange(self.n_cells)

        self._extract_faces(hex_data, sorted_indices)

    def _compute_morton_sort(self, lowers, uppers, centers) -> np.ndarray:
        b_min = np.min(lowers, axis=0)
        diff = np.where((d := np.max(uppers, axis=0) - b_min) == 0, 1, d)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)

        morton_keys = np.zeros(len(centers), dtype=int)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0] >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1] >> i) & 1) << (3 * i + 1)
            morton_keys |= ((norm_centers[:, 2] >> i) & 1) << (3 * i + 2)
        return np.argsort(morton_keys)

    def _extract_faces(self, hex_data: np.ndarray, sorted_indices: np.ndarray) -> None:
        self.face_to_cells = {}
        for new_id, orig_id in enumerate(sorted_indices):
            self._add_cell_faces(new_id, hex_data[orig_id])

        self.internal_faces_list = [
            c for c in self.face_to_cells.values() if len(c) == 2
        ]
        boundary_faces_all = {
            f: c[0] for f, c in self.face_to_cells.items() if len(c) == 1
        }

        self.boundary_faces_by_direction = {
            "+X": [],
            "-X": [],
            "+Y": [],
            "-Y": [],
            "+Z": [],
            "-Z": [],
        }
        for f, c_id in boundary_faces_all.items():
            self._classify_boundary_face(f, c_id)

    def _add_cell_faces(self, c_id: int, nodes: np.ndarray) -> None:
        faces = [
            (nodes[0], nodes[3], nodes[2], nodes[1]),
            (nodes[4], nodes[5], nodes[6], nodes[7]),
            (nodes[0], nodes[1], nodes[5], nodes[4]),
            (nodes[3], nodes[7], nodes[6], nodes[2]),
            (nodes[0], nodes[4], nodes[7], nodes[3]),
            (nodes[1], nodes[2], nodes[6], nodes[5]),
        ]
        for f in faces:
            self.face_to_cells.setdefault(tuple(sorted(f)), []).append(c_id)

    def _classify_boundary_face(self, f: tuple, c_id: int) -> None:
        pts = self.mesh.points[list(f)]
        cross = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        area = np.linalg.norm(cross)
        if area < self.GEOMETRY_TOLERANCE:
            return

        normal = cross / area
        if np.dot(np.mean(pts, axis=0) - self.c_center[c_id], normal) < 0:
            normal = -normal

        abs_n = np.abs(normal)
        if abs_n[2] >= abs_n[0] and abs_n[2] >= abs_n[1]:
            dir_key = "+Z" if normal[2] > 0 else "-Z"
        elif abs_n[0] >= abs_n[1]:
            dir_key = "+X" if normal[0] > 0 else "-X"
        else:
            dir_key = "+Y" if normal[1] > 0 else "-Y"

        self.boundary_faces_by_direction[dir_key].append((c_id, normal, area))

    def _init_cell_hydro_properties(self) -> None:
        m = self.c_is_fluid & (self.c_dynamic_viscosity > 0)
        if not np.any(m):
            return

        w, L, h = self.c_dims[m, 0], self.c_dims[m, 1], self.c_dims[m, 2]
        v = self.c_dynamic_viscosity[m]

        cond_eq = np.abs(h - w) < 1e-10
        cond_gt = h > w
        cond_lt = ~(cond_eq | cond_gt)

        hydroC = np.zeros_like(w)
        hydroC[cond_eq] = (0.42229 * h[cond_eq] ** 4) / (12 * v[cond_eq] * L[cond_eq])
        hydroC[cond_gt] = (
            (1 - 0.63 * (w[cond_gt] / h[cond_gt])) * w[cond_gt] ** 3 * h[cond_gt]
        ) / (12 * v[cond_gt] * L[cond_gt])
        hydroC[cond_lt] = (
            (1 - 0.63 * (h[cond_lt] / w[cond_lt])) * h[cond_lt] ** 3 * w[cond_lt]
        ) / (12 * v[cond_lt] * L[cond_lt])

        self.c_hydroC[m] = hydroC

    def _apply_boundary_conditions(self) -> None:
        for bc in self.config["boundary_conditions"]:
            if bc.get("type") != "pressure":
                continue

            pressure, temp = float(bc["pressure"]), float(bc.get("temperature", np.nan))
            for c_id, _, _ in self.boundary_faces_by_direction.get(
                bc.get("face", ""), []
            ):
                if self.c_is_fluid[c_id] and self.c_layer_name[c_id] == bc.get(
                    "target"
                ):
                    self.c_is_pressure_boundary[c_id] = True
                    self.c_pressure[c_id] = pressure
                    self.c_inlet_temperature[c_id] = temp

    def _solve_pressure_field(self) -> None:
        fluid_ids = np.where(self.c_is_fluid)[0]
        if len(fluid_ids) == 0:
            return

        n_fluid = len(fluid_ids)
        global_to_fluid = np.full(self.n_cells, -1, dtype=int)
        global_to_fluid[fluid_ids] = np.arange(n_fluid)

        rows, cols, data = [], [], []
        b_pressure, diag_C = np.zeros(n_fluid), np.zeros(n_fluid)
        is_p_bound = self.c_is_pressure_boundary[fluid_ids]

        # Bound inputs
        bound_idx = np.where(is_p_bound)[0]
        rows.extend(bound_idx)
        cols.extend(bound_idx)
        data.extend(np.ones(len(bound_idx)))
        b_pressure[bound_idx] = self.c_pressure[fluid_ids][bound_idx]

        for c0_id, c1_id in self.internal_faces_list:
            i0, i1 = global_to_fluid[c0_id], global_to_fluid[c1_id]
            if i0 == -1 or i1 == -1:
                continue

            sum_hc = self.c_hydroC[c0_id] + self.c_hydroC[c1_id]
            C_eff = (
                2.0 * self.c_hydroC[c0_id] * self.c_hydroC[c1_id] / sum_hc
                if sum_hc > 0
                else 0.0
            )

            if not self.c_is_pressure_boundary[c0_id]:
                rows.append(i0)
                cols.append(i1)
                data.append(C_eff)
                diag_C[i0] += C_eff
            if not self.c_is_pressure_boundary[c1_id]:
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
            solved_p = splinalg.spsolve(
                sp.csr_matrix((data, (rows, cols)), shape=(n_fluid, n_fluid)),
                b_pressure,
            )
            self.c_pressure[fluid_ids] = solved_p
        except Exception as e:
            print(f"[WARNING] Pressure solve failed: {e}")

    def _compute_nusselt(self, c_id: int) -> float:
        w, h = sorted([self.c_dims[c_id, 0], self.c_dims[c_id, 1]])
        AR = w / h if h > 0 else 1.0
        return 8.235 * (
            1
            - 2.0421 * AR
            + 3.0853 * AR**2
            - 2.4765 * AR**3
            + 1.0578 * AR**4
            - 0.1861 * AR**5
        )

    def _calc_resistance(self, c_a: int, c_b: int, axis: int, area: float) -> float:
        fluid_a, fluid_b = self.c_is_fluid[c_a], self.c_is_fluid[c_b]
        if fluid_a != fluid_b:
            f_id, s_id = (c_a, c_b) if fluid_a else (c_b, c_a)
            Nu = self._compute_nusselt(f_id)
            d_h = (
                2
                * self.c_dims[f_id, 0]
                * self.c_dims[f_id, 1]
                / (self.c_dims[f_id, 0] + self.c_dims[f_id, 1])
            )
            h_f = (Nu * self.c_k[f_id]) / d_h if d_h > 0 else 1e-6
            return self.c_dims[s_id, axis] / (2.0 * self.c_k[s_id] * area) + 1.0 / (
                h_f * area
            )

        return (self.c_dims[c_a, axis] / (2.0 * self.c_k[c_a] * area)) + (
            self.c_dims[c_b, axis] / (2.0 * self.c_k[c_b] * area)
        )

    def _get_face_conduction(self, c_a: int, c_b: int, tol: float) -> list:
        b_a, b_b = self.c_box[c_a], self.c_box[c_b]
        if (
            max(b_a[1], b_b[1]) > min(b_a[4], b_b[4]) + tol
            or max(b_a[2], b_b[2]) > min(b_a[5], b_b[5]) + tol
        ):
            return []

        res_list = []
        for axis in range(3):
            if not (
                abs(b_a[axis + 3] - b_b[axis]) < tol
                or abs(b_a[axis] - b_b[axis + 3]) < tol
            ):
                continue
            area = _overlap_area(b_a, b_b, axis)
            if area <= tol:
                continue

            res = self._calc_resistance(c_a, c_b, axis, area)
            if res > tol:
                res_list.append(1.0 / res)

        return res_list

    def _assemble_conduction_matrix(self) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        tol = self.GEOMETRY_TOLERANCE
        sorted_ids = np.argsort(self.c_box[:, 0])
        active_list = []

        for c_a in sorted_ids:
            active_list = [
                c for c in active_list if self.c_box[c, 3] >= self.c_box[c_a, 0] - tol
            ]
            for c_b in active_list:
                for g in self._get_face_conduction(c_a, c_b, tol):
                    rows.extend([c_a, c_b, c_a, c_b])
                    cols.extend([c_a, c_b, c_b, c_a])
                    data.extend([-g, -g, g, g])
            active_list.append(c_a)

        return sp.csr_matrix((data, (rows, cols)), shape=(self.n_cells, self.n_cells))

    def _assemble_advection_matrix(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n = self.n_cells
        rows, cols, data, rhs = [], [], [], np.zeros(n)
        fluid_ids = np.where(self.c_is_fluid)[0]

        if len(fluid_ids) == 0:
            return sp.csr_matrix((n, n)), rhs

        net_outflux = np.zeros(n)
        for c0_id, c1_id in self.internal_faces_list:
            if not (self.c_is_fluid[c0_id] and self.c_is_fluid[c1_id]):
                continue

            sum_hc = self.c_hydroC[c0_id] + self.c_hydroC[c1_id]
            C_eff = (
                2.0 * self.c_hydroC[c0_id] * self.c_hydroC[c1_id] / sum_hc
                if sum_hc > 0
                else 0.0
            )
            den_eff = (self.c_density[c0_id] + self.c_density[c1_id]) * 0.5
            mass_flux = (
                (self.c_pressure[c0_id] - self.c_pressure[c1_id]) * C_eff * den_eff
            )

            net_outflux[c0_id] += mass_flux
            net_outflux[c1_id] -= mass_flux

            if abs(mass_flux) > self.GEOMETRY_TOLERANCE:
                up_id, dn_id = (c0_id, c1_id) if mass_flux > 0 else (c1_id, c0_id)
                adv_term = abs(mass_flux) * self.c_cp[up_id]
                rows.extend([up_id, dn_id])
                cols.extend([up_id, up_id])
                data.extend([-adv_term, adv_term])

        for c_id in fluid_ids:
            influx = net_outflux[c_id]
            if influx > self.GEOMETRY_TOLERANCE and not np.isnan(
                self.c_inlet_temperature[c_id]
            ):
                rhs[c_id] += influx * self.c_cp[c_id] * self.c_inlet_temperature[c_id]
            elif influx < -self.GEOMETRY_TOLERANCE:
                rows.append(c_id)
                cols.append(c_id)
                data.append(influx * self.c_cp[c_id])

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _precompute_power_matrix(self) -> None:
        self._extract_active_units()
        if not self.active_units or self.n_cells == 0:
            self.power_matrix = sp.csr_matrix((self.n_cells, 0))
            return

        rows, cols, data = [], [], []
        for j, u in enumerate(self.active_units):
            vol = u["dx"] * u["dy"] * u["dz"]
            if vol <= 0:
                continue

            u_min = np.array([u["lx"], u["ly"], u["lz"]])
            u_max = u_min + np.array([u["dx"], u["dy"], u["dz"]])
            intersect = np.prod(
                np.maximum(
                    0,
                    np.minimum(self.c_box[:, 3:], u_max)
                    - np.maximum(self.c_box[:, :3], u_min),
                ),
                axis=1,
            )

            valid = np.where(intersect > self.GEOMETRY_TOLERANCE)[0]
            rows.extend(valid)
            cols.extend([j] * len(valid))
            data.extend(intersect[valid] / vol)

        self.power_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.n_cells, len(self.active_units))
        )

    def _extract_active_units(self) -> None:
        self.active_units = []
        z_cursor = 0.0
        for l in self.stackup:
            if l.active:
                for u in l.units:
                    self.active_units.append(
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
        self.unit_names = [u["name"] for u in self.active_units]

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n, rhs = self.n_cells, np.zeros(self.n_cells)
        rows, cols, data = [], [], []

        for bc in [
            b
            for b in self.config["boundary_conditions"]
            if b.get("type") == "convection"
        ]:
            h, t_inf, target = float(bc["h"]), float(bc["T_inf"]), bc.get("target")
            for c_id, _, area in self.boundary_faces_by_direction.get(
                bc.get("face", ""), []
            ):
                if target and target != self.c_layer_name[c_id]:
                    continue
                g = area / (
                    (0.5 * (self.c_vol[c_id] / area) / self.c_k[c_id]) + (1.0 / h)
                )
                rows.append(c_id)
                cols.append(c_id)
                data.append(-g)
                rhs[c_id] += g * t_inf

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def solve(self) -> None:
        self._init_cell_hydro_properties()
        self._apply_boundary_conditions()

        self.g_total = self._assemble_conduction_matrix()
        bc_mat, self.boundary_rhs = self._build_boundary_terms()
        self.g_total += bc_mat

        if np.any(self.c_is_fluid):
            self._solve_pressure_field()
            adv_mat, adv_rhs = self._assemble_advection_matrix()
            self.g_total += adv_mat
            self.boundary_rhs += adv_rhs

        ptrace_path = os.path.join(self.base_dir, self.config["ptrace_file_path"])
        self.ptrace_steps = []
        if os.path.exists(ptrace_path):
            with open(ptrace_path, "r") as f:
                headers = f.readline().split()
                self.ptrace_steps = [
                    dict(zip(headers, map(float, l.split()))) for l in f if l.strip()
                ]

        if self.config["simulation_type"] == "steady":
            self._solve_steady_state()
        else:
            self._solve_transient()

    def _solve_steady_state(self) -> None:
        mean_powers = np.array(
            [
                (
                    np.mean([s.get(n, 0.0) for s in self.ptrace_steps])
                    if self.ptrace_steps
                    else 0.0
                )
                for n in self.unit_names
            ]
        )

        self.c_temperature = splinalg.spsolve(
            -self.g_total, self.boundary_rhs + (self.power_matrix @ mean_powers)
        )
        print(
            f"[RESULT] T_min={np.min(self.c_temperature):.2f} K, T_max={np.max(self.c_temperature):.2f} K"
        )
        self.save("result.vtu")

    def _solve_transient(self) -> None:
        dt, total_time = self.config["timestep"], self.config["time"]
        n_steps = max(1, math.ceil(total_time / dt) if total_time > 0 else 1)
        ptrace = self.ptrace_steps or [{}] * n_steps

        c_mat = sp.diags(self.c_cp * self.c_vol) / dt
        solve_step = splinalg.factorized((c_mat - self.g_total).tocsc())

        self.c_temperature = np.full(
            self.n_cells, float(self.config["init_temperature"])
        )
        self._load_initial_temperature()

        for i, step_power in enumerate(ptrace):
            power_vec = np.array([step_power.get(n, 0.0) for n in self.unit_names])
            self.c_temperature = solve_step(
                (c_mat @ self.c_temperature)
                + self.boundary_rhs
                + (self.power_matrix @ power_vec)
            )
            if i % 10 == 0 or i == len(ptrace) - 1:
                print(
                    f"[STEP {i:4d}] T_min={np.min(self.c_temperature):.2f} K, T_max={np.max(self.c_temperature):.2f} K"
                )

        self.save("transient_result.vtu")

    def _load_initial_temperature(self) -> None:
        init_file = self.config["init_temperature_file_path"]
        if not init_file or not os.path.exists(os.path.join(self.base_dir, init_file)):
            return

        init_mesh = meshio.read(os.path.join(self.base_dir, init_file))
        hex_data = init_mesh.cell_data.get("Temperature_K", [])
        offset = 0
        for block, block_temps in zip(init_mesh.cells, hex_data):
            if block.type == "hexahedron":
                count = len(block_temps)
                valid_ids = np.arange(offset, offset + count)
                valid_mask = valid_ids < len(self.orig_to_new_id)
                nids = self.orig_to_new_id[valid_ids[valid_mask]]
                self.c_temperature[nids] = block_temps[valid_mask]
                offset += count

    def save(self, output_name: str) -> None:
        mapped = np.empty(self.n_cells)
        mapped[self.c_original_id] = self.c_temperature

        hex_blocks, temp_chunks, offset = [], [], 0
        for block in self.mesh.cells:
            if block.type == "hexahedron":
                count = len(block.data)
                hex_blocks.append(block)
                temp_chunks.append(mapped[offset : offset + count])
                offset += count

        meshio.Mesh(
            points=self.mesh.points,
            cells=hex_blocks,
            cell_data={"Temperature_K": temp_chunks},
        ).write(os.path.join(self.base_dir, output_name))
