import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import toml

from metahotspot.model25d import load_stackup


@dataclass(slots=True)
class Cell:
    """FVM cell representing a hexahedral mesh element.

    Cell types for microchannel:
        0 = SOLID (non-fluid)
        1 = FLUID (active fluid cell)
        2 = INLET (fluid cell with pressure BC)
        3 = OUTLET (fluid cell with pressure BC)
    """

    original_id: int
    id: int
    center: np.ndarray
    dims: np.ndarray
    box: np.ndarray
    k: float
    cp: float
    tag: int
    vol: float
    name: str = ""  # Unit name for BC matching
    layer_name: str = ""  # Layer name for BC matching
    cell_type: int = 0  # 0=solid, 1=fluid, 2=inlet, 3=outlet
    # Computed from pressure solve
    pressure: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Inlet temperature for advective BCs (set from pressure BC config)
    inlet_temp: float = 298.15


def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
    axes = [(1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 3, 4)][axis]
    d1 = min(box_a[axes[2]], box_b[axes[2]]) - max(box_a[axes[0]], box_b[axes[0]])
    d2 = min(box_a[axes[3]], box_b[axes[3]]) - max(box_a[axes[1]], box_b[axes[1]])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


class FVMSolver:
    """Finite Volume Method solver for 2.5D thermal simulation.

    Supports microchannel cooling with pressure-driven flow:
    - Build pressure matrix from hydraulic network
    - Solve for pressure at each fluid cell
    - Compute velocity from pressure gradient
    - Apply upwind advection scheme
    """

    GEOMETRY_TOLERANCE = 1e-12
    DEFAULT_INITIAL_TEMPERATURE = 318.15
    # Water properties for microchannel
    WATER_DENSITY = 1000.0  # kg/m^3
    WATER_VISCOSITY = 8.89e-4  # Pa·s

    def __init__(self, config_path: str) -> None:
        self.base_dir = os.path.dirname(config_path)
        self.config = toml.load(config_path)
        self.mesh_path = os.path.join(
            self.base_dir, self.config.get("mesh_file_path", "mesh.msh")
        )
        self.mesh = meshio.read(self.mesh_path)

        self._sanitize_config()
        self._init_materials_and_stackup()

        self.cells: List[Cell] = []
        self._prepare_mesh()
        self._precompute_power_matrix()

    def _sanitize_config(self) -> None:
        self.config["init_temperature"] = float(
            self.config.get("init_temperature", self.DEFAULT_INITIAL_TEMPERATURE)
        )
        self.config["timestep"] = float(self.config.get("timestep", 0.1))
        self.config["time"] = float(self.config.get("time", 0.0))
        self.config["simulation_type"] = str(
            self.config.get("simulation_type", "steady")
        )
        self.config["ptrace_file_path"] = str(self.config.get("ptrace_file_path", ""))
        self.config.setdefault("stackup", [])
        self.config.setdefault("boundary_conditions", [])
        self.config.setdefault("init_temperature_file_path", None)

    def _init_materials_and_stackup(self) -> None:
        self.materials = self.config.get("materials", {})
        self.stackup = load_stackup(self.config, self.base_dir)

    def _prepare_mesh(self) -> None:
        print("[INFO] Preparing mesh data...")
        hex_blocks = [b.data for b in self.mesh.cells if b.type == "hexahedron"]
        if not hex_blocks:
            raise ValueError("No hexahedron cells found in mesh")

        hex_data = np.vstack(hex_blocks)
        physical_tags = self.mesh.cell_data_dict.get("gmsh:physical", {}).get(
            "hexahedron", np.full(len(hex_data), -1)
        )

        coords = self.mesh.points[hex_data]
        lowers, uppers = np.min(coords, axis=1), np.max(coords, axis=1)
        centers = (lowers + uppers) * 0.5
        dims = uppers - lowers
        vols = np.prod(dims, axis=1)

        b_min, b_max = np.min(lowers, axis=0), np.max(uppers, axis=0)
        diff = np.where((b_max - b_min) == 0, 1, b_max - b_min)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)

        morton_keys = np.zeros(len(centers), dtype=int)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0] >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1] >> i) & 1) << (3 * i + 1)
            morton_keys |= ((norm_centers[:, 2] >> i) & 1) << (3 * i + 2)

        sorted_indices = np.argsort(morton_keys)

        mat_k_array, mat_cp_array = np.zeros(len(centers)), np.zeros(len(centers))
        cell_layer_names = np.array([""] * len(centers), dtype=object)

        # 核心改动：利用 2.5D Stackup 为 3D 网格中心点映射材料属性
        tol = self.GEOMETRY_TOLERANCE
        z_cursor = 0.0
        for layer in self.stackup:
            z_min = z_cursor
            z_max = z_cursor + layer.thickness
            z_cursor = z_max

            layer_mask = (centers[:, 2] >= z_min - tol) & (centers[:, 2] <= z_max + tol)
            if not np.any(layer_mask):
                continue

            def_mat = self.materials.get(
                layer.default_material, {"k": 1.0, "cp": 1.0e6}
            )
            mat_k_array[layer_mask] = float(def_mat["k"])
            mat_cp_array[layer_mask] = float(def_mat["cp"])
            cell_layer_names[layer_mask] = layer.name

            # 覆盖异构材料单元
            for u in layer.units:
                u_mask = (
                    layer_mask
                    & (centers[:, 0] >= u.lx - tol)
                    & (centers[:, 0] <= u.lx + u.dx + tol)
                    & (centers[:, 1] >= u.ly - tol)
                    & (centers[:, 1] <= u.ly + u.dy + tol)
                )
                if np.any(u_mask):
                    if u.k is not None:
                        mat_k_array[u_mask] = u.k
                        mat_cp_array[u_mask] = u.cp
                    elif u.material and u.material in self.materials:
                        mat_k_array[u_mask] = float(self.materials[u.material]["k"])
                        mat_cp_array[u_mask] = float(self.materials[u.material]["cp"])

        # Pre-compute layer z-bounds for cell matching
        layer_z_min = {}
        z_cursor = 0.0
        for layer in self.stackup:
            layer_z_min[layer.name] = z_cursor
            z_cursor += layer.thickness

        self.face_to_cells: Dict[tuple, List[int]] = {}
        for new_id, orig_id in enumerate(sorted_indices):
            nodes = hex_data[orig_id]

            # Initialize cell properties
            cell_type = 0  # Default: solid
            unit_name = ""
            layer_name = cell_layer_names[orig_id]

            # Find matching Unit2D from stackup
            # Use center-based matching to handle mesh refinement
            # Only search in the cell's own layer (determined by z-position)
            c_center = centers[orig_id]
            for layer in self.stackup:
                # Check if cell's z-center is within this layer's z-range
                z_min = layer_z_min[layer.name]
                z_max = z_min + layer.thickness
                if c_center[2] >= z_min - tol and c_center[2] <= z_max + tol:
                    # Cell belongs to this layer, search its units
                    for u in layer.units:
                        if (
                            c_center[0] >= u.lx - tol
                            and c_center[0] <= u.lx + u.dx + tol
                            and c_center[1] >= u.ly - tol
                            and c_center[1] <= u.ly + u.dy + tol
                        ):
                            cell_type = getattr(
                                u, "cell_type", 0
                            )  # 0=solid, 1=fluid, 2=inlet, 3=outlet
                            unit_name = u.name
                            break
                    break

            c = Cell(
                original_id=orig_id,
                id=new_id,
                center=centers[orig_id],
                dims=dims[orig_id],
                box=np.array([*lowers[orig_id], *uppers[orig_id]]),
                k=float(mat_k_array[orig_id]),
                cp=float(mat_cp_array[orig_id]),
                tag=int(physical_tags[orig_id]),
                vol=float(vols[orig_id]),
                name=unit_name,
                layer_name=layer_name,
                cell_type=cell_type,
            )
            self.cells.append(c)

            fs = [  # 6 faces of hexahedron
                tuple(sorted([nodes[0], nodes[3], nodes[2], nodes[1]])),  # -Z face
                tuple(sorted([nodes[4], nodes[5], nodes[6], nodes[7]])),  # +Z face
                tuple(sorted([nodes[0], nodes[1], nodes[5], nodes[4]])),  # -Y face
                tuple(sorted([nodes[3], nodes[7], nodes[6], nodes[2]])),  # +Y face
                tuple(sorted([nodes[0], nodes[4], nodes[7], nodes[3]])),  # -X face
                tuple(sorted([nodes[1], nodes[2], nodes[6], nodes[5]])),  # +X face
            ]
            for f in fs:
                if f not in self.face_to_cells:
                    self.face_to_cells[f] = []
                self.face_to_cells[f].append(new_id)

        # Build internal and boundary face maps
        self.internal_faces = {
            f: tuple(c_ids)
            for f, c_ids in self.face_to_cells.items()
            if len(c_ids) == 2
        }
        self.boundary_faces_all = {
            f: tuple(c_ids)
            for f, c_ids in self.face_to_cells.items()
            if len(c_ids) == 1
        }

        self.orig_to_new_id = {c.original_id: c.id for c in self.cells}
        self._extract_boundary_faces()

    def _extract_boundary_faces(self) -> None:
        """Extract boundary faces and compute their outward normal direction.

        Creates self.boundary_faces_by_direction:
            { "+Z": [(cell_id, face_normal, area), ...],
              "-Z": [...],
              "+X": [...],
              "-X": [...],
              "+Y": [...],
              "-Y": [...] }
        """
        self.boundary_faces_by_direction: Dict[str, List[tuple]] = {
            "+X": [],
            "-X": [],
            "+Y": [],
            "-Y": [],
            "+Z": [],
            "-Z": [],
        }

        if not self.boundary_faces_all:
            return

        tol = self.GEOMETRY_TOLERANCE

        for f, (c_id,) in self.boundary_faces_all.items():
            pts = self.mesh.points[list(f)]

            # Calculate face normal (pointing outward from cell)
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            cross_prod = np.cross(v1, v2)
            area = np.linalg.norm(cross_prod)

            if area < tol:
                continue

            normal = cross_prod / area

            # Get cell center to determine outward direction
            c = self.cells[c_id]
            face_center = np.mean(pts, axis=0)

            # Vector from cell center to face center
            vec = face_center - c.center

            # If vector points same direction as normal, normal is outward
            # Otherwise flip it
            if np.dot(vec, normal) < 0:
                normal = -normal

            # Determine direction label
            abs_normal = np.abs(normal)
            if abs_normal[2] >= abs_normal[0] and abs_normal[2] >= abs_normal[1]:
                direction = "+Z" if normal[2] > 0 else "-Z"
            elif abs_normal[0] >= abs_normal[1]:
                direction = "+X" if normal[0] > 0 else "-X"
            else:
                direction = "+Y" if normal[1] > 0 else "-Y"

            self.boundary_faces_by_direction[direction].append((c_id, normal, area))

    def _solve_pressure(self) -> None:
        """Build and solve the hydraulic pressure matrix for microchannel.

        Uses Hagen-Poiseuille equation for hydraulic conductance:
            hydroC = (1 - 0.63*(min/max)) * min^3 * max / (12 * viscosity * L)

        The pressure matrix is a Laplacian-like system where:
        - Each fluid cell is a node
        - Edges between adjacent fluid cells have conductance hydroC
        - Inlet cells have Dirichlet BC: pressure = pumping_pressure
        - Outlet cells have Dirichlet BC: pressure = 0
        """
        # Get microchannel pressure BCs
        pressure_bcs = []
        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") == "pressure":
                pressure_bcs.append(
                    {
                        "face": bc.get("face", ""),
                        "target": bc.get("target", ""),
                        "pressure": float(bc.get("pressure", 0.0)),
                        "temperature": bc.get("temperature"),
                    }
                )

        if not pressure_bcs:
            print("[INFO] No pressure BCs found, skipping pressure solve")
            return

        # Build fluid connectivity graph
        fluid_cells = [c for c in self.cells if c.cell_type in (1, 2, 3)]
        if not fluid_cells:
            print("[INFO] No fluid cells found, skipping pressure solve")
            return

        # Create mapping from cell id to pressure matrix index
        cell_to_idx = {c.id: i for i, c in enumerate(fluid_cells)}
        n_fluid = len(fluid_cells)

        # Compute hydraulic conductance for each cell
        avg_dims = np.mean([c.dims for c in fluid_cells], axis=0)
        h = avg_dims[2]  # thickness (Z direction - height of channel)
        w = avg_dims[0]  # width (X direction)
        L = avg_dims[1]  # length (Y direction)

        viscosity = self.WATER_VISCOSITY

        # Hagen-Poiseuille for rectangular channel
        if abs(h - w) < 1e-10:  # Square
            hydroC = (0.42229 * h**4) / (12 * viscosity * L)
        elif h > w:
            hydroC = ((1 - 0.63 * (w / h)) * w**3 * h) / (12 * viscosity * L)
        else:
            hydroC = ((1 - 0.63 * (h / w)) * h**3 * w) / (12 * viscosity * L)

        print(f"[INFO] Hydraulic conductance: {hydroC:.6e} m^3/(Pa·s)")
        print(f"[INFO] Fluid cells: {n_fluid}")

        # Build pressure matrix (Laplacian-like)
        rows, cols, data = [], [], []

        for c in fluid_cells:
            i = cell_to_idx[c.id]

            # Find neighboring fluid cells
            neighbors = []
            for f, (c0_id, c1_id) in self.internal_faces.items():
                if c0_id == c.id and c1_id in cell_to_idx:
                    neighbors.append(c1_id)
                elif c1_id == c.id and c0_id in cell_to_idx:
                    neighbors.append(c0_id)

            # Diagonal entry: negative sum of all conductances
            rows.append(i)
            cols.append(i)
            data.append(-len(neighbors) * hydroC)

            # Off-diagonal entries
            for neighbor_id in neighbors:
                j = cell_to_idx[neighbor_id]
                rows.append(i)
                cols.append(j)
                data.append(hydroC)

        A_pressure = sp.csr_matrix((data, (rows, cols)), shape=(n_fluid, n_fluid))
        b_pressure = np.zeros(n_fluid)

        # Apply boundary conditions based on geometric location, not cell_type
        # For each pressure BC, find cells on the specified face direction
        for bc in pressure_bcs:
            face = bc["face"]
            target = bc["target"]
            pressure = bc["pressure"]
            temperature = bc.get("temperature")

            # Find boundary faces matching this BC
            bc_faces = self.boundary_faces_by_direction.get(face, [])
            for c_id, normal, area in bc_faces:
                c = self.cells[c_id]
                # Only apply to cells in the target layer
                if c.layer_name != target:
                    continue
                # Only apply to fluid cells
                if c.cell_type not in (1, 2, 3):
                    continue

                i = cell_to_idx.get(c.id)
                if i is None:
                    continue

                # Fix pressure at this cell
                A_pressure[i, :] = 0
                A_pressure[i, i] = 1
                b_pressure[i] = pressure

                # Set inlet temperature if provided
                if temperature is not None:
                    c.inlet_temp = temperature

                print(
                    f"[INFO] Applied {face} BC: pressure={pressure} Pa to cell {c.id} (layer={c.layer_name})"
                )

        # Solve pressure system
        try:
            pressure = splinalg.spsolve(A_pressure, b_pressure)

            # Store pressure in cells
            for c in fluid_cells:
                c.pressure = pressure[cell_to_idx[c.id]]

            print(
                f"[INFO] Pressure solved. Range: {pressure.min():.2f} to {pressure.max():.2f} Pa"
            )

        except Exception as e:
            print(f"[WARNING] Pressure solve failed: {e}")
            # Set zero pressure as fallback
            for c in fluid_cells:
                c.pressure = 0.0

    def _compute_face_normal(self, pts: np.ndarray) -> np.ndarray:
        """Compute outward normal for a face given its points."""
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        cross_prod = np.cross(v1, v2)
        area = np.linalg.norm(cross_prod)
        if area > self.GEOMETRY_TOLERANCE:
            return cross_prod / area
        return np.array([0.0, 0.0, 1.0])

    def _compute_velocity_from_pressure(self) -> None:
        """Compute velocity at each fluid cell face from pressure gradient.

        Uses Darcy's law: v = -K * grad(P) / mu
        For simplicity, assumes velocity is proportional to pressure difference.
        """
        viscosity = self.WATER_VISCOSITY

        for c in self.cells:
            if c.cell_type not in (1, 2, 3):
                c.velocity = np.zeros(3)
                continue

            # Find pressure gradient from neighbors
            pressure_grad = np.zeros(3)
            count = 0

            for f, (c0_id, c1_id) in self.internal_faces.items():
                if c0_id == c.id:
                    c1 = self.cells[c1_id]
                    if c1.cell_type in (1, 2, 3):
                        # Vector from c to neighbor
                        dvec = c1.center - c.center
                        dist = np.linalg.norm(dvec)
                        if dist > self.GEOMETRY_TOLERANCE:
                            # Pressure difference in direction of neighbor
                            dP = c1.pressure - c.pressure
                            pressure_grad += dP * dvec / (dist * dist)
                            count += 1
                elif c1_id == c.id:
                    c0 = self.cells[c0_id]
                    if c0.cell_type in (1, 2, 3):
                        dvec = c0.center - c.center
                        dist = np.linalg.norm(dvec)
                        if dist > self.GEOMETRY_TOLERANCE:
                            dP = c0.pressure - c.pressure
                            pressure_grad += dP * dvec / (dist * dist)
                            count += 1

            if count > 0:
                pressure_grad /= count

            # Darcy's law: v = -k/mu * grad(P) where k is permeability
            # For a channel: k = hydroC * L / A
            # Simplified: velocity proportional to negative pressure gradient
            perm = 1e-10  # Approximate permeability
            c.velocity = -perm / viscosity * pressure_grad

            # Also check boundary faces for direction
            for f, (c_id,) in self.boundary_faces_all.items():
                if c_id != c.id:
                    continue
                pts = self.mesh.points[list(f)]
                normal = self._compute_face_normal(pts)
                area = np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))

                if c.cell_type == 2:  # INLET
                    # Flow enters from boundary, velocity points inward
                    c.velocity = -normal * np.abs(c.pressure) * 0.001
                elif c.cell_type == 3:  # OUTLET
                    # Flow exits to boundary, velocity points outward
                    c.velocity = normal * np.abs(c.pressure) * 0.001

    def _add_fluid_advection_generic(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Assemble fluid advection matrix using upwind scheme and computed velocity.

        Returns:
            Tuple of (advection_matrix, advection_rhs)
        """
        n = len(self.cells)
        rows, cols, data = [], [], []
        rhs = np.zeros(n)
        tol = self.GEOMETRY_TOLERANCE

        # 1. Compute internal fluid face fluxes using velocity from pressure
        for f, (c0_id, c1_id) in self.internal_faces.items():
            c0, c1 = self.cells[c0_id], self.cells[c1_id]

            # Skip if not both fluid (cell_type 1, 2, or 3)
            if c0.cell_type == 0 or c1.cell_type == 0:
                continue

            # Get face points from mesh
            pts = self.mesh.points[list(f)]

            # Calculate face normal and area
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            cross_prod = np.cross(v1, v2)
            area = np.linalg.norm(cross_prod)

            if area < tol:
                continue

            n_vec = cross_prod / area

            # Ensure normal points from c0 to c1 (c1 - c0 direction)
            vec_c0_c1 = c1.center - c0.center
            if np.dot(n_vec, vec_c0_c1) < 0:
                n_vec = -n_vec

            # Use velocity from pressure solve (stored in cell.velocity)
            v_avg = 0.5 * (c0.velocity + c1.velocity)

            # Volume flux: Q = dot(v_avg, n_vec) * area
            vol_flux = np.dot(v_avg, n_vec) * area

            # Mass flux: m_dot = vol_flux * density
            density = self.WATER_DENSITY

            # Determine upstream cell based on velocity direction
            if np.dot(v_avg, n_vec) > 0:
                # Flow from c0 to c1, c0 is upstream
                upstream, downstream = c0, c1
                upstream_id, downstream_id = c0_id, c1_id
            else:
                # Flow from c1 to c0, c1 is upstream
                upstream, downstream = c1, c0
                upstream_id, downstream_id = c1_id, c0_id

            mass_flux = vol_flux * density
            cp = upstream.cp
            advection_term = mass_flux * cp

            # Only add significant terms
            if abs(advection_term) > tol:
                # Donor cell loses energy (negative coefficient)
                rows.append(upstream_id)
                cols.append(upstream_id)
                data.append(-advection_term)

                # Receiver cell gains energy (positive coefficient)
                rows.append(downstream_id)
                cols.append(downstream_id)
                data.append(advection_term)

        # 2. Handle fluid boundary faces (inlet/outlet) with temperature
        for f, (c0_id,) in self.boundary_faces_all.items():
            c0 = self.cells[c0_id]

            # Skip if not fluid (cell_type 0 = solid) or no inlet temperature
            if c0.cell_type == 0 or c0.inlet_temp is None:
                continue

            # Get face points
            pts = self.mesh.points[list(f)]

            # Calculate face area
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            cross_prod = np.cross(v1, v2)
            area = np.linalg.norm(cross_prod)

            if area < tol:
                continue

            # Velocity at boundary
            vel_mag = np.linalg.norm(c0.velocity)
            if vel_mag < tol:
                continue

            # Mass flux at inlet
            density = self.WATER_DENSITY
            mass_flux = vel_mag * area * density

            # Inlet: energy enters system from inlet_temp
            rhs[c0_id] += mass_flux * c0.cp * c0.inlet_temp

            # Boundary cell loses energy via outflow
            rows.append(c0_id)
            cols.append(c0_id)
            data.append(-mass_flux * c0.cp)

        G_adv = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        return G_adv, rhs

    def _precompute_power_matrix(self) -> None:
        # 核心改动：从 2.5D Stackup 收集 active_units 的 3D 信息
        active_units_3d = []
        z_cursor = 0.0
        for layer in self.stackup:
            if layer.active:
                for u in layer.units:
                    active_units_3d.append(
                        {
                            "name": u.name,
                            "lx": u.lx,
                            "ly": u.ly,
                            "lz": z_cursor,
                            "dx": u.dx,
                            "dy": u.dy,
                            "dz": layer.thickness,
                        }
                    )
            z_cursor += layer.thickness

        self.unit_names = [u["name"] for u in active_units_3d]

        if not active_units_3d or not self.cells:
            self.power_matrix = sp.csr_matrix((len(self.cells), 0))
            return

        cell_boxes = np.array([c.box for c in self.cells])
        cell_lowers, cell_uppers = cell_boxes[:, :3], cell_boxes[:, 3:]
        rows, cols, data = [], [], []

        for unit_idx, unit in enumerate(active_units_3d):
            vol = unit["dx"] * unit["dy"] * unit["dz"]
            if vol <= 0:
                continue

            u_lower = np.array([unit["lx"], unit["ly"], unit["lz"]])
            u_upper = u_lower + np.array([unit["dx"], unit["dy"], unit["dz"]])

            overlap_lowers = np.maximum(cell_lowers, u_lower)
            overlap_uppers = np.minimum(cell_uppers, u_upper)
            overlap_dims = np.maximum(0, overlap_uppers - overlap_lowers)

            intersect_vols = np.prod(overlap_dims, axis=1)
            valid_mask = intersect_vols > self.GEOMETRY_TOLERANCE

            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                rows.extend(valid_indices)
                cols.extend([unit_idx] * len(valid_indices))
                data.extend(intersect_vols[valid_mask] / vol)

        self.power_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(self.cells), len(active_units_3d))
        )

    def _get_initial_temperatures(self, n_cells: int) -> np.ndarray:
        default_temp = self.config["init_temperature"]
        init_file = self.config["init_temperature_file_path"]
        if not init_file or init_file in {"(null)", "None", ""}:
            return np.full(n_cells, default_temp)
        init_path = os.path.join(self.base_dir, init_file)
        if not os.path.exists(init_path):
            return np.full(n_cells, default_temp)

        init_mesh = meshio.read(init_path)
        temps = np.zeros(n_cells)
        offset = 0
        hex_data = init_mesh.cell_data.get("Temperature_K", [])
        for block, block_temps in zip(init_mesh.cells, hex_data):
            if block.type != "hexahedron":
                continue
            for i, t in enumerate(block_temps):
                new_id = self.orig_to_new_id.get(offset + i)
                if new_id is not None:
                    temps[new_id] = t
            offset += len(block_temps)
        return temps

    def assemble_g_matrix(self) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        tol = self.GEOMETRY_TOLERANCE
        sorted_cells = sorted(self.cells, key=lambda c: c.box[0])
        active_list: List[Cell] = []

        for c_a in sorted_cells:
            active_list = [c for c in active_list if c.box[3] >= c_a.box[0] - tol]
            for c_b in active_list:
                if max(c_a.box[1], c_b.box[1]) > min(c_a.box[4], c_b.box[4]) + tol:
                    continue
                if max(c_a.box[2], c_b.box[2]) > min(c_a.box[5], c_b.box[5]) + tol:
                    continue
                for axis in range(3):
                    if not (
                        abs(c_a.box[axis + 3] - c_b.box[axis]) < tol
                        or abs(c_a.box[axis] - c_b.box[axis + 3]) < tol
                    ):
                        continue
                    area = _overlap_area(c_a.box, c_b.box, axis)
                    if area <= tol:
                        continue
                    res = (c_a.dims[axis] / (2.0 * c_a.k * area)) + (
                        c_b.dims[axis] / (2.0 * c_b.k * area)
                    )
                    if res <= tol:
                        continue
                    g = 1.0 / res
                    rows.extend([c_a.id, c_b.id, c_a.id, c_b.id])
                    cols.extend([c_a.id, c_b.id, c_b.id, c_a.id])
                    data.extend([-g, -g, g, g])
            active_list.append(c_a)
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(len(self.cells), len(self.cells))
        )

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Build boundary condition terms using direction and target-based selection.

        BC format with layer targeting:
            [[boundary_conditions]]
            name = "sink_conv"
            type = "convection"
            face = "+Z"
            target = "Sink"  # Layer name to apply this BC
            h = 2777.78
            T_inf = 318.15

        Cell-level override:
            [[boundary_conditions]]
            name = "cell_inlet"
            type = "inlet"
            unit_name = "microchannel_0"  # Specific unit
            face = "-X"
            temperature = 298.15
        """
        n = len(self.cells)
        rhs, rows, cols, data = np.zeros(n), [], [], []

        # Group BCs by direction
        bcs_by_direction: Dict[str, list] = {
            "+X": [],
            "-X": [],
            "+Y": [],
            "-Y": [],
            "+Z": [],
            "-Z": [],
        }

        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") == "convection":
                face = bc.get("face", "")
                if face in bcs_by_direction:
                    bcs_by_direction[face].append(bc)

        # Apply boundary conditions by direction
        for direction, bcs in bcs_by_direction.items():
            if not bcs:
                continue

            faces = self.boundary_faces_by_direction.get(direction, [])
            for cell_id, normal, area in faces:
                c = self.cells[cell_id]

                # Check for cell-level override first (unit_name match)
                cell_bc = None
                if c.name:  # Only check if cell has a name
                    for bc in self.config.get("boundary_conditions", []):
                        if bc.get("unit_name") and bc.get("unit_name") == c.name:
                            cell_bc = bc
                            break

                if cell_bc and cell_bc.get("type") == "convection":
                    h = float(cell_bc["h"])
                    t_inf = float(cell_bc["T_inf"])
                else:
                    # Find layer-level BC that matches this cell's layer
                    layer_bc = None
                    for bc in bcs:
                        target = bc.get("target", "")  # Layer name
                        if not target:
                            # No target specified, applies to all layers
                            layer_bc = bc
                        elif target == c.layer_name:
                            # Target matches this cell's layer
                            layer_bc = bc
                            break

                    if layer_bc is None:
                        continue

                    h = float(layer_bc["h"])
                    t_inf = float(layer_bc["T_inf"])

                dist = c.vol / area
                g = area / ((0.5 * dist / c.k) + (1.0 / h))
                rows.append(c.id)
                cols.append(c.id)
                data.append(-g)
                rhs[c.id] += g * t_inf

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _load_ptrace(self) -> List[dict]:
        ptrace_path = os.path.join(self.base_dir, self.config["ptrace_file_path"])
        if not os.path.exists(ptrace_path):
            return []
        with open(ptrace_path, "r", encoding="utf-8") as f:
            headers = f.readline().split()
            return [
                dict(zip(headers, map(float, line.split())))
                for line in f
                if line.strip()
            ]

    def solve(self) -> None:
        self.g_total = self.assemble_g_matrix() + self._build_boundary_terms()[0]
        self.boundary_rhs = self._build_boundary_terms()[1]

        # Check for fluid cells and solve pressure if present
        fluid_cells = [c for c in self.cells if c.cell_type in (1, 2, 3)]
        if fluid_cells:
            print(
                f"[INFO] Found {len(fluid_cells)} fluid cells, solving pressure-driven flow..."
            )
            # Solve pressure field first, then compute velocities
            self._solve_pressure()
            self._compute_velocity_from_pressure()

            # Assemble advection with computed velocities
            advection_mat, advection_rhs = self._add_fluid_advection_generic()
            self.g_total = self.g_total + advection_mat
            self.boundary_rhs = self.boundary_rhs + advection_rhs

        self.ptrace_steps = self._load_ptrace()
        if self.config["simulation_type"] == "steady":
            self._solve_steady_state()
        else:
            self._solve_transient()

    def _solve_steady_state(self) -> None:
        mean_powers = np.array(
            [
                (
                    np.mean([s.get(name, 0.0) for s in self.ptrace_steps])
                    if self.ptrace_steps
                    else 0.0
                )
                for name in self.unit_names
            ]
        )
        power_rhs = self.power_matrix @ mean_powers
        temperatures = splinalg.spsolve(-self.g_total, self.boundary_rhs + power_rhs)
        print(
            f"[RESULT] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
        )
        self.save(temperatures, "result.vtu")

    def _solve_transient(self) -> None:
        dt, total_time = self.config["timestep"], self.config["time"]
        n_steps = max(1, math.ceil(total_time / dt) if total_time > 0 else 1)
        ptrace = self.ptrace_steps or [{}] * n_steps
        c_mat = sp.diags([c.cp * c.vol for c in self.cells]) / dt
        solve_step = splinalg.factorized((c_mat - self.g_total).tocsc())
        temperatures = self._get_initial_temperatures(len(self.cells))

        for i, step_power in enumerate(ptrace):
            power_vec = np.array(
                [step_power.get(name, 0.0) for name in self.unit_names]
            )
            rhs = (
                (c_mat @ temperatures)
                + self.boundary_rhs
                + (self.power_matrix @ power_vec)
            )
            temperatures = solve_step(rhs)
            if i % 10 == 0 or i == len(ptrace) - 1:
                print(
                    f"[STEP {i:4d}] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
                )
        self.save(temperatures, "transient_result.vtu")

    def save(self, temperatures: np.ndarray, output_name: str) -> None:
        import meshio

        mapped = np.zeros(len(self.cells))
        for c in self.cells:
            mapped[c.original_id] = temperatures[c.id]
        hex_blocks, temp_chunks, offset = [], [], 0
        for block in self.mesh.cells:
            if block.type == "hexahedron":
                count = len(block.data)
                hex_blocks.append(block)
                temp_chunks.append(mapped[offset : offset + count])
                offset += count
        out_mesh = meshio.Mesh(
            points=self.mesh.points,
            cells=hex_blocks,
            cell_data={"Temperature_K": temp_chunks},
        )
        out_mesh.write(os.path.join(self.base_dir, output_name))
