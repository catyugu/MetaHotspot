from typing import Any, Dict, List, Tuple

import meshio
import numpy as np

from metahotspot.metahotspot_types import MeshTopology, PhysicalFields


class MeshPreprocessor:
    GEOMETRY_TOLERANCE = 1e-15

    def __init__(self, config: Dict[str, Any], stackup: List[Any]) -> None:
        self.config = config
        self.stackup = stackup

    def process(self, mesh_path: str) -> Tuple[MeshTopology, PhysicalFields]:
        print(f"[INFO] Processing mesh: {mesh_path}")
        mesh = meshio.read(mesh_path)
        topo = self._extract_geometry(mesh)
        fields = self._map_physical_properties(topo)
        return topo, fields

    def _extract_geometry(self, mesh: meshio.Mesh) -> MeshTopology:
        hex_blocks = [b.data for b in mesh.cells if b.type == "hexahedron"]
        if not hex_blocks:
            raise ValueError("No hexahedron cells found in mesh")
        hex_data = np.vstack(hex_blocks)
        coords = mesh.points[hex_data]
        lowers, uppers = np.min(coords, axis=1), np.max(coords, axis=1)
        centers, dims = (lowers + uppers) * 0.5, uppers - lowers
        vols = np.prod(dims, axis=1)
        sorted_indices = self._compute_morton_sort(lowers, uppers, centers)
        n_cells = len(centers)
        c_centers, c_dims, c_boxes, c_vols = (
            centers[sorted_indices],
            dims[sorted_indices],
            np.hstack([lowers[sorted_indices], uppers[sorted_indices]]),
            vols[sorted_indices],
        )
        orig_to_new_id = np.empty(n_cells, dtype=int)
        orig_to_new_id[sorted_indices] = np.arange(n_cells)
        internal_faces, boundary_faces = self._build_topology(
            mesh, hex_data, sorted_indices, c_centers
        )
        return MeshTopology(
            n_cells,
            c_centers,
            c_dims,
            c_boxes,
            c_vols,
            internal_faces,
            boundary_faces,
            sorted_indices,
            orig_to_new_id,
        )

    def _compute_morton_sort(self, lowers, uppers, centers) -> np.ndarray:
        b_min = np.min(lowers, axis=0)
        diff = np.where((d := np.max(uppers, axis=0) - b_min) == 0, 1, d)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)
        morton_keys = np.zeros(len(centers), dtype=np.int64)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0].astype(np.int64) >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1].astype(np.int64) >> i) & 1) << (
                3 * i + 1
            )
            morton_keys |= ((norm_centers[:, 2].astype(np.int64) >> i) & 1) << (
                3 * i + 2
            )
        return np.argsort(morton_keys)

    def _build_topology(
        self,
        mesh: meshio.Mesh,
        hex_data: np.ndarray,
        sorted_indices: np.ndarray,
        c_centers: np.ndarray,
    ) -> Tuple[list, dict]:
        face_to_cells = {}
        for new_id, orig_id in enumerate(sorted_indices):
            nodes = hex_data[orig_id]
            faces = [
                (nodes[0], nodes[3], nodes[2], nodes[1]),
                (nodes[4], nodes[5], nodes[6], nodes[7]),
                (nodes[0], nodes[1], nodes[5], nodes[4]),
                (nodes[3], nodes[7], nodes[6], nodes[2]),
                (nodes[0], nodes[4], nodes[7], nodes[3]),
                (nodes[1], nodes[2], nodes[6], nodes[5]),
            ]
            for f in faces:
                face_to_cells.setdefault(tuple(sorted(f)), []).append(new_id)
        internal_faces = [tuple(c) for c in face_to_cells.values() if len(c) == 2]
        boundary_faces_raw = {f: c[0] for f, c in face_to_cells.items() if len(c) == 1}
        boundary_faces = {"+X": [], "-X": [], "+Y": [], "-Y": [], "+Z": [], "-Z": []}
        for f, c_id in boundary_faces_raw.items():
            pts = mesh.points[list(f)]
            cross = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            area = np.linalg.norm(cross)
            if area < self.GEOMETRY_TOLERANCE:
                continue
            normal = cross / area
            if np.dot(np.mean(pts, axis=0) - c_centers[c_id], normal) < 0:
                normal = -normal
            abs_n = np.abs(normal)
            if abs_n[2] >= abs_n[0] and abs_n[2] >= abs_n[1]:
                dir_key = "+Z" if normal[2] > 0 else "-Z"
            elif abs_n[0] >= abs_n[1]:
                dir_key = "+X" if normal[0] > 0 else "-X"
            else:
                dir_key = "+Y" if normal[1] > 0 else "-Y"
            boundary_faces[dir_key].append((c_id, normal, area))
        return internal_faces, boundary_faces

    def _map_physical_properties(self, topo: MeshTopology) -> PhysicalFields:
        n, centers, tol = topo.n_cells, topo.centers, self.GEOMETRY_TOLERANCE
        k, cp, density, is_fluid, dynamic_viscosity = (
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n, dtype=bool),
            np.zeros(n),
        )
        layer_names, unit_names = np.empty(n, dtype=object), np.empty(n, dtype=object)
        def_mat = self.config["materials"]["default_solid"]
        (
            k[:],
            cp[:],
            density[:],
            is_fluid[:],
            dynamic_viscosity[:],
            layer_names[:],
            unit_names[:],
        ) = (
            def_mat["k"],
            def_mat["cp"],
            def_mat["density"],
            def_mat.get("fluid", False),
            def_mat.get("dynamic_viscosity", 0.0),
            "default_layer",
            "",
        )
        z_cursor = 0.0
        for layer in self.stackup:
            z_min, z_max = z_cursor, z_cursor + layer.thickness
            z_cursor = z_max
            l_mask = (centers[:, 2] >= z_min - tol) & (centers[:, 2] <= z_max + tol)
            if not np.any(l_mask):
                continue
            (
                k[l_mask],
                cp[l_mask],
                density[l_mask],
                is_fluid[l_mask],
                dynamic_viscosity[l_mask],
                layer_names[l_mask],
            ) = (
                layer.k,
                layer.cp,
                layer.density,
                layer.is_fluid,
                layer.dynamic_viscosity,
                layer.name,
            )
            for unit in layer.units:
                u_mask = (
                    l_mask
                    & (centers[:, 0] >= unit.lx - tol)
                    & (centers[:, 0] <= unit.lx + unit.dx + tol)
                    & (centers[:, 1] >= unit.ly - tol)
                    & (centers[:, 1] <= unit.ly + unit.dy + tol)
                )
                if np.any(u_mask):
                    (
                        k[u_mask],
                        cp[u_mask],
                        density[u_mask],
                        is_fluid[u_mask],
                        dynamic_viscosity[u_mask],
                        unit_names[u_mask],
                    ) = (
                        unit.k,
                        unit.cp,
                        unit.density,
                        unit.is_fluid,
                        unit.dynamic_viscosity,
                        unit.name,
                    )
        return PhysicalFields(
            k,
            cp,
            density,
            is_fluid,
            dynamic_viscosity,
            np.zeros(n),
            np.zeros(n),
            np.full(n, np.nan),
            layer_names,
            unit_names,
        )
