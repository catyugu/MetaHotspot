from typing import List, Tuple
import meshio
import numpy as np

from metahotspot.metahotspot_types import (
    MeshTopology,
    PhysicalFields,
    MaterialProps,
    LayerRegion,
)


class MeshPreprocessor:
    GEOMETRY_TOLERANCE = 1e-12

    def __init__(
        self, default_solid: MaterialProps, layer_regions: List[LayerRegion]
    ) -> None:
        self.default_solid = default_solid
        self.layer_regions = layer_regions

    def process(self, mesh_path: str) -> Tuple[MeshTopology, PhysicalFields]:
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

        internal_faces, boundary_faces = self._build_topology_vectorized(
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

    def _build_topology_vectorized(
        self,
        mesh: meshio.Mesh,
        hex_data: np.ndarray,
        sorted_indices: np.ndarray,
        c_centers: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        n_cells = len(sorted_indices)

        faces_def = np.array(
            [
                [0, 3, 2, 1],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [3, 7, 6, 2],
                [0, 4, 7, 3],
                [1, 2, 6, 5],
            ]
        )

        cell_ids = np.repeat(np.arange(n_cells), 6)
        all_faces_nodes = hex_data[sorted_indices][:, faces_def].reshape(-1, 4)

        sorted_faces = np.sort(all_faces_nodes, axis=1)
        sort_order = np.lexsort(
            (
                sorted_faces[:, 3],
                sorted_faces[:, 2],
                sorted_faces[:, 1],
                sorted_faces[:, 0],
            )
        )
        sorted_faces_lex = sorted_faces[sort_order]
        cell_ids_lex = cell_ids[sort_order]

        is_same = np.all(sorted_faces_lex[:-1] == sorted_faces_lex[1:], axis=1)
        internal_idx = np.where(is_same)[0]
        internal_faces = np.column_stack(
            (cell_ids_lex[internal_idx], cell_ids_lex[internal_idx + 1])
        )

        bound_mask = np.ones(len(sorted_faces_lex), dtype=bool)
        bound_mask[internal_idx] = False
        bound_mask[internal_idx + 1] = False

        bound_indices = np.where(bound_mask)[0]
        orig_bound_idx = sort_order[bound_indices]
        bound_c_ids = cell_ids[orig_bound_idx]
        bound_face_nodes = all_faces_nodes[orig_bound_idx]

        boundary_faces = {"+X": [], "-X": [], "+Y": [], "-Y": [], "+Z": [], "-Z": []}

        pts = mesh.points[bound_face_nodes]
        cross = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        areas = np.linalg.norm(cross, axis=1)

        valid = areas > self.GEOMETRY_TOLERANCE
        normals = cross[valid] / areas[valid, None]
        b_c_ids = bound_c_ids[valid]
        b_areas = areas[valid]

        centers_dir = np.mean(pts[valid], axis=1) - c_centers[b_c_ids]
        flip_mask = np.sum(centers_dir * normals, axis=1) < 0
        normals[flip_mask] *= -1

        abs_n = np.abs(normals)
        for i in range(len(b_c_ids)):
            n, a_n = normals[i], abs_n[i]
            if a_n[2] >= a_n[0] and a_n[2] >= a_n[1]:
                dir_key = "+Z" if n[2] > 0 else "-Z"
            elif a_n[0] >= a_n[1]:
                dir_key = "+X" if n[0] > 0 else "-X"
            else:
                dir_key = "+Y" if n[1] > 0 else "-Y"
            boundary_faces[dir_key].append((b_c_ids[i], n, b_areas[i]))

        return internal_faces, {
            k: (
                np.array([x[0] for x in v]),
                np.array([x[1] for x in v]),
                np.array([x[2] for x in v]),
            )
            for k, v in boundary_faces.items()
            if v
        }

    def _map_physical_properties(self, topo: MeshTopology) -> PhysicalFields:
        n, centers, tol = topo.n_cells, topo.centers, self.GEOMETRY_TOLERANCE

        k = np.zeros(n)
        cp = np.zeros(n)
        density = np.zeros(n)
        is_fluid = np.zeros(n, dtype=bool)
        dynamic_viscosity = np.zeros(n)

        layer_ids = np.zeros(n, dtype=np.int16)
        unit_ids = np.zeros(n, dtype=np.int16)

        layer_name_map = ["default_layer"]
        unit_name_map = [""]

        # 直接读取强类型 default_solid 属性
        k[:] = self.default_solid.k
        cp[:] = self.default_solid.cp
        density[:] = self.default_solid.density
        is_fluid[:] = self.default_solid.is_fluid
        dynamic_viscosity[:] = self.default_solid.dynamic_viscosity

        for layer in self.layer_regions:
            z_min, z_max = layer.lz, layer.lz + layer.dz
            l_mask = (centers[:, 2] >= z_min - tol) & (centers[:, 2] <= z_max + tol)

            if not np.any(l_mask):
                continue

            if layer.name not in layer_name_map:
                layer_name_map.append(layer.name)
            l_id = layer_name_map.index(layer.name)

            k[l_mask] = layer.props.k
            cp[l_mask] = layer.props.cp
            density[l_mask] = layer.props.density
            is_fluid[l_mask] = layer.props.is_fluid
            dynamic_viscosity[l_mask] = layer.props.dynamic_viscosity
            layer_ids[l_mask] = l_id

            for unit in layer.units:
                u_mask = (
                    l_mask
                    & (centers[:, 0] >= unit.lx - tol)
                    & (centers[:, 0] <= unit.lx + unit.dx + tol)
                    & (centers[:, 1] >= unit.ly - tol)
                    & (centers[:, 1] <= unit.ly + unit.dy + tol)
                )
                if np.any(u_mask):
                    if unit.name not in unit_name_map:
                        unit_name_map.append(unit.name)
                    u_id = unit_name_map.index(unit.name)

                    k[u_mask] = unit.props.k
                    cp[u_mask] = unit.props.cp
                    density[u_mask] = unit.props.density
                    is_fluid[u_mask] = unit.props.is_fluid
                    dynamic_viscosity[u_mask] = unit.props.dynamic_viscosity
                    unit_ids[u_mask] = u_id

        return PhysicalFields(
            k=k,
            cp=cp,
            density=density,
            is_fluid=is_fluid,
            dynamic_viscosity=dynamic_viscosity,
            hydroC=np.zeros((n, 3)),
            pressure=np.zeros(n),
            boundary_temperature=np.full(n, np.nan),
            layer_ids=layer_ids,
            unit_ids=unit_ids,
            layer_name_map=layer_name_map,
            unit_name_map=unit_name_map,
        )
