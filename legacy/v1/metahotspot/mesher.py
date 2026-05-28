import math
from collections import deque
import numpy as np
from typing import List, Tuple

from metahotspot.metahotspot_types import (
    MeshTopology,
    PhysicalFields,
    MaterialProps,
    LayerRegion,
)


class Mesher:
    GEOMETRY_TOLERANCE = 1e-15
    DEFAULT_MAX_MESH_SIZE = 0.01
    DEFAULT_MIN_MESH_SIZE = 0.0005
    DEFAULT_REFINEMENT_DISTANCE = 0.002

    def __init__(self, layer_regions: List[LayerRegion]):
        self.layer_regions = layer_regions

    def generate(
        self, mesh_params: dict = None
    ) -> Tuple[MeshTopology, PhysicalFields, np.ndarray, np.ndarray]:
        mesh_params = mesh_params or {}
        max_size = mesh_params.get("max_mesh_size", self.DEFAULT_MAX_MESH_SIZE)
        min_size = mesh_params.get("min_mesh_size", self.DEFAULT_MIN_MESH_SIZE)
        refine_dist = mesh_params.get(
            "refine_distance", self.DEFAULT_REFINEMENT_DISTANCE
        )

        heat_boxes = [
            (lr.lx, lr.ly, lr.lx + lr.dx, lr.ly + lr.dy)
            for lr in self.layer_regions
            if lr.is_active
        ]

        boxes_list = []
        k_list, cp_list, rho_list, fluid_list, visc_list = [], [], [], [], []
        l_ids, u_ids = [], []
        l_map, u_map = ["default_layer"], [""]

        # 直接将属性与 Box 生成绑定，消除后续的空间查找消耗
        for layer in self.layer_regions:
            if layer.name not in l_map:
                l_map.append(layer.name)
            l_id = l_map.index(layer.name)

            leaves = self._subdivide_layer(
                layer, max_size, min_size, refine_dist, heat_boxes
            )

            for x0, y0, x1, y1, unit in leaves:
                boxes_list.append([x0, y0, layer.lz, x1, y1, layer.lz + layer.dz])

                if unit.name not in u_map:
                    u_map.append(unit.name)
                u_id = u_map.index(unit.name)

                l_ids.append(l_id)
                u_ids.append(u_id)
                k_list.append(unit.props.k)
                cp_list.append(unit.props.cp)
                rho_list.append(unit.props.density)
                fluid_list.append(unit.props.is_fluid)
                visc_list.append(unit.props.dynamic_viscosity)

        c_boxes = np.array(boxes_list, dtype=np.float64)
        centers = (c_boxes[:, :3] + c_boxes[:, 3:]) * 0.5
        dims = c_boxes[:, 3:] - c_boxes[:, :3]
        vols = np.prod(dims, axis=1)

        # 空间莫顿排序优化访存局部性
        sorted_idx = self._compute_morton_sort(c_boxes[:, :3], c_boxes[:, 3:], centers)

        c_boxes = c_boxes[sorted_idx]
        centers = centers[sorted_idx]
        dims = dims[sorted_idx]
        vols = vols[sorted_idx]

        # 同步重排物理场数据
        k = np.array(k_list, dtype=np.float64)[sorted_idx]
        cp = np.array(cp_list, dtype=np.float64)[sorted_idx]
        rho = np.array(rho_list, dtype=np.float64)[sorted_idx]
        is_fluid = np.array(fluid_list, dtype=bool)[sorted_idx]
        visc = np.array(visc_list, dtype=np.float64)[sorted_idx]
        layer_ids = np.array(l_ids, dtype=np.int16)[sorted_idx]
        unit_ids = np.array(u_ids, dtype=np.int16)[sorted_idx]

        n_cells = len(c_boxes)

        # 向量化生成六面体节点数据 (8 nodes/cell)
        nodes = np.empty((n_cells, 8, 3), dtype=np.float64)
        nodes[:, 0] = c_boxes[:, [0, 1, 2]]
        nodes[:, 1] = c_boxes[:, [3, 1, 2]]
        nodes[:, 2] = c_boxes[:, [3, 4, 2]]
        nodes[:, 3] = c_boxes[:, [0, 4, 2]]
        nodes[:, 4] = c_boxes[:, [0, 1, 5]]
        nodes[:, 5] = c_boxes[:, [3, 1, 5]]
        nodes[:, 6] = c_boxes[:, [3, 4, 5]]
        nodes[:, 7] = c_boxes[:, [0, 4, 5]]

        flat_nodes = nodes.reshape(-1, 3)
        rounded_nodes = np.round(flat_nodes, decimals=9)
        points, inverse_idx = np.unique(rounded_nodes, axis=0, return_inverse=True)
        hex_data = inverse_idx.reshape(n_cells, 8)

        internal_faces, boundary_faces = self._build_topology_vectorized(
            points, hex_data, centers
        )

        topo = MeshTopology(
            n_cells=n_cells,
            centers=centers,
            dims=dims,
            boxes=c_boxes,
            volumes=vols,
            internal_faces=internal_faces,
            boundary_faces=boundary_faces,
        )

        fields = PhysicalFields(
            k=k,
            cp=cp,
            density=rho,
            is_fluid=is_fluid,
            dynamic_viscosity=visc,
            hydroC=np.zeros((n_cells, 3), dtype=np.float64),
            pressure=np.zeros(n_cells, dtype=np.float64),
            boundary_temperature=np.full(n_cells, np.nan, dtype=np.float64),
            layer_ids=layer_ids,
            unit_ids=unit_ids,
            layer_name_map=l_map,
            unit_name_map=u_map,
        )

        return topo, fields, points, hex_data

    def _subdivide_layer(self, layer, max_size, min_size, refine_dist, heat_boxes):
        leaves = []
        queue = deque([(u.lx, u.ly, u.lx + u.dx, u.ly + u.dy, u) for u in layer.units])

        while queue:
            x0, y0, x1, y1, u = queue.popleft()
            w, h = x1 - x0, y1 - y0
            needs_split = w > max_size or h > max_size

            if not needs_split and (w > min_size * 1.01 or h > min_size * 1.01):
                for hb in heat_boxes:
                    dist_x = max(0.0, x0 - hb[2], hb[0] - x1)
                    dist_y = max(0.0, y0 - hb[3], hb[1] - y1)
                    if math.hypot(dist_x, dist_y) <= refine_dist:
                        needs_split = True
                        break

            if needs_split:
                if w >= h:
                    mid = (x0 + x1) / 2.0
                    queue.extend([(x0, y0, mid, y1, u), (mid, y0, x1, y1, u)])
                else:
                    mid = (y0 + y1) / 2.0
                    queue.extend([(x0, y0, x1, mid, u), (x0, mid, x1, y1, u)])
            else:
                leaves.append((x0, y0, x1, y1, u))
        return leaves

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

    def _build_topology_vectorized(self, points, hex_data, centers):
        n_cells = len(hex_data)
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
        all_faces_nodes = hex_data[:, faces_def].reshape(-1, 4)

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

        pts = points[bound_face_nodes]
        cross = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        areas = np.linalg.norm(cross, axis=1)

        valid = areas > self.GEOMETRY_TOLERANCE
        normals = cross[valid] / areas[valid, None]
        b_c_ids = bound_c_ids[valid]
        b_areas = areas[valid]

        centers_dir = np.mean(pts[valid], axis=1) - centers[b_c_ids]
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
