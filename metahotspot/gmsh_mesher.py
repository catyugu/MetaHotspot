import math
from collections import deque
import gmsh
import meshio
import numpy as np
from typing import List

from metahotspot.metahotspot_types import LayerRegion, ActiveRegion


class GmshMesher:
    DEFAULT_MAX_MESH_SIZE = 0.01
    DEFAULT_MIN_MESH_SIZE = 0.0005
    DEFAULT_REFINEMENT_DISTANCE = 0.002

    def __init__(self, model_name: str = "MetaHotspotMesh") -> None:
        self.model_name = model_name
        self._node_id = 1
        self._elem_id = 1
        self._node_map: dict = {}
        self._global_node_coords: dict = {}

    def generate_mesh(
        self,
        layer_regions: List[LayerRegion],
        active_regions: List[ActiveRegion],
        mesh_params: dict = None,
    ) -> meshio.Mesh:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Mute stdout warnings optionally
        gmsh.model.add(self.model_name)

        mesh_params = mesh_params or {}
        max_mesh_size = mesh_params.get("max_mesh_size", self.DEFAULT_MAX_MESH_SIZE)
        min_mesh_size = mesh_params.get("min_mesh_size", self.DEFAULT_MIN_MESH_SIZE)
        refine_distance = mesh_params.get(
            "refine_distance", self.DEFAULT_REFINEMENT_DISTANCE
        )

        heat_boxes = [
            (ps.lx, ps.ly, ps.lx + ps.dx, ps.ly + ps.dy) for ps in active_regions
        ]

        for layer in layer_regions:
            discrete_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.addPhysicalGroup(3, [discrete_tag], layer.tag)

            lz, dz = layer.lz, layer.dz

            leaves = self._subdivide_layer(
                layer, max_mesh_size, min_mesh_size, refine_distance, heat_boxes
            )
            self._create_hex_elements(discrete_tag, lz, dz, leaves)

        mesh = self._extract_meshio_mesh()

        gmsh.finalize()
        self._node_map.clear()
        self._global_node_coords.clear()

        return mesh

    def _subdivide_layer(
        self, layer, max_mesh_size, min_mesh_size, refine_distance, heat_boxes
    ):
        leaves, queue = [], deque(
            [(u.lx, u.ly, u.lx + u.dx, u.ly + u.dy) for u in layer.units]
        )

        while queue:
            x0, y0, x1, y1 = queue.popleft()
            w, h = x1 - x0, y1 - y0
            needs_split = w > max_mesh_size or h > max_mesh_size

            if not needs_split and (
                w > min_mesh_size * 1.01 or h > min_mesh_size * 1.01
            ):
                for hb in heat_boxes:
                    dist_x, dist_y = max(0.0, x0 - hb[2], hb[0] - x1), max(
                        0.0, y0 - hb[3], hb[1] - y1
                    )
                    if math.hypot(dist_x, dist_y) <= refine_distance:
                        needs_split = True
                        break

            if needs_split:
                if w >= h:
                    mid = (x0 + x1) / 2.0
                    queue.extend([(x0, y0, mid, y1), (mid, y0, x1, y1)])
                else:
                    mid = (y0 + y1) / 2.0
                    queue.extend([(x0, y0, x1, mid), (x0, mid, x1, y1)])
            else:
                leaves.append((x0, y0, x1, y1))

        return leaves

    def _get_node(self, x: float, y: float, z: float) -> int:
        key = (round(x, 12), round(y, 12), round(z, 12))
        if key not in self._node_map:
            self._node_map[key] = self._node_id
            self._global_node_coords[self._node_id] = (x, y, z)
            self._node_id += 1
        return self._node_map[key]

    def _create_hex_elements(self, discrete_tag, lz, dz, leaves) -> None:
        element_tags, element_nodes, used_node_ids = [], [], set()

        for x0, y0, x1, y1 in leaves:
            nodes = [
                self._get_node(x0, y0, lz),
                self._get_node(x1, y0, lz),
                self._get_node(x1, y1, lz),
                self._get_node(x0, y1, lz),
                self._get_node(x0, y0, lz + dz),
                self._get_node(x1, y0, lz + dz),
                self._get_node(x1, y1, lz + dz),
                self._get_node(x0, y1, lz + dz),
            ]
            element_tags.append(self._elem_id)
            element_nodes.extend(nodes)
            used_node_ids.update(nodes)
            self._elem_id += 1

        if element_tags:
            layer_nodes_tags = sorted(used_node_ids)
            layer_nodes_coords = [
                coord
                for nid in layer_nodes_tags
                for coord in self._global_node_coords[nid]
            ]
            gmsh.model.mesh.addNodes(
                3, discrete_tag, layer_nodes_tags, layer_nodes_coords
            )
            gmsh.model.mesh.addElements(
                3, discrete_tag, [5], [element_tags], [element_nodes]
            )

    def _extract_meshio_mesh(self) -> meshio.Mesh:
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = np.array(node_coords).reshape(-1, 3)
        tag2idx = {tag: i for i, tag in enumerate(node_tags)}

        hex_data = []
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

        for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
            if etype == 5:
                # Ensure elements strictly match the order they were generated/tagged
                sort_idx = np.argsort(etags)
                sorted_enodes = np.array(enodes).reshape(-1, 8)[sort_idx]
                arr = np.array([tag2idx[t] for t in sorted_enodes.flat]).reshape(-1, 8)
                hex_data.append(arr)

        cells = [("hexahedron", np.vstack(hex_data))] if hex_data else []
        return meshio.Mesh(points=points, cells=cells)
