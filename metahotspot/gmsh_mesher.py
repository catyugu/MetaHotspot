import math
from collections import deque
from pathlib import Path
from typing import List

import gmsh
import toml
from metahotspot.model25d import load_stackup


class GmshMesher:
    """Mesher that takes a config TOML path and produces a .msh file.

    Decoupled from converter - call with config path after conversion.
    """

    DEFAULT_MAX_MESH_SIZE = 0.003
    DEFAULT_MIN_MESH_SIZE = 0.0005
    DEFAULT_REFINEMENT_DISTANCE = 0.001

    def __init__(self, model_name: str = "MetaHotspotMesh") -> None:
        gmsh.initialize()
        gmsh.model.add(model_name)
        self._node_id = 1
        self._elem_id = 1
        self._node_map: dict = {}
        self._global_node_coords: dict = {}

    def generate_mesh(self, config_path: str, mesh_params: dict = None) -> None:
        """Generate mesh from config TOML path.

        Args:
            config_path: Path to solver_config.toml
            mesh_params: Optional dict with max_mesh_size, min_mesh_size, refine_distance.
                        Defaults to GmshMesher.DEFAULT_* values.
        """
        if mesh_params is None:
            mesh_params = {}

        base_dir = str(Path(config_path).parent)
        config = toml.load(config_path)

        max_mesh_size = mesh_params.get("max_mesh_size", self.DEFAULT_MAX_MESH_SIZE)
        min_mesh_size = mesh_params.get("min_mesh_size", self.DEFAULT_MIN_MESH_SIZE)
        refine_distance = mesh_params.get(
            "refine_distance", self.DEFAULT_REFINEMENT_DISTANCE
        )

        stackup = load_stackup(config, base_dir)
        self._generate_2_5D_mesh(stackup, max_mesh_size, min_mesh_size, refine_distance)

    def _generate_2_5D_mesh(
        self,
        stackup,
        max_mesh_size: float,
        min_mesh_size: float,
        refine_distance: float,
    ) -> None:
        """Internal mesh generation logic."""

        # Collect heat source boxes for mesh refinement
        heat_boxes = []
        for layer in stackup:
            if layer.active:
                for u in layer.units:
                    heat_boxes.append((u.lx, u.ly, u.lx + u.dx, u.ly + u.dy))

        z_cursor = 0.0

        for layer in stackup:
            discrete_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.addPhysicalGroup(3, [discrete_tag], layer.tag)

            lz = z_cursor
            dz = layer.thickness
            z_cursor += dz

            leaves = self._subdivide_layer(
                layer, max_mesh_size, min_mesh_size, refine_distance, heat_boxes
            )
            self._create_hex_elements(layer, discrete_tag, lz, dz, leaves)

    def _subdivide_layer(
        self, layer, max_mesh_size, min_mesh_size, refine_distance, heat_boxes
    ):
        """Subdivide layer into quad leaves for hex mesh generation."""
        leaves = []
        queue = deque()

        for u in layer.units:
            queue.append((u.lx, u.ly, u.lx + u.dx, u.ly + u.dy))

        while queue:
            x0, y0, x1, y1 = queue.popleft()
            w = x1 - x0
            h = y1 - y0

            needs_split = False

            if w > max_mesh_size or h > max_mesh_size:
                needs_split = True
            elif w > min_mesh_size * 1.01 or h > min_mesh_size * 1.01:
                for hb in heat_boxes:
                    dist_x = max(0.0, x0 - hb[2], hb[0] - x1)
                    dist_y = max(0.0, y0 - hb[3], hb[1] - y1)
                    if math.hypot(dist_x, dist_y) <= refine_distance:
                        needs_split = True
                        break

            if needs_split:
                if w >= h:
                    mid = (x0 + x1) / 2.0
                    queue.append((x0, y0, mid, y1))
                    queue.append((mid, y0, x1, y1))
                else:
                    mid = (y0 + y1) / 2.0
                    queue.append((x0, y0, x1, mid))
                    queue.append((x0, mid, x1, y1))
            else:
                leaves.append((x0, y0, x1, y1))

        return leaves

    def _get_node(self, x: float, y: float, z: float) -> int:
        """Get or create a node at (x, y, z)."""
        key = (round(x, 12), round(y, 12), round(z, 12))
        if key not in self._node_map:
            self._node_map[key] = self._node_id
            self._global_node_coords[self._node_id] = (x, y, z)
            self._node_id += 1
        return self._node_map[key]

    def _create_hex_elements(self, layer, discrete_tag, lz, dz, leaves) -> None:
        """Create hex elements for a layer's quad leaves."""
        element_tags: List[int] = []
        element_nodes: List[int] = []
        used_node_ids = set()

        for x0, y0, x1, y1 in leaves:
            # Collect nodes for bottom face (-Z)
            n0 = self._get_node(x0, y0, lz)
            n1 = self._get_node(x1, y0, lz)
            n2 = self._get_node(x1, y1, lz)
            n3 = self._get_node(x0, y1, lz)

            # Collect nodes for top face (+Z)
            n4 = self._get_node(x0, y0, lz + dz)
            n5 = self._get_node(x1, y0, lz + dz)
            n6 = self._get_node(x1, y1, lz + dz)
            n7 = self._get_node(x0, y1, lz + dz)

            element_tags.append(self._elem_id)
            element_nodes.extend([n0, n1, n2, n3, n4, n5, n6, n7])
            used_node_ids.update([n0, n1, n2, n3, n4, n5, n6, n7])
            self._elem_id += 1

        if element_tags:
            # Build ordered node lists for addNodes
            layer_nodes_tags = sorted(used_node_ids)
            layer_nodes_coords = []
            for nid in layer_nodes_tags:
                x, y, z = self._global_node_coords[nid]
                layer_nodes_coords.extend([x, y, z])

            gmsh.model.mesh.addNodes(
                3, discrete_tag, layer_nodes_tags, layer_nodes_coords
            )
            gmsh.model.mesh.addElements(
                3, discrete_tag, [5], [element_tags], [element_nodes]
            )

    def finalize(self, output_path: str) -> None:
        """Write mesh file and cleanup gmsh."""
        gmsh.write(output_path)
        gmsh.finalize()
