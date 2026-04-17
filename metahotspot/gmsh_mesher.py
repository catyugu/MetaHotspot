from typing import Dict, List, Tuple

import gmsh
import numpy as np


class GmshMesher:
    def __init__(self, model_name: str = "MetaHotspotMesh") -> None:
        gmsh.initialize()
        gmsh.model.add(model_name)

    def generate_layer_mesh_unified(
        self,
        tag: int,
        layer_entities: List[dict],
        mesh_size: float,
        node_id_start: int,
        elem_id_start: int,
    ) -> Tuple[int, int]:
        x_min = min(unit["lx"] for unit in layer_entities)
        x_max = max(unit["lx"] + unit["dx"] for unit in layer_entities)
        y_min = min(unit["ly"] for unit in layer_entities)
        y_max = max(unit["ly"] + unit["dy"] for unit in layer_entities)
        z_min = min(unit["lz"] for unit in layer_entities)
        z_max = max(unit["lz"] + unit["dz"] for unit in layer_entities)

        xs = np.linspace(
            x_min, x_max, max(2, int(round((x_max - x_min) / mesh_size)) + 1)
        )
        ys = np.linspace(
            y_min, y_max, max(2, int(round((y_max - y_min) / mesh_size)) + 1)
        )
        zs = np.linspace(
            z_min, z_max, max(2, int(round((z_max - z_min) / mesh_size)) + 1)
        )

        discrete_tag = gmsh.model.addDiscreteEntity(3)
        gmsh.model.addPhysicalGroup(3, [discrete_tag], tag)

        node_id = node_id_start
        node_map: Dict[tuple, int] = {}
        node_tags: List[int] = []
        node_coords: List[float] = []

        for k in range(len(zs)):
            for j in range(len(ys)):
                for i in range(len(xs)):
                    node_tags.append(node_id)
                    node_coords.extend([xs[i], ys[j], zs[k]])
                    node_map[(i, j, k)] = node_id
                    node_id += 1

        gmsh.model.mesh.addNodes(3, discrete_tag, node_tags, node_coords)

        elem_id = elem_id_start
        element_tags: List[int] = []
        element_nodes: List[int] = []

        for k in range(len(zs) - 1):
            for j in range(len(ys) - 1):
                for i in range(len(xs) - 1):
                    nodes = [
                        node_map[(i, j, k)],
                        node_map[(i + 1, j, k)],
                        node_map[(i + 1, j + 1, k)],
                        node_map[(i, j + 1, k)],
                        node_map[(i, j, k + 1)],
                        node_map[(i + 1, j, k + 1)],
                        node_map[(i + 1, j + 1, k + 1)],
                        node_map[(i, j + 1, k + 1)],
                    ]
                    element_tags.append(elem_id)
                    element_nodes.extend(nodes)
                    elem_id += 1

        # 5 is Gmsh's hexahedron element type.
        gmsh.model.mesh.addElements(
            3, discrete_tag, [5], [element_tags], [element_nodes]
        )

        return node_id, elem_id

    def finalize(self, output_path: str) -> None:
        gmsh.write(output_path)
        gmsh.finalize()
