import math
from typing import Dict, List
from collections import deque

import gmsh


class GmshMesher:
    def __init__(self, model_name: str = "MetaHotspotMesh") -> None:
        gmsh.initialize()
        gmsh.model.add(model_name)

    def generate_2_5D_mesh(
        self,
        layers_entities: Dict[int, dict],
        power_units: List[dict],
        max_mesh_size: float = 0.006,
        min_mesh_size: float = 0.0005,
        refine_distance: float = 0.010,
    ) -> None:
        """
        局部剖分策略：
        1. 以每一层的实际 functional units 作为初始网格节点（完美贴合 unit 边界，绝不外延拉伸）。
        2. 若单元过大 (w or h > max_mesh_size)，对其长边进行中点切分。
        3. 若单元处于热源附近，继续对长边进行细化，直至逼近 min_mesh_size。
        """
        # 提前收集热源框，用于局部加密判定
        heat_boxes = [
            (u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"])
            for u in power_units
        ]

        node_id = 1
        elem_id = 1

        for tag, layer_data in layers_entities.items():
            discrete_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.addPhysicalGroup(3, [discrete_tag], tag)

            lz = layer_data["units"][0]["lz"]
            dz = layer_data["units"][0]["dz"]

            leaves = []
            queue = deque()

            # 初始化：直接以功能单元的物理边界框作为待细化的基础几何网格
            for u in layer_data["units"]:
                queue.append((u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"]))

            # 递归细分
            while queue:
                x0, y0, x1, y1 = queue.popleft()
                w = x1 - x0
                h = y1 - y0

                needs_split = False

                # 判定条件 1: 网格尺寸大于允许的最大尺寸限制
                if w > max_mesh_size or h > max_mesh_size:
                    needs_split = True
                # 判定条件 2: 网格在热源的影响范围内，且长边仍大于最小尺寸限制
                elif w > min_mesh_size * 1.01 or h > min_mesh_size * 1.01:
                    for hb in heat_boxes:
                        dist_x = max(0.0, x0 - hb[2], hb[0] - x1)
                        dist_y = max(0.0, y0 - hb[3], hb[1] - y1)
                        if math.hypot(dist_x, dist_y) <= refine_distance:
                            needs_split = True
                            break

                # 执行切分：永远沿着最长的边切分一刀
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

            # 根据最终的 leaves 构建当前层独立的 3D Hexahedrons (完全抛弃全层间的强行共形)
            layer_nodes_tags = []
            layer_nodes_coords = []
            node_map = {}

            def get_node(x: float, y: float, z: float) -> int:
                nonlocal node_id
                # 保持坐标精度位以防止浮点数误差产生冗余节点
                key = (round(x, 12), round(y, 12), round(z, 12))
                if key not in node_map:
                    node_map[key] = node_id
                    layer_nodes_tags.append(node_id)
                    layer_nodes_coords.extend([x, y, z])
                    node_id += 1
                return node_map[key]

            element_tags = []
            element_nodes = []

            for x0, y0, x1, y1 in leaves:
                n0 = get_node(x0, y0, lz)
                n1 = get_node(x1, y0, lz)
                n2 = get_node(x1, y1, lz)
                n3 = get_node(x0, y1, lz)

                n4 = get_node(x0, y0, lz + dz)
                n5 = get_node(x1, y0, lz + dz)
                n6 = get_node(x1, y1, lz + dz)
                n7 = get_node(x0, y1, lz + dz)

                element_tags.append(elem_id)
                element_nodes.extend([n0, n1, n2, n3, n4, n5, n6, n7])
                elem_id += 1

            if element_tags:
                gmsh.model.mesh.addNodes(
                    3, discrete_tag, layer_nodes_tags, layer_nodes_coords
                )
                gmsh.model.mesh.addElements(
                    3, discrete_tag, [5], [element_tags], [element_nodes]
                )

    def finalize(self, output_path: str) -> None:
        gmsh.write(output_path)
        gmsh.finalize()
