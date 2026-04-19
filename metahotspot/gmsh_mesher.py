import math
from typing import Dict, List

import gmsh
import numpy as np


class GmshMesher:
    def __init__(self, model_name: str = "MetaHotspotMesh") -> None:
        gmsh.initialize()
        gmsh.model.add(model_name)

    def generate_2_5D_mesh(
        self,
        layers_entities: Dict[int, dict],
        power_units: List[dict],
        base_mesh_size: float = 0.006,
        min_mesh_size: float = 0.0005,
        refine_distance: float = 0.010,
    ) -> None:
        """
        生成 2.5D 挤压网格：层内 Quadtree 自适应（非共形），层间严格拉伸（共形）
        """
        # 1. 计算全局 2D 包围盒
        x_min = min(u["lx"] for l in layers_entities.values() for u in l["units"])
        x_max = max(
            u["lx"] + u["dx"] for l in layers_entities.values() for u in l["units"]
        )
        y_min = min(u["ly"] for l in layers_entities.values() for u in l["units"])
        y_max = max(
            u["ly"] + u["dy"] for l in layers_entities.values() for u in l["units"]
        )

        # 2. 收集所有物理边界和热源边界，用于触发网格细化
        geometry_boxes = []
        for l in layers_entities.values():
            for u in l["units"]:
                geometry_boxes.append(
                    (u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"])
                )

        heat_boxes = []
        for u in power_units:
            heat_boxes.append((u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"]))

        leaves = []

        def refine(x0: float, y0: float, x1: float, y1: float) -> None:
            """递归生成 2D Quadtree"""
            dx = x1 - x0
            dy = y1 - y0

            # 停止条件 1：达到最小网格尺寸
            if dx <= min_mesh_size * 1.01 and dy <= min_mesh_size * 1.01:
                leaves.append((x0, y0, x1, y1))
                return

            needs_refinement = False

            # 触发条件 1：距离热源较近 (捕捉热流扩散)
            for hb in heat_boxes:
                dist_x = max(0.0, x0 - hb[2], hb[0] - x1)
                dist_y = max(0.0, y0 - hb[3], hb[1] - y1)
                if math.sqrt(dist_x**2 + dist_y**2) <= refine_distance:
                    needs_refinement = True
                    break

            # 触发条件 2：网格跨越了物理边界 (确保网格完美贴合所有层级模块的边缘)
            if not needs_refinement:
                for gb in geometry_boxes:
                    # 如果垂直边界穿过当前网格，并且 Y 方向有交集
                    if (x0 < gb[0] < x1 or x0 < gb[2] < x1) and (
                        max(y0, gb[1]) < min(y1, gb[3])
                    ):
                        needs_refinement = True
                        break
                    # 如果水平边界穿过当前网格，并且 X 方向有交集
                    if (y0 < gb[1] < y1 or y0 < gb[3] < y1) and (
                        max(x0, gb[0]) < min(x1, gb[2])
                    ):
                        needs_refinement = True
                        break

            if needs_refinement:
                mid_x = (x0 + x1) / 2.0
                mid_y = (y0 + y1) / 2.0
                split_x = dx > min_mesh_size * 1.01
                split_y = dy > min_mesh_size * 1.01

                xs = [x0, mid_x, x1] if split_x else [x0, x1]
                ys = [y0, mid_y, y1] if split_y else [y0, y1]

                for i in range(len(xs) - 1):
                    for j in range(len(ys) - 1):
                        refine(xs[i], ys[j], xs[i + 1], ys[j + 1])
            else:
                leaves.append((x0, y0, x1, y1))

        # 3. 初始化基础粗网格并启动递归划分
        nx = max(1, int(round((x_max - x_min) / base_mesh_size)))
        ny = max(1, int(round((y_max - y_min) / base_mesh_size)))
        xs = np.linspace(x_min, x_max, nx + 1)
        ys = np.linspace(y_min, y_max, ny + 1)

        for i in range(nx):
            for j in range(ny):
                refine(xs[i], ys[j], xs[i + 1], ys[j + 1])

        # 4. 将 2D Quadtree 向上拉伸 (Extrude) 到 3D 的每一层
        node_id = 1
        elem_id = 1

        for tag, layer_data in layers_entities.items():
            discrete_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.addPhysicalGroup(3, [discrete_tag], tag)

            # 提取当前层的 Z 轴高度和厚度
            lz = layer_data["units"][0]["lz"]
            dz = layer_data["units"][0]["dz"]

            layer_node_tags = []
            layer_node_coords = []
            node_map = {}

            def get_node(x: float, y: float, z: float) -> int:
                nonlocal node_id
                key = (round(x, 6), round(y, 6), round(z, 6))
                if key not in node_map:
                    node_map[key] = node_id
                    layer_node_tags.append(node_id)
                    layer_node_coords.extend([x, y, z])
                    node_id += 1
                return node_map[key]

            element_tags = []
            element_nodes = []

            for leaf in leaves:
                x0, y0, x1, y1 = leaf
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

                # 只有当 2D 叶子节点落在当前物理层的有效区域内时，才生成 3D 实体
                inside = False
                for u in layer_data["units"]:
                    if (
                        u["lx"] <= cx <= u["lx"] + u["dx"]
                        and u["ly"] <= cy <= u["ly"] + u["dy"]
                    ):
                        inside = True
                        break

                if inside:
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
                    3, discrete_tag, layer_node_tags, layer_node_coords
                )
                gmsh.model.mesh.addElements(
                    3, discrete_tag, [5], [element_tags], [element_nodes]
                )

    def finalize(self, output_path: str) -> None:
        gmsh.write(output_path)
        gmsh.finalize()
