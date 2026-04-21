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
    ) -> dict:
        """
        局部剖分策略：
        1. 以每一层的实际 functional units 作为初始网格节点（完美贴合 unit 边界，绝不外延拉伸）。
        2. 若单元过大 (w or h > max_mesh_size)，对其长边进行中点切分。
        3. 若单元处于热源附近，继续对长边进行细化，直至逼近 min_mesh_size。

        返回:
            boundary_info (dict): 包含所有外表面分组信息的元数据，供 Converter 过滤。
        """
        heat_boxes = [
            (u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"])
            for u in power_units
        ]

        node_id = 1
        elem_id = 1

        global_node_coords = {}
        all_hex_elements = []

        for tag, layer_data in layers_entities.items():
            discrete_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.addPhysicalGroup(3, [discrete_tag], tag)

            lz = layer_data["units"][0]["lz"]
            dz = layer_data["units"][0]["dz"]

            leaves = []
            queue = deque()

            for u in layer_data["units"]:
                queue.append((u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"]))

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

            layer_nodes_tags = []
            layer_nodes_coords = []
            node_map = {}

            def get_node(x: float, y: float, z: float) -> int:
                nonlocal node_id
                key = (round(x, 12), round(y, 12), round(z, 12))
                if key not in node_map:
                    node_map[key] = node_id
                    layer_nodes_tags.append(node_id)
                    layer_nodes_coords.extend([x, y, z])
                    global_node_coords[node_id] = (x, y, z)
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
                all_hex_elements.extend(element_nodes)

        # ---------------------------------------------------------
        # 边界自然分组与编号 (拓扑提取)
        # ---------------------------------------------------------
        faces_count = {}
        for i in range(0, len(all_hex_elements), 8):
            n = all_hex_elements[i : i + 8]
            # 六面体的六个面 (统一向内或向外的节点顺序并不影响判断重复面)
            fs = [
                tuple(sorted([n[0], n[3], n[2], n[1]])),
                tuple(sorted([n[4], n[5], n[6], n[7]])),
                tuple(sorted([n[0], n[1], n[5], n[4]])),
                tuple(sorted([n[3], n[7], n[6], n[2]])),
                tuple(sorted([n[0], n[4], n[7], n[3]])),
                tuple(sorted([n[1], n[2], n[6], n[5]])),
            ]
            for f in fs:
                faces_count[f] = faces_count.get(f, 0) + 1

        # 仅出现一次的面为外表面
        boundary_faces = [f for f, count in faces_count.items() if count == 1]

        # 按照共面性 (平面方程) 分组
        groups = {}
        for f in boundary_faces:
            pts = [global_node_coords[n_tag] for n_tag in f]
            xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

            # 因为是正交网格，根据坐标恒定值判断平面轴
            if max(xs) - min(xs) < 1e-9:
                axis, val = "X", round(xs[0], 6)
            elif max(ys) - min(ys) < 1e-9:
                axis, val = "Y", round(ys[0], 6)
            else:
                axis, val = "Z", round(zs[0], 6)

            groups.setdefault((axis, val), []).append(f)

        boundary_info = {}
        base_tag = 2000

        for (axis_name, val), faces in groups.items():
            base_tag += 1
            ent_tag = gmsh.model.addDiscreteEntity(2)
            gmsh.model.addPhysicalGroup(
                2, [ent_tag], base_tag, name=f"boundary_{axis_name}_{val}"
            )

            elem_tags = [elem_id + i for i in range(len(faces))]
            elem_id += len(faces)

            elem_nodes = []
            for f in faces:
                pts_with_id = [(global_node_coords[n_tag], n_tag) for n_tag in f]
                cx = sum(p[0][0] for p in pts_with_id) / 4.0
                cy = sum(p[0][1] for p in pts_with_id) / 4.0
                cz = sum(p[0][2] for p in pts_with_id) / 4.0

                # 为满足 Gmsh 四边形图元定义，将节点按照相对于重心的极角环向排序
                if axis_name == "X":
                    pts_with_id.sort(
                        key=lambda item: math.atan2(item[0][2] - cz, item[0][1] - cy)
                    )
                elif axis_name == "Y":
                    pts_with_id.sort(
                        key=lambda item: math.atan2(item[0][2] - cz, item[0][0] - cx)
                    )
                else:
                    pts_with_id.sort(
                        key=lambda item: math.atan2(item[0][1] - cy, item[0][0] - cx)
                    )

                elem_nodes.extend([item[1] for item in pts_with_id])

            gmsh.model.mesh.addElements(2, ent_tag, [3], [elem_tags], [elem_nodes])

            boundary_info[base_tag] = {
                "axis": axis_name,
                "val": val,
                "name": f"boundary_{axis_name}_{val}",
            }

        return boundary_info

    def finalize(self, output_path: str) -> None:
        gmsh.write(output_path)
        gmsh.finalize()
