import toml
import os
import re
import gmsh
import numpy as np
import shutil


class HotSpotParser:
    @staticmethod
    def parse_flp(file_path):
        units = []
        if not os.path.exists(file_path):
            return units
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) >= 5:
                    units.append(
                        {
                            "name": parts[0],
                            "width": float(parts[1]),
                            "height": float(parts[2]),
                            "left_x": float(parts[3]),
                            "bottom_y": float(parts[4]),
                        }
                    )
        return units

    @staticmethod
    def parse_config(file_path):
        config = {}
        if not os.path.exists(file_path):
            return config
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(r"^-(\w+)\s+([^#]+)", line)
                if match:
                    key, val = match.groups()
                    val = val.strip()
                    try:
                        config[key] = float(val)
                    except:
                        config[key] = val
        return config

    @staticmethod
    def parse_materials(file_path):
        materials = {}
        if not os.path.exists(file_path):
            return materials
        with open(file_path, "r") as f:
            lines = [
                l.strip() for l in f if l.strip() and not l.strip().startswith("#")
            ]
            i = 0
            while i < len(lines):
                name = lines[i]
                m_type = lines[i + 1]
                k = float(lines[i + 2])
                cp = float(lines[i + 3])
                if m_type.lower() == "fluid":
                    v = float(lines[i + 4])
                    materials[name] = {
                        "k": k,
                        "cp": cp,
                        "fluid": True,
                        "dynamic_viscosity": v,
                    }
                    i += 5
                else:
                    materials[name] = {"k": k, "cp": cp, "fluid": False}
                    i += 4
        return materials

    @staticmethod
    def parse_lcf(file_path):
        layers = []
        if not os.path.exists(file_path):
            return layers
        with open(file_path, "r") as f:
            lines = [
                l.strip() for l in f if l.strip() and not l.strip().startswith("#")
            ]
            i = 0
            while i < len(lines):
                layer_num = int(lines[i])
                lateral, power = (
                    lines[i + 1].upper() == "Y",
                    lines[i + 2].upper() == "Y",
                )
                val_3 = lines[i + 3]
                try:
                    cp = float(val_3)
                    res = float(lines[i + 4])
                    thick = float(lines[i + 5])
                    flp = lines[i + 6]
                    layers.append(
                        {
                            "id": layer_num,
                            "power": power,
                            "cp": cp,
                            "k": 1.0 / res if res != 0 else 0,
                            "thickness": thick,
                            "flp_file": flp,
                            "type": "numeric",
                        }
                    )
                    i += 7
                except ValueError:
                    thick = float(lines[i + 4])
                    flp = lines[i + 5]
                    layers.append(
                        {
                            "id": layer_num,
                            "power": power,
                            "material": val_3,
                            "thickness": thick,
                            "flp_file": flp,
                            "type": "named",
                        }
                    )
                    i += 6
        return layers


class Mesher:
    def __init__(self):
        gmsh.initialize()
        gmsh.model.add("MetaHotspotMesh")

    def generate_layer_mesh_unified(
        self, tag, layer_entities, mesh_size, node_id_start, elem_id_start
    ):
        x_min, x_max = min(u["lx"] for u in layer_entities), max(
            u["lx"] + u["dx"] for u in layer_entities
        )
        y_min, y_max = min(u["ly"] for u in layer_entities), max(
            u["ly"] + u["dy"] for u in layer_entities
        )
        z_min, z_max = min(u["lz"] for u in layer_entities), max(
            u["lz"] + u["dz"] for u in layer_entities
        )
        xs = np.linspace(
            x_min, x_max, max(2, int(round((x_max - x_min) / mesh_size)) + 1)
        )
        ys = np.linspace(
            y_min, y_max, max(2, int(round((y_max - y_min) / mesh_size)) + 1)
        )
        zs = np.linspace(
            z_min, z_max, max(2, int(round((z_max - z_min) / mesh_size)) + 1)
        )
        d_tag = gmsh.model.addDiscreteEntity(3)
        gmsh.model.addPhysicalGroup(3, [d_tag], tag)
        node_id = node_id_start
        node_map = {}
        all_nt, all_nc = [], []
        for k in range(len(zs)):
            for j in range(len(ys)):
                for i in range(len(xs)):
                    all_nt.append(node_id)
                    all_nc.extend([xs[i], ys[j], zs[k]])
                    node_map[(i, j, k)] = node_id
                    node_id += 1
        gmsh.model.mesh.addNodes(3, d_tag, all_nt, all_nc)
        elem_id = elem_id_start
        el_ids, n_conns = [], []
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
                    el_ids.append(elem_id)
                    n_conns.extend(nodes)
                    elem_id += 1
        gmsh.model.mesh.addElements(3, d_tag, [5], [el_ids], [n_conns])
        return node_id, elem_id

    def finalize(self, output_path):
        gmsh.write(output_path)
        gmsh.finalize()


def convert_hotspot_to_metahotspot(example_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    parser = HotSpotParser()
    config = parser.parse_config(os.path.join(example_dir, "example.config"))
    materials = parser.parse_materials(os.path.join(example_dir, "example.materials"))

    std_mats = {
        "silicon": {"k": 130.0, "cp": 1.63e6, "fluid": False},
        "copper": {"k": 400.0, "cp": 3.44e6, "fluid": False},
        "aluminum": {"k": 237.0, "cp": 2.42e6, "fluid": False},
        "tim": {"k": 4.0, "cp": 4.0e6, "fluid": False},
    }
    for m, p in std_mats.items():
        if m not in materials:
            materials[m] = p

    total_w, total_h = 0.0, 0.0
    for root, _, files in os.walk(example_dir):
        for f in files:
            if f.endswith(".flp"):
                for u in parser.parse_flp(os.path.join(root, f)):
                    total_w = max(total_w, u["left_x"] + u["width"])
                    total_h = max(total_h, u["bottom_y"] + u["height"])
    if total_w == 0:
        total_w, total_h = 0.01, 0.01

    ptrace_src = next(
        (
            os.path.join(example_dir, f)
            for f in os.listdir(example_dir)
            if f.endswith(".ptrace")
        ),
        None,
    )
    ptrace_name = os.path.basename(ptrace_src) if ptrace_src else ""
    if ptrace_src:
        shutil.copy(ptrace_src, os.path.join(output_dir, ptrace_name))

    layers_entities, power_units, domain_assignment, z_cursor = {}, [], {}, 0.0
    lcf_path = next(
        (
            os.path.join(example_dir, f)
            for f in os.listdir(example_dir)
            if f.endswith(".lcf")
        ),
        None,
    )
    if lcf_path:
        for layer in parser.parse_lcf(lcf_path):
            tag = layer["id"] + 1
            mat = (
                f"layer_{layer['id']}_mat"
                if layer["type"] == "numeric"
                else layer["material"]
            )
            if layer["type"] == "numeric":
                materials[mat] = {"k": layer["k"], "cp": layer["cp"], "fluid": False}
            if mat not in domain_assignment:
                domain_assignment[mat] = []
            domain_assignment[mat].append(tag)
            m_size = 0.0005 if layer["power"] else 0.001
            layers_entities[tag] = {"mesh_size": m_size, "units": []}
            for u in parser.parse_flp(os.path.join(example_dir, layer["flp_file"])):
                ent = {
                    "name": u["name"],
                    "lx": u["left_x"],
                    "ly": u["bottom_y"],
                    "lz": z_cursor,
                    "dx": u["width"],
                    "dy": u["height"],
                    "dz": layer["thickness"],
                }
                layers_entities[tag]["units"].append(ent)
                if layer["power"]:
                    power_units.append(ent)
            z_cursor += layer["thickness"]
    else:
        flp_path = next(
            (
                os.path.join(example_dir, f)
                for f in os.listdir(example_dir)
                if f.endswith(".flp")
            ),
            None,
        )
        if flp_path:
            tag, t_chip = 1, config.get("t_chip", 0.00015)
            domain_assignment["silicon"] = [tag]
            layers_entities[tag] = {"mesh_size": 0.0005, "units": []}
            for u in parser.parse_flp(flp_path):
                ent = {
                    "name": u["name"],
                    "lx": u["left_x"],
                    "ly": u["bottom_y"],
                    "lz": z_cursor,
                    "dx": u["width"],
                    "dy": u["height"],
                    "dz": t_chip,
                }
                layers_entities[tag]["units"].append(ent)
                power_units.append(ent)
            z_cursor += t_chip

    def add_p(name, thick, side, mat, tag, ms):
        nonlocal z_cursor
        lx, ly = (total_w - side) / 2, (total_h - side) / 2
        layers_entities[tag] = {
            "mesh_size": ms,
            "units": [
                {
                    "name": name,
                    "lx": lx,
                    "ly": ly,
                    "lz": z_cursor,
                    "dx": side,
                    "dy": side,
                    "dz": thick,
                }
            ],
        }
        if mat not in domain_assignment:
            domain_assignment[mat] = []
        domain_assignment[mat].append(tag)
        z_cursor += thick

    add_p("TIM", config.get("t_tim", 0.00002), total_w, "tim", 1000, 0.001)
    add_p(
        "Spreader",
        config.get("t_spreader", 0.001),
        config.get("s_spreader", 0.03),
        "copper",
        1001,
        0.003,
    )
    add_p(
        "Sink",
        config.get("t_sink", 0.0069),
        config.get("s_sink", 0.06),
        "aluminum",
        1002,
        0.006,
    )

    toml_data = {
        "simulation_type": config.get("simulation_type", "steady"),
        "time": config.get("time", 0.1),
        "timestep": config.get("timestep", 0.01),
        "sampling_intvl": config.get("sampling_intvl", 0.01),
        "proc_freq": config.get("proc_freq", 3.0e9),
        "materials": materials,
        "domain_material_assignment": domain_assignment,
        "mesh_file_path": "mesh.msh",
        "ptrace_file_path": ptrace_name,
        "power_units": power_units,
        "ambient": config.get("ambient", 318.15),
        "boundary_conditions": [
            {
                "name": "sink_conv",
                "type": "convection",
                "h": 1.0
                / (config.get("r_convec", 0.1) * (config.get("s_sink", 0.06) ** 2)),
                "T_inf": config.get("ambient", 318.15),
                "selection": [1002],
            }
        ],
    }
    with open(os.path.join(output_dir, "solver_config.toml"), "w") as f:
        toml.dump(toml_data, f)
    m = Mesher()
    ni, ei = 1, 1
    for tag, data in layers_entities.items():
        ni = m.generate_layer_mesh_unified(
            tag, data["units"], data["mesh_size"], ni, ei
        )[0]
    m.finalize(os.path.join(output_dir, "mesh.msh"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python adapter.py <in> <out>")
    else:
        convert_hotspot_to_metahotspot(sys.argv[1], sys.argv[2])
