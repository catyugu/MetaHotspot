import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ==========================================
# 单一真相：全局默认配置与标准材料库
# ==========================================
DEFAULT_CONFIG = {
    "simulation_type": "steady",
    "ambient": 318.15,
    "init_temperature": 318.15,
    "t_chip": 0.00015,
    "t_tim": 0.00002,
    "t_spreader": 0.001,
    "t_sink": 0.0069,
    "base_proc_freq": 3.0e9,
    "r_convec": 0.1,
    "material_interface": "tim",
    "material_spreader": "copper",
    "material_sink": "copper",
    "init_file": "",
    "sampling_intvl": 0.01,
    "time": 0.01,
    "timestep": 0.01,
    "mesh_file_path": "mesh.msh",
    "ptrace_file_path": "",
    "init_temperature_file_path": "",
    "pumping_pressure": 52000.0,
    "inlet_temperature": 298.15,
    "boundary_conditions": [],
    "stackup": [],
    "materials": {},
}

STANDARD_MATERIALS = {
    "silicon": {"k": 130.0, "cp": 1.63e6, "fluid": False, "density": 2330.0},
    "copper": {"k": 400.0, "cp": 3.44e6, "fluid": False, "density": 8960.0},
    "aluminum": {"k": 237.0, "cp": 2.42e6, "fluid": False, "density": 2700.0},
    "tim": {"k": 4.0, "cp": 4.0e6, "fluid": False, "density": 1000.0},
    "water": {
        "k": 0.6069,
        "cp": 4.17e6,
        "fluid": True,
        "dynamic_viscosity": 8.89e-4,
        "density": 1000.0,
    },
    "default_solid": {"k": 1.0, "cp": 1.0e6, "fluid": False, "density": 1000.0},
}


@dataclass
class Unit2D:
    """2D layout unit for FVM mesh generation."""

    name: str
    lx: float
    ly: float
    dx: float
    dy: float
    material: str
    k: float
    cp: float
    is_fluid: bool


@dataclass
class Layer25D:
    """2.5D layer definition with fully resolved properties."""

    name: str
    tag: int
    thickness: float
    material: str
    k: float
    cp: float
    is_fluid: bool
    active: bool
    units: List[Unit2D] = field(default_factory=list)
    lx: float = 0.0
    ly: float = 0.0
    dx: float = 0.01
    dy: float = 0.01


def load_config(config_path: str) -> Dict[str, Any]:
    """统一配置加载入口：确保下游直接读取到清洗完毕、具有绝对信任度的配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    return merge_with_defaults(raw_config)


def merge_with_defaults(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """配置关口：将原始配置与默认值合并。一处更改，处处有效。"""
    config = dict(DEFAULT_CONFIG)

    for k, v in raw_config.items():
        if k in config and type(config[k]) is not type(v):
            try:
                if v not in {"(null)", "null", "None", ""}:
                    config[k] = type(config[k])(v)
            except ValueError:
                config[k] = v
        else:
            config[k] = v

    config["t_interface"] = raw_config.get("t_interface", config["t_tim"])
    config["time"] = raw_config.get("time", max(config["sampling_intvl"], 0.01))
    config["timestep"] = raw_config.get("timestep", config["sampling_intvl"])

    if "init_temp" in raw_config:
        config["init_temperature"] = float(raw_config["init_temp"])

    for mat_name, mat_props in STANDARD_MATERIALS.items():
        if mat_name not in config["materials"]:
            config["materials"][mat_name] = dict(mat_props)

    return config


def load_stackup(config: Dict[str, Any], base_dir: str) -> List[Layer25D]:
    """Load 2.5D stackup model and resolve all properties according to priority rules."""
    layers = []
    stackup_data = config.get("stackup", [])
    materials = config.get("materials", {})
    default_solid = materials.get("default_solid", STANDARD_MATERIALS["default_solid"])

    for i, layer_cfg in enumerate(stackup_data):
        tag = int(layer_cfg.get("tag", i + 100))
        name = str(layer_cfg.get("name", f"layer_{tag}"))
        lx, ly = float(layer_cfg.get("lx", 0.0)), float(layer_cfg.get("ly", 0.0))
        dx, dy = float(layer_cfg.get("dx", 0.01)), float(layer_cfg.get("dy", 0.01))

        layer_mat_name = layer_cfg.get("material", "silicon")
        layer_mat = materials.get(layer_mat_name, default_solid)
        layer_k = float(layer_mat.get("k", default_solid["k"]))
        layer_cp = float(layer_mat.get("cp", default_solid["cp"]))
        layer_is_fluid = bool(layer_mat.get("fluid", False))

        units = []
        layout_file = layer_cfg.get("layout_file", "")

        if layout_file and layout_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, layout_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for u in json.load(f):
                        unit_mat_name = u.get("material", layer_mat_name)
                        unit_mat = materials.get(unit_mat_name, layer_mat)

                        # Priority: Unit direct > Unit Material > Layer Material
                        u_k = u.get("k")
                        u_k = (
                            float(u_k)
                            if u_k is not None
                            else float(unit_mat.get("k", layer_k))
                        )

                        u_cp = u.get("cp")
                        u_cp = (
                            float(u_cp)
                            if u_cp is not None
                            else float(unit_mat.get("cp", layer_cp))
                        )

                        u_is_fluid = bool(
                            u.get("is_fluid", unit_mat.get("fluid", layer_is_fluid))
                        )

                        units.append(
                            Unit2D(
                                name=u["name"],
                                lx=float(u["lx"]),
                                ly=float(u["ly"]),
                                dx=float(u["dx"]),
                                dy=float(u["dy"]),
                                material=unit_mat_name,
                                k=u_k,
                                cp=u_cp,
                                is_fluid=u_is_fluid,
                            )
                        )

        if not units:
            units.append(
                Unit2D(
                    name=f"{name}_bulk",
                    lx=lx,
                    ly=ly,
                    dx=dx,
                    dy=dy,
                    material=layer_mat_name,
                    k=layer_k,
                    cp=layer_cp,
                    is_fluid=layer_is_fluid,
                )
            )

        layers.append(
            Layer25D(
                name=name,
                tag=tag,
                thickness=float(layer_cfg["thickness"]),
                material=layer_mat_name,
                k=layer_k,
                cp=layer_cp,
                is_fluid=layer_is_fluid,
                active=bool(layer_cfg.get("active", False)),
                units=units,
                lx=lx,
                ly=ly,
                dx=dx,
                dy=dy,
            )
        )

    return layers
