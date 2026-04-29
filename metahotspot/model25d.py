import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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
}

STANDARD_MATERIALS = {
    "silicon": {"k": 130.0, "cp": 1.63e6, "fluid": False},
    "copper": {"k": 400.0, "cp": 3.44e6, "fluid": False},
    "aluminum": {"k": 237.0, "cp": 2.42e6, "fluid": False},
    "tim": {"k": 4.0, "cp": 4.0e6, "fluid": False},
    "water": {"k": 0.6069, "cp": 4.17e6, "fluid": True, "dynamic_viscosity": 8.89e-4},
    "default_solid": {"k": 1.0, "cp": 1.0e6, "fluid": False},
}


@dataclass
class Unit2D:
    """2D layout unit for FVM mesh generation."""

    name: str
    lx: float
    ly: float
    dx: float
    dy: float
    material: Optional[str] = None
    k: Optional[float] = None
    cp: Optional[float] = None
    is_fluid: bool = False


@dataclass
class Layer25D:
    name: str
    tag: int
    thickness: float
    default_material: str
    active: bool
    units: List[Unit2D] = field(default_factory=list)
    lx: float = 0.0
    ly: float = 0.0
    dx: float = 0.01
    dy: float = 0.01


def merge_with_defaults(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """配置关口：将原始配置与默认值合并，保证下游读取时绝对安全。"""
    config = dict(DEFAULT_CONFIG)
    for k, v in raw_config.items():
        if k in config:
            config[k] = (
                type(config[k])(v) if v not in {"(null)", "null", "None"} else ""
            )
        else:
            config[k] = v

    # 特殊依赖回退处理 (一处处理，处处有效)
    if "t_interface" not in raw_config:
        config["t_interface"] = config["t_tim"]
    if "time" not in raw_config:
        config["time"] = max(config["sampling_intvl"], 0.01)
    if "timestep" not in raw_config:
        config["timestep"] = config["sampling_intvl"]
    if "init_temp" in raw_config:
        config["init_temperature"] = float(raw_config["init_temp"])

    return config


def load_stackup(config: Dict[str, Any], base_dir: str) -> List[Layer25D]:
    """Load 2.5D stackup model from strict config."""
    layers = []
    stackup_data = config.get("stackup", [])

    for i, layer_cfg in enumerate(stackup_data):
        tag = int(layer_cfg.get("tag", i + 100))
        name = str(layer_cfg.get("name", f"layer_{tag}"))
        lx, ly = float(layer_cfg.get("lx", 0.0)), float(layer_cfg.get("ly", 0.0))
        dx, dy = float(layer_cfg.get("dx", 0.01)), float(layer_cfg.get("dy", 0.01))
        material = layer_cfg.get("material", "silicon")

        units = []
        layout_file = layer_cfg.get("layout_file", "")

        if layout_file and layout_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, layout_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for u in json.load(f):
                        units.append(
                            Unit2D(
                                name=u["name"],
                                lx=float(u["lx"]),
                                ly=float(u["ly"]),
                                dx=float(u["dx"]),
                                dy=float(u["dy"]),
                                material=u.get("material"),
                                k=u.get("k"),
                                cp=u.get("cp"),
                                is_fluid=bool(u.get("is_fluid", False)),
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
                    material=material,
                    is_fluid=False,
                )
            )

        layers.append(
            Layer25D(
                name=name,
                tag=tag,
                thickness=float(layer_cfg["thickness"]),
                default_material=material,
                active=bool(layer_cfg.get("active", False)),
                units=units,
                lx=lx,
                ly=ly,
                dx=dx,
                dy=dy,
            )
        )

    return layers
