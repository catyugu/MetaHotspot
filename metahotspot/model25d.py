import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

from metahotspot.metahotspot_types import SolverConfig, BoundaryCondition, MaterialProps

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
    "silicon": {
        "k": 130.0,
        "cp": 1.63e6,
        "fluid": False,
        "density": 2330.0,
        "dynamic_viscosity": 0.0,
    },
    "copper": {
        "k": 400.0,
        "cp": 3.44e6,
        "fluid": False,
        "density": 8960.0,
        "dynamic_viscosity": 0.0,
    },
    "aluminum": {
        "k": 237.0,
        "cp": 2.42e6,
        "fluid": False,
        "density": 2700.0,
        "dynamic_viscosity": 0.0,
    },
    "tim": {
        "k": 4.0,
        "cp": 4.0e6,
        "fluid": False,
        "density": 1000.0,
        "dynamic_viscosity": 0.0,
    },
    "water": {
        "k": 0.6069,
        "cp": 4.17e6,
        "fluid": True,
        "density": 1000.0,
        "dynamic_viscosity": 8.89e-4,
    },
    "default_solid": {
        "k": 1.0,
        "cp": 1.0e6,
        "fluid": False,
        "density": 1000.0,
        "dynamic_viscosity": 0.0,
    },
}


@dataclass(slots=True)
class Unit2D:
    """2D layout unit for FVM mesh generation with full property resolution."""

    name: str
    lx: float
    ly: float
    dx: float
    dy: float
    material: str
    k: float
    cp: float
    density: float
    dynamic_viscosity: float
    is_fluid: bool


@dataclass(slots=True)
class Layer25D:
    """2.5D layer definition with fully resolved properties."""

    name: str
    tag: int
    thickness: float
    material: str
    k: float
    cp: float
    density: float
    dynamic_viscosity: float
    is_fluid: bool
    active: bool
    units: List[Unit2D] = field(default_factory=list)
    lx: float = 0.0
    ly: float = 0.0
    dx: float = 0.01
    dy: float = 0.01


def load_config(config_path: str) -> Dict[str, Any]:
    """读取并合并默认值的底层方法（返回弱类型字典给外部框架使用）"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    return merge_with_defaults(raw_config)


def build_solver_config(raw_config: Dict[str, Any]) -> SolverConfig:
    """将弱类型的 JSON 配置字典转为内部核心使用的强类型 SolverConfig (单向屏障)"""
    def_mat = raw_config.get("materials", {}).get(
        "default_solid", STANDARD_MATERIALS["default_solid"]
    )

    default_solid = MaterialProps(
        k=float(def_mat.get("k", 1.0)),
        cp=float(def_mat.get("cp", 1.0e6)),
        density=float(def_mat.get("density", 1000.0)),
        is_fluid=bool(def_mat.get("fluid", False)),
        dynamic_viscosity=float(def_mat.get("dynamic_viscosity", 0.0)),
    )

    boundary_conditions = []
    for bc in raw_config.get("boundary_conditions", []):
        boundary_conditions.append(
            BoundaryCondition(
                name=str(bc.get("name", "")),
                type=str(bc.get("type", "")),
                face=str(bc.get("face", "")),
                target=str(bc.get("target", "")),
                parameters={
                    str(k): float(v) for k, v in bc.get("parameters", {}).items()
                },
            )
        )

    return SolverConfig(
        simulation_type=str(raw_config.get("simulation_type", "steady")),
        timestep=float(raw_config.get("timestep", 0.01)),
        init_temperature=float(raw_config.get("init_temperature", 318.15)),
        mesh_file_path=str(raw_config.get("mesh_file_path", "mesh.msh")),
        ptrace_file_path=str(raw_config.get("ptrace_file_path", "")),
        init_temperature_file_path=str(
            raw_config.get("init_temperature_file_path", "")
        ),
        default_solid=default_solid,
        boundary_conditions=boundary_conditions,
    )


def merge_with_defaults(raw_config: Dict[str, Any]) -> Dict[str, Any]:
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


def _resolve_prop(
    key: str, unit_data: dict, unit_mat: dict, layer_mat: dict, default_mat: dict
) -> Any:
    if key in unit_data and unit_data[key] is not None:
        return unit_data[key]
    if key in unit_mat and unit_mat[key] is not None:
        return unit_mat[key]
    if key in layer_mat and layer_mat[key] is not None:
        return layer_mat[key]
    return default_mat.get(key)


def load_stackup(config: Dict[str, Any], base_dir: str) -> List[Layer25D]:
    layers = []
    stackup_data = config.get("stackup", [])
    materials = config.get("materials", {})
    def_mat = materials.get("default_solid", STANDARD_MATERIALS["default_solid"])

    for i, layer_cfg in enumerate(stackup_data):
        tag = int(layer_cfg.get("tag", i + 100))
        name = str(layer_cfg.get("name", f"layer_{tag}"))
        lx, ly = float(layer_cfg.get("lx", 0.0)), float(layer_cfg.get("ly", 0.0))
        dx, dy = float(layer_cfg.get("dx", 0.01)), float(layer_cfg.get("dy", 0.01))

        layer_mat_name = layer_cfg.get("material", "silicon")
        layer_mat = materials.get(layer_mat_name, def_mat)
        layout_file = layer_cfg.get("layout_file", "")
        units = []

        if layout_file and layout_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, layout_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for u in json.load(f):
                        umat_name = u.get("material", layer_mat_name)
                        umat = materials.get(umat_name, layer_mat)

                        units.append(
                            Unit2D(
                                name=u["name"],
                                lx=float(u["lx"]),
                                ly=float(u["ly"]),
                                dx=float(u["dx"]),
                                dy=float(u["dy"]),
                                material=umat_name,
                                k=float(
                                    _resolve_prop("k", u, umat, layer_mat, def_mat)
                                ),
                                cp=float(
                                    _resolve_prop("cp", u, umat, layer_mat, def_mat)
                                ),
                                density=float(
                                    _resolve_prop(
                                        "density", u, umat, layer_mat, def_mat
                                    )
                                ),
                                dynamic_viscosity=float(
                                    _resolve_prop(
                                        "dynamic_viscosity", u, umat, layer_mat, def_mat
                                    )
                                ),
                                is_fluid=bool(
                                    _resolve_prop("fluid", u, umat, layer_mat, def_mat)
                                ),
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
                    k=float(_resolve_prop("k", {}, {}, layer_mat, def_mat)),
                    cp=float(_resolve_prop("cp", {}, {}, layer_mat, def_mat)),
                    density=float(_resolve_prop("density", {}, {}, layer_mat, def_mat)),
                    dynamic_viscosity=float(
                        _resolve_prop("dynamic_viscosity", {}, {}, layer_mat, def_mat)
                    ),
                    is_fluid=bool(_resolve_prop("fluid", {}, {}, layer_mat, def_mat)),
                )
            )

        layers.append(
            Layer25D(
                name=name,
                tag=tag,
                thickness=float(layer_cfg["thickness"]),
                material=layer_mat_name,
                k=float(_resolve_prop("k", {}, {}, layer_mat, def_mat)),
                cp=float(_resolve_prop("cp", {}, {}, layer_mat, def_mat)),
                density=float(_resolve_prop("density", {}, {}, layer_mat, def_mat)),
                dynamic_viscosity=float(
                    _resolve_prop("dynamic_viscosity", {}, {}, layer_mat, def_mat)
                ),
                is_fluid=bool(_resolve_prop("fluid", {}, {}, layer_mat, def_mat)),
                active=bool(layer_cfg.get("active", False)),
                units=units,
                lx=lx,
                ly=ly,
                dx=dx,
                dy=dy,
            )
        )

    return layers
