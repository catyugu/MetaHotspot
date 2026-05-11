import json
import os
from typing import List, Dict, Any, Tuple

from metahotspot.metahotspot_types import (
    SolverConfig,
    BoundaryCondition,
    MaterialProps,
    LayerRegion,
    UnitRegion,
)

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


def parse_computational_model(
    config_path: str,
) -> Tuple[SolverConfig, List[LayerRegion]]:
    base_dir = os.path.dirname(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    config = merge_with_defaults(raw_config)

    def_mat = config.get("materials", {}).get(
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
    for bc in config.get("boundary_conditions", []):
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

    solver_config = SolverConfig(
        simulation_type=str(config.get("simulation_type", "steady")),
        timestep=float(config.get("timestep", 0.01)),
        init_temperature=float(config.get("init_temperature", 318.15)),
        ptrace_file_path=str(config.get("ptrace_file_path", "")),
        init_temperature_file_path=str(config.get("init_temperature_file_path", "")),
        default_solid=default_solid,
        boundary_conditions=boundary_conditions,
    )

    layer_regions: List[LayerRegion] = []

    materials = config.get("materials", {})
    stackup_data = config.get("stackup", [])
    z_cursor = 0.0

    for i, layer_cfg in enumerate(stackup_data):
        tag = int(layer_cfg.get("tag", i + 100))
        name = str(layer_cfg.get("name", f"layer_{tag}"))
        thickness = float(layer_cfg.get("thickness", 0.0))
        lx, ly = float(layer_cfg.get("lx", 0.0)), float(layer_cfg.get("ly", 0.0))
        dx, dy = float(layer_cfg.get("dx", 0.01)), float(layer_cfg.get("dy", 0.01))
        active = bool(layer_cfg.get("active", False))

        layer_mat_name = layer_cfg.get("material", "silicon")
        layer_mat = materials.get(layer_mat_name, def_mat)
        layout_file = layer_cfg.get("layout_file", "")

        units: List[UnitRegion] = []
        if layout_file and layout_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, layout_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for u in json.load(f):
                        umat_name = u.get("material", layer_mat_name)
                        umat = materials.get(umat_name, layer_mat)

                        u_props = MaterialProps(
                            k=float(_resolve_prop("k", u, umat, layer_mat, def_mat)),
                            cp=float(_resolve_prop("cp", u, umat, layer_mat, def_mat)),
                            density=float(
                                _resolve_prop("density", u, umat, layer_mat, def_mat)
                            ),
                            is_fluid=bool(
                                _resolve_prop("fluid", u, umat, layer_mat, def_mat)
                            ),
                            dynamic_viscosity=float(
                                _resolve_prop(
                                    "dynamic_viscosity", u, umat, layer_mat, def_mat
                                )
                            ),
                        )

                        units.append(
                            UnitRegion(
                                name=u["name"],
                                lx=float(u["lx"]),
                                ly=float(u["ly"]),
                                dx=float(u["dx"]),
                                dy=float(u["dy"]),
                                props=u_props,
                            )
                        )

        if not units:
            l_props = MaterialProps(
                k=float(_resolve_prop("k", {}, {}, layer_mat, def_mat)),
                cp=float(_resolve_prop("cp", {}, {}, layer_mat, def_mat)),
                density=float(_resolve_prop("density", {}, {}, layer_mat, def_mat)),
                is_fluid=bool(_resolve_prop("fluid", {}, {}, layer_mat, def_mat)),
                dynamic_viscosity=float(
                    _resolve_prop("dynamic_viscosity", {}, {}, layer_mat, def_mat)
                ),
            )
            units.append(
                UnitRegion(
                    name=f"{name}_bulk", lx=lx, ly=ly, dx=dx, dy=dy, props=l_props
                )
            )

        l_props_layer = MaterialProps(
            k=float(_resolve_prop("k", {}, {}, layer_mat, def_mat)),
            cp=float(_resolve_prop("cp", {}, {}, layer_mat, def_mat)),
            density=float(_resolve_prop("density", {}, {}, layer_mat, def_mat)),
            is_fluid=bool(_resolve_prop("fluid", {}, {}, layer_mat, def_mat)),
            dynamic_viscosity=float(
                _resolve_prop("dynamic_viscosity", {}, {}, layer_mat, def_mat)
            ),
        )

        layer_regions.append(
            LayerRegion(
                name=name,
                tag=tag,
                lx=lx,
                ly=ly,
                lz=z_cursor,
                dx=dx,
                dy=dy,
                dz=thickness,
                props=l_props_layer,
                units=units,
                is_active=active,
            )
        )

        z_cursor += thickness

    return solver_config, layer_regions
