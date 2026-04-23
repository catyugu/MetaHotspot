import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Unit2D:
    name: str
    lx: float
    ly: float
    dx: float
    dy: float
    material: Optional[str] = None
    k: Optional[float] = None
    cp: Optional[float] = None


@dataclass
class Layer25D:
    name: str
    tag: int
    thickness: float
    default_material: str
    active: bool
    units: List[Unit2D] = field(default_factory=list)
    # 对于没有 layout 文件的层（如封装层），使用全局尺寸
    lx: float = 0.0
    ly: float = 0.0
    dx: float = 0.01
    dy: float = 0.01


def load_stackup(config: Dict[str, Any], base_dir: str) -> List[Layer25D]:
    """从配置和拆分的独立版图文件中动态加载 2.5D 堆叠模型"""
    layers = []
    stackup_cfg = config.get("stackup", [])

    for i, layer_cfg in enumerate(stackup_cfg):
        tag = layer_cfg.get("tag", i + 100)
        name = layer_cfg.get("name", f"layer_{tag}")
        thickness = float(layer_cfg["thickness"])
        default_material = layer_cfg.get("material", "silicon")
        active = bool(layer_cfg.get("active", False))

        lx = float(layer_cfg.get("lx", 0.0))
        ly = float(layer_cfg.get("ly", 0.0))
        dx = float(layer_cfg.get("dx", 0.01))
        dy = float(layer_cfg.get("dy", 0.01))

        units = []
        layout_file = layer_cfg.get("layout_file")

        if layout_file and layout_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, layout_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    layout_data = json.load(f)
                    for u in layout_data:
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
                            )
                        )
            else:
                print(
                    f"[WARNING] Layout file {full_path} not found. Falling back to bulk layer."
                )

        # 如果没有有效的版图单元，则用一个完整的 Bulk Unit 代表这一层
        if not units:
            units.append(
                Unit2D(
                    name=f"{name}_bulk",
                    lx=lx,
                    ly=ly,
                    dx=dx,
                    dy=dy,
                    material=default_material,
                )
            )

        layers.append(
            Layer25D(
                name=name,
                tag=tag,
                thickness=thickness,
                default_material=default_material,
                active=active,
                units=units,
                lx=lx,
                ly=ly,
                dx=dx,
                dy=dy,
            )
        )

    return layers
