import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from metahotspot.hotspot_parser import HotSpotParser


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
    """Load 2.5D stackup from stackup config with per-layer FLP files."""
    layers = []
    stackup_cfg = config.get("stackup", [])
    parser = HotSpotParser()

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
        flp_file = str(layer_cfg.get("flp_file", "")).strip()

        if flp_file and flp_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, flp_file)
            if os.path.exists(full_path):
                flp_data = parser.parse_flp(full_path)
                if flp_data:
                    ox = lx
                    oy = ly

                    for u in flp_data:
                        units.append(
                            Unit2D(
                                name=u["name"],
                                lx=float(u["left_x"]) + ox,
                                ly=float(u["bottom_y"]) + oy,
                                dx=float(u["width"]),
                                dy=float(u["height"]),
                                material=None,
                                k=float(u["k"]) if "k" in u else None,
                                cp=(
                                    float(u["specific_heat"])
                                    if "specific_heat" in u
                                    else None
                                ),
                            )
                        )
            else:
                print(
                    f"[WARNING] FLP file {full_path} not found. Falling back to bulk layer."
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
