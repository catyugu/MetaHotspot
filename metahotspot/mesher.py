"""网格生成器。"""

import numpy as np
from metahotspot.metahotspot_types import ModelConfig, MeshCoordinates, MeshTopology, CellGeometry
from metahotspot.units import UnitConverter

def generate_mesh(config: ModelConfig) -> MeshTopology:
    """基于 XML 提供的坐标系生成离散网格拓扑 (采用 Cell-centered FVM 方案)。"""
    coords = config.mesh_coords
    uc = UnitConverter(config.length_unit)
    
    # 获取网格节点数量
    n_x_nodes = len(coords.x)
    n_y_nodes = len(coords.y)
    n_z_nodes = len(coords.z)
    
    # 单元格数量 = 节点数量 - 1
    nx = n_x_nodes - 1
    ny = n_y_nodes - 1
    nz = n_z_nodes - 1
    n_cells = nx * ny * nz
    
    # 转换为国际单位 (米)
    x_m = uc.to_m(coords.x)
    y_m = uc.to_m(coords.y)
    z_m = uc.to_m(coords.z)
    
    # 计算单元格中心和尺寸
    cx = (x_m[:-1] + x_m[1:]) / 2.0
    cy = (y_m[:-1] + y_m[1:]) / 2.0
    cz = (z_m[:-1] + z_m[1:]) / 2.0
    
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    dz = np.diff(z_m)
    
    # 构建三维网格
    CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing='ij')
    DX, DY, DZ = np.meshgrid(dx, dy, dz, indexing='ij')
    
    # 展平为 SoA 格式，shape: (n_cells,)
    centers = np.column_stack([CX.ravel(), CY.ravel(), CZ.ravel()])
    volumes = (DX * DY * DZ).ravel()
    
    # 占位层和块 ID
    layer_ids = np.zeros(n_cells, dtype=np.int32)
    block_ids = np.zeros(n_cells, dtype=np.int32)
    
    cell_geom = CellGeometry(
        centers=centers,
        volumes=volumes,
        layer_ids=layer_ids,
        block_ids=block_ids
    )
    
    return MeshTopology(
        n_cells=n_cells,
        n_x=n_x_nodes,  
        n_y=n_y_nodes,
        n_z=n_z_nodes,
        coords=coords,
        cell_geom=cell_geom,
        layer_names=[layer.name for layer in config.layers],
        material_names=list(config.materials.keys())
    )