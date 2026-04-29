# 工作清单

## 项目背景，约束和说明

* 参见[project.md](project.md)
* 目前我们的项目全面使用Python进行编程。
* 但是必须保证程序本身的通用性。
* 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
* **在进入下一个步骤之前，你必须保证前一个步骤的全部任务得到完成。**

## 运行环境

```bash
conda activate numerical
```
这个conda环境中已经安装好了基础的编译器和常用库，如果有其他需要，可在里面安装。

## 当前任务

为了将 `metahotspot` 从一个简单的 Python 脚本向真正意义上的工业级（或类似 C++ 求解器）架构演进，我们需要彻底摒弃“基于坐标排序”或“硬编码特定面”等妥协做法。

未来的 C++ 求解器核心应该完全依赖于**非结构化网格的拓扑关系（Topological Connectivity）**。不论微流道怎么弯折，只要我们赋予流体单元正确的速度矢量（Velocity Vector），迎风格式（Upwind Scheme）就可以基于共用面的法向量（Face Normal）自动推导出能量的流动方向和大小。

以下是具体的架构改进方案与代码重构指南：

### 核心演进思路

1. **几何与边界条件彻底解耦**：不再通过 `Z_max` 或 `X_min` 硬编码找边界。将外部对流换热（散热器）、微流道入口（Inlet）和出口（Outlet）统一抽象为网格中的“边界物理面（Boundary Physical Surfaces）”。
2. **通用的流固共轭传热（FVM Advection）**：抛弃原来的一维 `flow_dir` 排序法。引入速度矢量场（Velocity Field）。利用相邻网格单元的中心连线与公共面法向量进行点乘运算，自动计算质量流量并组装对流矩阵。这天然支持了任意形状、弯折的流道。
3. **微流道层级抽象（Example 4 适配）**：在转换 `example4` 的 CSV 时，不再将每个像素生成一个极小的 `Unit2D`，而是通过算法将连续的流体网格合并为整条微流道的“几何版图单元（FLP Unit）”，极大降低网格复杂度和求解矩阵维度。

---

### Phase 1: 数据结构升级 (改进 `model25d.py`)

我们需要将原先粗糙的 `flow_dir` 字符串升级为真正的物理场量：**速度矢量**。同时添加边界条件的标记能力。

```python
# metahotspot/model25d.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

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
    
    # --- 流体与共轭传热升级 ---
    is_fluid: bool = False
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0) # 速度矢量 (vx, vy, vz)
    density: float = 1000.0 # 密度
    
    # 边界标记，允许转换器动态挂载边界条件
    inlet_temp: Optional[float] = None 
```

---

### Phase 2: 真正的 FVM 对流项组装 (改进 `fvm_solver.py`)

这是向 C++ 求解器迁移的最核心一步。你需要构建网格单元的**面-邻居拓扑图（Face-to-Cells Graph）**。通过面的通量（Flux）来决定能量传递，这使得代码对几何形状完全免疫。

首先，在 `_prepare_mesh` 中建立面与左右网格的映射关系：

```python
# fvm_solver.py - 修改 _prepare_mesh 建立完整拓扑
self.face_to_cells = {}
for new_id, orig_id in enumerate(sorted_indices):
    # ... 原有初始化 Cell 的代码 ...
    
    # 构建面拓扑
    fs = [ ... ] # 提取六面体的6个面
    for f in fs:
        if f not in self.face_to_cells:
            self.face_to_cells[f] = []
        self.face_to_cells[f].append(new_id)

# 区分内部面和边界侧面
self.internal_faces = {f: c_ids for f, c_ids in self.face_to_cells.items() if len(c_ids) == 2}
self.boundary_faces_all = {f: c_ids for f, c_ids in self.face_to_cells.items() if len(c_ids) == 1}
```

然后，用真正的**通量迎风格式**重构 `_add_fluid_advection`：

```python
def _add_fluid_advection_generic(self) -> Tuple[sp.csr_matrix, np.ndarray]:
    n = len(self.cells)
    rows, cols, data = [], [], []
    rhs = np.zeros(n)
    tol = self.GEOMETRY_TOLERANCE

    # 1. 计算内部流体面通量 (Internal Convection)
    for f, (c0_id, c1_id) in self.internal_faces.items():
        c0, c1 = self.cells[c0_id], self.cells[c1_id]
        if c0.is_fluid and c1.is_fluid:
            # 计算面面积与法向量 (近似从 c0 指向 c1)
            pts = self.mesh.points[list(f)]
            cross_prod = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            area = np.linalg.norm(cross_prod)
            n_vec = cross_prod / (area + 1e-16)
            
            vec_c0_c1 = c1.center - c0.center
            if np.dot(n_vec, vec_c0_c1) < 0:
                n_vec = -n_vec # 确保法向指向 c1
                
            # 取迎风侧速度进行通量计算
            v_avg = 0.5 * (np.array(c0.velocity) + np.array(c1.velocity))
            vol_flux = np.dot(v_avg, n_vec) * area # 体积流量 m^3/s
            mass_flux = vol_flux * c0.density # 质量流量 kg/s
            
            cp = c0.cp # 近似取上游比热容
            advection_term = mass_flux * cp
            
            if advection_term > tol:
                # c0 流向 c1
                rows.extend([c0_id, c1_id])
                cols.extend([c0_id, c0_id])
                data.extend([-advection_term, advection_term])
            elif advection_term < -tol:
                # c1 流向 c0
                rows.extend([c1_id, c0_id])
                cols.extend([c1_id, c1_id])
                data.extend([-abs(advection_term), abs(advection_term)])

    # 2. 处理流体边界 (Inlet & Outlet)
    for f, (c0_id,) in self.boundary_faces_all.items():
        c0 = self.cells[c0_id]
        if c0.is_fluid and c0.inlet_temp is not None:
            # 简化：如果定义了 inlet_temp，我们强制将外部焓流注入此单元
            # 实际 C++ 中会计算边界法向判断是流入还是流出
            velocity_mag = np.linalg.norm(c0.velocity)
            area = ... # 计算面面积
            mass_flux = velocity_mag * area * c0.density
            
            # 边界流出（损失能量）
            rows.append(c0_id)
            cols.append(c0_id)
            data.append(-mass_flux * c0.cp)
            
            # 边界流入（源项）
            rhs[c0_id] += mass_flux * c0.cp * c0.inlet_temp

    G_adv = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return G_adv, rhs
```
*这种写法的巨大优势在于：不管网格长什么样，只要分配了合法的速度场，系统就会自动推导传热矩阵，这与 OpenFOAM/Fluent 等成熟求解器的内核逻辑完全一致。*

---

### Phase 3: 转换器支持微流道宏单元化 (改进 `converter.py`)

对应 `example4` 的需求，我们需要在解析 `horizontal.csv` 时，**将连续的像素拼接成真正的流道单元**，而不是生成几万个碎小的 FLP unit。

在 `converter.py` 的 `_build_microchannel_layer` 中引入“寻点生长”或“行列扫描”算法：

```python
def _build_microchannel_layer(self, csv_path: str, flp_path: str, layer_cfg: dict) -> List[dict]:
    grid = self.parser.parse_microchannel_csv(csv_path)
    if not grid: return []
    
    # ... 获取 dx, dy ...
    rows, cols = len(grid), len(grid[0])
    units = []
    
    # 算法：将水平方向相邻的 1 连成一条完整的微流道
    for row in range(rows):
        col = 0
        while col < cols:
            if grid[row][col] == 1: # 发现流体
                start_col = col
                while col < cols and grid[row][col] == 1:
                    col += 1
                end_col = col - 1
                
                # 构建宏观长条流道单元
                y = (rows - 1 - row) * dy
                length = (end_col - start_col + 1) * dx
                
                unit = {
                    "name": f"Channel_row{row}_{start_col}",
                    "width": length,
                    "height": dy,
                    "left_x": start_col * dx,
                    "bottom_y": y,
                    "is_fluid": True,
                    "material": "water",
                    "k": 0.6,
                    "cp": 4.17e6,
                    "velocity": [0.0, 0.1, 0.0],  # 通过外部参数控制整体流向和流速
                    "density": 1000.0,
                    "inlet_temp": 298.15 # 可仅赋予起始单元，或者利用 solver 内部判别边界
                }
                units.append(unit)
            else:
                col += 1
                
    # 其余空白区域可以作为一个巨大的 Solid Bulk 填充，或者在转换为 JSON 时依赖 Default Material 自动填充
    return units
```

### 总结

通过上述修改，你的 Python Demo 已经具备了工业求解器的数据流向结构：
1. **GmshMesher** 负责输出纯粹的节点和物理分组（Physical Groups）。
2. **FVMSolver** 通过读取拓扑面（Face）和单元（Cell）的数组，利用 CSR 稀疏矩阵组装通用微分方程。
3. **Converter** 负责将各种奇葩的历史格式（如像素 CSV）转化为优雅的几何图元。
