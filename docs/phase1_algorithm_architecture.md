# 第一步交付：算法学习与架构确立

## 1. 参考系统学习结论

### 1.1 Hotspot（当前对齐基线）

关键点（来自 `Hotspot/src/temperature_block.c`、`Hotspot/src/temperature_grid.c`、`Hotspot/src/hotspot.c`）：

- 采用热-电类比的紧凑 RC 网络。
- 稳态解本质是线性方程组求解。
- 瞬态采用离散时间推进（等价于后向欧拉形式）。
- `grid_steady_file`、`grid_transient_file` 的文本格式已固定：
  - 稳态：按 `Layer i:` 分段，每个网格点一行 `index temperature`。
  - 瞬态：按 `t = ...` + `Layer i:` 多帧写出。

### 1.2 ARTSim（非共形与异构处理参考）

关键点（来自 `ARTSim/src/nub_ctm.py`）：

- 通过单元细分 + 显式几何关系判断（邻接、重叠）构建连接关系。
- 共享界面面积用于热阻计算，核心形式：
  - $R=\frac{L}{kA}$
- 跨层连接与层内连接统一转成导热支路后组装全局矩阵。

### 1.3 3D-ICE（流体与流固耦合参考）

关键点（来自 `3d-ice/sources/channel.c`、`3d-ice/sources/thermal_grid.c`、`3d-ice/sources/system_matrix.c`、`3d-ice/sources/thermal_data.c`）：

- 冷却层与固体层统一到同一热网络中。
- 使用有效对流项与换热系数耦合流体-固体界面。
- 瞬态时矩阵对角项中包含 $C/\Delta t$ 项。
- 流量更新会触发系统矩阵重建和再分解（便于后续流量-温度联动迭代）。

## 2. MetaHotspot 采用的核心方程

### 2.1 稳态

令 $G$ 为导热矩阵，$T$ 为温度向量，$b$ 为功率和边界源项，则：

$$
-GT=b
$$

其中每条单元-单元连接导热系数：

$$
G_{ij}=\frac{1}{\frac{\Delta x_i}{2k_iA_{ij}}+\frac{\Delta x_j}{2k_jA_{ij}}}
$$

### 2.2 瞬态（后向欧拉）

$$
\left(\frac{C}{\Delta t}-G\right)T^{n+1}=\frac{C}{\Delta t}T^n+b^{n+1}
$$

其中 $C$ 是体积热容对角矩阵，$C_i=(\rho c_p V)_i$。

### 2.3 非共形连接规则

- 邻接判定：面接触且其余两个方向存在正重叠面积。
- 界面面积：使用几何重叠面积 $A_{ij}$。
- 功率耦合：按体积交比把功率单元分配到网格单元。

## 3. Hotspot 到 MetaHotspot 的配置映射规则（第二步准备）

### 3.1 参数映射

- `-xxx value` 形式参数映射到 `solver_config.toml` 对应字段。
- `materials` 和 `lcf` 中材料属性转为统一 `materials` 表。

### 3.2 坐标系规则（重点）

- Hotspot `flp` 内功率单元坐标是“层局部左下角坐标系”。
- MetaHotspot 使用全局三维坐标：
  - $(x,y)$ 保留平面坐标；
  - 通过层厚累加得到 $z$ 和 $dz$。

### 3.3 边界实体组规则（重点）

- Hotspot 网格/CSV 的入口、出口、流体、固体分类，统一转换为边界或域分组编号。
- 在 `boundary_conditions` 中通过 `selection=[group_ids]` 指定边界条件挂载位置。

## 4. 程序架构与接口规约

当前已经按分层重构为以下模块：

- `metahotspot/hotspot_parser.py`
  - 职责：解析 `flp/config/materials/lcf`。
- `metahotspot/gmsh_mesher.py`
  - 职责：生成 Gmsh 网格并维护离散域/物理组。
- `metahotspot/converter.py`
  - 职责：Hotspot 示例目录 -> `solver_config.toml` + `mesh.msh` + `ptrace`。
- `metahotspot/fvm_solver.py`
  - 职责：读取网格与配置、组装矩阵、执行稳态/瞬态求解并输出 `vtu`。
- CLI 入口：
  - `scripts/compare_hotspot.py`
  - 职责：对比 Hotspot 输出与 MetaHotspot 输出，输出误差指标与通过判定。
  - `adapter.py`
  - `solver.py`
  - `visualize.py`
 
## 5. 正确性验证流程（可直接执行）

以下流程用于例1~例4逐步验收。

1. 生成 Hotspot 基准输出。
2. 运行 `adapter.py` 生成 MetaHotspot 输入。
3. 运行 `solver.py` 生成 MetaHotspot 输出。
4. 用 `scripts/compare_hotspot.py` 对比误差。

示例命令（以某个案例目录为例）：

```bash
python adapter.py Hotspot/examples/example1 outputs/example1_meta
python solver.py outputs/example1_meta/solver_config.toml
python scripts/compare_hotspot.py Hotspot/examples/example1/outputs/gcc.steady outputs/example1_meta/result.vtu --threshold-k 1.0
```

对比策略：

- 若向量长度一致：输出 `max_abs_error / mean_abs_error / rmse`。
- 若长度不一致：输出分布分位误差（用于网格尺度不一致时的阶段验收）。

验收门槛（第三步目标）：

- 关键层最大绝对误差 `<= 1.0 K`。

## 6. 本阶段结论

- 已建立统一的算法框架与分层代码结构。
- 已形成可执行的验证链路和对比脚本。
- 下一步可进入“第二步：四个案例前处理转换”并做案例级回归。
