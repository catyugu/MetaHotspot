# 数据流与设计原则

## 核心对象与生命周期

| 对象 | 职责 | 生命周期 |
| --- | --- | --- |
| `ModelDefinition` | 连续几何、材料、热源、边界、探针和参考温度 | 可变输入；与网格配置分开 |
| `MeshSpec` | `ConformalMeshSpec` 或程序构造的 `NonConformalMesh` | 编译输入，坐标使用 SI 米 |
| `Model` | 网格、Cell 属性、外边界、源投影、编译表达式和冻结流场 | 编译后独立拥有，重复装配和求解时只读 |
| `Operators {K, C, f}` | 当前状态和时刻的线性化系统 `C dx/dt + K x = f` | 装配工作区持有；导出的算子独立拥有 |
| `Solution` | 单元状态、历史和探针记录 | 独立拥有求解结果 |

`compile` 不修改输入；`assemble` 和 `solve` 不修改编译模型。C opaque handle 和 Python wrapper 负责所有权与转换。Python 返回的 NumPy/SciPy 数据在 native handle 销毁后仍有效。

## 网格与单元

`ConformalMesh` 保存 `x_vertices`、`y_vertices`、`z_vertices` 三个一维坐标数组。Z 索引变化最快。完整网格不保存单元几何、内部 Connection 或索引映射；Cell 的包围盒、中心、体积与相邻编号按索引计算。存在非活动位置时，才保存 `logical_to_cell` 和 `cell_to_logical`，使活动 CellId 保持连续。两个映射的元素均为 32 位 `CellId`。空洞暴露的表面也生成外边界。

无内部几何切分的单元直接在中心分类并生成源项，不分配切分数组；有内部几何边界时才切分以验证材料与活动域一致性，并计算源项子体积积分。

`NonConformalMesh` 保存轴对齐六面体的 `CellGeometry` 和一份无向 `Connection {a,b}` 列表。连接方向可任意，每对面接触单元必须且只能出现一次。编译器验证有限且递增的包围盒、体积重叠、非法/重复编号、零面积接触和遗漏的面接触。临时 BVH 用于空间查询，编译期一次构造的紧凑 CSR 邻接表用于边界裁剪；编译结束后均释放。

`Mesh` 是两类网格的 variant。`cell_count`、`cell_geometry` 和 `for_each_connection` 提供共同访问方式。热装配在网格级分派，采用相同的 TPFA 数值逻辑。Connection 不保存面积、法向、中心或半距离，装配时由两侧包围盒计算，工作区也不缓存这些几何。

`Model::cells` 保存 `Cell {material}`；流体身份由材料类别确定。`Facet` 表示包含 minus/plus、法向轴、平面坐标和切向矩形范围的有向积分片。内部 Facet 由 Connection 临时推导；`Model::boundary_facets` 仅存外边界片，owner 和外法向符号由缺失的一侧推导。边界片段按 owner 排序。不保留内部 Facet 列表、Cell–Facet 关联或持久邻接 CSR。稀疏矩阵的 CSR 是求解算子的结构。

## 编译与装配

| 阶段 | 输入 | 输出与规则 |
| --- | --- | --- |
| PTA 适配 | Layer / Block / Rect、单位、几何变量 | 应用层转换为 SI 物理区域与 `ConformalMeshSpec` |
| 网格编译 | 网格配置、有序物理区域 | 检查真实生效的材料/活动域界面；创建 Cell 属性和外边界 |
| 源投影 | 各物理子体积与源表 | `SourceTerm {cell, source, weight, point}`；中点积分 |
| 边界细分 | 外边界与热/流体区域 | 对矩形交叠细分，条件后者覆盖前者，独立绑定到 BoundaryFacetId |
| 表达式编译 | 材料、源、边界表达式 | `CompiledExpression`；线程模型见 [expr-api.md](expr-api.md) |
| 流体预处理 | 流体 Cell、Connection、压力/质量流边界 | 临时压力系统；输出紧凑的流体连接流量、边界输运与膜阻力因子 |
| 容量与源装配 | 单元几何、状态和时间 | `rho c V` 和源项积分 |
| 热连接装配 | 两侧单元、按需接触几何 | 每条连接计算一次，对两侧写入守恒的 TPFA 贡献 |
| 边界装配 | 外边界 Facet、物理条件、当前状态 | 求解表面迹并凝聚到单元方程 |
| 流体输运装配 | 冻结流量、当前 rho/cp | 上风能量通量与边界焓流 |

每个 Cell 必须完整属于活动物理域且具有唯一材料；跨越真实材料界面或活动域边界时明确报错。编译器按几何盒切割形成临时子体积，检查最终覆盖关系，因此被后续区域覆盖的无效盒边界不要求与网格对齐。热源可以在同材料 Cell 内分区，各子体积独立投影；未对齐材料界面不会被中心采样静默忽略。非共形输入中的每个显式 Cell 都必须有效，不会静默删除输入编号。

域内均使用 TPFA，包括非共形粗细单元接触。对非共形网格，切向中心错位可能产生离散误差；本轮没有 NCC、多域或跨域连接。固–流体传热沿用流体侧膜阻力。热、流体边界使用共同的几何切分，投影面积只计一次；多个入口分别累积。

没有流体时，所有流体数组为空。有流体时，`global_to_fluid` / `fluid_to_global` 提供紧凑压力编号映射；`FluidConnection {cells, volume_flux}` 仅记录流体之间的接触，流量正方向是 `a -> b`。`FluidBoundaryTerm` 保存 BoundaryFacetId 和 `variant<FluidInflow, FluidOutflow>`，质量流率均非负，只有流入记录含温度。类型由实际流向决定，支持压力边界反向流动。缺省流入温度取参考温度。

## 求解与输出

公开 C++ 入口 `mhs::solver::solve(model, study, ...)` 位于 `solver/solver.hpp`。`ThermalAssembler` 一次建立 K/C 模式与材料工作区，后续刷新数值。非线性求解器消费只读 `LinearSystemView`。瞬态构造 K/C 的联合模式，再更新 `A=K+αC`；时间步由 AdaptiveTimeStepper 的 BDF1/2 控制，输出时刻独立于实际积分步。

梯度重构遍历 Connection，向两个 Cell 累积 Green–Gauss 贡献，再加入外边界迹并除以体积，不建立单元邻接表。每个探针输出时刻重构一次全部梯度，供所有探针复用；边界采样按 owner 二分定位外边界片段。探针初始化选择编号顺序中最后一个包含该点的 Cell。

现有 CLI VTU 写出通过统一 Cell 几何访问支持两类网格；PTA XML 结果回写继续使用原始共形采样轴。本轮不增加 SDK/Python VTU Writer，也不增加非共形 XML reader。

## C API 与 Python 数组

- `mhs_model_set_conformal_mesh` / `Model.set_conformal_mesh(x,y,z)` 设置坐标轴。
- `mhs_model_set_nonconformal_mesh` / `Model.set_nonconformal_mesh(cell_min,cell_max,connections)` 设置 `(N,3)` 包围盒与 `(E,2)` CellId 对，输入由库复制。
- 旧 `set_cartesian_mesh`、全体 Facet 的 minus/plus 导出及 `DiscreteMesh` 已删除，不保留兼容层。
- `Compiled.fields` 按 CellId 导出 `(N,3)` cell_min/max、材料索引与源项投影；只导出外边界的 boundary_owner/side/axis/coordinate/bounds。bounds 为 `(B,4)`，顺序为 `[a_min,a_max,b_min,b_max]`；切向轴是 X→YZ、Y→XZ、Z→XY。side 为 int32 的 -1/+1，owner 为 uint32 CellId。
- `Compiled.connections()` 按需导出 `(E,2)` CellId 对；共形连接在调用时生成，不在 fields 或模型内缓存。宏模型子域切分据此找到人工切口并计算交叠矩形；外边界端口使用 BoundaryFacetId。
- `Compiled.info` 只读取标量；`Compiled.fields` 首次访问才复制数组，两者独立缓存。
- `mhs_operators_copy_k/c` 导出 CSR：`row_ptr[N+1]`、`col_idx[nnz]`、`values[nnz]`。Python `assemble()` 返回 SciPy CSR。
- `Compiled.source_operator(state,time)` 保留各物理源列。`SourcePort` 指定 source_ids 或 cells 选择器之一。
- `BoundaryGroup` 使用外边界片段及投影面积。Robin 参数仍是组平均有效仿射系数。

数量与偏移使用 size_t；实体引用使用 uint32（无效值 UINT32_MAX）；CSR 索引使用 int32。矩阵构建前检查稀疏索引容量。输入和输出所有权、失败状态与线程局部错误语义保持既有 C API 约定；成功的 status 调用清空错误。API 成功与数值求解收敛分别报告。
