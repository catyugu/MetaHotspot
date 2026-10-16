# 数据流与设计原则

## 核心对象与生命周期

| 对象 | 职责 | 生命周期 |
| --- | --- | --- |
| `ModelDefinition` | 几何、材料、热源、边界、探针和 study 设置 | 可变输入；由 XML reader 或 C/Python 构造接口填充 |
| `mhs::core::Model` | 显式网格、材料与积分数据、编译表达式和冻结流场 | `build_model` 的独立结果；供重复装配和求解使用 |
| `Operators {K, C, f}` | 当前状态和时刻的线性化系统 `C dx/dt + K x = f` | 每次装配的值对象 |
| `Solution` | 最终状态、输出时刻的状态历史和探针记录 | 独立拥有求解结果 |

`compile` 不修改输入；`assemble` 和 `solve` 不修改编译模型。C opaque handle 和 Python wrapper 负责所有权与转换。Python 返回的 NumPy/SciPy 数据在 native handle 销毁后仍有效。

## 编译与装配

| 阶段 | 输入 | 输出与规则 |
| --- | --- | --- |
| 几何解析 | Layer / Block / Rect 和几何表达式 | SI 米下的区域；同层后出现的 Block 覆盖前者 |
| 网格输入 | Cartesian 坐标轴或调用方提供的控制体、面片 | `DiscreteMesh`：单元边界、唯一有向面片和单元—面片 CSR 关联 |
| 网格校验 | 控制体与面片几何、连接 | 检查体积重叠、面覆盖、法向距离和 TPFA 正交条件 |
| 体积投影 | 网格与物理区域 | `cell_material` 和 `source_terms`；跨材料或物理域边界的控制体被拒绝 |
| 边界投影 | 外露面片与有序 `BoundaryPatch` | 生成引用原始 FaceId 的 `boundary_terms`；积分矩形通过 `condition` 引用 `boundary_conditions` 中的 `BoundaryCondition` variant（`Dirichlet` / `Neumann` / `Convection`），不改变拓扑，后出现的边界覆盖前者，未覆盖部分使用默认边界 |
| 表达式编译 | 材料、源、边界表达式 | `CompiledExpression`；线程模型见 [expr-api.md](expr-api.md) |
| 流体预处理 | 流体材料、连接和流体边界 | 独立按面积投影流体边界；临时压力解转换为内部面体积流量、边界片质量通量与入口温度、流固换热因子 |
| 容量与体热源装配 | 单元几何、材料、体积投影、状态和时间 | `rho c V` 及各投影片上的热源积分 |
| 面片装配 | 面片、两侧物理类型、边界描述 | 固固导热、流固换热、外边界通量；内部面只装配一次，两侧热流相反 |
| 流体能量装配 | 冻结流量、状态和材料 | 上风对流和入口能量贡献，与导热项合并为 `Operators` |

当前 TPFA 适用于轴对齐长方体、相邻中心沿面法向对齐及对角导热张量。数值内核消费已有网格，不生成或加密网格。

`SourceTerm` 仅保存 CellId、源表索引、体积权重和积分点；一个控制体可以接收多个源。常数体热源按体积守恒，空间表达式在各交叠片上采用中点积分。材料归属仅保存每单元一个材料索引。热与流体边界分别采用后出现条件覆盖前者的面积投影，重叠矩形按并集计量；同一单元上的不同入口分别累计质量流量与焓。压力条件施加于投影面积，质量流量按选中面积分配；未指定的入口温度取初始温度。

## 求解与输出

`solve_system` 通过 `SystemAssembler(state, time)` 请求完整算子，执行非线性迭代和时间推进。稳态在 `t=0` 求解；瞬态使用 BDF1/BDF2，`StepController` 控制 Adaptive/Fixed 步长并截短到输出时刻，observer 接收真实积分状态。全局状态不做时间插值。

探针初始化时定位控制体，记录时由局部邻域重建温度；共享面上的点选择下角坐标按字典序最大的控制体。温度状态直接按 CellId 访问。`Model::output` 保存展示用顶点与单元连接，VTU 可输出外部箱形网格；XML 输出要求 Cartesian 采样形状。

## 输入与数组约定

- C++ 的 `build_model(definition, discrete_mesh)`、C 的 `mhs_model_set_discrete_mesh` 和 Python 的 `Model.set_discrete_mesh` 接收完整的控制体与面片连接。外部网格坐标始终使用 SI 米，与物理定义的长度单位无关。
- Python 网格边界数组为 `(count, 3)`；面片 owner/neighbor、axis/sign 为 `(face_count,)`。外边界 neighbor 为 `SIZE_MAX`，Python 输入也接受有符号 `-1`。面片 sign 表示 owner 的外法向。
- 几何、材料、`K/C/f`、`state` 和 `state_history` 统一按 CellId 排列；该编号只在当前编译模型内有效。Python `Solution.temperature` 和 `temperature_history` 直接共享状态数组。
- `Compiled.source_operator(state, time)` 按物理源返回稀疏功率列。`SourcePort` 必须且只能指定 `source_ids` 或物理 `cells` 选择器之一。
- `BoundaryGroup(surface=geometry.surface(Face.ZP), h_range=...)` 使用实际 FaceId 和投影面积，同一控制体上的面片贡献累加。Robin 参数采用组平均的有效仿射系数，并非物理 HTC 的精确仿射表示。
