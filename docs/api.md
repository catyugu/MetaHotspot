# C ABI 与 Python API

## 生命周期

`Model` 是可变定义；`compile()` 生成独立的只读 `Compiled` 快照，供重复装配与求解。每次 `solve()` 的 `Study`、初值和选项独立传入。算子与 `Solution` 保留自己的数据；删除上游句柄不影响已有快照。

C 端维持四种独立拥有型句柄：`mhs_model_t`、`mhs_compiled_t`、`mhs_operators_t`、`mhs_solution_t`。传入的字符串与数组由库复制；输出数组要求调用方按元数据分配精确长度。各 `destroy` 可接受 NULL。失败返回 `MHS_ERROR`，详细原因通过当前线程的 `mhs_last_error()` 获取。

## Component 与物理外部契约

模型从零个 Component 开始。C 使用 `mhs_model_add_cartesian_component` 或 `mhs_model_add_reduced_component` 创建；Python 分别调用 `add_cartesian_component()` 与 `add_reduced_component(K, C, f)`。Component ID 按创建顺序稳定编号。FVM 和 ROM 可以混合；两者共享 Port、Connection、热边界和物理温度观测契约。

C 使用单一 `mhs_model_add_port`；Cartesian 端口的 `outward_sign=0, patches=NULL, patch_count=0`，其物理迹由网格生成。Reduced 端口须提供 ±1 外法向符号和覆盖整个区域的 patches。Patch 包含 `trace · z + offset` 温度迹、法向半热阻（m² K/W）和可选切向梯度；跨非匹配接口的 NCC 需要梯度映射。

C 热边界采用 `mhs_thermal_bc_t` 与 `mhs_model_add_thermal_boundary`、`mhs_model_set_default_thermal_boundary`。Python 使用 `Dirichlet(temperature)`、`Neumann(heat_flux)`、`Convection(coefficient, ambient_temperature)` 值对象。外边界允许覆盖的区域按声明次序处理；Reduced 的外边界只施加在未连接的 Port 上。

C 物理观测使用 `mhs_model_add_observation`。Cartesian Component 只传空间位置（weights=NULL、offset=0），Reduced Component 提供长度为降阶阶数的模态权重与可选 offset。Python 两者均由 `Model.add_observation(component, name, position, ...)` 创建，输出沿同一 `Solution.probes` 时间轴记录。

## Python 最短示例

```python
from metahotspot import Dirichlet, Model, SurfaceRegion
from metahotspot.enums import Axis

with Model() as model:
    component = model.add_cartesian_component()
    model.set_cartesian_mesh([0, 0.001], [0, 0.001], [0, 0.001], component=component)
    material = model.add_material("solid", "1", "1", "1", "1000", "500")
    volume = model.add_volume(material, component=component)
    model.add_box(volume, (0, 0, 0), (0.001, 0.001, 0.001), component=component)
    bottom = [SurfaceRegion(Axis.Z, 0, 0, 0.001, 0, 0.001)]
    model.add_thermal_boundary(bottom, Dirichlet("300"), component=component)
    model.add_observation(component, "center", (0.0005, 0.0005, 0.0005))
    compiled = model.compile()

with compiled:
    solution = compiled.solve()
    print(solution.probes[0].values[-1])
```

## 数组与物理约定

所有几何坐标使用 SI 米。矩形切向轴依次为 X→(y,z)、Y→(x,z)、Z→(x,y)；Connection 热流符号正向为 A 侧向外。全局状态按 `FieldLayout` 中连续的全局 DOF 区间排列，可能同时包含温度、压力、速度及 Reduced 坐标。C 的 `mhs_compiled_info_t.component_count` 和 `mhs_component_info_t.kind` 明确 Component 数量与数值种类；Python 对应 `Compiled.info.component_count`、`Compiled.components[i].kind` 和 `Compiled.field_layouts[i].component`。

`Compiled.fields` 返回按 Component 顺序拼接的笛卡尔几何；ROM 没有虚构的 Cell。算子 `K/C` 为 CSR，`f` 与 `Solution.state` 遵循相同 DOF 顺序。 `Solution.state_history` 为 (时间记录数, 状态数)。`Compiled.default_state()` 将笛卡尔温度按求解初温初始化，Reduced 坐标按所声明的模态初值初始化。完整数值布局与非匹配界面规则见 [状态及连接设计](design/state-layout.md)。

C ABI 不提供外部离散网格输入；PTA 文件格式的网格解析属于 CLI 应用侧。
