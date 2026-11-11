# ModelDefinition 与 PTA 适配

`model/model_definition.hpp` 是库的建模契约。所有几何、网格坐标和探针坐标使用 SI 米；材料及边界表达式使用规范变量 `x/y/z/T/t`，编译器原样解析。材料中的 `T` 表示单元温度，热边界中的 `T` 表示表面迹温度。

`ModelDefinition` 保存 reference_temperature、functions、materials、sources、volumes、thermal_boundaries、default_thermal_boundary、point_probes 和 fluid_boundaries。网格规格由独立的 `MeshSpec = variant<ConformalMeshSpec, NonConformalMesh>` 保存。材料和热源采用有序表，`VolumeSpec` 使用表内 ID 引用；`core::invalid_id` 表示无热源。

## 体几何

`VolumeSpec { material, source, geometry }` 的 geometry 为三维轴对齐 `BoxOperation { operation, min, max }` 序列。每个 box 的 min/max 是绝对 SI 坐标，必须有限且严格递增。操作按顺序执行：最后命中的 Add/Subtract 决定点是否属于体；后添加的体覆盖前面的体。当前网格离散保持笛卡尔六面体，不引入 CAD 对象树。

`ConformalMeshSpec` 接收严格递增的 x/y/z 坐标轴，运行时的 `ConformalMesh` 按索引计算 Cell 几何与内部连接。`NonConformalMesh` 接收显式 Cell 包围盒和全部面接触的 `Connection {a,b}`，支持粗细单元一对多接触。真实生效的材料或活动域边界必须沿 Cell 面；无效或被覆盖的内部盒边界允许不对齐。热源可在同材料单元内部按子体积分区。非法几何、重复/遗漏连接和体积重叠均报错。

`SurfaceRegion { axis, coordinate, rectangle }` 描述 SI 平面区域。编译器验证轴、坐标有限性及矩形范围的严格递增，并将热学和流体区域用于同一套外部 BoundaryFacet 切分。ThermalBoundarySpec 按添加顺序覆盖，default_thermal_boundary 覆盖未命中的外表面。流体边界共用 SurfaceRegion，并采用 PressureInlet、MassFlowInlet、PressureOutlet variant；入口缺省温度采用模型初温。

## 应用层

PTA 的 XML、LengthUnit、Layer、Block、Rect、FaceKey、几何变量及表达式形式参数规则全部位于 `apps/metahotspot`。`pta::normalize` 使用独立的算术求值器解析几何表达式，将层厚度/偏移/矩形展开成绝对三维 SI box，解析材料名，换算边界、网格和探针坐标，并完成 PTA 的 `x→T/t` 重写。随后才调用 `compiler::build_model`。

CLI 兼容 PTA 时保留从 XML 的 `Results/Mesh` 或 `GeneratedMesh` 读取网格坐标轴的行为。读取与单位换算属于应用层，转换后通过独立的 `ConformalMeshSpec` 传入编译器；内核与公开 API 不依赖 PTA。非共形网格通过程序构造，不提供 XML reader；CLI 缺少共形坐标轴时显式报错。

PTA `VolumetricHeatSource` 使用 W/m³；引用裸函数名时，适配器转换为 `function(t)`。PTA 的周期分段常数函数 Interval 为每段时长，周期为 Interval × Values.Count。

`OutputMesh`、节点插值、PTA XML 结果回写和 VTU 写出均属于应用私有实现。运行期 Model 只保存计算数据。SDK 不依赖 tinyxml2，不安装应用层头文件。

## C ABI / Python

使用 add_material 返回的 ID、add_source 返回的 ID、add_volume 和 volume_add_box（Python 为 add_box）构造模型。所有坐标直接使用 SI。调用方通过编译网格与 raw Solution 接口取得几何和场数据。

C ABI 使用 `mhs_surface_region_t`、`mhs_model_set_conformal_mesh` 和 `mhs_model_add_point_probe` 建模；点探针传入 `const double xyz[3]`。求解配置通过 `mhs_compiled_solve` 的 `mhs_study_t` 参数传入。

Python 的 `set_conformal_mesh`、`add_point_probe(name, position)` 分别对应建模入口；`Compiled.solve(study=Study(...))` 指定求解类型和时刻。热与流体边界统一接受非空 `Sequence[SurfaceRegion]`。
以下示例创建一个可编译、可求解的模型：

```python
from metahotspot import Model, SurfaceRegion
from metahotspot.enums import Axis

with Model() as model:
    model.set_conformal_mesh([0, 0.001], [0, 0.001], [0, 0.001])
    material = model.add_material("solid", "1", "1", "1", "1000", "500")
    volume = model.add_volume(material)
    model.add_box(volume, (0, 0, 0), (0.001, 0.001, 0.001))
    bottom = [SurfaceRegion(Axis.Z, 0, 0, 0.001, 0, 0.001)]
    model.add_dirichlet(bottom, temperature="300")
    model.add_point_probe("center", (0.0005, 0.0005, 0.0005))
    compiled = model.compile()
    solution = compiled.solve()
```

`SurfaceRegion` 表示一个连续矩形，切向坐标顺序为 X→(y,z)、Y→(x,z)、Z→(x,y)。
多个矩形直接组成区域序列。默认热边界通过 `set_default_*` 指定；空区域调用报错。
流体边界分别使用 `add_pressure_inlet(regions, pressure, inlet_temperature=None, pump_resistance=0.0)`、`add_mass_flow_inlet(regions, mass_flow_rate, inlet_temperature=None)` 和 `add_pressure_outlet(regions, pressure)`；C ABI 提供对应的 `mhs_model_add_pressure_inlet`、`mhs_model_add_mass_flow_inlet` 和 `mhs_model_add_pressure_outlet`。泵阻单位为 Pa·s/m³，作用于整个入口总流量。

`mhs_model_t` 是可变建模入口，保存模型定义与点探针；study 在求解调用时提供。`compile` 创建独立、
不可变的 compiled handle。材料通过 `MaterialKind::{Solid, Fluid}` 显式分类；
`dynamic_viscosity` 只是流体材料的必需物性，不承担类型判定。C ABI 使用
`mhs_material_kind_t`，Python 使用 `MaterialKind`。PTA 适配器直接把 XML 的
`FluidMaterial` 映射到该分类，并继续在 app 侧展开多矩形 FaceKey。

编译器先根据热与流体区域的联合切分生成外部 `BoundaryFacet`，热学与流体分别通过 `BoundaryFacetId` 绑定条件。内部连接仅为 CellId 对，共形网格中由索引生成；不构造内部 Facet 或 Cell–Facet CSR。

`mhs_compiled_copy_fields` 导出单元、外边界和热源投影。外边界使用 `boundary_owner`、`boundary_side`、`boundary_axis`、`boundary_coordinate`、`boundary_bounds`；`boundary_side` 为外法向符号 -1/+1。删除旧的全体 Facet minus/plus 格式，不保留旧 API 名称。`Compiled.connections()` 按需导出连接编号，支持宏模型的人工子域切口计算。

程序构造非共形网格无需经过 PTA：

```python
model.set_nonconformal_mesh(
    cell_min=[[0, 0, 0], [1, 0, 0], [1, 1, 0]],
    cell_max=[[1, 2, 1], [2, 1, 1], [2, 2, 1]],
    connections=[[0, 1], [0, 2], [1, 2]],
)
```

这些单元仍由 `ModelDefinition` 中的连续体区域赋予材料、源和外边界条件。域内使用 TPFA，尚不包含多域连接和 NCC。本轮保留现有 CLI VTU 写出，不增加 SDK/Python Writer。
