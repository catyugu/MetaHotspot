# ModelDefinition 与 PTA 适配

`model/model_definition.hpp` 是库的建模契约。所有几何、网格坐标和探针坐标使用 SI 米；材料及边界表达式使用规范变量 `x/y/z/T/t`，编译器原样解析。材料中的 `T` 表示单元温度，热边界中的 `T` 表示表面迹温度。

`ModelDefinition` 保存 settings、mesh、variables、functions、materials、sources、volumes、thermal_boundaries、default_thermal_boundary、point_probes 和 fluid_boundaries。材料和热源采用有序表，`VolumeSpec` 使用表内 ID 引用；`core::invalid_id` 表示无热源。

## 体几何

`VolumeSpec { material, source, geometry }` 的 geometry 为三维轴对齐 `BoxOperation { operation, min, max }` 序列。每个 box 的 min/max 是绝对 SI 坐标，必须有限且严格递增。操作按顺序执行：最后命中的 Add/Subtract 决定点是否属于体；后添加的体覆盖前面的体。当前网格离散保持笛卡尔六面体，不引入 CAD 对象树。

编译器消费 `ModelDefinition::domain.mesh` 中显式给定、严格递增的 Cartesian 网格轴，构造内部 `DiscreteMesh`。热源按实际几何交叠体积积分，不要求源区域边界与单元边界重合。

`SurfaceRegion { axis, coordinate, rectangle }` 描述 SI 平面区域。ThermalBoundarySpec 按添加顺序覆盖，default_thermal_boundary 覆盖未命中的外表面。流体边界共用 SurfaceRegion，并采用 PressureInlet、MassFlowInlet、PressureOutlet variant；入口缺省温度采用模型初温。

## 应用层

PTA 的 XML、LengthUnit、Layer、Block、Rect、FaceKey 和表达式形式参数规则全部位于 `apps/metahotspot`。`pta::normalize` 将层厚度/偏移/矩形展开成绝对三维 SI box，解析材料名，换算边界、网格和探针坐标，并完成 PTA 的 `x→T/t` 重写。随后才调用 `compiler::build_model`。

CLI 兼容 PTA 时保留从 XML 的 `Results/Mesh` 或 `GeneratedMesh` 读取网格坐标轴的行为。读取与单位换算属于应用层，转换后通过通用的 `ModelDefinition::domain.mesh` 传入编译器；内核与公开 API 不包含 PTA 网格导入逻辑或外部 `DiscreteMesh` 输入入口。

PTA `VolumetricHeatSource` 使用 W/m³；引用裸函数名时，适配器转换为 `function(t)`。PTA 的周期分段常数函数 Interval 为每段时长，周期为 Interval × Values.Count。

`OutputMesh`、节点插值、PTA XML 结果回写和 VTU 写出均属于应用私有实现。运行期 Model 只保存计算数据。SDK 不依赖 tinyxml2，不安装应用层头文件。

## C ABI / Python

使用 add_material 返回的 ID、add_source 返回的 ID、add_volume 和 volume_add_box（Python 为 add_box）构造模型。所有坐标直接使用 SI。调用方通过编译网格与 raw Solution 接口取得几何和场数据。

C ABI 使用
`mhs_surface_region_t`、`mhs_model_set_cartesian_mesh`、`mhs_model_set_study`
和 `mhs_model_add_point_probe`。点探针传入 `const double xyz[3]`。

Python 的 `set_cartesian_mesh`、`set_study` 和 `add_point_probe(name, position)`
对应这些入口。热边界与流体边界统一接受非空 `Sequence[SurfaceRegion]`。
以下示例创建一个可编译、可求解的模型：

```python
from metahotspot import Model, SurfaceRegion
from metahotspot.enums import Axis

with Model() as model:
    model.set_study()
    model.set_cartesian_mesh([0, 0.001], [0, 0.001], [0, 0.001])
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
流体边界同样使用 `add_fluid_boundary(regions, kind, value, inlet_temperature=None)`。

`mhs_model_t` 是可变建模入口，持有 study 设置与点探针。`compile` 创建独立、
不可变的 compiled handle。内核中的材料以 `dynamic_viscosity` 是否存在区分固体/流体，
不维护额外布尔标记。PTA 适配器在 app 侧校验 `FluidMaterial`，展开多矩形 FaceKey。

几何投影只处理区域与网格的交叠，返回 cell、axis、sign、bounds 和不透明 tag。
热与流体编译器分别解释 tag；投影算法不依赖热边界 ID 或条件类型。
