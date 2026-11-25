# FomDefinition 与 PTA 适配

`model/fom_definition.hpp` 是库的建模契约。所有几何、网格坐标和探针坐标使用 SI 米；材料及边界表达式使用规范变量 `x/y/z/T/t`，编译器原样解析。材料中的 `T` 表示单元温度，热边界中的 `T` 表示表面迹温度。

`FomDefinition` 保存 reference_temperature、functions、materials、sources、volumes、thermal_boundaries、default_thermal_boundary、point_probes 和 fluid_boundaries。Cartesian 网格规格由独立的 `MeshSpec` 保存。材料和热源采用有序表，`VolumeSpec` 使用表内 ID 引用；`core::invalid_id` 表示无热源。

## 体几何

`VolumeSpec { material, source, geometry }` 的 geometry 为三维轴对齐 `BoxOperation { operation, min, max }` 序列。每个 box 的 min/max 是绝对 SI 坐标，必须有限且严格递增。操作按顺序执行：最后命中的 Add/Subtract 决定点是否属于体；后添加的体覆盖前面的体。当前网格离散保持笛卡尔六面体，不引入 CAD 对象树。

编译器独立接收 `MeshSpec` 中显式给定、严格递增的 Cartesian 网格轴，构造`ConformalMesh`。当前共形编译路径要求网格内部的 Box 边界坐标与对应坐标轴顶点对齐（容差 `geometry_eps`）；不符合时抛出 `std::invalid_argument`。体热源按单元中心中点积分，权重为完整单元体积。热与流体边界矩形可局部细分外表面。

`SurfaceRegion { axis, coordinate, rectangle }` 描述 SI 平面区域。编译器验证轴、坐标有限性及矩形范围的严格递增，并将热学和流体区域用于同一套外部 Facet 切分。ThermalBoundarySpec 按添加顺序覆盖，default_thermal_boundary 覆盖未命中的外表面。流体边界共用 SurfaceRegion，并采用 PressureInlet、MassFlowInlet、PressureOutlet variant；入口缺省温度采用模型初温。

## 应用层

PTA 的 XML、LengthUnit、Layer、Block、Rect、FaceKey、几何变量及表达式形式参数规则全部位于应用共享的 `apps/common/pta`。`definition.hpp/.cpp` 负责定义与规范化（含几何求值），`io.hpp/.cpp` 负责 XML 读取和结果写回；`xml.hpp` 仅为需要直接操作 XML 的应用实现提供内部入口。`pta::normalize` 使用独立的算术求值器解析几何表达式，将层厚度/偏移/矩形展开成绝对三维 SI box，解析材料名，换算边界、网格和探针坐标，并完成 PTA 的 `x→T/t` 重写。随后才调用 `compiler::compile_fom`。

CLI 兼容 PTA 时保留从 XML 的 `Results/Mesh` 或 `GeneratedMesh` 读取网格坐标轴的行为。读取与单位换算属于应用层，转换后通过独立的 `MeshSpec` 传入编译器。

PTA `VolumetricHeatSource` 使用 W/m³；引用裸函数名时，适配器转换为 `function(t)`。PTA 的周期分段常数函数 Interval 为每段时长，周期为 Interval × Values.Count。

`OutputMesh`、节点插值、PTA XML 结果回写和 VTU 写出均属于应用私有实现。运行期 `Fom` 只保存计算数据。原生 EROM 读取归 io 组件，入口为 `mhs::io::read_erom`。io 将 tinyxml2 作为私有实现对象编入静态库，供原生 EROM XML 和应用内部 XML 操作复用；model 不处理文件格式，也不依赖 io 或 tinyxml2。SDK 不安装或查找 tinyxml2，也不安装 PTA 应用层头文件。

## C ABI / Python

使用 add_material 返回的 ID、add_source 返回的 ID、add_volume 和 volume_add_box（Python 为 add_box）构造模型。所有坐标直接使用 SI。调用方通过编译网格与 raw Solution 接口取得几何和场数据。

C ABI 使用 `mhs_surface_region_t`、`mhs_model_set_cartesian_mesh` 和 `mhs_model_add_point_probe` 建模；点探针传入 `const double xyz[3]`。求解配置通过 `mhs_compiled_solve` 的 `mhs_study_t` 参数传入。

Python 的 `set_cartesian_mesh`、`add_point_probe(name, position)` 分别对应建模入口；`Compiled.solve(study=Study(...))` 指定求解类型和时刻。热与流体边界统一接受非空 `Sequence[SurfaceRegion]`。
以下示例创建一个可编译、可求解的模型：

```python
from metahotspot import Fom, SurfaceRegion
from metahotspot.enums import Axis

with Fom() as model:
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
流体边界分别使用 `add_pressure_inlet(regions, pressure, inlet_temperature=None, pump_resistance=0.0)`、`add_mass_flow_inlet(regions, mass_flow_rate, inlet_temperature=None)` 和 `add_pressure_outlet(regions, pressure)`；C ABI 提供对应的 `mhs_model_add_pressure_inlet`、`mhs_model_add_mass_flow_inlet` 和 `mhs_model_add_pressure_outlet`。泵阻单位为 Pa·s/m³，作用于整个入口总流量。

`mhs_model_t` 是可变建模入口，保存模型定义与点探针；study 在求解调用时提供。`compile` 创建独立、
不可变的 compiled handle。材料通过 `MaterialKind::{Solid, Fluid}` 显式分类；
`dynamic_viscosity` 只是流体材料的必需物性，不承担类型判定。C ABI 使用
`mhs_material_kind_t`，Python 使用 `MaterialKind`。PTA 适配器直接把 XML 的
`FluidMaterial` 映射到该分类，并继续在 app 侧展开多矩形 FaceKey。

编译器联合热与流体边界区域切分外表面，并按所属单元存放 `ExteriorFacet`。热边界组随面片保存，流体标签用于压力预处理。C API 的 `mhs_compiled_copy_fields` 将单元、外表面和热源投影复制到调用者的缓冲区。
