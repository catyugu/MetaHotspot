# ModelDefinition 与 PTA 适配

`model/model_definition.hpp` 是库的建模契约。所有几何、网格坐标和探针坐标使用 SI 米；材料及边界表达式使用规范变量 `x/y/z/T/t`，编译器原样解析。材料中的 `T` 表示单元温度，热边界中的 `T` 表示表面迹温度。

`ModelDefinition` 使用 `materials`、`bodies` 和 `volume_sources` 三个独立列表。材料复用 `MaterialSpec`；`BodySpec {geometry, material}` 保存实体几何和材料引用；`VolumeSourceSpec {geometry, power_density}` 直接保存独立体热源及体积功率密度表达式（W/m³）。没有独立的热源表达式表或总功率模式。

## 体几何

`VolumeGeometry` 保存绝对 SI 坐标下的有序 `BoxOperation {operation, min, max}`。同一对象的 Add/Subtract 操作按顺序生效；后定义 Body 的材料覆盖前定义 Body。各 VolumeSource 的功率直接叠加，热源对象不创建计算域，也不改变材料分布。

编译器根据 Body 与显式 Cartesian 网格轴创建有效 Cell、赋予材料，再逐个计算 VolumeSource 与有效 Cell 的几何交叠，生成 `SourceTerm`。源边界可以穿过 Cell，重叠源的积分项独立保留。材料界面必须与单元边界对齐；出现单元内部材料变化时拒绝编译。

`SurfaceRegion { axis, coordinate, rectangle }` 描述 SI 平面区域。编译器验证轴、坐标有限性及矩形范围的严格递增，并将热学和流体区域用于同一套外部 Facet 切分。ThermalBoundarySpec 按添加顺序覆盖，default_thermal_boundary 覆盖未命中的外表面。流体边界共用 SurfaceRegion，并采用 PressureInlet、MassFlowInlet、PressureOutlet variant；入口缺省温度采用模型初温。

## 应用层

PTA 的 XML、LengthUnit、Layer、Block、Rect、FaceKey、几何变量及表达式形式参数规则全部位于 `apps/metahotspot`。`pta::normalize` 使用独立的算术求值器解析几何表达式，将层厚度/偏移/矩形展开成绝对三维 SI box，解析材料名，换算边界、网格和探针坐标，并完成 PTA 的 `x→T/t` 重写。PTA Block 的功耗按照 Block 优先级裁剪可见体积后转换成独立 Source；这一旧格式的覆盖语义不进入核心编译器。随后才调用 `compiler::build_model`。

CLI 兼容 PTA 时保留从 XML 的 `Results/Mesh` 或 `GeneratedMesh` 读取网格坐标轴的行为。读取与单位换算属于应用层，转换后通过通用的 `ModelDefinition::mesh` 传入编译器；内核与公开 API 不包含 PTA 网格导入逻辑或外部 `DiscreteMesh` 输入入口。

PTA `VolumetricHeatSource` 使用 W/m³；引用裸函数名时，适配器转换为 `function(t)`。PTA 的周期分段常数函数 Interval 为每段时长，周期为 Interval × Values.Count。

`OutputMesh`、节点插值、PTA XML 结果回写和 VTU 写出均属于应用私有实现。运行期 Model 只保存计算数据。SDK 不依赖 tinyxml2，不安装应用层头文件。

## C ABI / Python

C ABI 使用 `mhs_model_add_body` / `mhs_model_body_add_box` 定义实体，使用 `mhs_model_add_volume_source` / `mhs_model_volume_source_add_box` 定义独立体热源。Python 使用 `add_body` / `body_add_box` 和 `add_volume_source` / `volume_source_add_box`。旧的 Volume 和 add_source 接口已移除。

C ABI 使用 `mhs_surface_region_t`、`mhs_model_set_cartesian_mesh` 和 `mhs_model_add_point_probe` 建模；点探针传入 `const double xyz[3]`。求解配置通过 `mhs_compiled_solve` 的 `mhs_study_t` 参数传入。

Python 的 `set_cartesian_mesh`、`add_point_probe(name, position)` 分别对应建模入口；`Compiled.solve(study=Study(...))` 指定求解类型和时刻。热与流体边界统一接受非空 `Sequence[SurfaceRegion]`。
以下示例创建一个可编译、可求解的模型：

```python
from metahotspot import Model, SurfaceRegion
from metahotspot.enums import Axis

with Model() as model:
    model.set_cartesian_mesh([0, 0.001], [0, 0.001], [0, 0.001])
    material = model.add_material("solid", "1", "1", "1", "1000", "500")
    body = model.add_body(material)
    model.body_add_box(body, (0, 0, 0), (0.001, 0.001, 0.001))
    source = model.add_volume_source("1e6")
    model.volume_source_add_box(source, (0, 0, 0), (0.001, 0.001, 0.001))
    bottom = [SurfaceRegion(Axis.Z, 0, 0, 0.001, 0, 0.001)]
    model.add_dirichlet(bottom, temperature="300")
    model.add_point_probe("center", (0.0005, 0.0005, 0.0005))
    compiled = model.compile()
    solution = compiled.solve()
```

`SurfaceRegion` 表示一个连续矩形，切向坐标顺序为 X→(y,z)、Y→(x,z)、Z→(x,y)。
多个矩形直接组成区域序列。默认热边界通过 `set_default_*` 指定；空区域调用报错。
流体边界同样使用 `add_fluid_boundary(regions, kind, value, inlet_temperature=None)`。

`mhs_model_t` 是可变建模入口，保存模型定义与点探针；study 在求解调用时提供。`compile` 创建独立、
不可变的 compiled handle。材料通过 `MaterialKind::{Solid, Fluid}` 显式分类；
`dynamic_viscosity` 只是流体材料的必需物性，不承担类型判定。C ABI 使用
`mhs_material_kind_t`，Python 使用 `MaterialKind`。PTA 适配器直接把 XML 的
`FluidMaterial` 映射到该分类，并继续在 app 侧展开多矩形 FaceKey。

编译器先根据热与流体边界区域的联合切分生成唯一的外部 `Facet`，再构建 Cell–Facet 关联。热学与流体分别通过 `FacetId` 绑定条件；`facet_coordinate` 和 `facet_bounds` 是面片几何的事实来源。C ABI 通过 `mhs_compiled_copy_fields` 将单元、Facet 和热源投影一次性复制到调用者缓冲区。
