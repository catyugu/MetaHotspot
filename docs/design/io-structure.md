# ModelDefinition 与 PTA 适配

`model/model_definition.hpp` 是库的建模契约。所有几何、网格坐标和探针坐标使用 SI 米；材料及边界表达式使用规范变量 `x/y/z/T/t`，编译器原样解析。材料中的 `T` 表示单元温度，热边界中的 `T` 表示表面迹温度。

`ModelDefinition` 同时容纳连续物理定义和独立的 `MeshSpec`。连续定义包括材料/表达式表、`regions`、`material_assignments`、`source_assignments`、热与流体边界及探针。更换网格无需修改这些物理区域。当前仍只生成全局 Cartesian 网格，局部加密和非共形网格尚未实现。

## 体几何与物理赋值

`RegionSpec { geometry }` 是可复用的连续区域，其 `RegionId` 是 `regions` 中的索引，与 CellId/FacetId 无关。`geometry` 为轴对齐 `BoxOperation { operation, min, max }` 序列；SI 坐标必须有限且严格递增。操作按顺序执行，最后命中的 Add/Subtract 决定点是否属于区域。

`MaterialAssignment { region, material }` 定义单值材料场。材料赋值区域的并集构成计算域，重叠处后赋值者优先；未绑定材料的区域不会增加计算域。每个有效控制体必须完全属于同一个材料 ID，允许同材料的多个区域共同覆盖一个单元。跨材料或计算域边界的单元会报错，包括中心落在域外、但仍有部分体积与域相交的单元。前端应在有效材料/域边界处提供切割；编译器采用现有 `geometry_eps` 容差进行局部几何切分。

`SourceAssignment { region, source, operation }` 独立定义热源。默认 `SourceOperation::Add` 将所有命中的源项相加；`Replace` 在其作用区域内清除此前的源项，再应用当前源项。仅 Replace 允许 `source = core::invalid_id`，表示局部清除。源项只在材料域内积分，源区域可以局部覆盖单元，也可与材料赋值复用同一个 RegionId。积分使用区域交叠子体积上的中点求积，保持现有空间表达式的求积规则。

例如，C++ 输入可以为一个材料域绑定两个独立热源：

```cpp
ModelDefinition d;
d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
d.materials.push_back({"solid", MaterialSpec {}});
d.sources = {"10", "20*t"};
d.regions.push_back({{{GeometryOperation::Add, {0, 0, 0}, {1, 1, 1}}}});
d.material_assignments.push_back({0, 0});
d.source_assignments = {{0, 0}, {0, 1}};
```

C++ 调用方将原先 `volumes` 的 geometry 移入 `regions`，材料与热源分别绑定；需要保留旧体覆盖语义时，为每个体显式添加 Replace 源赋值，包括无源体的清除操作。C ABI 和 Python 的既有 `add_volume`/`add_box` 调用形式保持不变，由入口完成此转换。

编译器对每个候选单元建立一次局部几何划分，共享区域包含关系以判定材料、域占用和热源积分，复用切分与分类缓冲区。全部位于域外的单元被跳过。随后构造唯一 Facet 与 Cell–Facet 关联；运行期 Model 的数据格式保持不变。探针在活动单元确定后通过网格轴二分查询定位，共享平面仍选择原扫描顺序中最后一个包含该点的活动单元。

`SurfaceRegion { axis, coordinate, rectangle }` 描述 SI 平面区域。编译器验证轴、坐标有限性及矩形范围的严格递增，并将热学和流体区域用于同一套外部 Facet 切分。ThermalBoundarySpec 按添加顺序覆盖，default_thermal_boundary 覆盖未命中的外表面。流体边界共用 SurfaceRegion，并采用 PressureInlet、MassFlowInlet、PressureOutlet variant；入口缺省温度采用模型初温。

## 应用层

PTA 的 XML、LengthUnit、Layer、Block、Rect、FaceKey、几何变量及表达式形式参数规则全部位于 `apps/metahotspot`。`pta::normalize` 使用独立的算术求值器解析几何表达式，将层厚度/偏移/矩形展开成绝对三维 SI box，解析材料名，换算边界、网格和探针坐标，并完成 PTA 的 `x→T/t` 重写。随后将每个 Block 转成一个 Region、一个材料赋值和一个 Replace 热源赋值，再调用 `compiler::build_model`；因此后 Block 覆盖前 Block 的材料与热源，包括无源覆盖。

CLI 兼容 PTA 时保留从 XML 的 `Results/Mesh` 或 `GeneratedMesh` 读取网格坐标轴的行为。读取与单位换算属于应用层，转换后通过通用的 `ModelDefinition::mesh` 传入编译器；内核与公开 API 不包含 PTA 网格导入逻辑或外部 `DiscreteMesh` 输入入口。

PTA `VolumetricHeatSource` 使用 W/m³；引用裸函数名时，适配器转换为 `function(t)`。PTA 的周期分段常数函数 Interval 为每段时长，周期为 Interval × Values.Count。

`OutputMesh`、节点插值、PTA XML 结果回写和 VTU 写出均属于应用私有实现。运行期 Model 只保存计算数据。SDK 不依赖 tinyxml2，不安装应用层头文件。

## C ABI / Python

使用 add_material 返回的 ID、add_source 返回的 ID、add_volume 和 volume_add_box（Python 为 add_box）构造模型。所有坐标直接使用 SI。调用方通过编译网格与 raw Solution 接口取得几何和场数据。

C ABI 使用 `mhs_surface_region_t`、`mhs_model_set_cartesian_mesh` 和 `mhs_model_add_point_probe` 建模；点探针传入 `const double xyz[3]`。求解配置通过 `mhs_compiled_solve` 的 `mhs_study_t` 参数传入。

Python 的 `set_cartesian_mesh`、`add_point_probe(name, position)` 分别对应建模入口；`Compiled.solve(study=Study(...))` 指定求解类型和时刻。热与流体边界统一接受非空 `Sequence[SurfaceRegion]`。
以下示例创建一个可编译、可求解的模型：

```python
from metahotspot import Model, SurfaceRegion
from metahotspot.enums import Axis

with Model() as model:
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

`mhs_model_t` 是可变建模入口，保存模型定义与点探针；study 在求解调用时提供。`compile` 创建独立、
不可变的 compiled handle。材料通过 `MaterialKind::{Solid, Fluid}` 显式分类；
`dynamic_viscosity` 只是流体材料的必需物性，不承担类型判定。C ABI 使用
`mhs_material_kind_t`，Python 使用 `MaterialKind`。PTA 适配器直接把 XML 的
`FluidMaterial` 映射到该分类，并继续在 app 侧展开多矩形 FaceKey。

编译器先根据热与流体边界区域的联合切分生成唯一的外部 `Facet`，再构建 Cell–Facet 关联。热学与流体分别通过 `FacetId` 绑定条件；`facet_coordinate` 和 `facet_bounds` 是面片几何的事实来源。C ABI 通过 `mhs_compiled_copy_fields` 将单元、Facet 和热源投影一次性复制到调用者缓冲区。

## 本轮验证（2026-11-10）

基线为 `dev` 的 `861230b5ee3126278963857db3548a0e32829ecf`，本地使用 GCC 14.2、CMake Release、相同依赖构建前后两版。

- CTest：81/81；Python：26/26。既有跨域用例改为验证未对齐时报错、补充切割后保持正确源项体积，没有新增测试文件或测试项。
- `run_cases.py`：10 个稳态、非线性、瞬态和微流体案例全部通过；前后生成的 10 份 XML 和 10 份 VTU 逐文件完全一致。
- 临时独立 C++ 验证覆盖 Add/Replace/清除、部分源区、材料优先级、域边界穿单元、源区域不扩大计算域和无效引用；324 个含共享网格平面、空洞和域外点的探针位置与穷举定位一致。
- 合成编译计时：单立方体材料域，三个轴均为 `linspace(0,1,31)`，27,000 单元；探针为 `(q,q,q)`，`q=(i+0.5)/4000`。仅计 Python `Model.compile()`，不计建模及求解，每组 15 次编译取中位数：无探针 11.102→6.663 ms；4,000 探针 88.173→6.209 ms。这是该容器与该简单模型的观测值，不代表复杂多区域模型的加速比例。
- 修改过的 C++ 文件通过 clang-format 和 `git diff --check`。跨平台构建仍由 PR 的 CI 验证。
