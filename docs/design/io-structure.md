# ModelDefinition 与 PTA 适配

`model/model_definition.hpp` 是库的建模契约。所有几何、网格坐标和探针坐标使用 SI 米；材料及边界表达式使用规范变量 `x/y/z/T/t`，编译器原样解析。材料中的 `T` 表示单元温度，热边界中的 `T` 表示表面迹温度。

`ModelDefinition` 保存 settings、mesh、variables、functions、materials、sources、volumes、boundaries、default_boundary、observation_points 和 fluid_boundaries。材料和热源采用有序表，`VolumeSpec` 使用表内 ID 引用；`core::invalid_id` 表示无热源。

## 体几何

`VolumeSpec { material, source, geometry }` 的 geometry 为三维轴对齐 `BoxOperation { operation, min, max }` 序列。每个 box 的 min/max 是绝对 SI 坐标，必须有限且严格递增。操作按顺序执行：最后命中的 Add/Subtract 决定点是否属于体；后添加的体覆盖前面的体。当前网格离散保持笛卡尔六面体，不引入 CAD 对象树。

编译器消费 `ModelDefinition::mesh` 中显式给定、严格递增的 Cartesian 网格轴，构造内部 `DiscreteMesh`。热源按实际几何交叠体积积分，不要求源区域边界与单元边界重合。

`FaceRegion { axis, coordinate, rectangles }` 描述 SI 平面区域。BoundaryPatch 按添加顺序覆盖，default_boundary 覆盖未命中的外表面。流体边界共用 FaceRegion，并采用 PressureInlet、MassFlowInlet、PressureOutlet variant；入口缺省温度采用模型初温。

## 应用层

PTA 的 XML、LengthUnit、Layer、Block、Rect、FaceKey 和表达式形式参数规则全部位于 `apps/metahotspot`。`pta::normalize` 将层厚度/偏移/矩形展开成绝对三维 SI box，解析材料名，换算边界、网格和探针坐标，并完成 PTA 的 `x→T/t` 重写。随后才调用 `compiler::build_model`。

CLI 兼容 PTA 时保留从 XML 的 `Results/Mesh` 或 `GeneratedMesh` 读取网格坐标轴的行为。读取与单位换算属于应用层，转换后通过通用的 `ModelDefinition::mesh` 传入编译器；内核与公开 API 不包含 PTA 网格导入逻辑或外部 `DiscreteMesh` 输入入口。

PTA `VolumetricHeatSource` 使用 W/m³；引用裸函数名时，适配器转换为 `function(t)`。PTA 的周期分段常数函数 Interval 为每段时长，周期为 Interval × Values.Count。

`OutputMesh`、节点插值、PTA XML 结果回写和 VTU 写出均属于应用私有实现。运行期 Model 只保存计算数据。SDK 不依赖 tinyxml2，不安装应用层头文件。

## C ABI / Python

使用 add_material 返回的 ID、add_source 返回的 ID、add_volume 和 volume_add_box（Python 为 add_box）构造模型。所有坐标直接使用 SI。XML 读写、层/块/矩形和 VTU 导出接口已删除；调用方通过编译网格与 raw Solution 接口取得几何和场数据。

## 编译和求解边界

`ModelDefinition::settings` 和 `observation_points` 保留建模输入；`build_model` 仅编译物理模型。C++ 调用方将 settings 中的 study_type、transient_duration、transient_output_interval 组成 `solver::StudyOptions`，并将 `observation_points`（`solver::ProbeSpec` 数组）传给 `solve`。这些请求不进入 `model::Model`。

compiler 内部的 `SurfaceRegion` 引用原始 `FaceRegion` 并携带调用方 tag；`project_surface_regions` 返回纯几何 `SurfacePatch`。后添加区域覆盖前面的区域，未命中部分使用 fallback，invalid fallback 省略未覆盖部分。热边界在投影后转换为 `BoundaryFace`；流体直接使用 tag 查找自己的边界表。输入引用只在编译期间有效，不保存在运行期模型中。
