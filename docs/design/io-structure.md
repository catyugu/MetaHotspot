# Authoring Model

`src/core/model_definition.hpp` 是唯一建模数据契约，位于 `mhs::model`，为 header-only 轻量类型。它不镜像 XML schema，也不依赖 tinyxml2、muparser、Eigen、TBB 或 spdlog。

XML reader 和外部代码都直接填充 `ModelDefinition` 结构，随后调用 `mhs::sim::build_model()` 编译为运行期 `mhs::core::Model`。

## 顶层结构

```cpp
struct ModelDefinition {
    ModelSettings settings;
    MeshSpec mesh;
    std::vector<VariableSpec> variables;
    std::vector<NamedFunction> functions;
    std::vector<NamedMaterial> materials;
    std::vector<LayerSpec> layers;
    std::vector<BoundaryPatch> boundaries;
    ThermalBoundary default_boundary;
    std::vector<ObservationPointSpec> observation_points;
    std::vector<FluidBoundarySpec> fluid_boundaries;
};
```

材料库与函数库使用有序 vector 作为事实存储。Compiler 可以临时建立名称索引，但不得用无序容器取代输入顺序。

## 几何

```cpp
struct RectOperation {
    GeometryOperation operation; // Add / Subtract
    RectSpec rect;
};

struct BlockSpec {
    std::string material;
    Expression volumetric_heat_source;
    Expression x_offset;
    Expression y_offset;
    std::optional<Expression> thickness;
    std::vector<RectOperation> geometry;
};

struct LayerSpec {
    Expression thickness;
    Expression x_offset;
    Expression y_offset;
    std::vector<BlockSpec> blocks;
};
```

顺序是模型契约：

- RectOperation 按 append 顺序执行；一个点最后命中的 Add/Subtract 决定其是否属于 Block。
- 同一 Layer 中后出现的 Block 覆盖前者，并同时提供材料和体热源。
- `layers[0]` 保持为最上层，后续层依次向下堆叠。

## 材料与函数

`MaterialSpec` 使用领域名 `conductivity_x/y/z`、`density`、`specific_heat` 和可选 `dynamic_viscosity`。旧 XML 名称只允许出现在 `src/io/xml_model_reader.cpp` 的 tag 解析中。

函数以 `NamedFunction {name, value}` 有序保存。`ExpressionFunctionSpec`、`DoubleExponentialFunctionSpec`、`GaussFunctionSpec`、`SineFunctionSpec` 和 `PiecewiseFunctionSpec` 组成 `FunctionSpec` variant。

## 结构化边界

```cpp
struct FaceRegion {
    Axis axis;
    double coordinate;
    std::vector<RegionRect> rectangles;
};

struct BoundaryPatch {
    std::vector<FaceRegion> regions;
    ThermalBoundary condition;
};
```

BoundaryPatch 按 append 顺序覆盖，后出现者获胜；`default_boundary` 仅在没有显式区域命中时生效。热边界和流体边界共享 `FaceRegion`，模型编译器不解析外部格式字符串。

旧 XML 的 FaceKey 编码仅由 `src/io/face_region_parser.cpp` 转换为 `FaceRegion`，不会传播到建模层和数值层。

## 流体输入

材料 XML 仅接受 `ThermalConductivity`、`Density`、`SpecificHeatCapacity`、
`DynamicViscosity` 和 `FluidMaterial`。每条材料记录都必须显式给出
`FluidMaterial`；流体材料还必须提供非空的 `DynamicViscosity`。通过 C API 添加
材料时，非空的动态黏度参数会将材料标记为流体。

主输入文件是流体材料、流体块、函数和流体边界的唯一输入来源。流体边界类型由字段组合决定：
`InletTemperature` 与 `Pressure` 对应 `PressureInletBoundary`，
`InletTemperature` 与 `MassFlowRate` 对应 `MassFlowInletBoundary`，仅有
`Pressure` 对应 `OutletBoundary`。一个边界可包含多个 `FaceKeys/string` 面区域。
`MassFlowRate` 表示该边界所有匹配面的总流量
(kg/s)，并在这些面之间平均分配；正值表示质量流入计算域。压力按输入值使用。
冻结流模型按单元采用恒定密度且视为不可压缩，入口与出口之间的压力差驱动流动。

入口温度用于随入口流入的质量。出口采用零梯度流出；若出口发生回流，则使用相邻
单元温度，因为没有为回流指定温度。

`VolumetricHeatSource` 是单位为 W/m3 的体热源。积分功率等于热源值乘以材料块占据的
体积 (m3)。该字段引用的表达式函数使用热源时间变量。
`PeriodicPiecewiseConstantFunction.Interval` 表示一个分段的时长；各值按左闭右开区间
取值，并以 `Interval * Values.Count` 为周期重复。

C API 和 Python 的流体边界枚举按 `None`、`PressureInlet`、`MassFlowInlet`、`Outlet`
定义。移除旧的 `Velocity` 边界是一次有意的不兼容接口变更。

## 附录：ModelDefinition 构造

`ModelDefinition` 目前由 C API 的直接 Handle 函数（如 `mhs_model_add_material`、`mhs_model_add_block`）和 XML reader (`mhs::io::read_xml`) 共同填充。两者操作同一套数据结构，不经过中间 Builder。
`BlockSpec`、`LayerSpec`、`BoundaryPatch` 等类型定义见前文。C API 的 `mhs_model_t` 是唯一的外部构造入口。
