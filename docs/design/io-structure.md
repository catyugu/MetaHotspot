# Authoring Model

`src/common/model_definition.hpp` 是唯一建模数据契约，位于 `mhs::model`，为 header-only 轻量类型。它不镜像 XML schema，也不依赖 tinyxml2、muparser、Eigen、TBB 或 spdlog。

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

## 附录：ModelDefinition 构造

`ModelDefinition` 目前由 C API 的直接 Handle 函数（如 `mhs_model_add_material`、`mhs_model_add_block`）和 XML reader (`mhs::io::read_xml`) 共同填充。两者操作同一套数据结构，不经过中间 Builder。
`BlockSpec`、`LayerSpec`、`BoundaryPatch` 等类型定义见前文。C API 的 `mhs_model_t` 是唯一的外部构造入口。
