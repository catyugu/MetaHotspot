# Authoring Model

`src/model/model_definition.hpp` 是唯一建模数据契约，位于 `mhs::model`，由独立的轻量 `mhs_model` 拥有。它不镜像 XML schema，也不依赖 tinyxml2、muparser、Eigen、TBB 或 spdlog。

`mhs_model` 本身不链接第三方目标。XML 适配、表达式编译和线性求解分别留在 `mhs_io`、`mhs_expression` 与 `mhs_linear`；因此修改 builder 或模型契约时，不会重新编译这些较慢的第三方依赖及其封装目标。

XML reader 和外部代码都生成同一个 `ModelDefinition`，随后调用 `mhs::sim::build_model()` 编译为运行期 `mhs::core::Model`。

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

## ModelBuilder

`ModelBuilder` 是未来 C opaque-pointer API 的内部落点。C ABI 将公开 `typedef void *mhs_model_handle` 这类纯指针句柄，内部再转换为私有的 `ModelBuilder`；不会在公共头中声明或暴露 opaque struct tag。

当前 `ModelBuilder` 只提供 append-only 建模：`add_layer(LayerParams)` 与 `add_block(LayerId, BlockParams)` 的参数不含子节点，Block 和 Rect 只能通过各自的追加操作进入模型；最后由 `finish() &&` 移出 `ModelDefinition`。当前阶段不承担校验、删除、插入或重排序，也不保留旧 C++ 建模结构的兼容别名。
