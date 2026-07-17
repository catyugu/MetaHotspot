# `ModelDefinition` 输入结构

`src/data/model_definition.hpp` 定义模型构建所需的领域输入。它不是 XML DOM 的逐字段镜像：XML 读取器把外部格式裁剪并转换成这些 POD，外部代码也可以直接构造同一结构，然后调用 `mhs::sim::build_model()`。

求解输出不在这里；`Solution` 和 `ProbeTrace` 位于 `src/data/solution.hpp`。

## 顶层结构

```cpp
namespace mhs::core {

struct ModelDefinition {
    StudyType study_type = StudyType::Steady;
    LengthUnit length_unit = LengthUnit::M;
    double initial_temperature = 300.0;

    std::vector<Variable> variables;
    std::vector<Layer> layers;
    std::unordered_map<std::string, Material> materials;
    std::vector<Boundary> boundaries;

    double transient_duration = 0.0;
    double transient_time_step = 1.0; // output interval
    std::variant<FirstTypeThermalBC,
                 SecondTypeThermalBC,
                 ThirdTypeThermalBC> other_bc;

    std::vector<double> mesh_vertex_x;
    std::vector<double> mesh_vertex_y;
    std::vector<double> mesh_vertex_z;

    std::unordered_map<std::string, Function> functions;
    std::vector<ObservationPoint3D> observation_points;
    std::vector<FluidBoundary> fluid_boundaries;
};

}
```

已删除不会参与模型构建的 XML/UI 字段，例如对象显示名称、绘图区间、周期显示参数、瞬态时间单位和未实现的维度枚举。

## 几何、材料与边界

```cpp
struct Variable { std::string name; std::string value; };

struct Rect {
    bool add_sub = true;
    std::string width_expr, height_expr;
    std::string x_expr, y_expr;
};

struct Block {
    std::vector<Rect> all_rects;
    std::string material_name;
    std::string x_offset_expr, y_offset_expr;
    std::string ti_reyuan_expr;
    std::string thickness_expr;
};

struct Layer {
    std::vector<Block> blocks;
    std::string thickness_expr;
    std::string x_offset_expr, y_offset_expr;
};

struct Material {
    std::string kx = "0.0", ky = "0.0", kz = "0.0";
    std::string midu = "0.0";
    std::string bi_rerong = "0.0";
    std::string dynamic_viscosity; // 空字符串表示固体
};

struct Boundary {
    std::vector<std::string> face_keys;
    std::variant<FirstTypeThermalBC,
                 SecondTypeThermalBC,
                 ThirdTypeThermalBC> bc;
};
```

材料名称只作为 `ModelDefinition::materials` 的 key 和 `Block::material_name` 的引用存在，不在 `Material` 内重复存储。

`DaoreXishu` 的 XML 解析规则保持不变：单表达式同时赋给 `kx/ky/kz`；三个逗号分隔表达式依次赋给三轴；其他段数报错。

## 流体输入

流体数据直接属于模型定义，不再存在 `FluidOverlay` 领域类型：

```cpp
struct FluidBoundary {
    std::vector<std::string> face_keys;
    FluidBCType kind = FluidBCType::None;
    double value = 0.0;
    double inlet_temperature =
        std::numeric_limits<double>::quiet_NaN();
};
```

- 材料的 `dynamic_viscosity` 非空时，`build_model()` 将该材料编译为流体材料。
- `fluid_boundaries` 与热边界一样直接参与模型构建。
- 兼容的附加流体 XML 由 `merge_fluid_xml(path, definition)` 一次性合并到材料表和 `fluid_boundaries`，之后与代码直接构造的定义走相同的 `build_model()` 路径。

## 表达式函数

`Function` 是以下五种输入的 `std::variant`：

- `ExpressionFunction { expression }`
- `DoubleExponentialFunction { a, alpha, beta }`
- `GaussFunction { a, tau, x0 }`
- `SineFunction { a, omega, phi }`
- `PieceWiseFunction { points }`

绘图范围属于 UI 元数据，不参与表达式编译，因此不在 `ModelDefinition` 中。

## Face key

格式为 `Axis|Category|Coordinate|RectList`。坐标和矩形范围使用 `length_unit`，在 `build_model()` 中统一转换成 SI：

- Z 面矩形：`xmin,xmax,ymin,ymax;...`
- X/Y 面矩形：`min1|max1|min2|max2`

例如：`Z|E|0|0,50,50,100;50,100,0,50`。
