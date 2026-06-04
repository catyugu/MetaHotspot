# IO 模型结构

直接映射 XML schema，仅用于序列化/反序列化。`src/common/io_model.hpp`。**单位**：`LengthUnit` 在预处理阶段转 SI 米。

```cpp
namespace mhs {

struct Variable { std::string name; std::string value; };

struct Rect {
    bool add_sub;
    std::string width_expr;    // 几何表达式（字符串）
    std::string height_expr;
    std::string x_expr;
    std::string y_expr;
    std::string x_size_expr;
    std::string y_size_expr;
    std::string x_interval_expr;
    std::string y_interval_expr;
    std::string name;
};

struct Block {
    std::vector<Rect> all_rects;
    std::string material_name;
    std::string x_offset_expr;
    std::string y_offset_expr;
    std::string ti_reyuan_expr;  // 体热源表达式 [W/m³]，如 "1e9" 或 "1e8+0.5*x"
    std::string name;
    bool is_normal_material = true;
};

struct Layer {
    std::vector<Block> blocks;
    std::string name;
    std::string thickness_expr;
    std::string x_offset_expr;
    std::string y_offset_expr;
    std::string period_width_expr;
    int period_width = 10;
    bool is_top_layer = false;
};

// 边界
enum class BoundaryCategory { Electrical };
enum class ThermalBCType    { FirstType, SecondType, ThirdType };
struct FirstTypeThermalBC  { std::string temperature = "300.0"; };   // Dirichlet
struct SecondTypeThermalBC { std::string heat_flux = "0.0"; };       // Neumann
struct ThirdTypeThermalBC  { std::string convection_coeff = "0.0";
                             std::string T_inf = "300.0"; };         // Cauchy
struct Boundary { BoundaryCategory category; std::string name;
                  std::vector<std::string> face_keys;
                  ThermalBCType bc_type;
                  FirstTypeThermalBC first; SecondTypeThermalBC second; ThirdTypeThermalBC third; };

// 材料
struct Material { std::string name;
                  std::string daore_xishu = "0.0";   // k
                  std::string midu        = "0.0";   // rho
                  std::string bi_rerong   = "0.0"; }; // c

// 元数据
enum class StudyType  { Steady, Transient };
enum class LengthUnit { M, Mm, Um, Nm, Inch, Mil };
enum class Dimension  { Dimension2D, Dimension3D };  // Dimension2D 在预处理 panic

struct IOStructure {
    StudyType  study_type;
    Dimension  dimension;
    LengthUnit length_unit;
    double initial_temperature  = 300.0;
    double ambient_temperature  = 300.0;

    std::vector<Variable> variables;
    std::vector<Layer>    layers;
    std::unordered_map<std::string, Material> materials;
    std::vector<Boundary> boundaries;

    double transient_duration     = 0.0;
    double transient_time_step    = 1.0;
    std::string transient_time_unit = "s";

    ThermalBCType     other_bc_type  = ThermalBCType::SecondType;
    FirstTypeThermalBC  other_bc_first;
    SecondTypeThermalBC other_bc_second;
    ThirdTypeThermalBC  other_bc_third;

    std::vector<double> mesh_vertex_x, mesh_vertex_y, mesh_vertex_z;

    // 已注册为 native function 的 C++ 求值器。
    // ⚠️ TODO(Function types): 当前为扁平 map；
    //    计划从 XML 解析结构化函数定义（Gauss、PieceWise、…）→ 编译为 CompiledExpression / FieldEvaluator。
    //    计划类型（未实现）：enum class FunctionType { Expression, DoubleExponential, Gauss, Sine, PieceWise }
    //    + ExpressionFunction / DoubleExponentialFunction / GaussFunction / SineFunction / PieceWiseFunction / Function。
    std::unordered_map<std::string, FieldEvaluator> functions;
};

} // namespace mhs
```

---

## 2.2 表达式函数类型（TODO：尚未定义）

> **注意**：以下类型体系尚未定义——连头文件声明也不存在。当前 `IOStructure.functions` 为 `unordered_map<string, FieldEvaluator>` 的扁平 map。
> 未来计划定义完整的 Function 类型体系，支持从 XML 解析结构化函数定义（Gauss、PieceWise 等），
> 并在预处理阶段将它们转换为 `CompiledExpression` 或 `FieldEvaluator`。

```cpp
namespace mhs {

enum class FunctionType { Expression, DoubleExponential, Gauss, Sine, PieceWise };

struct ExpressionFunction {
    std::string expression;  // 如 "20*(x+1)-exp(x)"
    double draw_min_x = 0.0;
    double draw_max_x = 100.0;
};

struct DoubleExponentialFunction {
    double a = 0.0, alpha = 0.0, beta = 0.0;
    double draw_min_x = 0.0;
    double draw_max_x = 100.0;
};

struct GaussFunction {
    double a = 0.0, tau = 0.0, x0 = 0.0;
    double draw_min_x = 0.0;
    double draw_max_x = 100.0;
};

struct SineFunction {
    double a = 0.0, omega = 0.0, phi = 0.0;
    double draw_min_x = 0.0;
    double draw_max_x = 100.0;
};

struct PieceWiseFunction {
    struct Point { double x = 0.0, y = 0.0; };
    std::vector<Point> points;
    double draw_min_x = 0.0;
    double draw_max_x = 100.0;
};

struct Function {
    std::string key;
    FunctionType type;
    ExpressionFunction expression;
    DoubleExponentialFunction double_exp;
    GaussFunction gauss;
    SineFunction sine;
    PieceWiseFunction piecewise;
};

} // namespace mhs
```

---

## 字段说明

### BC 类型

| 类型       | 数学                          | XML 元素                       |
| ---------- | ----------------------------- | ------------------------------ |
| FirstType  | `T = T₀`                      | `FirstTypeThermalBoundary`     |
| SecondType | `-k ∂T/∂n = q₀`               | `SecondTypeThermalBoundary`    |
| ThirdType  | `-k ∂T/∂n = h(T - T_∞)`       | `ThirdTypeThermalBoundary`     |

### Face key 格式

```text
Face|Direction|CoordValue|RectList
```

- `Face`: `Z` / `Y` / `X` — 法向轴
- `Direction`: `E` — 类别（当前仅 Electrical，未使用）
- `CoordValue`: 边界面的**空间坐标**（SI 前乘 `si_scale`）。**不是层索引**。
- `RectList`:
    - Z-face: `xmin,xmax,ymin,ymax;xmin2,xmax2,ymin2,ymax2;…`
    - X/Y-face: `Min1|Max1|Min2|Max2`（pipe-delimited single rect）

示例: `Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100`

### `ti_reyuan_expr`（体热源）

每个 Block 一个体热源密度表达式 `[W/m³]`，由 `preprocessor` 去重后编入 `heat_source_table`。可以是常数（`"1e9"`）、空间函数（`"1e8 + 0.5*x"`），或任意 exprtk 表达式。

### Block 不含 Z 字段

Block 仅在 XY 平面通过 add/sub Rect 定义形状，Z 范围完全继承父 Layer 的 `z_start/z_end`。
