# IO 模型结构

IO 结构直接映射 XML schema，仅用于序列化/反序列化。

---

## 2.1 顶层结构

```cpp
namespace mhs::model::io {

struct Variable { std::string name; double value; };

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
    std::string thickness_expr;
    std::string mesh_size_x_expr;
    std::string mesh_size_y_expr;
    std::string mesh_size_z_expr;
    std::string x_offset_expr;
    std::string y_offset_expr;
    std::string z_offset_expr;
    std::string ti_reyuan_expr;  // 体热源表达式 [W/m³]，如 "1e9" 或 "1e8+0.5*x"
    std::string name;
    bool is_normal_material = true;
};

struct Layer {
    std::vector<Block> blocks;
    std::string name;
    std::string thickness_expr;
    std::string mesh_size_x_expr;
    std::string mesh_size_y_expr;
    std::string mesh_size_z_expr;
    std::string x_offset_expr;
    std::string y_offset_expr;
    std::string period_width_expr;
    int period_width = 10;
    bool is_top_layer = false;
};

enum class BoundaryCategory { Electrical };
enum class ThermalBCType { FirstType, SecondType, ThirdType };

struct FirstTypeThermalBC  { double temperature = 300.0; };    // Dirichlet：固定温度
struct SecondTypeThermalBC { double heat_flux = 0.0; };         // Neumann：固定热通量
struct ThirdTypeThermalBC  { double convection_coeff = 0.0; double environment_temp = 300.0; };  // Cauchy：换热

struct Boundary {
    BoundaryCategory category;
    std::string name;
    std::vector<std::string> face_keys;  // 原始面键字符串
    ThermalBCType bc_type;
    FirstTypeThermalBC  first;
    SecondTypeThermalBC second;
    ThirdTypeThermalBC  third;
};

struct Material {
    std::string name;
    double daore_xishu = 0.0;       // 导热系数 k
    double midu = 0.0;              // 密度 rho（可选）
    double bi_rerong = 0.0;         // 比热容 c（可选）
};

enum class StudyType { Steady, Transient };
enum class LengthUnit { Mm, Cm, M };
enum class Dimension { Dimension2D, Dimension3D };  // Dimension2D 触发 panic

struct Structure {
    // 元数据
    std::string software_mode;
    StudyType study_type;
    Dimension dimension;
    LengthUnit length_unit;
    double initial_temperature = 300.0;
    double ambient_temperature = 300.0;
    int die_layer_num = 0;

    // 几何变量
    std::vector<Variable> variables;

    // 层和材料
    std::vector<Layer> layers;
    std::unordered_map<std::string, Material> materials;

    // 边界条件
    std::vector<Boundary> boundaries;

    // 瞬态设置（仅当 study_type == Transient 时使用）
    double transient_duration = 0.0;
    double transient_time_step = 1.0;
    std::string transient_time_unit = "s";

    // 未指定面的默认 BC（OtherThermalBoundary）。
    // 通常为 SecondType（Neumann，HeatFlux=0 = 绝热）或 ThirdType。
    // 预处理阶段将此类默认 BC 应用于所有未明确指定的面。
    ThermalBCType other_bc_type = ThermalBCType::SecondType;
    SecondTypeThermalBC other_bc_second;
    ThirdTypeThermalBC other_bc_third;

    // 结果（用于从 XML 读取参考值进行回归测试）
    std::vector<double> result_values;  // 温度值扁平数组
    std::vector<double> result_x;
    std::vector<double> result_y;
    std::vector<double> result_z;
};

} // namespace mhs::model::io
```

---

## 2.2 表达式函数类型

```cpp
namespace mhs::model::io {

enum class FunctionType { Expression, DoubleExponential, Gauss, Sine, PieceWise };

struct ExpressionFunction {
    std::string expression;  // 如 "20*(x+1)-exp(x)"
    double draw_min_x = 0.0;
    double draw_max_x = 100.0;
};

struct DoubleExponentialFunction {
    double a = 0.0, alpha = 0.0, beta = 0.0;
    double draw_min_x = 0.0, double draw_max_x = 100.0;
};

struct GaussFunction {
    double a = 0.0, tau = 0.0, x0 = 0.0;
    double draw_min_x = 0.0, double draw_max_x = 100.0;
};

struct SineFunction {
    double a = 0.0, omega = 0.0, phi = 0.0;
    double draw_min_x = 0.0, double draw_max_x = 100.0;
};

struct PieceWiseFunction {
    struct Point { double x = 0.0, y = 0.0; };
    std::vector<Point> points;
    double draw_min_x = 0.0, double draw_max_x = 100.0;
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

} // namespace mhs::model::io
```

---

## 字段说明

### 热边界条件类型

| 类型       | 中文名                 | 数学描述                            | XML 类型                    |
| ---------- | ---------------------- | ----------------------------------- | --------------------------- |
| FirstType  | 第一类（Dirichlet）    | `T = T₀`（固定温度）                | `FirstTypeThermalBoundary`  |
| SecondType | 第二类（Neumann）      | `-k ∂T/∂n = q₀`（固定热通量）       | `SecondTypeThermalBoundary` |
| ThirdType  | 第三类（Cauchy/Robin） | `-k ∂T/∂n = h(T - T_∞)`（对流换热） | `ThirdTypeThermalBoundary`  |

### 面键格式

边界面规格格式：`Face|Direction|LayerIndex|X_min,Y_min,X_max,Y_max;...`

示例：`Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100`

- `Face`：Z/Y/X 表示法向方向
- `Direction`：E 表示电气边界（暂未用）
- `LayerIndex`：层索引
- 后续坐标描述一个或多个矩形区域

### ti_reyuan_expr（热源）

每个 Block 有一个体热源密度表达式，单位 W/m³。可以是：

- 常数：如 `"1e9"`
- 空间表达式：如 `"1e8 + 0.5*x"`（上下文为 `{x, y, z, T, t}`）
- 任意 exprtk 支持的表达式
