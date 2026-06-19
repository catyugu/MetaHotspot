# IO 模型结构

直接映射 XML schema，仅用于序列化/反序列化。`src/data/io_model.hpp`。**单位**：`LengthUnit` 在预处理阶段转 SI 米。

```cpp
namespace mhs::core {

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
    std::string thickness_expr;  // Block 自己的厚度表达式（layer 0 可变厚度）
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
                  std::string kx = "0.0";   // 热导率 X 方向 [W/(m·K)]
                  std::string ky = "0.0";   // 热导率 Y 方向 [W/(m·K)]
                  std::string kz = "0.0";   // 热导率 Z 方向 [W/(m·K)]
                  std::string midu        = "0.0";   // rho
                  std::string bi_rerong   = "0.0"; }; // c

// 元数据
enum class StudyType  { Steady, Transient };
enum class LengthUnit { M, Mm, Um, Nm, Inch, Mil };
enum class Dimension  { Dimension2D, Dimension3D };  // Dimension2D 当前未实现，预处理不读取

// 3D 探针（观察点）：用户坐标系下的固定位置，坐标以 muparser 表达式形式给出
// （如 "chip_w/2 + 0.1"），由 preprocessor 在加载时一次性求值到 InternalModel。
struct ObservationPoint3D {
    std::string name;
    std::string x;
    std::string y;
    std::string z;
};

// 探针温度时间序列：与 ObservationPoint3D::name 一一对应。
struct ProbeTrace {
    std::string name;
    std::vector<double> times;
    std::vector<double> values;
};

struct IOStructure {
    StudyType  study_type;
    Dimension  dimension;
    LengthUnit length_unit;
    double initial_temperature  = 300.0;

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
    std::unordered_map<std::string, Function> functions;

    // 3D 观察点（探针）列表，默认空：稳态 case 不会有此项。
    std::vector<ObservationPoint3D> observation_points;
};

} // namespace mhs::core
```

---

## 2.2 表达式函数类型

`IOStructure.functions` 是 `unordered_map<string, Function>`，按 `Function.type` 分发到 `mhs::sim::function_helpers` 的 5 个闭包构造器之一。预处理阶段用 `mhs::sim::register_all_functions` 把整张表注册为 expr 全局 native。

```cpp
namespace mhs::core {

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
    FunctionType type = FunctionType::Expression;
    ExpressionFunction expression;
    DoubleExponentialFunction double_exp;
    GaussFunction gauss;
    SineFunction sine;
    PieceWiseFunction piecewise;
};

} // namespace mhs::core
```

---

## 字段说明

### BC 类型

| 类型       | 数学                    | XML 元素                    |
|------------|-------------------------|-----------------------------|
| FirstType  | `T = T₀`                | `FirstTypeThermalBoundary`  |
| SecondType | `-k ∂T/∂n = q₀`         | `SecondTypeThermalBoundary` |
| ThirdType  | `-k ∂T/∂n = h(T - T_∞)` | `ThirdTypeThermalBoundary`  |

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

每个 Block 一个体热源密度表达式 `[W/m³]`，由 `preprocessor` 去重后编入 `heat_source_table`。可以是常数（`"1e9"`）、空间函数（`"1e8 + 0.5*x"`），或任意 muparser 表达式。

### `kx / ky / kz`（各向异性热导率）

材料热导率按笛卡尔三轴拆分（**W/(m·K)**），与装配时面法向匹配（X 面用 `kx`，Y 面用 `ky`，Z 面用 `kz`）。`io` 模块在解析 `DaoreXishu` 节点时按以下规则拆分：

- **单表达式**（如 `DaoreXishu>100</DaoreXishu>`）→ `kx = ky = kz = "100"`，退化为各向同性。
- **三表达式**（`DaoreXishu>kx_expr, ky_expr, kz_expr</DaoreXishu>`）→ 按逗号分隔（容忍空白）分别赋给 `kx / ky / kz`。
- 其它段数（2 段、4 段及以上）经 `MHS_FATAL` panic。

### Block 不含 XY 以外的空间字段

Block 仅在 XY 平面通过 add/sub Rect 定义形状，Z 范围默认继承父 Layer 的 `z_start/z_end`（layer 0 支持 Block 级 `thickness_expr` 实现可变 Z 厚度）。
