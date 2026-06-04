# PRD: 单变元函数类别（Functions）解析

## 目标

支持 `cases/simple_steady_tests/case3.xml` 定义的 5 类**单变元**函数：

| 类别                        | XML 元素                        | 字段                                  | 数学形式                                   |
| --------------------------- | ------------------------------- | ------------------------------------- | ------------------------------------------ |
| `ExpressionFunction`        | `<b:ExpressionFunction>`        | `Expression`, `DrawMinX`, `DrawMaxX`  | 用户提供的任意 exprtk 字符串（自变量 `x`） |
| `DoubleExponentialFunction` | `<b:DoubleExponentialFunction>` | `A`, `Alpha`, `Beta`, `DrawMinX/MaxX` | `A·exp(α·x) + A·exp(β·x)` 或类似           |
| `GaussFunction`             | `<b:GaussFunction>`             | `A`, `Tau`, `X0`, `DrawMinX/MaxX`     | `A·exp(-((x - X0)/Tau)²)`                  |
| `SineFunction`              | `<b:SineFunction>`              | `A`, `Omega`, `Phi`, `DrawMinX/MaxX`  | `A·sin(Omega·x + Phi)`                     |
| `PieceWiseFunction`         | `<b:PieceWiseFunction>`         | `Points: {X, Y}[]`, `DrawMinX/MaxX`   | 折线分段插值（点集按 X 升序）              |

> 端到端目标：用户可以写 `<TiReyuan>test_gaussian(x)</TiReyuan>`（在体热源中以 `t` 代入），
> `<DaoreXishu>test_sine(T)</DaoreXishu>`（在材料参数中以 `T` 代入），
> `<ConvectionCoefficient>test_double_exponential(T)</ConvectionCoefficient>`（在 BC 中以 `T` 代入）。
> `cases/simple_steady_tests/case3.xml` 与 `cases/simple_transient_tests/case1.xml`
> 的现有 case 应能正确解析并产生与 reference 一致的结果。

## 自变量映射规则

**前端表达**：用户写 `function_name(x)`，字面量 `x`。

**后端代入**：在 **io 层或 preprocessor 层** 做一次**变量重写**：

| 引用位置                                                                              | 自变量 `x` 重写为 |
| ------------------------------------------------------------------------------------- | ----------------- |
| 体热源表达式（`ti_reyuan_expr`）                                                      | `t`               |
| 材料 k / ρ / c（`daore_xishu` / `midu` / `bi_rerong`）                                | `T`               |
| BC：Dirichlet `temperature`、Neumann `heat_flux`、Cauchy `convection_coeff` / `T_inf` | `T`               |

实现策略：在 preprocessor 拿到所有 Functions 后，对每个用到的 `function_name` 调用
重写生成一条 exprtk 表达式（`function_name(t)` 或 `function_name(T)`），与原表达式合并：

- 选项 A：在 preprocessor 阶段用 `expr::register_function(name, "function_name(argname)")` 透明重写
    - 注册名保留 `function_name`，但表达式变成对底层 native 函数的薄包装
- 选项 B：在 preprocessor 阶段把 `function_name(x)` 在 `ti_reyuan_expr` 中字面替换为 `function_name(t)`，再编译
    - 字符串替换，简单但脆弱（用户 `function_name` 出现在注释中会误伤）
- 选项 C：直接注册 `function_name` 为 native，**native 实现里根据调用现场的 FieldContext 取合适变量**
    - 即"统一用 `T` 表达"——这与材料/BC 一致；但体热源需要 `t`
    - 不推荐：体热源本质是时间函数，强行用 `T` 语义错

**推荐选项 B + 配套改进**：在 preprocessor 阶段，对每条 `ti_reyuan_expr` / `daore_xishu` / `midu` /
`bi_rerong` / BC 字符串做一次 `function_name(x)` → `function_name(t or T)` 的字面替换
（用 token 边界正则或简单 sregex 替换），然后再 `expr::parse`。

> 替换规则：仅替换形如 `name(x)` 的**完整 token**（前后是非标识符字符），不替换文本中其他位置。

## 数据流

```text
XML <Functions><a:KeyValueOfstringFunctionAdzryM2O>
        <a:Key>name</a:Key>
        <a:Value i:type="b:GaussFunction">
            <b:A>5</b:A><b:Tau>10</b:Tau><b:X0>20</b:X0>
            <b:DrawMinX>0</b:DrawMinX><b:DrawMaxX>100</b:DrawMaxX>
        </a:Value>
        ...
    </a:KeyValueOfstringFunctionAdzryM2O>
  </Functions>

  → io::read_xml
    解析为 IOStructure::functions
    IOStructure::functions: std::unordered_map<std::string, Function>
    （替换现有的 FieldEvaluator map，参考 docs/design/io-model.md 的 Function 类型体系）

  → Preprocessor::load
    对每个 Function：
      1. expr::register_native(name, FieldEvaluator)
         - ExpressionFunction: FieldEvaluator 内部 eval 时把 ctx.t 代入 x，然后解析 underlying expression
         - DoubleExponential: f(t) = A*(exp(α·t) - exp(β·t))  // 见决策点
         - Gauss: f(t) = A·exp(-((t - X0)/Tau)²)
         - Sine: f(t) = A·sin(Omega·t + Phi)
         - PieceWise: f(t) = 折线插值（X 升序；首尾延伸用首/末值）
    对每个引用位置（ti_reyuan_expr / daore_xishu / midu / bi_rerong / BC 字符串）：
      2. 字面替换 name(x) → name(t) 或 name(T)
         - 体热源 → t
         - 材料 / BC → T
      3. expr::parse(替换后字符串) 编译为 CompiledExpression
```

## 新增数据结构

- `src/common/io_model.hpp` — 把 `IOStructure.functions: unordered_map<string, FieldEvaluator>`
  改为 `unordered_map<string, Function>`（参考 `docs/design/io-model.md` 第 110-156 行的 Function 类型体系）
    - 引入 `FunctionType` 枚举 + 五个 POD struct
    - 引入 `Function` 联合体（用 union 内存紧凑，或简单 struct with all 字段）
- `src/common/internal_model.hpp` — 同样 `InternalModel.functions: unordered_map<string, Function>`
- `src/expr/expr.hpp/cpp`
    - `register_function(name, native_fn)` 已存在；`register_native` 已存在
    - 新增 `register_function_definition(name, Function fn)` 把 5 类函数之一包装为 native
    - 或：在 preprocessor 直接用现有 `register_native` 注册（**推荐**——expr 层无依赖）

## 行为细节

### 1. XML 解析（io.cpp）

- 在 `Materials` 块后增加 `<Functions>` 块解析
- 子节点 `a:KeyValueOfstringFunctionAdzryM2O`：
    - `<a:Key>` → 函数名
    - `<a:Value i:type="b:ExpressionFunction">` 等 → 选对应子 struct
    - 缺失字段 → 用默认 0.0（与现有 `BiRerong i:nil="true"` 行为一致）
- `<b:ExpressionFunction>` 的 `Expression` 是 exprtk 字符串，自变量名为 `x`
- `<b:PieceWiseFunction>` 的 `Points` 是 `b:PieceWiseFunction.Point` 子元素列表：
    - `<b:X>` / `<b:Y>`（不是大写 `XArray/YArray` 那种）

### 2. native 函数注册（preprocessor）

每个 Function 转 `FieldEvaluator = [fn](const FieldContext& ctx) -> double`：

```cpp
// ExpressionFunction
FieldEvaluator wrap(const std::string& inner_expr) {
    auto ce = mhs::expr::parse(inner_expr);   // 自变量名仍然是 x
    return [ce](const FieldContext& ctx) {
        FieldContext inner{ ctx.x, ctx.y, ctx.z, ctx.t, ctx.t };  // 把 t 当 x
        return ce.eval(inner);
    };
}
// DoubleExponential
FieldEvaluator wrap(double A, double alpha, double beta) {
    return [=](const FieldContext& c) { return A * (std::exp(alpha*c.t) - std::exp(beta*c.t)); };
}
// Gauss
FieldEvaluator wrap(double A, double tau, double x0) {
    return [=](const FieldContext& c) {
        double u = (c.t - x0) / tau;
        return A * std::exp(-u*u);
    };
}
// Sine
FieldEvaluator wrap(double A, double omega, double phi) {
    return [=](const FieldContext& c) { return A * std::sin(omega*c.t + phi); };
}
// PieceWise
FieldEvaluator wrap(std::vector<Point> pts) {
    return [pts=std::move(pts)](const FieldContext& c) {
        double x = c.t;
        // 二分 / 顺序扫描找到段
        if (x <= pts.front().x) return pts.front().y;
        if (x >= pts.back().x)  return pts.back().y;
        auto it = std::upper_bound(pts.begin(), pts.end(), x,
                                   [](double v, const Point& p){ return v < p.x; });
        const auto& p1 = *(it-1); const auto& p2 = *it;
        double t = (x - p1.x) / (p2.x - p1.x);
        return p1.y + t * (p2.y - p1.y);
    };
}
```

> **关键点**：所有 native 函数的语义都"读 ctx.t"——这是因为我们把引用处
> 的 `name(x)` 字面替换为 `name(t)` / `name(T)`，但 native 函数本身对自变量名不敏感，
> 它读的是 FieldContext 的 `t` 槽。
>
> 也就是说：在 preprocessor 阶段字面替换后，函数调用写 `test_gaussian(t)` 或
> `test_gaussian(T)`，但 native 实现里都从 `ctx.t` 取值（无所谓哪个名字，exprtk 解析时
> 会绑定到符号 `t`）。

### 3. 字面替换规则（preprocessor）

> **核心动作**：在每条 `ti_reyuan_expr` / `daore_xishu` / `midu` / `bi_rerong` / BC 字符串中，
> 把字符 `x`（视作"待替换的自变量"）替换为 `t` 或 `T`。
> **函数名 `name(...)` 本身不动**（`test_gaussian` 仍是 `test_gaussian`）——它通过
> `expr::register_native` 注册到 exprtk 引擎。

```cpp
// 对每个引用字符串 s：
//   扫描每个位置 i：
//     if s[i] == 'x' && (i==0 || (!isalpha(s[i-1]) && s[i-1] != '_'))
//                     && (i+1==n || (!isalpha(s[i+1]) && s[i+1] != '_'))
//        → 标记为"孤立 x"
//   倒序把每个孤立 x 替换为 argname
//   其中 argname = "t" (体热源) 或 "T" (材料 / BC)
```

判定规则：**字符 `x` 的前面和后面都不是字母或下划线**（即不是 `[A-Za-z_]`）即为
"孤立自变量"。

- 字符串首（`i==0`）和字符串尾（`i+1==n`）视作非字母非下划线 → 边界 `x` 算孤立
- `_` 是标识符字符 → 视作"非孤立"，`x_next` / `x_max` 中的 `x` **不替换**
- 数字、运算符、括号、空格等非字母非下划线 → 不影响孤立判定

用户视角示例：

- `test_gaussian(x)` → `test_gaussian(t)`（在体热源中）✓
- `test_gaussian(x)/(x*0.01+1)` → `test_gaussian(t)/(t*0.01+1)` ✓
- `2*x` → `2*t`（前 `2` 数字）✓
- `x*0.01+1` → `t*0.01+1`（前 字符串首）✓
- `x_next` 中的 `x` **不替换**（后 `_`）
- `xx`、`axb` 中的 `x` 都不替换（前后都是字母）

实现方式：单次扫描字符串，记录每个孤立 `x` 位置，倒序替换为 argname（避免位置偏移）。
**不使用 std::regex**（避免 ECMAScript 引擎对 lookbehind / `\b` 的支持问题）。

### 4. 错误模式

- XML 中 `Functions` 块缺失 → 现有 iostructure.functions 空 map，行为不变
- 引用处出现未在 functions 注册的 `name(x)` → `expr::parse` 在 exprtk 编译时失败，
  按现有 `parse` 行为回退常量 0.0——这不对！
    - **修复**：在 parse 之前对所有 `function_name(x)` 引用做"已注册校验"，未注册 → `panic`
- 数值字段解析失败 → 走 `parse_double`（已存在），空串→0
- 同一 name 出现两次 → panic / 取先出现的

### 5. 性能 / 内存

- 5 个 native 闭包注册到 `expr::register_native`，主线程持锁。一次性代价。
- 字面替换在 preprocessor 阶段，每个引用字符串一次 regex 替换。
- `ExpressionFunction` 的 `parse` 是 inner exprtk 字符串解析，只在 preprocessor 阶段跑一次，
  闭包内捕获的 `CompiledExpression` 走 TLS，**不**重新解析。

## 验收

- `cases/simple_transient_tests/case1.xml`：现有 `test_gaussian` 函数被解析并注册，
  `<TiReyuan>test_gaussian(x)/(x*0.01+1)</TiReyuan>` 编译通过，结果与 reference 一致
- `cases/simple_steady_tests/case3.xml`：5 类函数全部被解析并注册；
  引用处（如 `<DaoreXishu>test_sine(T)</DaoreXishu>`）按"T"代入
- 单元测试：
    - 5 类函数的 FieldEvaluator 数学正确性（已知参数 → 已知值）
    - 字面替换正确性（`sin(x)` 不被 `sine(x)` 误匹配；`name(T)` 在 T 槽取 ctx.T）
    - 非法引用 panic
- 现有 5 个 steady case 回归无损（无函数引用时不变）

## 不在范围内

- 多变元函数（2D / 3D 输入）
- 表达式函数（`<b:ExpressionFunction>`）的 underlying 字符串含额外 `function_name(x)` 引用
    - 简化为：嵌套调用在第一层替换后即视作字符串原样编译；如有更深层引用，由 exprtk 报错
    - 不在主体 PRD 范围
- 改变 `FieldContext` 字段（仍是 `{x, y, z, T, t}`）
- 改 `expr::parse` 接口
- 改动非线性 / 线性求解器
- 在 docs/xsd/ 改 schema 文档（如有）

## 涉及文件

- `src/common/io_model.hpp` — Function 类型体系（实现文档中已写好的）
- `src/common/internal_model.hpp` — `InternalModel.functions`
- `src/io/io.cpp` — 解析 `<Functions>` 块
- `src/preprocessor/preprocessor.cpp` — 注册 native + 字面替换 + 编译
- `tests/...` — 5 类函数 evaluator 单测、字面替换单测、集成测试
- `cases/...` — 现有 case3.xml 应能直接跑通；可能新增 case 验证 T 代入路径
- `docs/adr/0008-function-categories.md`（后续 issue）

## 关联

- PRD-各向异性 k：不冲突
- PRD-瞬态探针：case1.xml 的瞬态 case 会顺带覆盖函数解析
