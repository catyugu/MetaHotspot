# expr 模块接口

表达式解析与求值模块，封装 `exprtk`。处理所有场/边界条件表达式，上下文为 `{x, y, z, T, t}`。

---

## FieldContext

```cpp
namespace mhs::expr {

// 表达式求值上下文
struct FieldContext {
    double x = 0.0, y = 0.0, z = 0.0;  // 空间位置
    double T = 0.0;                     // 该位置的温度
    double t = 0.0;                     // 当前仿真时间
};

} // namespace mhs::expr
```

---

## FieldExpression

所有表达式（材料属性、BC 参数、热源）预编译后均为 `FieldExpression` 对象。

```cpp
namespace mhs::expr {

// 预编译的场表达式（使用 exprtk）
// 内部模型中不存储原始表达式字符串，只存储可求值的 FieldExpression。
class FieldExpression {
public:
    // 构造常数表达式（无需求值，直接返回常数）
    static FieldExpression make_constant(double value);

    // 从字符串构造，如 "1.5 + 0.002*T" 或 "sin(x) + exp(-y)"
    // 注册 x, y, z, T, t 符号；在函数池中查找函数名
    static FieldExpression from_string(const std::string& expr);

    // 在给定上下文处求值
    double eval(const FieldContext& ctx) const;

    bool is_constant() const { return is_constant_; }
    double constant_value() const { return constant_value_; }

private:
    void* exprtk_expr_ = nullptr;  // exprtk 不透明句柄
    bool is_constant_ = false;
    double constant_value_ = 0.0;
};

} // namespace mhs::expr
```

---

## Native Function

某些函数形式难以用字符串表达（如分片常数、分片线性、查表数据），支持直接注册 C++ 函数。

```cpp
namespace mhs::expr {

// Native function 类型 — 接收完整 FieldContext，返回 double
using NativeFunc = std::function<double(const FieldContext&)>;

// 注册一个 native C++ 函数到模块级函数池。
// 示例：
//   register_native("my_piecewise", [](const FieldContext& ctx) {
//       if (ctx.x < 1.0) return 0.0;
//       if (ctx.x < 2.0) return 1.0;
//       return 2.0;
//   });
//
// 注册后，可在字符串表达式中调用：FieldExpression::from_string("my_piecewise(x, y, z, T, t)")
// exprtk 会将 my_piecewise 绑定到此函数。
void register_native(const std::string& name, NativeFunc func);

// 注册用户定义的 exprtk 表达式函数。
//   用户定义函数的表达式中可以出现：
//  - x, y, z, t, T — 预定义上下文变量
//  - 其他函数（如 test_gaussian）
//  - 其他定义过的常数
void register_function(const std::string& name, const std::string& expression);

} // namespace mhs::expr
```

### Native Function 使用场景

| 场景         | 示例             | 表达难度      |
| ------------ | ---------------- | ------------- |
| 分片常数     | 空间分区常数热源 | 字符串难表达  |
| 分片线性     | 非均匀材料层     | 字符串难表达  |
| 查表函数     | 实验数据表驱动   | native 更高效 |
| 复杂几何判断 | 距某点距离函数   | native 更直观 |

---

## 求值上下文生命周期

`FieldExpression::eval(ctx)` 在以下两个场景被调用：

1. **预处理阶段**（一次性）：编译时做常量折叠（`make_constant` 的基础）
2. **组装阶段**（高频）：每个 Newton 迭代、每个单元调用一次

对于热循环中的高频调用，优先使用 `make_constant()` 创建常数表达式，避免 exprtk 求值开销。

---

## exprtk 配置要点

- 启用单精度浮点支持（`ETK_FAST_FLOATING_POINT`）
- 禁用科学计数法解析（避免歧义）
- 启用 `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `floor`, `ceil` 等标准函数
- 自定义函数通过 `register_native` 注入到 exprtk 的符号表
