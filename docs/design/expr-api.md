# expr 模块接口

表达式解析与求值模块。处理所有场/边界条件表达式，上下文为 `{x, y, z, T, t}`。

---

## FieldContext

```cpp
namespace mhs::expr {

// 表达式求值上下文
struct FieldContext {
    double x = 0.0, y = 0.0, z = 0.0;  // 空间位置（SI 单位）
    double T = 0.0;                     // 该位置的温度
    double t = 0.0;                     // 当前仿真时间
};

} // namespace mhs::expr
```

---

## CompiledExpression

所有表达式（材料属性、BC 参数、热源）预编译后均为 `CompiledExpression` 对象。

```cpp
namespace mhs::expr {

class CompiledExpression {
public:
    double eval(const FieldContext& ctx) const;
    bool is_constant() const;
    double constant_value() const;

    // 创建常数表达式
    static CompiledExpression make_constant(double value);

private:
    // implementation detail
};

} // namespace mhs::expr
```

---

## 表达式注册表（模块内部，对外无感）

线程安全的全局注册表，在预处理阶段由 `ModelBuilder` 填充：

```cpp
namespace mhs::expr {

// 几何变量（注册后 eval_geometry 可直接使用）
void set_variable(const std::string& name, double value_in_SI);

// 注册 native C++ 函数到模块级函数池
// 示例：
//   expr::register_native("my_piecewise", [](const FieldContext& ctx) {
//       if (ctx.x < 1.0) return 0.0;
//       if (ctx.x < 2.0) return 1.0;
//       return 2.0;
//   });
//
// 注册后，可在字符串表达式中调用：parse("my_piecewise(x, y, z, T, t)")
void register_native(const std::string& name, FieldEvaluator func);

// 注册用户定义的 exprtk 表达式函数
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

## 表达式解析与求值

```cpp
namespace mhs::expr {

// 解析字符串表达式为 CompiledExpression
// 注册表须已包含所有引用的变量和函数
CompiledExpression parse(const std::string& formula);

// 求值几何表达式（不需要上下文）
// 所有几何变量须已通过 set_variable() 注册
double eval_geometry(const std::string& formula);

} // namespace mhs::expr
```

### 使用示例

```cpp
// 预处理阶段（ModelBuilder 中）
expr::set_variable("w_top", 10.0);      // mm -> SI 已在 ModelBuilder 转换
expr::set_variable("h_middle", 2.0);
expr::register_native("my_piecewise", [](const FieldContext&) { ... });
expr::register_function("test_gaussian", "exp(-((x-x0)^2+(y-y0)^2)/sigma)");

// 求值几何表达式
double half_width = expr::eval_geometry("w_top/2");

// 编译场表达式
CompiledExpression k = expr::parse("k_copper + 0.01*T");
double k_val = k.eval({x: 0.01, y: 0.02, z: 0.0, T: 350.0, t: 1.0});
```

---

## 求值上下文生命周期

`CompiledExpression::eval(ctx)` 在以下两个场景被调用：

1. **预处理阶段**（一次性）：编译时做常量折叠
2. **组装阶段**（高频）：每个 Newton 迭代、每个单元调用一次

高频调用时，优先使用常数表达式（`parse("1.5")` 返回常数）避免求值开销。

---

## 线程安全

- `set_variable()`, `register_native()`, `register_function()`: 互斥锁保护
- `parse()`: 编译时读取注册表，互斥锁保护
- `CompiledExpression::eval()`: 无锁（函数指针在解析时已捕获）
- `eval_geometry()`: 无锁（变量在解析时已内联）
