# expr 模块接口

表达式解析与求值模块。处理所有场/边界条件表达式，上下文为 `{x, y, z, T, t}`。

---

## FieldContext

```cpp
namespace mhs {

// 表达式求值上下文（定义在 types.hpp，非 mhs::expr）
struct FieldContext {
    double x = 0.0, y = 0.0, z = 0.0;  // 空间位置（SI 单位）
    double T = 0.0;                     // 该位置的温度
    double t = 0.0;                     // 当前仿真时间
};

} // namespace mhs
```

---

## CompiledExpression

所有表达式（材料属性、BC 参数、热源）预编译后均为 `CompiledExpression` 对象。

```cpp
namespace mhs::expr {

// Move-only 独占类型，每个实例持有独立 ExprTKCompiled（pimpl）
class CompiledExpression {
public:
    CompiledExpression();  // 默认构造为常数 0.0
    ~CompiledExpression();

    // Move-only：每个实例独占 unique_ptr<ExprTKCompiled>，不可复制
    CompiledExpression(CompiledExpression&&) noexcept;
    CompiledExpression& operator=(CompiledExpression&&) noexcept;
    CompiledExpression(const CompiledExpression&) = delete;
    CompiledExpression& operator=(const CompiledExpression&) = delete;

    double eval(const FieldContext& ctx) const;

    bool is_constant() const;
    double constant_value() const;

    static CompiledExpression make_constant(double value);
    static CompiledExpression make_evaluator(std::unique_ptr<ExprTKCompiled> impl);

private:
    std::unique_ptr<ExprTKCompiled> impl_;
    bool is_const_ = false;
    double const_val_ = 0.0;
};

} // namespace mhs::expr
```

特点：

- **Move-only 独占类型**：每个实例持有 `unique_ptr<ExprTKCompiled>`，表达式编译后只传递所有权，不复制
- **无虚函数**：`ExprTKCompiled` 为 pimpl 内部类，对外不可见
- **独占实例**：每个 `CompiledExpression` 拥有独立的 `ExprTKCompiled`，不共享缓存
- **eval() 无锁**：因为实例独立，`eval()` 天然线程安全，无需 mutex

---

## 表达式注册表（模块内部，对外无感）

线程安全的全局注册表，在预处理阶段由 `Preprocessor` 填充：

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

// Get a registered native function
FieldEvaluator get_native(const std::string& name);

// Clear all registered variables, native functions, and user functions
// Called at the start of Preprocessor::load() to reset state
void clear_registry();

// Parse a field expression string — creates a fresh ExprTKCompiled instance (no caching)
CompiledExpression parse(const std::string& formula);

// Evaluate a geometry expression (no context needed, uses registered variables)
double eval_geometry(const std::string& formula);

} // namespace mhs::expr
```

### 使用示例

```cpp
// 预处理阶段（Preprocessor::load() 中）
expr::set_variable("w_top", 10.0);      // mm -> SI 已在 Preprocessor 转换
expr::set_variable("h_middle", 2.0);
expr::register_native("my_piecewise", [](const FieldContext&) { ... });
expr::register_function("test_gaussian", "exp(-((x-x0)^2+(y-y0)^2)/sigma)");

// 求值几何表达式
double half_width = expr::eval_geometry("w_top/2");

// 编译场表达式
CompiledExpression k = expr::parse("k_copper + 0.01*T");
double k_val = k.eval({0.01, 0.02, 0.0, 350.0, 1.0});  // FieldContext 字段顺序
```

---

## 求值上下文生命周期

`CompiledExpression::eval(ctx)` 在以下两个场景被调用：

1. **预处理阶段**（一次性）：编译时做常量折叠
2. **组装阶段**（高频）：每个 Newton 迭代、每个单元调用一次

高频调用时，优先使用常数表达式（`parse("1.5")` 返回常数）避免求值开销。

---

### 线程安全

- `set_variable()`, `register_native()`, `register_function()`: 互斥锁保护
- `parse()`: 编译时读取注册表，互斥锁保护；每次 `parse()` 创建新实例，不做缓存
- `CompiledExpression::eval()`: **无锁**——每个实例持有独立的 `ExprTKCompiled`，各实例天然线程安全。常数表达式（`make_constant`）同样无锁。
- `eval_geometry()`: 互斥锁保护（访问注册表变量）

### 注意事项

- `clear_registry()` 必须在每次 `Preprocessor::load()` 开头调用，以清除上一次运行残留的变量和函数
- `CompiledExpression` 为 move-only 类型——表达式编译后只能通过 `std::move` 传递，不可复制。所有权转移发生在 `MaterialProps`、`BCParamTable`、`CellFields.heat_source` 等内部模型结构中
