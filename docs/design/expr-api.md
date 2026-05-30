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

// 值类型，无堆分配，无虚函数（eval 内联）
class CompiledExpression {
public:
    CompiledExpression() : is_const_(true), const_val_(0.0) { }
    CompiledExpression(const CompiledExpression&) = default;
    CompiledExpression(CompiledExpression&&) = default;
    CompiledExpression& operator=(const CompiledExpression&) = default;
    CompiledExpression& operator=(CompiledExpression&&) = default;
    ~CompiledExpression() = default;

    double eval(const FieldContext& ctx) const;

    bool is_constant() const { return is_const_; }
    double constant_value() const { return const_val_; }

    static CompiledExpression make_constant(double value);
    static CompiledExpression make_evaluator(FieldEvaluator eval);

private:
    FieldEvaluator eval_;
    bool is_const_ = false;
    double const_val_ = 0.0;

    CompiledExpression(FieldEvaluator eval, bool is_const, double const_val)
        : eval_(std::move(eval)), is_const_(is_const), const_val_(const_val)
    {
    }
};

} // namespace mhs::expr
```

特点：

- **值类型**：`std::function` + 2 个标量，无堆分配
- **无虚函数**：求值内联，无 vtable 开销
- **可复制/移动**：`= default` 默认行为

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

// Get a registered native function
FieldEvaluator get_native(const std::string& name);

// Clear all registered variables, native functions, and user functions
// Called at the start of Preprocessor::load() to reset state
void clear_registry();

// Parse a field expression string (thread-safe during compilation)
CompiledExpression parse(const std::string& formula);

// Evaluate a geometry expression (no context needed, uses registered variables)
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

### 线程安全

- `set_variable()`, `register_native()`, `register_function()`: 互斥锁保护
- `parse()`: 编译时读取注册表，互斥锁保护
- `CompiledExpression::eval()`: **exprtk 缓存内的表达式有互斥锁保护**（`ExprTKCompiled` 在缓存中为单例，`eval()` 须锁保护共享的 x_/y_/z_/T_/t_ 成员）。常数表达式（`make_constant`）无锁开销。
- `eval_geometry()`: 互斥锁保护（访问注册表变量）

### 注意事项

- `clear_registry()` 必须在每次 `Preprocessor::load()` 开头调用，以清除上一次运行残留的变量和函数
- `ExprTKCompiled` 的 `eval_mutex_` 意味着缓存命中时多个 `CompiledExpression` 对象共享同一个 exprtk 实例，且求值被串行化。这对 TBB 并行组装有性能影响——常数表达式无此问题
