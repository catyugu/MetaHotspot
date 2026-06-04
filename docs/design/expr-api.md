# expr 模块接口

表达式解析与求值模块。处理所有场/边界条件表达式，上下文为 `{x, y, z, T, t}`。

---

## FieldContext

`FieldContext` 是 expr 引擎向上层暴露的数据契约：调用方在求值时必须以这个结构提供 `(x, y, z, T, t)`。**类型定义在 `src/expr/expr.hpp`，命名空间 `mhs::expr`**；`src/common/types.hpp` 通过 `using` 别名将同名符号重新导出到 `mhs` 根命名空间，供 assembler/preprocessor 等上层代码直接使用而不必每次写 `expr::FieldContext`。

```cpp
// 定义位置：src/expr/expr.hpp
namespace mhs::expr {

struct FieldContext {
    double x = 0.0;  // 空间位置（SI 单位）
    double y = 0.0;
    double z = 0.0;
    double T = 0.0;  // 该位置的温度
    double t = 0.0;  // 当前仿真时间
};

using FieldEvaluator = std::function<double(const FieldContext&)>;

} // namespace mhs::expr

// 重新导出位置：src/common/types.hpp
namespace mhs {
    using FieldContext = expr::FieldContext;
    using FieldEvaluator = expr::FieldEvaluator;
    // ...
} // namespace mhs
```

---

## CompiledExpression

所有表达式（材料属性、BC 参数、热源）预编译后均为 `CompiledExpression` 对象。

```cpp
namespace mhs::expr {

// 可复制 / 可移动的轻量句柄
// 内部通过 shared_ptr<ExprTKCompiledTLS> 共享一份公式字符串，
// 但每个线程通过 tbb::enumerable_thread_specific 自动获得独立 ExprTK AST
class CompiledExpression {
public:
    CompiledExpression();  // 默认构造为常数 0.0
    ~CompiledExpression();

    // 浅复制：仅复制 shared_ptr，AST 在首次 eval 时按需懒构造
    CompiledExpression(const CompiledExpression&) = default;
    CompiledExpression& operator=(const CompiledExpression&) = default;
    CompiledExpression(CompiledExpression&&) noexcept = default;
    CompiledExpression& operator=(CompiledExpression&&) noexcept = default;

    double eval(const FieldContext& ctx) const;

    bool is_constant() const;
    double constant_value() const;

    // 工厂：构造常数表达式（无锁、无 AST 分配）
    static CompiledExpression make_constant(double value);

    // 工厂：构造求值器（懒构造 per-thread ExprTK AST）
    static CompiledExpression make_evaluator(const std::string& formula);

private:
    bool is_const_ = false;
    double const_val_ = 0.0;
    std::shared_ptr<ExprTKCompiledTLS> tls_impl_;
};

} // namespace mhs::expr
```

特点：

- **可复制 / 可移动的轻量句柄**：`shared_ptr` 共享公式字符串的句柄，单份公式在容器中（`vector<MaterialProps>`、`BCParamTable.*`）只产生一次字符串存储
- **懒构造 per-thread AST**：底层 `tbb::enumerable_thread_specific<ExprTKCompiled>` 内部按需为每个访问线程构造独立的 ExprTK AST（formula 字符串作为构造参数捕获），无锁、无 false sharing
- **无虚函数**：`ExprTKCompiled` 与 `ExprTKCompiledTLS` 均为 pimpl 内部类，对外不可见
- **eval() 无锁**：通过 `tls.local()` 取得当前线程的私有 AST，`x_/y_/z_/T_/t_` 符号槽仅由本线程读写，无需任何同步
- **常数表达式短路**：`is_const_` 为 true 时直接返回 `const_val_`，不触达 `tls_impl_`

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

// Parse a field expression string.
// 若 formula 为纯数字字面量（如 "1.5"），返回 make_constant 短路结果；
// 否则先在主线程做一次试编译（尽早暴露语法错误），再返回 make_evaluator 句柄。
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

1. **预处理阶段**（一次性）：`parse()` 内部做主线程试编译以捕获语法错误
2. **组装阶段**（高频）：每个迭代、每个单元调用一次 —— 此时由 TBB 工作线程首次访问 `tls.local()`，触发该线程的 ExprTK AST 懒构造

高频调用时，优先使用常数表达式（`parse("1.5")` 返回常数）避免求值开销。

---

### 线程安全

- `set_variable()`, `register_native()`, `register_function()`, `clear_registry()`, `eval_geometry()`: 互斥锁保护
- `parse()`: 主线程调用；持有 registry 互斥锁读取变量表以做语法试编译；每次 `parse()` 返回新的 `CompiledExpression` 句柄（共享同一 `shared_ptr<ExprTKCompiledTLS>` 等价于共享公式字符串）
- `CompiledExpression::eval()`: **无锁**。`shared_ptr<ExprTKCompiledTLS>::tls.local()` 取得当前线程的私有 `ExprTKCompiled`，该 AST 的 `x_/y_/z_/T_/t_` 符号槽仅由本线程读写。常量表达式（`make_constant`）同样无锁
- TBB 并行 for 内部：所有工作线程首次访问同一 `CompiledExpression` 时，各自懒构造自己的 ExprTK AST；之后整个模拟期间零同步开销

### 实现原理（TBB enumerable_thread_specific）

`ExprTKCompiledTLS` 是 expr 模块内部的 pimpl 包装：

```cpp
// 内部实现，对外不可见
struct ExprTKCompiled {
    // 持有 exprtk::symbol_table + expression + x_/y_/z_/T_/t_ 槽位
    // 构造时绑定 formula，编译失败 valid_ = false
};

struct ExprTKCompiledTLS {
    tbb::enumerable_thread_specific<ExprTKCompiled> tls;
    explicit ExprTKCompiledTLS(const std::string& formula)
        : tls([formula]() { return ExprTKCompiled(formula); }) {}
};
```

- 公式字符串在 `tbb::enumerable_thread_specific` 的 lambda 构造器中被按值捕获一次
- 每个 TBB 工作线程首次 `tls.local()` 时按公式构造独立 AST
- 线程退出时 ETS 自动析构其专属 AST —— 无显式清理代码
- 多个 `CompiledExpression` 持有同一公式时（即 `parse()` 同一字符串两次），由于各自持有独立 `shared_ptr<ExprTKCompiledTLS>`，TBB 仍为各线程各构造一份 AST；如需真正跨句柄共享 AST，可改造为进程级 `unordered_map<formula, shared_ptr<ExprTKCompiledTLS>>` 缓存 —— 当前实现优先保证单次 `parse()` 句柄的独立性

### 注意事项

- `clear_registry()` 必须在每次 `Preprocessor::load()` 开头调用，以清除上一次运行残留的变量和函数
- `CompiledExpression` 现在是**可复制**的轻量句柄（`shared_ptr`），可在 `vector`、`MaterialProps`、`BCParamTable` 中自由复制与移动，无需 `std::move`
- 跨线程共享同一 `CompiledExpression` 句柄是安全的 —— 线程隔离发生在 ETS 层而非句柄层
