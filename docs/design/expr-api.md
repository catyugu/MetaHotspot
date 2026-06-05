# expr 模块接口

封装 exprtk。`src/expr/expr.hpp`。**所有场/BC 表达式**走此模块；几何表达式走 `eval_geometry()` 走同名注册表但语法更窄。

---

## FieldContext 与 FieldEvaluator

**定义**：`mhs::core` (`src/expr/expr.hpp`)。依赖方向 `mhs::sim → mhs::core`，从不超过这个方向。

```cpp
namespace mhs::core {
    struct FieldContext { double x = 0.0, y = 0.0, z = 0.0, T = 0.0, t = 0.0; };
    using FieldEvaluator = std::function<double(const FieldContext&)>;
}
```

## CompiledExpression

轻量句柄。可复制 / 可移动。底层 `shared_ptr<ExprTKCompiledTLS>` 包装 `tbb::enumerable_thread_specific<ExprTKCompiled>`。

```cpp
namespace mhs::core {
    class CompiledExpression {
    public:
        CompiledExpression() = default;                    // = make_constant(0.0)
        CompiledExpression(const CompiledExpression&) = default;
        CompiledExpression& operator=(const CompiledExpression&) = default;
        CompiledExpression(CompiledExpression&&) noexcept = default;
        CompiledExpression& operator=(CompiledExpression&&) noexcept = default;
        ~CompiledExpression();

        double eval(const FieldContext& ctx) const;        // lock-free
        bool   is_constant()   const;
        double constant_value() const;

        static CompiledExpression make_constant(double value);
        static CompiledExpression make_evaluator(const std::string& formula);

    private:
        bool is_const_ = false;
        double const_val_ = 0.0;
        std::shared_ptr<ExprTKCompiledTLS> tls_impl_;
    };
}
```

特点：

- **可复制轻量句柄** — `shared_ptr` 共享公式字符串；可放 `vector<MaterialProps>` / `BCParamTable` / `heat_source_table` 中自由复制移动
- **懒构造 per-thread AST** — 公式字符串在 ETS 构造器中按值捕获；无锁、无 false sharing
- **常数短路** — `is_const_` 为 true 时直接返回 `const_val_`，不触达 `tls_impl_`

## 注册表

线程安全；`Preprocessor::load()` 在每次开始时调 `clear_registry()` 清空。

```cpp
namespace mhs::core {
    // 几何变量（mm/Mm/... 已在 Preprocessor 转换为 SI 米）
    void set_variable(const std::string& name, double value_in_SI);

    // 注册 native C++ 函数。注册后可在字符串公式中调用：
    //   register_native("piecewise_T", [](const FieldContext&){ ... });
    //   parse("piecewise_T(x, y, z, T, t)");
    void register_native(const std::string& name, FieldEvaluator func);

    // 注册用户定义的 exprtk 公式（按名字查）
    void register_function(const std::string& name, const std::string& expression);

    FieldEvaluator get_native(const std::string& name);

    // 清除全部变量、native、user function
    void clear_registry();
}
```

### 使用场景

| 场景         | 示例             | 表达方式 |
| ------------ | ---------------- | -------- |
| 分片常数     | 空间分区常数热源 | native   |
| 分片线性     | 非均匀材料层     | native   |
| 查表函数     | 实验数据表驱动   | native   |
| 复杂几何判断 | 距某点距离函数   | native   |
| 解析公式     | `1e9 + 0.5*x`    | 字符串   |

## 解析与求值

```cpp
namespace mhs::core {
    // 纯数字字面量 → make_constant；否则主线程试编译后返回 make_evaluator 句柄
    CompiledExpression parse(const std::string& formula);

    // 仅依赖已 set_variable 的几何变量；无 FieldContext
    double eval_geometry(const std::string& formula);
}
```

### 典型用法

```cpp
// Preprocessor
mhs::core::clear_registry();
mhs::core::set_variable("w_top", 10.0);
mhs::core::set_variable("h_middle", 2.0);
for (auto& [name, ev] : ios.functions)
    mhs::core::register_native(name, std::move(ev));
mhs::core::register_function("test_gaussian", "exp(-((x-x0)^2+(y-y0)^2)/sigma)");

// 几何
double half_w = mhs::core::eval_geometry("w_top/2");

// 场
auto k = mhs::core::parse("k_copper + 0.01*T");
double v = k.eval({0.01, 0.02, 0.0, 350.0, 1.0});   // (x, y, z, T, t)
```

## 线程安全

| 操作                                                                                      | 同步                           |
| ----------------------------------------------------------------------------------------- | ------------------------------ |
| `set_variable`, `register_native`, `register_function`, `clear_registry`, `eval_geometry` | 互斥锁                         |
| `parse()`                                                                                 | 主线程；持锁做试编译           |
| `CompiledExpression::eval()`                                                              | **无锁**（ETS 每线程独立 AST） |

TBB 并行 `assemble()` 内部：所有工作线程首次 `tls.local()` 时懒构造自己线程专属的 ExprTK AST；之后整个仿真期间零同步。

## 注意事项

- `clear_registry()` 必须在每次 `Preprocessor::load()` 开头调用 — 清除上次运行残留
- `CompiledExpression` 是可复制句柄（`shared_ptr`），容器中无需 `std::move`
- 跨线程共享同一句柄是安全的 — 线程隔离在 ETS 层，不在句柄层
