# expr 模块接口

封装 muparser。`src/expr/expr.hpp`。**所有场/BC 表达式**走此模块；几何表达式走 `eval_geometry()`，参数与 `parse()` 共享同一个 `SymbolTable` 但语法更窄（仅依赖 `variables` map）。

---

## FieldContext 与 FieldEvaluator

**定义**：`mhs::core` (`src/expr/expr.hpp`)。依赖方向 `mhs::sim → mhs::core`，从不超过这个方向。

```cpp
namespace mhs::core {
    struct FieldContext { double x = 0.0, y = 0.0, z = 0.0, T = 0.0, t = 0.0; };
    // args = muparser 已先行求值好的实参列表（如 fn(a, b) 中的 a, b）；
    // ctx  = 当前物理上下文（x, y, z, T, t 的真实值），供 native 参考。
    using FieldEvaluator = std::function<double(const double* args, int nargs,
                                               const FieldContext& ctx)>;
}
```

## CompiledExpression

轻量句柄。可复制 / 可移动。底层 `shared_ptr<MuCompiledTLS>` 包装 `tbb::enumerable_thread_specific<std::unique_ptr<MuCompiled>>`——用 `unique_ptr` 包住 AST 是为了锁住 `MuCompiled` 的内存地址（`NativeFnCtx` 通过裸指针引用其 `current_ctx_` 槽，必须地址稳定）。

`MuCompiled` 自身禁用拷贝/移动；构造时 `current_ctx_` 入栈、muparser 通过 `DefineVar` 按指针绑定 `x/y/z/T/t`，之后每次 `eval(ctx)` 仅覆写 `current_ctx_`，muparser 在 `Eval()` 中自动读到新值。

`MuCompiledTLS` 在构造时按值持有 `SymbolTable` 副本，闭包到 `MuCompiled` 的 `tbb::enumerable_thread_specific` 构造器 lambda 中 — 所以 `CompiledExpression` 一旦构造完成，对全局状态零依赖，可安全跨线程共享。

```cpp
namespace mhs::core {
    class CompiledExpression {
    public:
        CompiledExpression();                          // = make_constant(0.0)
        CompiledExpression(const CompiledExpression&) = default;
        CompiledExpression& operator=(const CompiledExpression&) = default;
        CompiledExpression(CompiledExpression&&) noexcept = default;
        CompiledExpression& operator=(CompiledExpression&& other) noexcept = default;
        ~CompiledExpression();

        double eval(const FieldContext& ctx) const;        // lock-free
        bool   is_constant()   const;
        double constant_value() const;

        static CompiledExpression make_constant(double value);
        static CompiledExpression make_evaluator(const std::string& formula,
                                                 const SymbolTable& symbols);

    private:
        bool is_const_ = false;
        double const_val_ = 0.0;
        std::shared_ptr<MuCompiledTLS> tls_impl_;
    };
}
```

特点：

- **可复制轻量句柄** — `shared_ptr` 共享公式字符串 + SymbolTable 副本 + ETS 基础设施；可放 `vector<MaterialProps>` / `BCParamTable` / `heat_source_table` 中自由复制移动
- **懒构造 per-thread AST** — 公式字符串与 `SymbolTable` 在 ETS 构造器中按值捕获；无锁、无 false sharing
- **AST 地址稳定** — ETS 元素类型是 `unique_ptr<MuCompiled>`，移动 `MuCompiledTLS` 不会搬动内部 AST；`NativeFnCtx` 持有的 `FieldContext*` 永远有效
- **常数短路** — `is_const_` 为 true 时直接返回 `const_val_`，不触达 `tls_impl_`

## SymbolTable

`SymbolTable` 是 `parse()` / `eval_geometry()` 的显式输入。值类型，按值传递 / 持有；持有一份**快照**而非引用，所以 `CompiledExpression` 构造后 `SymbolTable` 本身的生命周期与 `CompiledExpression` 解耦。

```cpp
namespace mhs::core {
    struct SymbolTable {
        std::unordered_map<std::string, double> variables;     // 几何变量（SI 米）
        std::unordered_map<std::string, FieldEvaluator> natives; // 注册 C++ 函数
    };
}
```

- `variables` — 在 `eval_geometry()` 中被 substring 扫描 + `DefineVar` 绑定；在 `parse()` 中**不参与**绑定（场/BC 公式只引用 `x/y/z/T/t`）。
- `natives` — 在 `parse()` 时通过 `DefineFunUserData()` 绑进每个线程的 muparser 实例；闭包副本在 AST 析构时随 `MuCompiledTLS` 释放。

Native 内部桥接：模块以 `mu::multfun_userdata_type` (`value_type(*)(void*, const value_type*, int)`) 注册到 muparser 的 `DefineFunUserData()` 槽，muparser 调用时先把所有实参独立求值，再以 `double* + int` 形式回调 `native_fn_bridge` 静态函数，连同当前 TLS `FieldContext*` 一起转发给用户 `FieldEvaluator`。

### 使用场景

| 场景         | 示例             | 表达方式 |
|--------------|------------------|----------|
| 分片常数     | 空间分区常数热源 | native   |
| 分片线性     | 非均匀材料层     | native   |
| 查表函数     | 实验数据表驱动   | native   |
| 复杂几何判断 | 距某点距离函数   | native   |
| 解析公式     | `1e9 + 0.5*x`    | 字符串   |

## 解析与求值

```cpp
namespace mhs::core {
    // 纯数字字面量 → make_constant；否则主线程试编译后返回 make_evaluator 句柄
    CompiledExpression parse(const std::string& formula, const SymbolTable& symbols);

    // 仅依赖 symbols.variables 中的几何变量；无 FieldContext
    double eval_geometry(const std::string& formula, const SymbolTable& symbols);
}
```

### 典型用法

```cpp
// build_model：构造本地 SymbolTable 贯穿 setup 路径
mhs::core::SymbolTable symbols;
mhs::sim::register_all_functions(symbols, ios.functions);  // typed Function → FieldEvaluator

// 几何
double half_w = mhs::core::eval_geometry("w_top/2", symbols);

// 场
auto k = mhs::core::parse("k_copper + 0.01*T", symbols);
double v = k.eval({0.01, 0.02, 0.0, 350.0, 1.0});   // (x, y, z, T, t)
```

## 线程安全

| 操作                                                     | 同步                           |
|----------------------------------------------------------|--------------------------------|
| `parse()` / `eval_geometry()` / `register_all_functions` | 主线程，构造本地 `SymbolTable` |
| `CompiledExpression::eval()`                             | **无锁**（ETS 每线程独立 AST） |

`SymbolTable` 是 setup 期间单线程构造的本地值；运行时（`CompiledExpression::eval`）无任何全局状态访问，因此多个 `build_model()` 调用互不干扰。

TBB 并行 `assemble()` 内部：所有工作线程首次 `tls.local()` 时懒构造自己线程专属的 muparser 实例；之后整个仿真期间零同步。

## 注意事项

- `SymbolTable` 按值传递；构造 `CompiledExpression` 时 `parse()` 内部复制一份到 `MuCompiledTLS`，与原 SymbolTable 解耦。
- `CompiledExpression` 是可复制句柄（`shared_ptr`），容器中无需 `std::move`
- 跨线程共享同一句柄是安全的 — 线程隔离在 ETS 层，不在句柄层
- 不存在 `set_variable` / `register_native` / `get_native` / `clear_registry` 等全局 API — 全部数据走 `SymbolTable`
