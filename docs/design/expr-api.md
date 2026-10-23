# expr 模块接口

封装 muparser。`include/metahotspot/numerics/expression/expr.hpp`。**所有场/BC 表达式**走此模块；几何表达式走 `eval_geometry()`，与 `parse()` 接收同一类 `SymbolTable`，但只绑定 `variables`，不绑定场上下文或 native 函数。

---

## FieldContext 与 FieldEvaluator

**定义**：`mhs::core` (`include/metahotspot/numerics/expression/expr.hpp`)。依赖方向 `mhs::sim → mhs::core`，从不超过这个方向。

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

`MuCompiled` 自身禁用拷贝/移动；每个实例持有自己的 `current_ctx_`，muparser 通过 `DefineVar` 按指针绑定 `x/y/z/T/t`，之后每次 `eval(ctx)` 仅覆写 `current_ctx_`，muparser 在 `Eval()` 中自动读到新值。

`MuCompiledTLS` 按值持有 `SymbolTable` 副本。ETS 构造器 lambda 按值捕获公式，通过 `this` 访问该副本；共享句柄维持 `MuCompiledTLS` 的生命周期。构造时立即调用 `tls.local()` 完成当前线程的解析和验证，后续求值复用该解析器。其他线程在首次求值时构造各自的解析器。

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

        double eval(const FieldContext& ctx) const;        // per-thread parser
        bool   is_constant()   const;
        double constant_value() const;

        static CompiledExpression make_constant(double value);
        static CompiledExpression make_evaluator(const std::string& formula,
                                                 const SymbolTable& symbols);

    private:
        std::variant<double, std::shared_ptr<MuCompiledTLS>> value_ {0.0};
    };
}
```

特点：

- **可复制轻量句柄** — `shared_ptr` 共享公式字符串 + SymbolTable 副本 + ETS 基础设施；可放 `vector<MaterialProps>` / `BoundaryCondition` / `heat_source_table` 中自由复制移动
- **懒构造 per-thread AST** — 公式字符串按值捕获，`SymbolTable` 由 `MuCompiledTLS` 持有；创建线程立即初始化，其余线程首次使用时初始化
- **AST 地址稳定** — ETS 元素类型是 `unique_ptr<MuCompiled>`，复制或移动句柄不搬动内部 AST；`NativeFnCtx` 持有的 `FieldContext*` 永远有效
- **常数短路** — `value_` 持有 `double` 时直接返回该常数，不访问 TLS

## SymbolTable

`SymbolTable` 是 `parse()` / `eval_geometry()` 的显式输入，以 const 引用传入。非恒定 `CompiledExpression` 的实现保存符号表的值副本，其生命周期与原符号表解耦。

```cpp
namespace mhs::core {
    struct SymbolTable {
        std::unordered_map<std::string, double> variables;     // 几何变量，长度按输入单位
        std::unordered_map<std::string, FieldEvaluator> natives; // 注册 C++ 函数
    };
}
```

- `variables` — 在 `eval_geometry()` 中被 substring 扫描 + `DefineVar` 绑定；在 `parse()` 中**不参与**绑定（场/BC 公式只引用 `x/y/z/T/t`）。
- `natives` — 在各线程解析器构造时通过 `DefineFunUserData()` 绑定；每个 `MuCompiled` 的预留 vector 持有 `NativeFnCtx` 和闭包副本，`MuCompiledTLS` 另持有符号表快照。

Native 内部桥接：模块以 `mu::multfun_userdata_type` (`value_type(*)(void*, const value_type*, int)`) 注册到 muparser 的 `DefineFunUserData()` 槽，muparser 调用时先把所有实参独立求值，再以 `double* + int` 形式回调 `native_fn_bridge` 静态函数，连同当前 TLS `FieldContext*` 一起转发给用户 `FieldEvaluator`。

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
    // 空串或纯数字字面量 → make_constant；否则初始化当前线程解析器并返回句柄
    CompiledExpression parse(const std::string& formula, const SymbolTable& symbols);

    // 仅依赖 symbols.variables 中的几何变量；无 FieldContext
    double eval_geometry(const std::string& formula, const SymbolTable& symbols);
}
```

### 典型用法

```cpp
// build_model：构造本地 SymbolTable 贯穿 setup 路径
mhs::core::SymbolTable symbols;
mhs::sim::register_all_functions(symbols, definition.functions);  // typed Function → FieldEvaluator

// 几何：此例的变量值以米表示
symbols.variables["w_top"] = 0.02;
double half_w = mhs::core::eval_geometry("w_top/2", symbols);

// 场
auto k = mhs::core::parse("400 + 0.01*T", symbols);
double v = k.eval({0.01, 0.02, 0.0, 350.0, 1.0});   // (x, y, z, T, t)
```

## 线程安全

| 操作                                                     | 同步                                   |
| -------------------------------------------------------- | -------------------------------------- |
| `parse()` / `eval_geometry()` / `register_all_functions` | setup 阶段使用各自的本地 `SymbolTable` |
| `CompiledExpression::eval()`                             | 每线程独立 AST，无应用层互斥锁         |

`SymbolTable` 是 setup 期间单线程构造的本地值；运行时（`CompiledExpression::eval`）无任何全局状态访问，因此多个 `build_model()` 调用互不干扰。

TBB 并行 `assemble()` 内部：所有工作线程首次 `tls.local()` 时懒构造自己线程专属的 muparser 实例；之后复用该线程的解析器；常数句柄直接返回数值。

## 注意事项

- 非恒定句柄的 `MuCompiledTLS` 保存 `SymbolTable` 副本；常数句柄只保存一个 `double`。
- `CompiledExpression` 可复制或移动；非恒定句柄的副本共享 `MuCompiledTLS`。
- 跨线程共享同一句柄是安全的 — 线程隔离在 ETS 层，不在句柄层
- 不存在 `set_variable` / `register_native` / `get_native` / `clear_registry` 等全局 API — 全部数据走 `SymbolTable`
