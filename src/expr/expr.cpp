// src/expr/expr.cpp
#include "expr.hpp"
#define exprtk_disable_caseinsensitivity
#include <tbb/enumerable_thread_specific.h>

#include <exprtk/exprtk.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mhs::core {

    namespace detail { // registry state: cross-file internal within mhs::core
        std::mutex& mutex()
        {
            static std::mutex m;
            return m;
        }

        std::unordered_map<std::string, double>& variables()
        {
            static std::unordered_map<std::string, double> vars;
            return vars;
        }

        std::unordered_map<std::string, FieldEvaluator>& native_functions()
        {
            static std::unordered_map<std::string, FieldEvaluator> funcs;
            return funcs;
        }

        std::unordered_map<std::string, std::string>& user_functions()
        {
            static std::unordered_map<std::string, std::string> funcs;
            return funcs;
        }
    } // namespace detail

    template <typename T> class NativeFn : public exprtk::ivararg_function<T> {
    public:
        // 构造时注入当前 TLS 专属的上下文指针
        explicit NativeFn(FieldEvaluator fe, const FieldContext* ctx_ptr)
            : exprtk::ivararg_function<T>() // 启用变长/多参数支持
            , fe_(std::move(fe))
            , ctx_ptr_(ctx_ptr)
        {
        }

        // 当 ExprTk 计算诸如 test(sine(x), y) 时，会先独立算出 sine(x) 和 y，
        // 然后打包成 args 传入这个重载函数。
        T operator()(const std::vector<T>& args) override
        {
            // 直接将 ExprTk 求值的参数与当前的物理上下文结合，传给用户回调
            return static_cast<T>(fe_(args, *ctx_ptr_));
        }

    private:
        FieldEvaluator fe_;
        const FieldContext* ctx_ptr_;
    };

    class ExprTKCompiled {
    public:
        explicit ExprTKCompiled(const std::string& formula)
        {
            using namespace exprtk;

            sym_table_ = std::make_unique<symbol_table<double>>();
            expr_ = std::make_unique<expression<double>>();

            sym_table_->add_variable("x", current_ctx_.x);
            sym_table_->add_variable("y", current_ctx_.y);
            sym_table_->add_variable("z", current_ctx_.z);
            sym_table_->add_variable("T", current_ctx_.T);
            sym_table_->add_variable("t", current_ctx_.t);

            {
                std::lock_guard<std::mutex> lock(detail::mutex());
                for (const auto& [name, fe] : detail::native_functions()) {
                    auto slot = std::make_shared<NativeFn<double>>(fe, &current_ctx_);
                    native_slots_[name] = slot;
                    // 强制路由至独立的变参函数存储区
                    sym_table_->add_reserved_function(name, *slot);
                }
            }

            expr_->register_symbol_table(*sym_table_);

            parser<double> parser;
            valid_ = parser.compile(formula, *expr_);
        }

        // 删除移动和拷贝构造，确保 current_ctx_ 内存地址绝对稳定，防止 ExprTk 野指针
        ExprTKCompiled(const ExprTKCompiled&) = delete;
        ExprTKCompiled& operator=(const ExprTKCompiled&) = delete;
        ExprTKCompiled(ExprTKCompiled&&) = delete;
        ExprTKCompiled& operator=(ExprTKCompiled&&) = delete;

        bool valid() const { return valid_; }

        double eval(const FieldContext& ctx)
        {
            if (!valid_)
                return 0.0;

            // 每次求值前，只更新这一个结构体。
            // ExprTk 的 symbol_table 因为引用绑定，会自动读到最新值；
            // 嵌套的 NativeFn 也会通过指针读到它。
            current_ctx_ = ctx;
            return expr_->value();
        }

    private:
        bool valid_ = true;
        FieldContext current_ctx_; // 统一的 TLS 上下文状态

        std::unique_ptr<exprtk::symbol_table<double>> sym_table_;
        std::unique_ptr<exprtk::expression<double>> expr_;
        std::unordered_map<std::string, std::shared_ptr<NativeFn<double>>> native_slots_;
    };

    // TLS 包装器：确保每个访问表达式的线程都能获得一个独立的 AST。
    // 使用 std::unique_ptr 确保内部的 ExprTKCompiled 在移动时不改变内存地址。
    struct ExprTKCompiledTLS {
        tbb::enumerable_thread_specific<std::unique_ptr<ExprTKCompiled>> tls;

        explicit ExprTKCompiledTLS(const std::string& formula)
            : tls([formula]() { return std::make_unique<ExprTKCompiled>(formula); })
        {
        }
    };

    CompiledExpression::CompiledExpression() : is_const_(true), const_val_(0.0) { }

    CompiledExpression::~CompiledExpression() = default;

    double CompiledExpression::eval(const FieldContext& ctx) const
    {
        if (is_const_)
            return const_val_;
        if (!tls_impl_)
            return 0.0;
        // 无锁获取当前线程的专属 AST 副本进行求值
        return tls_impl_->tls.local()->eval(ctx);
    }

    CompiledExpression CompiledExpression::make_constant(double value)
    {
        CompiledExpression e;
        e.is_const_ = true;
        e.const_val_ = value;
        return e;
    }

    CompiledExpression CompiledExpression::make_evaluator(const std::string& formula)
    {
        CompiledExpression e;
        e.is_const_ = false;
        e.tls_impl_ = std::make_shared<ExprTKCompiledTLS>(formula);
        return e;
    }

    void set_variable(const std::string& name, double value)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        detail::variables()[name] = value;
    }

    void register_native(const std::string& name, FieldEvaluator func)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        detail::native_functions()[name] = std::move(func);
    }

    void register_function(const std::string& name, const std::string& expression)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        detail::user_functions()[name] = expression;
    }

    FieldEvaluator get_native(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        auto it = detail::native_functions().find(name);
        if (it != detail::native_functions().end()) {
            return it->second;
        }
        return nullptr;
    }

    void clear_registry()
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        detail::variables().clear();
        detail::native_functions().clear();
        detail::user_functions().clear();
    }

    CompiledExpression parse(const std::string& formula)
    {
        char* end = nullptr;
        double val = std::strtod(formula.c_str(), &end);
        if (end != formula.c_str() && *end == '\0') {
            return CompiledExpression::make_constant(val);
        }

        // 主线程进行一次试编译，尽早捕获语法错误
        {
            ExprTKCompiled test_compile(formula);
            if (!test_compile.valid()) {
                return CompiledExpression::make_constant(0.0);
            }
        }
        return CompiledExpression::make_evaluator(formula);
    }

    double eval_geometry(const std::string& formula)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        const auto& vars = detail::variables();

        auto var_it = vars.find(formula);
        if (var_it != vars.end()) {
            return var_it->second;
        }

        using namespace exprtk;
        expression<double> exprtk_expr;
        symbol_table<double> sym_table;

        std::vector<std::pair<std::string, double>> active_vars;
        active_vars.reserve(vars.size());

        for (const auto& [name, val] : vars) {
            if (formula.find(name) != std::string::npos) {
                active_vars.emplace_back(name, val);
                sym_table.add_variable(active_vars.back().first, active_vars.back().second);
            }
        }

        exprtk_expr.register_symbol_table(sym_table);
        parser<double> parser;
        if (parser.compile(formula, exprtk_expr)) {
            return exprtk_expr.value();
        }
        return 0.0;
    }

} // namespace mhs::core