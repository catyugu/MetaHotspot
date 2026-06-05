#include "expr.hpp"
#define exprtk_disable_caseinsensitivity
#include <tbb/enumerable_thread_specific.h>

#include <exprtk/exprtk.hpp>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mhs::expr {

    namespace {
        // Centralized registry avoids boilerplate and multiple static function calls
        struct Registry {
            std::mutex mtx;
            std::unordered_map<std::string, double> variables;
            std::unordered_map<std::string, FieldEvaluator> native_functions;
            std::unordered_map<std::string, std::string> user_functions;

            static Registry& get()
            {
                static Registry instance;
                return instance;
            }
        };

        // Direct thread_local variable avoids the overhead of a function wrapper
        thread_local FieldContext g_thread_ctx {};
    }

    // Adapter for native functions
    class NativeFn : public exprtk::ifunction<double> {
    public:
        explicit NativeFn(FieldEvaluator fe)
            : exprtk::ifunction<double>(1) // 1 argument
            , fe_(std::move(fe))
        {
        }

        double operator()(const double& /*arg0*/) override
        {
            // Read from thread-local context directly
            return fe_(g_thread_ctx);
        }

    private:
        FieldEvaluator fe_;
    };

    class ExprTKCompiled {
    public:
        explicit ExprTKCompiled(const std::string& formula)
        {
            sym_table_.add_variable("x", x_);
            sym_table_.add_variable("y", y_);
            sym_table_.add_variable("z", z_);
            sym_table_.add_variable("T", T_);
            sym_table_.add_variable("t", t_);

            auto& reg = Registry::get();
            {
                std::lock_guard<std::mutex> lock(reg.mtx);
                for (const auto& [name, fe] : reg.native_functions) {
                    // std::list guarantees memory addresses remain stable,
                    // which is required since sym_table_ takes a reference.
                    native_slots_.emplace_back(fe);
                    sym_table_.add_function(name, native_slots_.back());
                }
            }

            expr_.register_symbol_table(sym_table_);

            exprtk::parser<double> parser;
            valid_ = parser.compile(formula, expr_);
        }

        ExprTKCompiled(ExprTKCompiled&&) = default;
        ExprTKCompiled& operator=(ExprTKCompiled&&) = default;

        bool valid() const { return valid_; }

        double eval(const FieldContext& ctx)
        {
            if (!valid_)
                return 0.0;

            x_ = ctx.x;
            y_ = ctx.y;
            z_ = ctx.z;
            T_ = ctx.T;
            t_ = ctx.t;

            g_thread_ctx = ctx; // Update thread-local before evaluation
            return expr_.value();
        }

    private:
        bool valid_ = true;
        double x_ = 0, y_ = 0, z_ = 0, T_ = 0, t_ = 0;

        // Stored as values rather than unique_ptrs to reduce heap indirection
        exprtk::symbol_table<double> sym_table_;
        exprtk::expression<double> expr_;
        std::list<NativeFn> native_slots_;
    };

    struct ExprTKCompiledTLS {
        tbb::enumerable_thread_specific<ExprTKCompiled> tls;

        explicit ExprTKCompiledTLS(const std::string& formula)
            : tls([formula]() { return ExprTKCompiled(formula); }) { }
    };

    CompiledExpression::CompiledExpression() : is_const_(true), const_val_(0.0) { }

    CompiledExpression::~CompiledExpression() = default;

    double CompiledExpression::eval(const FieldContext& ctx) const
    {
        if (is_const_)
            return const_val_;
        return tls_impl_ ? tls_impl_->tls.local().eval(ctx) : 0.0;
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
        auto& reg = Registry::get();
        std::lock_guard<std::mutex> lock(reg.mtx);
        reg.variables[name] = value;
    }

    void register_native(const std::string& name, FieldEvaluator func)
    {
        auto& reg = Registry::get();
        std::lock_guard<std::mutex> lock(reg.mtx);
        reg.native_functions[name] = std::move(func);
    }

    void register_function(const std::string& name, const std::string& expression)
    {
        auto& reg = Registry::get();
        std::lock_guard<std::mutex> lock(reg.mtx);
        reg.user_functions[name] = expression;
    }

    FieldEvaluator get_native(const std::string& name)
    {
        auto& reg = Registry::get();
        std::lock_guard<std::mutex> lock(reg.mtx);
        if (auto it = reg.native_functions.find(name); it != reg.native_functions.end()) {
            return it->second;
        }
        return nullptr;
    }

    void clear_registry()
    {
        auto& reg = Registry::get();
        std::lock_guard<std::mutex> lock(reg.mtx);
        reg.variables.clear();
        reg.native_functions.clear();
        reg.user_functions.clear();
    }

    CompiledExpression parse(const std::string& formula)
    {
        char* end = nullptr;
        double val = std::strtod(formula.c_str(), &end);
        if (end != formula.c_str() && *end == '\0') {
            return CompiledExpression::make_constant(val);
        }

        // Test compilation
        if (ExprTKCompiled test_compile(formula); !test_compile.valid()) {
            return CompiledExpression::make_constant(0.0);
        }

        return CompiledExpression::make_evaluator(formula);
    }

    double eval_geometry(const std::string& formula)
    {
        auto& reg = Registry::get();
        std::lock_guard<std::mutex> lock(reg.mtx);

        if (auto it = reg.variables.find(formula); it != reg.variables.end()) {
            return it->second;
        }

        exprtk::symbol_table<double> sym_table;

        // Add variables as constants to eliminate external referencing and substring matching
        for (const auto& [name, val] : reg.variables) {
            sym_table.add_constant(name, val);
        }

        exprtk::expression<double> exprtk_expr;
        exprtk_expr.register_symbol_table(sym_table);

        exprtk::parser<double> parser;
        if (parser.compile(formula, exprtk_expr)) {
            return exprtk_expr.value();
        }
        return 0.0;
    }

} // namespace mhs::expr